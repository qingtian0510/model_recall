import tensorflow as tf
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from dssm_v2.utils.layer_utils import (
    softmax_loss,
    mlp_net,
    multi_head_attention_net,
    fixed_size_emb_lookup_cl,
    add_time_weight_to_loss_for_bottom_page,
)
from dssm_v2.utils.train_utils import LOG, fully_connect, get_optimizer


def get_model_fn(user_ids, item_ids, filter_default=False, distributed=False):
    def model_fn(features, labels, mode=None, params=None):
        LOG("*" * 50, "focus_page_multi_interest")
        learning_rate = params.learning_rate
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        user_side_embs, item_side_embs, user_item_key_ids_dict = fixed_size_emb_lookup_cl(
            user_ids=user_ids,
            item_ids=item_ids,
            features=features,
            filter_default=filter_default,
            params=params,
            cl_dropout_rate=0.0,
        )
        user_side_embs_cl, item_side_embs_cl, user_item_key_ids_dict_cl = fixed_size_emb_lookup_cl(
            user_ids=user_ids,
            item_ids=item_ids,
            features=features,
            filter_default=filter_default,
            params=params,
            cl_dropout_rate=params.cl_dropout_rate,
        )

        LOG("user_side_embs, item_side_embs", user_side_embs, item_side_embs)
        if bool(params.use_feature_mha):
            with tf.variable_scope("mha_embs_layer", reuse=tf.AUTO_REUSE):
                user_side_embs = multi_head_attention_net(
                    input_x=user_side_embs,
                    name_scope="user_mha",
                    is_training=is_training,
                    drop_rate=params.drop_rate,
                    heads=params.attention_heads,
                    batch_norm=params.batch_norm,
                )

                item_side_embs = multi_head_attention_net(
                    input_x=item_side_embs,
                    name_scope="item_mha",
                    is_training=is_training,
                    drop_rate=params.drop_rate,
                    heads=params.attention_heads,
                    batch_norm=params.batch_norm,
                )
        #
        # user_side_mha_out:[None,user_side_feature_num,dense_dim]
        # flatten it
        user_side_embs = tf.layers.flatten(
            user_side_embs, name="user_side_embs_flatten"
        )
        item_side_embs = tf.layers.flatten(
            item_side_embs, name="item_side_embs_flatten"
        )
        user_side_embs_cl = tf.layers.flatten(
            user_side_embs_cl, name="user_side_embs_flatten_cl"
        )
        item_side_embs_cl = tf.layers.flatten(
            item_side_embs_cl, name="item_side_embs_flatten_cl"
        )
        if is_training:
            cl_dropout_rate = params.cl_dropout_rate
            user_side_embs_cl = tf.nn.dropout(user_side_embs_cl, rate=cl_dropout_rate, name="user_side_embs_cl_drop_{}".format(cl_dropout_rate))
            item_side_embs_cl = tf.nn.dropout(item_side_embs_cl, rate=cl_dropout_rate, name="item_side_embs_cl_drop_{}".format(cl_dropout_rate))


        # item_side_embs = tf.layers.flatten(item_side_embs)
        LOG("user_side_mha_out,item_side_mha_out", user_side_embs, item_side_embs)

        with tf.variable_scope("dense_layers", reuse=tf.AUTO_REUSE):
            user_side_mlp_out = mlp_net(
                input_x=user_side_embs,
                name_scope="user_side_mlp",
                layer_nums=params.user_dnn_layers,
                is_training=is_training,
                drop_rate=params.drop_rate,
                batch_norm=params.batch_norm,
                activation=tf.nn.swish,
            )

            item_side_mlp_out = mlp_net(
                input_x=item_side_embs,
                name_scope="item_side_mlp",
                layer_nums=params.item_dnn_layers,
                is_training=is_training,
                drop_rate=params.drop_rate,
                batch_norm=params.batch_norm,
                activation=tf.nn.swish,
            )
            user_side_mlp_out_cl = mlp_net(
                input_x=user_side_embs_cl,
                name_scope="user_side_mlp",
                layer_nums=params.user_dnn_layers,
                is_training=is_training,
                drop_rate=params.drop_rate,
                batch_norm=params.batch_norm,
                activation=tf.nn.swish,
            )

            item_side_mlp_out_cl = mlp_net(
                input_x=item_side_embs_cl,
                name_scope="item_side_mlp",
                layer_nums=params.item_dnn_layers,
                is_training=is_training,
                drop_rate=params.drop_rate,
                batch_norm=params.batch_norm,
                activation=tf.nn.swish,
            )

        LOG(
            "user_side_mlp_out, item_side_mlp_out", user_side_mlp_out, item_side_mlp_out
        )
        # user_side_mlp_out:[None,mlp_net_last_dim]
        # keep emb size same
        # out_dim = params.output_size * params.recall_heads
        # LOG('recall_heads,output_size', params.output_size, params.recall_heads)
        with tf.variable_scope("fully_connect_out", reuse=tf.AUTO_REUSE):
            user_side_fc_out = fully_connect(
                user_side_mlp_out,
                params.output_size,
                is_training=is_training,
                axis=1,
                name="user_side_fc",
                activation=tf.identity,
            )

            item_side_fc_out = fully_connect(
                item_side_mlp_out,
                params.output_size,
                is_training=is_training,
                axis=1,
                name="item_side_fc",
                activation=tf.identity,
            )
            LOG("user_side_fc_out,item_side_fc_out", user_side_fc_out, item_side_fc_out)
            user_side_fc_out_cl = fully_connect(
                user_side_mlp_out_cl,
                params.output_size,
                is_training=is_training,
                axis=1,
                name="user_side_fc",
                activation=tf.identity,
            )

            item_side_fc_out_cl = fully_connect(
                item_side_mlp_out_cl,
                params.output_size,
                is_training=is_training,
                axis=1,
                name="item_side_fc",
                activation=tf.identity,
            )

        with tf.name_scope("dot_predict"):
            if params.sim_type == "cosine":
                user_side_out = tf.nn.l2_normalize(
                    user_side_fc_out, axis=-1, epsilon=1e-6, name="user_side_out"
                )
                item_side_out = tf.nn.l2_normalize(
                    item_side_fc_out, axis=-1, epsilon=1e-6, name="item_side_out"
                )
                user_side_out_cl = tf.nn.l2_normalize(
                    user_side_fc_out_cl, axis=-1, epsilon=1e-6, name="user_side_out_cl"
                )
                item_side_out_cl = tf.nn.l2_normalize(
                    item_side_fc_out_cl, axis=-1, epsilon=1e-6, name="item_side_out_cl"
                )

            if params.loss_type == "softmax":
                LOG("MODEL", "SOFTMAX", "params.log_tau:{}".format(params.log_tau))
                if mode == tf.estimator.ModeKeys.TRAIN:
                    item_freq = features["item_freq"]
                else:
                    item_freq = None
                if bool(params.mask_false_negative):
                    assert user_item_key_ids_dict.get("user_history", None) is not None
                    assert user_item_key_ids_dict.get("item_newsid", None) is not None
                    loss, predictions = softmax_loss(
                        user_side=user_side_out,
                        item_side=item_side_out,
                        log_tau=params.log_tau,
                        freq_weight=params.freq_weight,
                        user_history=user_item_key_ids_dict["user_history"],
                        item_newsid=user_item_key_ids_dict["item_newsid"],
                        item_freq=item_freq,
                        is_training=is_training,
                    )
                else:
                    loss, predictions = softmax_loss(
                        user_side=user_side_out,
                        item_side=item_side_out,
                        item_newsid=user_item_key_ids_dict["item_newsid"],
                        log_tau=params.log_tau,
                        freq_weight=params.freq_weight,
                        item_freq=item_freq,
                    )
            else:
                raise NotImplementedError("loss function not implement")

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions["user_id"] = features["user_id"]
            predictions["item_newsid"] = features["item_newsid"]
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        if params.add_time_weight.lower() == "true":
            LOG("ADD", "TIME_WEIGHT", "add_time_weight_to_loss_for_bottom_page")
            loss = add_time_weight_to_loss_for_bottom_page(
                loss=loss, label=labels, watch_time=features["watch_time"], greater_30_weight=params.greater_30_weight
            )
        # 只取正样本
        watch_time = tf.identity(features["watch_time"], name="watch_time")
        watch_time_th = tf.cast(tf.greater_equal(watch_time, params.watch_time_th), tf.float32, name="watch_time_th")
        labels = tf.identity(labels, name="label")
        labels = labels * watch_time_th
        loss = loss * labels
        # loss = tf.reduce_mean(loss)
        loss = tf.divide(
            tf.reduce_sum(loss), tf.reduce_sum(labels) + 1e-10, name="loss"
        )

        with tf.name_scope("contrastive_learning"):
            # ===============================contrastive loss ===========================
            if params.cl_u2i_rate > 0:
                u2i_cl_loss_1, u2i_cl_predictions_1 = softmax_loss(
                    user_side=user_side_out_cl,
                    item_side=item_side_out,
                    log_tau=params.log_tau,
                    freq_weight=params.freq_weight,
                    user_history=user_item_key_ids_dict["user_history"],
                    item_newsid=user_item_key_ids_dict["item_newsid"],
                    item_freq=item_freq,
                    is_training=is_training,
                )
                u2i_cl_loss_2, u2i_cl_predictions_2 = softmax_loss(
                    user_side=user_side_out,
                    item_side=item_side_out_cl,
                    log_tau=params.log_tau,
                    freq_weight=params.freq_weight,
                    user_history=user_item_key_ids_dict["user_history"],
                    item_newsid=user_item_key_ids_dict["item_newsid"],
                    item_freq=item_freq,
                    is_training=is_training,
                )
                u2i_cl_loss = (u2i_cl_loss_1 + u2i_cl_loss_2) * labels
                u2i_cl_loss = tf.divide(tf.reduce_sum(u2i_cl_loss), tf.reduce_sum(labels) + 1e-10, name="u2i_cl_loss")
                LOG("u2u_cl_loss: {}, weight: {}".format(u2i_cl_loss, params.cl_u2i_rate))
                loss += u2i_cl_loss * params.cl_u2i_rate
            if params.cl_u2u_rate > 0:
                u2u_cl_loss, u2u_cl_predictions = softmax_loss(
                    user_side=user_side_out,
                    item_side=user_side_out_cl,
                    log_tau=params.log_tau,
                    freq_weight=params.freq_weight,
                    user_history=None,
                    item_newsid=features["user_id"],
                    item_freq=None,
                    is_training=is_training,
                )
                u2u_cl_loss = tf.reduce_mean(u2u_cl_loss, name="u2u_cl_loss")
                LOG("u2u_cl_loss: {}, weight: {}".format(u2u_cl_loss, params.cl_u2u_rate))
                loss += u2u_cl_loss * params.cl_u2u_rate
            if params.cl_i2i_rate > 0:
                i2i_cl_loss, i2i_cl_predictions = softmax_loss(
                    user_side=item_side_out_cl,
                    item_side=item_side_out,
                    log_tau=params.log_tau,
                    freq_weight=params.freq_weight,
                    user_history=None,
                    item_newsid=user_item_key_ids_dict["item_newsid"],
                    item_freq=None,
                    is_training=is_training,
                )
                i2i_cl_loss = tf.reduce_mean(i2i_cl_loss, name="i2i_cl_loss")
                LOG("i2i_cl_loss: {}, weight: {}".format(i2i_cl_loss, params.cl_i2i_rate))
                loss += i2i_cl_loss * params.cl_i2i_rate
            # ===============================contrastive loss ===========================

        auc_op = tf.metrics.auc(labels, predictions["score"])
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {"auc": auc_op}
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
            )

        optimizer = get_optimizer(
            optimizer_name=params.optimizer, learning_rate=learning_rate
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_or_create_global_step()
            )

        # only for distributed training
        if distributed:
            return train_op, loss, auc_op

        # Provide an estimator spec for `ModeKeys.TRAIN` modes
        if mode == tf.estimator.ModeKeys.TRAIN:
            tensor_to_log = {
                "loss": "loss",
                "contrastive_learning/u2i_cl_loss": "contrastive_learning/u2i_cl_loss",
                "contrastive_learning/i2i_cl_loss": "contrastive_learning/i2i_cl_loss",
                "contrastive_learning/u2u_cl_loss": "contrastive_learning/u2u_cl_loss",
                # "watch_time_th": "watch_time_th",
                # "label": "label",
                # "watch_time": "watch_time",
                # "greater_30": "greater_30",
            }
            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensor_to_log, every_n_iter=100
            )
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook]
            )
        return train_op, loss, auc_op

    return model_fn
