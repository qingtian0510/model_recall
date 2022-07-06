import json
import math

import numpy as np
import tensorflow as tf

from dssm_v2.utils.train_utils import LOG, fully_connect


def attention_layer(query, key, value, is_training, mask=None, drop_rate=0.0):
    """
    compute 'Scaled dot product Attention'
    :param query(tf.tensor): of shape(h*batch,q_size,d)
    :param key(tf.tensor):of shape(h*batch,k_size,d)
    :param value(tf.tensor):of shape(h*batch,k_size,d_v)
    :param mask(tf.tensor): of shape(h*batch,q_size,k_size)
    :param drop_rate: dropout rate
    :return: tf.tensor (batch,q_size,d_v)
    """

    d_k = query.get_shape().as_list()[-1] * 1.0
    # print(d_k)
    scores = tf.matmul(query, key, transpose_b=True) / math.sqrt(d_k)
    if mask is not None:
        scores = tf.multiply(scores, mask) + (1.0 - mask) * (1e-10)
    scores = tf.nn.softmax(scores, axis=-1)
    if is_training:
        scores = tf.nn.dropout(scores, rate=drop_rate)
    out = tf.matmul(scores, value)
    return out


def multi_head_attention(
    queries,
    keys,
    values,
    key_masks,
    is_training,
    drop_rate=0.0,
    num_heads=8,
    scope="multi_head_attention",
):
    """

    :param queries:[batch_size,len,dim] 3d tensor
    :param keys:
    :param values:
    :param key_masks:
    :param is_training:
    :param drop_rate:
    :param scope:
    :return:
    """

    q_dim = queries.get_shape().as_list()[-1]
    d_model = q_dim * num_heads
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N,t_q,d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True)  # (N,t_k,d_model)
        V = tf.layers.dense(values, d_model, use_bias=True)  # (N,t_k,d_model)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N,T_q,d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N,T_k,d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N,T_k,d_model/h)

        outputs = attention_layer(
            query=Q_,
            key=K_,
            value=V_,
            is_training=is_training,
            mask=key_masks,
            drop_rate=drop_rate,
        )  # (h*N,T_k,d_model/h)
        # restore outputs
        outputs = tf.concat(
            tf.split(outputs, num_heads, axis=0), axis=2
        )  # (N,T_k,d_model)

    # outputs = tf.reduce_mean(tf.layers.Dense(outputs, q_dim),axis=1)
    return outputs


def run_attention():
    x = np.random.rand(1, 2, 3)
    print(x)
    tx = tf.constant(x)
    out = attention_layer(tx, tx, tx, True)

    with tf.Session() as sess:
        vout = sess.run([out])
        print(vout)
        # print(vp)


def run_multihead_attention():
    x = np.random.rand(4, 2, 3)
    print(x)
    tx = tf.constant(x)
    out = multi_head_attention(
        tx, tx, tx, None, True, drop_rate=0.0, num_heads=8, scope="multi_head_attention"
    )
    out2 = tf.reduce_mean(out, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vout, vout2 = sess.run([out, out2])
        print(vout.shape, out2.shape)


def emb_lookup_layer_dense_simple(feat_ids, emb_table, filter_value=0, name=""):
    """
    filter -1 paddings in training data
    :param feat_ids:
    :param emb_table:
    :param filter_value:
    :return:
    """
    lookup_ori = tf.nn.embedding_lookup(emb_table, feat_ids, name=name)
    return lookup_ori


def emb_lookup_layer_dense(feat_ids, emb_table, filter_value=0):
    """
    filter -1 paddings in training data
    :param feat_ids:
    :param emb_table:
    :param filter_value:
    :return:
    """
    default_value_mask_neg = 1 - tf.cast(tf.less(feat_ids, filter_value), tf.int64)
    emb_default_mask_neg_float = tf.cast(
        tf.expand_dims(default_value_mask_neg, axis=2), dtype=tf.float32
    )
    feat_dnn_ids_filtered = tf.multiply(feat_ids, default_value_mask_neg)
    lookup_ori = tf.nn.embedding_lookup(emb_table, feat_dnn_ids_filtered)
    # zero also has embedding, so we need to mask them out
    lookup = tf.multiply(lookup_ori, emb_default_mask_neg_float)

    return lookup


def fixed_size_emb_lookup(user_ids, item_ids, features, filter_default, params):
    """
    :param user_ids:
    :param item_ids:
    :param features:
    :param filter_default:
    :param params:
    :return:[None,user_feature_length,emb_size],[None,item_feature_length,emb_size]
    """
    # embedding bias to avoid neg value
    EMB_BIAS = 3
    with tf.variable_scope("user_and_item_emb", reuse=tf.AUTO_REUSE):
        user_side_feature_list = []
        item_side_feature_list = []
        user_item_key_ids_dict = dict()
        for ky, value in list(user_ids.items()) + list(item_ids.items()):
            # only support same emb size
            emb_size = max(params.embedding_size_dnn, 2)
            # [bs,len]
            feat_dnn_ids = features[ky]
            his_slices = []
            # dense feature
            if "value_type" in value and value["value_type"] == "dense":
                if "is_weight" in value and value["is_weight"] != "true":
                    # [bs,len] ->  [bs,1,len]
                    feat_dnn_ids = tf.expand_dims(feat_dnn_ids, axis=1)
                    user_item_key_ids_dict[ky] = feat_dnn_ids
                    # [bs,1,dim]
                    deep_v = tf.layers.dense(
                        feat_dnn_ids,
                        units=emb_size,
                        kernel_initializer=tf.glorot_normal_initializer(),
                        bias_initializer=tf.glorot_normal_initializer(),
                        name="emb_lookup_{}_dim_{}".format(ky, emb_size),
                    )
                    # [bs,1,emb_size]
                    lookup = tf.reduce_mean(deep_v, axis=1, keep_dims=True)
                else:
                    LOG("EMB_LOOKUP", "dense weight, skip", feat_dnn_ids)
                    continue
            else:
                # sparse feature
                if value.get("value_type", "") == "string":
                    feat_dnn_ids = tf.strings.to_hash_bucket_fast(
                        feat_dnn_ids, value["hash_size"], name="{}_to_hash".format(ky)
                    )
                user_item_key_ids_dict[ky] = feat_dnn_ids
                deep_v = tf.get_variable(
                    name="emb_table_{}".format(value["emb_table"]),
                    shape=[value["hash_size"] + EMB_BIAS, emb_size],
                    initializer=tf.glorot_normal_initializer(),
                )
                if filter_default:
                    # LOG('LOOKUP DENSE')
                    feat_dnn_ids = feat_dnn_ids + EMB_BIAS
                    lookup_filtered = emb_lookup_layer_dense_simple(
                        feat_ids=feat_dnn_ids,
                        emb_table=deep_v,
                        name="{}_emb_look_up".format(ky),
                    )

                    # has weights
                    # deep_v =[bs,len,hash_dim]
                    if "embedding_weight" in value and value["embedding_weight"]:
                        # [bs, len]
                        deep_v_weight = features[value["embedding_weight"]]
                        lookup_filtered = tf.multiply(
                            lookup_filtered, tf.expand_dims(deep_v_weight, axis=-1)
                        )
                        LOG("EMBEEDING_WEIGHTS", lookup_filtered, deep_v_weight)

                    # attention is not a default
                    if (
                        value["attention_heads"] > 0
                        and params.enable_attention.lower() == "true"
                    ):
                        # only support one attention now
                        lookup_filtered = attention_layer(
                            query=lookup_filtered,
                            key=lookup_filtered,
                            value=lookup_filtered,
                            is_training=False,
                            mask=None,
                            drop_rate=0.0,
                        )
                        LOG(
                            "ATTENTION_LOOKUP",
                            ky,
                            value["attention_heads"],
                            feat_dnn_ids,
                        )
                    # [bs,1,hash_dim]
                    if ky == "user_history" and bool(params.add_more_his):
                        indices_all = [
                            [0],
                            [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        ]
                        for indices in indices_all:
                            lookup_filtered_his = tf.gather(
                                lookup_filtered, indices=indices, axis=1
                            )
                            his_slices.append(
                                tf.reduce_mean(
                                    lookup_filtered_his, axis=1, keep_dims=True
                                )
                            )
                    lookup = tf.reduce_mean(lookup_filtered, axis=1, keep_dims=True)
                else:
                    raise NotImplementedError("NOT SUPPORT SPARSE FEATURE")

            if ky in user_ids:
                LOG("user_features", ky, value, deep_v)
                user_side_feature_list.append(lookup)
                user_side_feature_list.extend(his_slices[:])
                his_slices = []
            else:
                LOG("item_features", ky, value, deep_v)
                item_side_feature_list.append(lookup)

    # [bs,feature_num,dim]
    user_side_inputs = tf.concat(user_side_feature_list, 1)
    item_side_inputs = tf.concat(item_side_feature_list, 1)

    return user_side_inputs, item_side_inputs, user_item_key_ids_dict


def inbatch_neg_sampling_loss(user_side, item_side, labels, tau=1, weights=None):
    """

    :param user_side: tensor, [bs,d]
    :param item_side: tensor,[bs,d]
    :param labels:tensor,[bs,1]
    :param tau:
    :return:
    """
    # [1,bs]
    # [bs,d] * [d,bs] - > [bs,bs]
    similarity_matrix = tf.matmul(user_side, item_side, transpose_b=True)
    bs = tf.shape(user_side)[0]
    select_index = tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, 2])

    # filter negative samples
    softmax_similarity_matrix = tf.nn.softmax(similarity_matrix, axis=-1)
    pos_prob = tf.expand_dims(
        tf.gather_nd(softmax_similarity_matrix, select_index), axis=1
    )

    pos_loss = -tf.log(pos_prob)

    loss = tf.reduce_mean(pos_loss)
    return loss, pos_prob, softmax_similarity_matrix, select_index


def select():
    temp_var = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    label = tf.Variable([[1], [0], [1], [0]])
    # idx = tf.constant([0, 2])

    rows = tf.gather(temp_var, label)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    print("1", sess.run(tf.where(tf.squeeze(label))))
    print("2", sess.run(tf.gather(temp_var, tf.squeeze(tf.where(tf.squeeze(label))))))


# print('3',sess.run(rows))  # ==> [[1, 2, 3], [7, 8, 9]]
def run_range():
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    print(sess.run(tf.tile(tf.reshape(tf.range(9), [-1, 1]), [1, 2])))


def run_inbatch_neg():
    bs = 10
    dim = 2
    # user_side = tf.constant(np.random.rand(bs, dim))
    # item_side = tf.constant(np.random.rand(bs, dim))
    # labels = tf.constant(np.random.randint(2, size=(bs, 1)))

    # print(user_side)
    user_side = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    item_side = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    # print(sess.run([positive_user, select_index, p_ui, ]))
    loss1 = inbatch_neg_sampling_loss(
        user_side=user_side, item_side=item_side, labels=labels
    )
    # loss1 = inbatch_neg_sampling_loss(user_side=user_side, item_side=item_side, labels=labels)
    for i in range(1, 2):
        feed_dict = {
            user_side: np.random.rand(i * 5, dim),
            item_side: np.random.rand(i * 5, dim),
            labels: np.random.randint(low=1, high=2, size=(i * 5, 1)),
        }
        print(sess.run([item_side, loss1, labels], feed_dict=feed_dict))


def add_time_weight_to_loss(loss, label, item_type, watch_time):
    VIDEO_AVG = 1 / 120.0
    TUWEN_AVG = 1 / 90.0
    VIDEO_TYPE_VALUE = 152
    is_video = tf.cast(
        tf.equal(item_type, tf.constant(VIDEO_TYPE_VALUE, dtype=tf.int64)), tf.float32
    )
    loss_weights = (
        label
        * (is_video * VIDEO_AVG + (1.0 - is_video) * TUWEN_AVG)
        * tf.clip_by_value(watch_time, 10.0, 600.0)
        + (1.0 - label) * 1.0
    )
    LOG(
        "add_time_weight_to_loss,VIDEO_AVG,VIDEO_TYPE_VALUE,TUWEN_AVG,loss_weights",
        VIDEO_AVG,
        VIDEO_TYPE_VALUE,
        TUWEN_AVG,
        loss_weights,
    )
    loss_ret = loss * loss_weights
    return loss_ret
    # return label,loss,item_type,watch_time,loss_weights,loss_ret,is_video


def add_time_weight_to_loss_for_bottom_page(loss, label, watch_time):
    # VIDEO_AVG = 1 / 3.8
    VIDEO_AVG = 1 / 9.4
    TUWEN_AVG = 1 / 90.0
    VIDEO_TYPE_VALUE = 152
    # is_video = tf.cast(tf.equal(item_type, tf.constant(VIDEO_TYPE_VALUE, dtype=tf.int64)), tf.float32)
    # loss_weights = label * VIDEO_AVG *tf.math.log(tf.clip_by_value(watch_time, 10.0, 600.0)) + (1.0 - label) * 1.0
    loss_weights = (
        label * VIDEO_AVG * tf.math.sqrt(tf.clip_by_value(watch_time, 10.0, 600.0))
        + (1.0 - label) * 1.0
    )
    loss_ret = loss * loss_weights
    return loss_ret
    # return label,loss,item_type,watch_time,loss_weights,loss_ret,is_video


def add_video_time_weight_to_loss_for_bottom_page(loss, vtime):

    VIDEO_AVG = 1 / 18.6
    # VIDEO_AVG = 1 / 384
    loss_weights = VIDEO_AVG * tf.math.sqrt(tf.clip_by_value(vtime, 10.0, 600.0))
    loss_ret = loss * loss_weights
    return loss_ret


def run_add_time_weight_to_loss():
    sess = tf.Session()
    BN = 4
    loss = tf.constant(np.random.rand(BN, 1), dtype=tf.float32)
    label = tf.constant(
        np.random.randint(low=0, high=2, size=(BN, 1)), dtype=tf.float32
    )
    item_type = tf.constant(
        np.random.randint(low=0, high=2, size=(BN, 1)) * 152, dtype=tf.int64
    )
    watch_time = tf.constant(
        np.random.randint(low=0, high=100, size=(BN, 1)), dtype=tf.float32
    )
    sess.run(tf.global_variables_initializer())

    x = add_time_weight_to_loss(
        loss=loss, label=label, item_type=item_type, watch_time=watch_time
    )
    print(sess.run(x))
    # print('is_video:{}\n loss_weights:{}\n loss:{}\n watch_time:{}\n item_type:{}\nloss_ret:{}\nlabel:{}\n'.format(*sess.run(x)))


def run_elu():
    sess = tf.Session()
    BN = 4
    loss = tf.constant(np.random.rand(BN, 1) - 0.5, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    print(sess.run([tf.nn.elu(loss), loss]))


def run_l2_reg():
    # import tensorflow as tf
    sess = tf.Session()
    weight_decay = 1.0
    tmp = tf.constant([0, 1, 2, 3], dtype=tf.float32)
    # """?\
    l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
    a = tf.get_variable("I_am_a", regularizer=l2_reg, initializer=tmp)
    b = tf.get_variable("I_am_b", regularizer=l2_reg, initializer=tmp)
    # """
    # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    # **上面代码的等价代码
    # a = tf.get_variable("I_am_a", initializer=tmp,regularizer=)

    # a2 = tf.reduce_sum(a * a) * weight_decay / 2;
    # a3 = tf.get_variable(a.name.split(":")[0] + "/Regularizer/l2_regularizer", initializer=a2)
    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, a2)
    # **
    sess.run(tf.global_variables_initializer())
    keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    for key in keys:
        print("%s : %s" % (key.name, sess.run(key)))

    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.add_n(reg_variables)
    print(sess.run(reg_term))
    # reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)


def mlp_net(
    input_x,
    name_scope,
    layer_nums,
    is_training,
    drop_rate,
    batch_norm,
    activation=tf.nn.swish,
):
    LOG("mlp_net", name_scope, layer_nums, input_x)
    residual_x = fully_connect(
        input_x=input_x,
        out_dim=layer_nums[-1],
        is_training=is_training,
        axis=1,
        batch_norm=batch_norm,
        activation=tf.identity,
        name="{}_residual".format(name_scope),
    )
    LOG("residual_x", residual_x)

    for i, num_node in enumerate(layer_nums):
        layer_name = "{}_{}_{}".format(name_scope, i, num_node)
        LOG("layer_name", layer_name)
        input_x = fully_connect(
            input_x=input_x,
            out_dim=num_node,
            is_training=is_training,
            axis=1,
            batch_norm=batch_norm,
            name="{}_{}".format(name_scope, layer_name),
            activation=activation,
        )
        if is_training:
            input_x = tf.nn.dropout(input_x, rate=drop_rate)
    LOG("input_x", input_x)
    return input_x + residual_x


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def multi_head_attention_net(
    input_x, name_scope, is_training, drop_rate, heads, batch_norm=False
):
    """
    :param input_x: 3d tensor,[BN,len,dim] # [bs,feature_num,dim]
    :param name_scope:
    :param layer_nums:
    :param is_training:
    :param drop_rate:
    :param batch_norm:
    :param heads:
    :return:
    """
    lask_dim = input_x.get_shape().as_list()[-1]
    LOG("multi head attention net ", "last_dim", lask_dim)

    # [BN,len,dim*heads]
    mha_x1 = multi_head_attention(
        queries=input_x,
        keys=input_x,
        values=input_x,
        key_masks=None,
        is_training=is_training,
        drop_rate=drop_rate,
        num_heads=heads,
        scope=name_scope,
    )
    LOG("multi_head_attention_net", name_scope, mha_x1)

    # [BN,len,dim]
    mha_x2 = tf.layers.dense(
        mha_x1,
        units=lask_dim,
        kernel_initializer=tf.glorot_normal_initializer(),
        bias_initializer=tf.glorot_normal_initializer(),
        name="{}_reshape".format(name_scope),
    )
    # norm in the last dim
    if batch_norm:
        mha_x2 = tf.layers.batch_normalization(
            mha_x2, axis=-1, training=is_training, name="{}_bn".format(name_scope)
        )

    # dropout
    if is_training:
        mha_x2 = tf.nn.dropout(mha_x2, rate=drop_rate)

    # residual
    out_x1 = mha_x2 + input_x

    return out_x1


def dot_loss(user_side, item_side, labels, log_tau, mode):
    LOG("dot_loss", "params.log_tau:{}".format(log_tau), user_side, item_side)
    dot = tf.reduce_sum(
        tf.multiply(user_side, item_side), keepdims=True, axis=1, name="cos"
    )
    logits = log_tau * dot
    predict = tf.nn.sigmoid(logits)

    predictions = {
        "logits": logits,
        "score": predict,
        "user_norm": user_side,
        "item_norm": item_side,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        loss = None
    else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    return loss, predictions


def mse_loss(user_side, item_side, labels, log_tau, mode):
    LOG("mse_loss", "params.log_tau:{}".format(log_tau), user_side, item_side)
    dot = tf.reduce_sum(
        tf.multiply(user_side, item_side), keepdims=True, axis=1, name="cos"
    )
    logits = log_tau * dot
    predict = tf.nn.sigmoid(logits)

    LOG("mse loss", labels, predict)
    predictions = {
        "logits": logits,
        "score": predict,
        "user_norm": user_side,
        "item_norm": item_side,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        loss = None
    else:
        loss = tf.square(predict - labels)
    LOG("mse loss", predict, loss)

    return loss, predictions


def softmax_mask1d(item_newsid):
    bs = tf.shape(item_newsid)[0]
    item_newsid_reshape = tf.reshape(item_newsid, shape=[1, bs])
    equal_mask = tf.equal(item_newsid_reshape, item_newsid)
    diag_ones = tf.matrix_diag(tf.ones(bs))
    equal_mask_float = tf.math.multiply(
        tf.cast(equal_mask, tf.float32), (1 - diag_ones)
    )
    equal_mask_bool = tf.cast(equal_mask_float, tf.bool)
    equal_mask_float_mask = tf.ones_like(equal_mask_bool, dtype=tf.float32) - 100000
    equal_mask = tf.cast(
        tf.where(equal_mask_bool, equal_mask_float_mask, equal_mask_float), tf.float32
    )
    # equal_mask = equal_mask - 1
    equal_mask = tf.identity(equal_mask, name="softmax_mask1d_equal_mask")
    return tf.stop_gradient(equal_mask)


def softmax_mask2d(user_history, item_newsid):
    bs = tf.shape(item_newsid)[0]
    expand_ids = tf.tile(tf.expand_dims(user_history, axis=1), [1, bs, 1])
    equal_ids = tf.equal(expand_ids, item_newsid)
    equal_mask = tf.math.reduce_any(equal_ids, axis=-1)
    diag_ones = tf.matrix_diag(tf.ones(bs))
    diag_zeros = 1 - diag_ones
    equal_mask_float = tf.math.multiply(tf.cast(equal_mask, tf.float32), diag_zeros)
    equal_mask_bool = tf.cast(equal_mask_float, tf.bool)  # diag cannot be -100000!
    equal_mask_float_mask = tf.ones_like(equal_mask_float) - 1000000
    equal_mask = tf.cast(
        tf.where(equal_mask_bool, equal_mask_float_mask, equal_mask_float), tf.float32
    )
    equal_mask = tf.identity(equal_mask, name="softmax_mask2d_equal_mask")
    return tf.stop_gradient(equal_mask)


def softmax_loss(
    user_side,
    item_side,
    log_tau,
    freq_weight,
    user_history=None,
    item_newsid=None,
    item_freq=None,
    is_training=False,
):
    # item_freq: (bs, 1) -> (bs,) (to ensure the right direction of broadcasting)
    similarity_matrix = tf.matmul(user_side, item_side, transpose_b=True)
    bs = tf.shape(user_side)[0]
    select_index = tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, 2])
    adjusted_similarity_matrix = log_tau * similarity_matrix
    if item_freq is not None:
        item_freq = tf.squeeze(item_freq)
        log_epsilon = tf.fill(tf.shape(item_freq), 1e-9)  # in case of log(0) or log(-1)
        item_freq = tf.where(item_freq > 0.0, item_freq, log_epsilon)
        adjusted_similarity_matrix = (
            adjusted_similarity_matrix - freq_weight * tf.math.log(item_freq)
        )
    # filter negative samples
    if user_history is not None and item_newsid is not None and is_training:
        print(
            "mask false negative with user_history: {}, item_newsid: {}".format(
                user_history, item_newsid
            )
        )
        softmax_mask_his = softmax_mask2d(
            user_history=user_history, item_newsid=item_newsid
        )
        adjusted_similarity_matrix = tf.math.add(
            adjusted_similarity_matrix,
            softmax_mask_his,
            name="adjusted_similarity_matrix_add_his_mask",
        )
    softmax_mask_item = softmax_mask1d(item_newsid=item_newsid)
    adjusted_similarity_matrix = tf.math.add(
        adjusted_similarity_matrix,
        softmax_mask_item,
        name="adjusted_similarity_matrix_add_item_mask",
    )
    softmax_similarity_matrix = tf.nn.softmax(
        adjusted_similarity_matrix, axis=-1, name="softmax_similarity_matrix"
    )
    # mask false negative

    pos_prob = tf.expand_dims(
        tf.gather_nd(softmax_similarity_matrix, select_index), axis=1
    )
    loss = -tf.log(pos_prob)
    predictions = {
        "logits": pos_prob,
        "score": pos_prob,
        "user_norm": user_side,
        "item_norm": item_side,
    }

    return loss, predictions


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def multi_interest_softmax_loss(user_side, item_side, log_tau):
    """
    :param user_side:[bs,heads_num,d]
    :param item_side:[bs,d]
    :param log_tau: temperature
    :param user_heads: user head num
    :return: loss,[bs,1]
    """
    LOG(
        "multi_interest_softmax_loss",
        "params.log_tau:{}".format(log_tau),
        user_side,
        item_side,
    )
    # user_side_r = tf.reshape(user_side, [None, dim])  # [bs*user_heads,dim]
    user_side_r = tf.transpose(user_side, [1, 0, 2])  # [heads_num,bs,d]
    similarity_matrix = tf.matmul(
        user_side_r, item_side, transpose_b=True
    )  # [heads_num,bs,bs]
    adjusted_similarity_matrix = log_tau * similarity_matrix

    softmax_similarity_matrix = tf.nn.softmax(adjusted_similarity_matrix, axis=-1)
    softmax_similarity_matrix1 = tf.reduce_max(
        softmax_similarity_matrix, axis=0, keepdims=False
    )  # [bs,bs]

    bs = tf.shape(item_side)[0]
    select_index = tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, 2])
    pos_prob = tf.expand_dims(
        tf.gather_nd(softmax_similarity_matrix1, select_index), axis=1
    )
    loss = -tf.log(pos_prob)
    predictions = {
        "logits": pos_prob,
        "score": pos_prob,
        "user_norm": tf.reshape(user_side, [bs, -1]),
        "item_norm": item_side,
    }

    return loss, predictions

    # return user_side, item_side, user_side_r, similarity_matrix, softmax_similarity_matrix, softmax_similarity_matrix1, select_index, pos_prob


def run_multi_interest_softmax_loss():
    dim = 4
    head_num = 3
    np.random.seed(1024)
    user_side = tf.placeholder(shape=[None, head_num, dim], dtype=tf.float32)
    item_side = tf.placeholder(shape=[None, dim], dtype=tf.float32)

    user_side_l2 = tf.nn.l2_normalize(
        user_side, axis=-1, epsilon=1e-6, name="user_side_out"
    )
    tf.logging.set_verbosity(tf.logging.INFO)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    mi_ret = multi_interest_softmax_loss(
        user_side=user_side, item_side=item_side, log_tau=10
    )
    for i in range(1, 2):
        feed_dict = {
            user_side: np.random.rand(i * 5, head_num, dim),
            item_side: np.random.rand(i * 5, dim),
        }
        ret = sess.run(mi_ret, feed_dict=feed_dict)
        for x in ret:
            LOG(json.dumps(x, cls=NumpyEncoder))

        ret = sess.run([user_side, user_side_l2], feed_dict=feed_dict)
        for x in ret:
            LOG(json.dumps(x, cls=NumpyEncoder))


def run_softmax_loss():
    bs = 10
    dim = 2
    # user_side = tf.constant(np.random.rand(bs, dim))
    # item_side = tf.constant(np.random.rand(bs, dim))
    # labels = tf.constant(np.random.randint(2, size=(bs, 1)))

    # print(user_side)
    user_side = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    item_side = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    # print(sess.run([positive_user, select_index, p_ui, ]))
    loss1 = softmax_loss(user_side=user_side, item_side=item_side, log_tau=10)
    # loss1 = inbatch_neg_sampling_loss(user_side=user_side, item_side=item_side, labels=labels)
    for i in range(1, 2):
        feed_dict = {
            user_side: np.random.rand(i * 5, dim),
            item_side: np.random.rand(i * 5, dim),
            labels: np.random.randint(low=1, high=2, size=(i * 5, 1)),
        }
        print(sess.run([item_side, loss1, labels], feed_dict=feed_dict))


def run_dot_loss():
    dim = 4
    head_num = 3
    np.random.seed(1024)
    user_side = tf.placeholder(shape=[None, dim], dtype=tf.float32)
    item_side = tf.placeholder(shape=[None, dim], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    user_side_l2 = tf.nn.l2_normalize(
        user_side, axis=-1, epsilon=1e-6, name="user_side_out"
    )
    tf.logging.set_verbosity(tf.logging.INFO)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    dret = dot_loss(
        user_side=user_side,
        item_side=item_side,
        labels=labels,
        log_tau=10,
        mode=tf.estimator.ModeKeys.TRAIN,
    )

    for i in range(1, 2):
        feed_dict = {
            user_side: np.random.rand(i * 5, dim),
            item_side: np.random.rand(i * 5, dim),
            labels: np.random.randint(low=1, high=2, size=(i * 5, 1)),
        }
        print(sess.run(dret, feed_dict=feed_dict))


if __name__ == "__main__":
    # run_multihead_attention()
    # run_range()
    # run_inbatch_neg()
    # run_range()
    run_dot_loss()
