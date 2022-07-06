import importlib

import tensorflow as tf
import logging
import tensorflow as tf

logger = logging.getLogger("tensorflow")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

for h in logger.handlers:
    h.setFormatter(formatter)
# get TF logger
log = logging.getLogger("tensorflow")
log.setLevel(logging.DEBUG)


def add_log_file(file_name):
    fh = logging.FileHandler(file_name)
    log.addHandler(fh)


def LOG(*args):
    msg = "\t".join([str(x) for x in args])
    return tf.logging.info(msg)


def DEBUG(*args):
    msg = "[" + "] [ ".join([str(x) for x in args]) + "]"
    return tf.logging.debug(msg)


def ERROR(*args):
    msg = "[" + "] [ ".join([str(x) for x in args]) + "]"
    return tf.logging.error(msg)


def input_fn(
    file_names,
    label_name,
    data_format,
    batch_size,
    num_epochs=1,
    num_threads=64,
    prefetch=1,
    perform_shuffle=False,
    GZIP=False,
):
    def decode_tfrecord(serial_example):
        try:
            example = tf.parse_example(serial_example, features=data_format)
            return example
        except tf.errors.DataLossError:
            logging.info("skip data loss error!")

        return None

    if GZIP:
        dataset = (
            tf.data.TFRecordDataset(
                file_names, compression_type="GZIP", buffer_size=1024000, num_parallel_reads=num_threads
            )
            .batch(batch_size)
            .map(decode_tfrecord, num_parallel_calls=num_threads)
            .prefetch(prefetch)
        )
    else:
        dataset = (
            tf.data.TFRecordDataset(file_names, buffer_size=1024000, num_parallel_reads=num_threads)
            .batch(batch_size)
            .map(decode_tfrecord, num_parallel_calls=num_threads)
            .prefetch(prefetch)
        )

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size)
    # epochs from blending together.
    print("dataset repeat num_epochs: {}".format(num_epochs))
    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()
    batch_features = iterator.get_next()
    batch_labels = batch_features[label_name]
    return batch_features, batch_labels


def fully_connect(
    input_x,
    out_dim,
    is_training,
    axis=1,
    name="",
    activation=tf.nn.relu,
    batch_norm=False,
    regularizer=None,
):
    x1 = tf.layers.dense(
        input_x,
        units=out_dim,
        trainable=True,
        kernel_initializer=tf.glorot_normal_initializer(),
        bias_initializer=tf.glorot_normal_initializer(),
        kernel_regularizer=regularizer,
        name=name,
    )
    if batch_norm:
        x2 = tf.layers.batch_normalization(
            x1, axis=axis, training=is_training, name=name + "_batch_normalization"
        )
        x3 = activation(x2)
    else:
        x3 = activation(x1)
    return x3


def residual_fully_connect(
    input_x,
    out_dim,
    is_training,
    drop_rate=0.2,
    axis=1,
    name="",
    activation=tf.nn.relu,
    batch_norm=False,
    regularizer=None,
):
    """
    residual fully connect , x + (x->(MLP->BN->leakyRELU->Dropout)*2)
    https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
    :param input_x:
    :param out_dim:
    :param is_training:
    :param axis:
    :param name:
    :param activation:
    :param batch_norm:
    :param regularizer:
    :return:
    """
    x1 = fully_connect(
        input_x=input_x,
        out_dim=out_dim,
        is_training=is_training,
        axis=axis,
        name=name + "_residual_1",
        activation=activation,
        batch_norm=batch_norm,
        regularizer=regularizer,
    )
    if is_training:
        x1 = tf.nn.dropout(x1, rate=drop_rate)
    x2 = fully_connect(
        input_x=x1,
        out_dim=out_dim,
        is_training=is_training,
        axis=axis,
        name=name + "_residual_2",
        activation=activation,
        batch_norm=batch_norm,
        regularizer=regularizer,
    )

    return input_x + x2


def get_optimizer(optimizer_name="sgd", learning_rate=0.001):
    if optimizer_name == "Adam":
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
    elif optimizer_name == "Adagrad":
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=learning_rate, initial_accumulator_value=1e-8
        )
    elif optimizer_name == "Momentum":
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.95
        )
    elif optimizer_name == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    else:  # 'sgd'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    return optimizer


def parse_data_config(feature_dict, skip_features=None, use_features=""):
    user_ids = {}
    item_ids = {}

    for k, x in feature_dict.items():
        if use_features:
            if x["name"] not in use_features:
                print("feature name {} not in use_features. skipped".format(x["name"]))
                continue
        else:
            if x["using"] != 1:
                print("feature name {} using!=1. skipped".format(x["name"]))
                continue
            if skip_features and x["name"] in skip_features:
                LOG("SKIP FEATURES", x["name"])
                continue
        feature_id = x["name"]
        emb_table_name = "{}".format(
            x["shared_field"] if x["shared_field"] else x["name"]
        )
        x["emb_table"] = emb_table_name
        ky = "{}".format(feature_id)
        if x["type"] == "user":
            table = user_ids
        elif x["type"] == "item":
            table = item_ids
        else:
            print("feautre filed key is {}, type is: {}, passed".format(k, x["type"]))
            continue
        table[ky] = x.copy()

    data_format = {}
    export_format = {}

    for k, v in list(user_ids.items()) + list(item_ids.items()):

        if "value_type" in v and v["value_type"] == "dense":
            df = tf.FixedLenFeature([v["field_size"]], tf.float32)
            ef = tf.placeholder(
                dtype=tf.float32, shape=[None, v["field_size"]], name=v["name"]
            )
        elif "value_type" in v and v["value_type"] == "string":
            df = tf.FixedLenFeature([v["field_size"]], tf.string)
            ef = tf.placeholder(
                dtype=tf.string, shape=[None, v["field_size"]], name=v["name"]
            )

        else:
            df = tf.FixedLenFeature([v["field_size"]], tf.int64)
            ef = tf.placeholder(
                dtype=tf.int64, shape=[None, v["field_size"]], name=v["name"]
            )

        data_format[k] = df
        export_format[k] = ef

    return user_ids, item_ids, data_format, export_format


def load_module_class(name, module_name="dssm_v2.models"):
    mod = importlib.import_module(module_name)
    tar_class = getattr(mod, name)
    return tar_class


if __name__ == "__main__":
    pass
    # print(load_module_class('double_tower').get_model_fn(None,None))
