from dssm_v2.utils.train_utils import LOG, parse_data_config
import tensorflow as tf


def get_data_format_fn_and_ids(args, feature_dict=dict()):
    if len(feature_dict) < 1:
        print("config error!")
        raise ValueError("config error!")
    skip_features = args.skip_features
    use_features = args.use_features
    LOG("filed_dict", feature_dict)
    user_ids, item_ids, data_format, export_format = parse_data_config(
        feature_dict=feature_dict,
        skip_features=skip_features,
        use_features=use_features,
    )
    data_format["label"] = tf.FixedLenFeature([1], tf.float32)
    data_format["item_freq"] = tf.FixedLenFeature([1], tf.float32)
    data_format["watch_time"] = tf.FixedLenFeature([1], tf.float32)
    data_format["vtime"] = tf.FixedLenFeature([1], tf.float32)
    return data_format, user_ids, item_ids, "label", export_format
