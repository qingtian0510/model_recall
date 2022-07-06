import tensorflow as tf

if tf.__version__.startswith("2"):
    print("tensorflow version: {}".format(tf.__version__))
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    print("tensorflow version: {}".format(tf.__version__))
import argparse

from dssm_v2.data import data_realtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_file", default="", type=str)
    parser.add_argument("--feature_file", default="", type=str)
    parser.add_argument("--skip_features", default=None)
    parser.add_argument("--use_features", default="", type=str)
    args = parser.parse_args()
    sess = tf.Session()
    # records = [tf.train.Example.FromString(example) for example in tf.python_io.tf_record_iterator(args.tfrecord_file)]
    features = []
    with open(args.feature_file) as f:
        features = f.readlines()
    features = [x.strip("\n").split("\t") for x in features]
    (
        data_format,
        user_ids,
        item_ids,
        label,
        export_format,
    ) = data_realtime.get_data_format_fn_and_ids(args)
    num_threads = 12
    batch_size = 12
    prefetch = 4

    def decode_tfrecord(serial_example):
        try:
            example = tf.parse_example(serial_example, features=data_format)
            return example
        except tf.errors.DataLossError:
            print("skip data loss error!")

        return None

    dataset = (
        tf.data.TFRecordDataset(args.tfrecord_file, num_parallel_reads=num_threads)
        .batch(batch_size)
        .map(decode_tfrecord, num_parallel_calls=num_threads)
        .prefetch(prefetch)
    )
    iterator = dataset.make_one_shot_iterator()
    batch_features = iterator.get_next()
    user_ids.update(item_ids)
    for i in range(10000):
        batch_features_results = sess.run(batch_features)

        for key, np_value in batch_features_results.items():
            try:
                if key == "userid_ori":
                    print(key)
                if key == "label":
                    print(key)
                hash_size = user_ids[key]["hash_size"]
                value_type = user_ids[key]["value_type"]
                if value_type == "sparse":
                    check_sum = (np_value >= hash_size + 3).sum()
                    if check_sum > 0:
                        print(
                            "key: {}, check_sum: {}, hash_size: {}, value: {}".format(
                                key, check_sum, hash_size, np_value
                            )
                        )
            except:
                pass
        print("test")
