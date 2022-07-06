import tensorflow as tf

import pickle
import os
import numpy as np


def unpickle(file):
    with open(file, "rb") as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


def _int64_feature(value):
    # print(value)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def padding(x, size, default_value):
    print(size - len(x))
    return np.append(x, np.array([default_value] * (size - len(x))))


def generate_test_data(
    user_ids, item_ids, label_name, pickle_file, tfrecord_file, batch_size=10240
):
    sample = []

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(batch_size):
        info = {}
        ti = {}
        for k, v in user_ids.items():
            shape = np.random.randint(low=1, high=user_ids[k]["field_size"] + 1)
            v = np.random.randint(low=0, high=user_ids[k]["hash_size"], size=[shape])
            info[k] = v
            ti[k] = padding(v, user_ids[k]["field_size"], -1)

        sample.append(ti)
        for k, v in item_ids.items():
            shape = np.random.randint(low=1, high=item_ids[k]["field_size"] + 1)
            info[k] = np.random.randint(
                low=0, high=item_ids[k]["hash_size"], size=[shape]
            )

        info[label_name] = [i]

        tf_info = {x: _int64_feature(info[x]) for x in info.keys() if x != label_name}
        tf_info[label_name] = _float32_feature(info[label_name])
        one_example = tf.train.Example(features=tf.train.Features(feature=tf_info))
        writer.write(one_example.SerializeToString())

    print(pickle_file)
    pickle.dump(sample, open(pickle_file, "wb"), 0)
    nps = {}
    for x in sample:
        for k, v in x.items():
            if k not in nps:
                nps[k] = []
            nps[k].append(np.expand_dims(v, axis=0))

    for k, v in nps.items():
        nps[k] = np.concatenate(nps[k], axis=0)
        # print(nps[k])
        np.save(os.path.join(os.path.dirname(pickle_file), k + ".npy"), nps[k])

    inputs = []
    for k in nps.keys():
        inputs.append("{}=./{}.npy".format(k, k))

    print(";".join(inputs))


# ax = np.array([1,2,3])
# print(padding(ax,3,-1))
