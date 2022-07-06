# -*- coding: utf-8 -*-
import sys
import traceback
import json
import hashlib
import codecs
import re
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.storagelevel import StorageLevel
import traceback
from random import random
import pyspark.sql.functions as F
from dssm_v2.configs.config_realtime_groupId import config_realtime, string_padding, feature_id_padding, float_padding, \
    format_float, format_features, format_string
config = config_realtime

dtype_dict = {"string": ArrayType(StringType(), True), "dense": ArrayType(FloatType(), True), "sparse": ArrayType(IntegerType(), True)}
FIELDS = [StructField(key, dtype_dict[value["value_type"]]) for key, value in config.items()]
print("FIELDS: {}".format(FIELDS))

def deletPath_b(sc, path):
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    URI = sc._gateway.jvm.java.net.URI
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fs = FileSystem.get(URI(path), sc._jsc.hadoopConfiguration())
    if fs.exists(Path(path)):
        fs.delete(Path(path), True)


def createPath(sc, path, num):
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    URI = sc._gateway.jvm.java.net.URI
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fs = FileSystem.get(URI(path), sc._jsc.hadoopConfiguration())
    f = fs.create(Path(path), True)
    f.write(str(num).encode())
    f.close()

def parse_for_tfrecord(feature):
    try:
        label = float(feature[config["label"]["feature_index"]])
        if label != 0.0 and label != 1.0:
            return None
        list_size = len(FIELDS)
        outlist = []
        for k in range(1, list_size):
            key_name = FIELDS[k].name
            fea_len = config[key_name]["field_size"]
            input_fea = feature[config[key_name]["feature_index"]]
            split_index = config[key_name].get("split_index", 0)
            format_type = config[key_name]["value_type"]
            if format_type == "string":
                outlist.append(format_string(input_fea, fea_len, split_index))
            elif format_type == "dense":
                outlist.append(format_float(input_fea, fea_len, split_index))
            elif format_type == "sparse":
                outlist.append(format_features(input_fea, fea_len, split_index))
        result = ([label], *outlist)
        return result
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    input_root = sys.argv[1]
    output_root = sys.argv[2]
    output_root_feature = output_root + "_feature"

    ss = SparkSession \
        .builder \
        .appName("dnn_recall_write2tfrecord") \
        .getOrCreate()
    sc = ss.sparkContext
    # mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/bottom_realtime/neg_sample/%YYYYMMDDHH% mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/bottom_realtime/tfrecord/%YYYYMMDDHH%

    try:
        deletPath_b(sc, output_root)
        deletPath_b(sc, output_root_feature)
    except:
        print("No file to remove")

    features_rdd = sc.textFile(input_root)
    for x in features_rdd.take(10):
        print(x)

    print("before length filter: {}".format(features_rdd.count()))
    features_rdd = features_rdd.map(lambda x: x.split("\t")).filter(lambda x: len(x) >= 152)
    print("after length filter: {}".format(features_rdd.count()))

    count_num = features_rdd.count()
    print("input data size: %d" % count_num)
    for x in features_rdd.take(10):
        print(x)

    schema = StructType(FIELDS)
    createPath(sc, output_root + '/record_size', count_num)
    # tf_in_rdd = features_rdd.map(parse_for_tfrecord).filter(lambda item: item is not None)
    tf_in_rdd = features_rdd.map(parse_for_tfrecord).filter(lambda item: item is not None)
    print("output size:: %d" % tf_in_rdd.count())

    df_1 = ss.createDataFrame(tf_in_rdd, schema)

    # add one colume 'rand'
    df_2 = df_1.withColumn('rand', F.rand(seed=42))
    df_2.show(100)
    # random shuffle
    df_rnd = df_2.orderBy(df_2.rand)
    df_rnd.show(100)
    # drop the colume 'rand'
    df = df_rnd.drop(df_rnd.rand)
    print('data_frame:', (df.count(), len(df.columns)))
    # print("save features to :{}".format(output_root_feature))
    # df.repartition(200).rdd.saveAsTextFile(output_root_feature)
    print("save tfrecords to: {}".format(output_root))
    df.repartition(200).write.format("tfrecords").save(output_root + '/train.tfrecord')

    # df.filter(df['label'][0] > 0.0).repartition(200).write.format("tfrecords").option("recordType", "Example").save(
    #     output_root + '/inbatch.tfrecord')
    sc.stop()

