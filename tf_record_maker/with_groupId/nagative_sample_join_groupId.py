# coding=utf-8
import sys
import traceback
import json

from pyspark import SparkContext
from pyspark.sql import SparkSession
import traceback
import random
import os
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from dssm_v2.configs.config_realtime_groupId import config_realtime, placeholder_str
# function: random sample some negative samples for training
iterm_key_values = {key:value for key, value in config_realtime.items() if value["type"] == "item"}
max_feature_index = max([value["feature_index"] for key, value in iterm_key_values.items()])
print("iterm_key_values: {}, max_feature_index: {}".format(iterm_key_values, max_feature_index))
print("test git to venus!")


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


def get_newest_files(sc, input_path, date_limit):
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    FileStatus = sc._gateway.jvm.org.apache.hadoop.fs.FileStatus
    fs = FileSystem.get(URI(input_path), sc._jsc.hadoopConfiguration())
    listdir = fs.listStatus(Path(input_path))

    avaliable_dirs = [x.getPath().toString() for x in listdir if os.path.basename(x.getPath().toString()) <= date_limit]

    return avaliable_dirs


# sampled items from positive and negtive samples
def sample_item(neg_num, linelist):
    # linelist = line.split("\t")
    # if len(linelist) < 103:
    #     return []
    outlist = []

    CLICK = config_realtime["label"]["feature_index"]
    label = float(linelist[CLICK])
    # if label not satisfy, return empyt list
    if label != 1.0 and label != 0.0:
        return []
    # if label is negative sample, reutrn None
    if label == 0.0:
        return []

    # if label == 1.0:
    outlist.append("\t".join(linelist))


    # these keys must set to 0.0, becase these are negative samples
    ori_default_dict = {"watch_time": config_realtime["watch_time"]["feature_index"], "vtime": config_realtime["vtime"]["feature_index"]}

    N_sample_list = random.sample(bc_item_id.value, neg_num)
    for cmsid in N_sample_list:
        N_linelist = linelist
        N_linelist[CLICK] = "0"
        itemlist = bc_item_info.value[cmsid].split("\t")
        # itemlist = bc_item_info.value[cmsid]
        for key in iterm_key_values.keys():

            iterm_index = iterm_key_values[key]["feature_index"]
            iterm_feature = itemlist[iterm_index]
            assert iterm_feature != placeholder_str
            N_linelist[iterm_index] = iterm_feature

        for key in ori_default_dict.keys():
            N_linelist[ori_default_dict[key]] = "0.0"

        one_n_sample = "\t".join(N_linelist)
        outlist.append(one_n_sample)

    return outlist


def generate_new_label(vtime, watch_time):
    label = 0
    if vtime > 0:
        label = 1 if watch_time > 30 or watch_time / vtime > 0.8 else 0
    return label


def line_split(line):
    line_arr = line.split('\t')
    devid = line_arr[2]
    if len(devid) == 0:
        return None

    return (devid.split("_")[0], line_arr)


def split_item_line(line):
    line_arr = line.strip("\n").split("\t")[:max_feature_index+1]
    line_arr = [x if x != placeholder_str else "" for x in line_arr]
    return (line_arr[iterm_key_values["item_newsid"]["feature_index"]], "\t".join(line_arr))


def neg_sample_with_ratio(line, ratio):
    if random.random() < ratio:
        return "\t".join(line)
    else:
        return None

def get_newest_file(sc, input_path):
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    FileStatus = sc._gateway.jvm.org.apache.hadoop.fs.FileStatus
    fs = FileSystem.get(URI(input_path), sc._jsc.hadoopConfiguration())
    listdir = fs.listStatus(Path(input_path))
    if len(listdir) > 0:
        return listdir[-1].getPath().toString()
    else:
        return "no_file"


def getUserGroup(sample, group):
    if group is None:
        return None

    user_info = group.split("_")
    if len(user_info) < 4:
        return None
    age = user_info[0]
    sex = user_info[1]
    phone_price = user_info[2]
    city = user_info[3]

    sample[2] = group
    sample[20] = city
    sample[28] = phone_price

    return "\t".join(sample)

if __name__ == "__main__":
    sample_input_file = "mdfs://cloudhdfs/newspluginalgo/data/dylantian/lowActive/model_recall/model_first_version/neg_sample"
    user_extend_info = "mdfs://cloudhdfs/newspluginalgo/data/dylantian/lowActive/user_info_expo_all"
    output_file = "mdfs://cloudhdfs/newspluginalgo/data/dylantian/lowActive/model_recall/model_first_version/neg_sample_with_userGroup"

    current_partition = sys.argv[1]

    sample_input_file = sample_input_file + "/" + current_partition
    output_file = output_file + "/" + current_partition

    print("sample_input_fils is: %s" % sample_input_file)
    print("outfile is: %s" % output_file)
    print("partition_idx is: %s" % current_partition)

    ss = SparkSession \
        .builder \
        .appName("dnn_recall_write2tfrecord") \
        .getOrCreate()
    sc = ss.sparkContext

    try:
        deletPath_b(sc, output_file)
        print('delete data path', output_file)
    except:
        print("No file to remove")

    user_extend_info = get_newest_file(sc, user_extend_info)
    print("user_extend_info:", user_extend_info)

    user_info_rdd = sc.textFile(user_extend_info).map(lambda x: (x.split("\t")[0], x.split("\t")[1])).reduceByKey(lambda x, y: x)
    print("afetr reduce user_info_rdd count:", user_info_rdd.count())
    for x in user_info_rdd.take(5):
        print(x)

    original_sample_rdd = sc.textFile(sample_input_file)
    # print("original_sample_rdd count:%d" % original_sample_rdd.count())
    original_sample_rdd = original_sample_rdd.map(lambda line: line_split(line))
    print("original_sample_rdd count:", original_sample_rdd.count())
    for x in original_sample_rdd.take(5):
        print(x)

    sample_rdd = original_sample_rdd.join(user_info_rdd).map(lambda x:getUserGroup(x[1][0], x[1][1])).filter(lambda x: x is not None)
    print("after join user info, sample_rdd count:", sample_rdd.count())
    for x in sample_rdd.take(5):
        print(x)

    sample_rdd.repartition(200).saveAsTextFile(output_file)

    sc.stop()
