# coding=utf-8
import sys
import traceback
import json

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
import traceback
import random
import os
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from dssm_v2.configs.config_realtime import config_realtime, placeholder_str

# function: random sample some negative samples for training
iterm_key_values = {
    key: value for key, value in config_realtime.items() if value["type"] == "item"
}
max_feature_index = max(
    [value["feature_index"] for key, value in iterm_key_values.items()]
)
print(
    "iterm_key_values: {}, max_feature_index: {}".format(
        iterm_key_values, max_feature_index
    )
)
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


def get_newest_file(sc, input_path, date_limit):
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    FileStatus = sc._gateway.jvm.org.apache.hadoop.fs.FileStatus
    fs = FileSystem.get(URI(input_path), sc._jsc.hadoopConfiguration())
    listdir = fs.listStatus(Path(input_path))

    avaliable_dirs = [
        x.getPath().toString()
        for x in listdir
        if os.path.basename(x.getPath().toString()) < date_limit
    ]

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
    ori_default_dict = {
        "watch_time": config_realtime["watch_time"]["feature_index"],
        "vtime": config_realtime["vtime"]["feature_index"],
    }

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


def add_freq_to_data(line: str) -> str:
    fields = line.strip().split("\t")
    ID = config_realtime["item_newsid"]["feature_index"]  # index of cmsid
    FREQ = config_realtime["item_freq"][
        "feature_index"
    ]  # index of item frequency (position to fill in)
    cmsid = fields[ID]

    item_clicks = g_item_clicks.value

    item_freq = item_clicks.get(cmsid, 1e-9)
    fields[FREQ] = str(item_freq)
    return "\t".join(fields)


def generate_new_label(vtime, watch_time):
    label = 0
    if vtime > 0:
        label = 1 if watch_time > 30 or watch_time / vtime > 0.8 else 0
    return label


def line_split_and_gen_label(line):
    line_arr = line.split("\t")
    label = 0
    try:
        label = generate_new_label(float(line_arr[90]), float(line_arr[53]))
    except:
        pass
    line_arr[17] = str(label)
    return line_arr


def split_item_line(line):
    line_arr = line.strip("\n").split("\t")[: max_feature_index + 1]
    line_arr = [x if x != placeholder_str else "" for x in line_arr]
    return (
        line_arr[iterm_key_values["item_newsid"]["feature_index"]],
        "\t".join(line_arr),
    )


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

def exchange_deviceId(line):
    assert len(line) > 153, "line length: {}, error".format(len(line))
    line[config_realtime["user_id"]["feature_index"]] = line[152]
    return line

if __name__ == "__main__":
    sample_input_file = sys.argv[1]
    item_sample_root = sys.argv[2]
    sample_num = int(sys.argv[3])
    output_file = sys.argv[4]
    # partition_idx = sys.argv[5]
    current_partition = sys.argv[5]
    is_landing = False
    if len(sys.argv) == 7 and sys.argv[6] == "landing":
        print("process for landing model")
        is_landing = True
    # mdfs://cloudhdfs/mttsparknew/data/szrecone/PCG/newshotalgo/user/shueli/landing_bottom_realtime/sample_dedup/%YYYYMMDDHH% mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/bottom_add_feature/only_item_feature 4 mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/bottom_add_feature/neg_sample_bottom/%YYYYMMDDHH% %YYYYMMDDHH%
    # mdfs://cloudhdfs/mttsparknew/data/szrecone/PCG/newshotalgo/user/shueli/landing_bottom_realtime/sample_wxlanding_dedup/%YYYYMMDDHH% mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/bottom_add_feature/only_item_feature 4 mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/landing_add_feature/neg_sample_bottom/%YYYYMMDDHH% %YYYYMMDDHH% landing
    print("sample_input_fils is: %s" % sample_input_file)
    print("item_sample_root is: %s" % item_sample_root)
    print("sample num is: %d" % sample_num)
    print("outfile is: %s" % output_file)
    # current_partition = os.path.basename(sample_input_file)
    print("partition_idx is: %s" % current_partition)

    ss = SparkSession.builder.appName("dnn_recall_write2tfrecord").getOrCreate()
    sc = ss.sparkContext

    try:
        deletPath_b(sc, output_file)
        print("delete data path", output_file)
    except:
        print("No file to remove")

    item_file_list = item_sample_root.split(",")
    item_file_newest = get_newest_file(sc, item_file_list[0])
    item_info_rdd = sc.textFile(item_file_newest)
    print("item_info_rdd count: {}".format(item_info_rdd.count()))
    for x in item_info_rdd.take(5):
        print(x)
    item_info_rdd = item_info_rdd.map(lambda line: split_item_line(line)).reduceByKey(
        lambda x, y: x
    )
    # print("item info rdd count is: %d" % item_info_rdd.count())
    print(
        "item_info_rdd after split_item_line  count: {}".format(item_info_rdd.count())
    )
    for x in item_info_rdd.take(5):
        print(x)
    item_id_rdd = item_info_rdd.map(lambda item: item[0])

    bc_item_info = sc.broadcast(item_info_rdd.collectAsMap())
    bc_item_id = sc.broadcast(item_id_rdd.collect())

    original_sample_rdd = sc.textFile(sample_input_file)
    # print("original_sample_rdd count:%d" % original_sample_rdd.count())
    original_sample_rdd = original_sample_rdd.map(
        lambda line: line_split_and_gen_label(line)
    )

    if is_landing:
        original_sample_rdd = original_sample_rdd.map(lambda line: exchange_deviceId(line))

    pos_sample_rdd = original_sample_rdd.filter(
        lambda larr: len(larr) > 102 and float(larr[17]) == 1.0
    )
    pos_sample_rdd_count = pos_sample_rdd.count()
    print("pos_sample_rdd_count :%d" % pos_sample_rdd_count)

    neg_sample_rdd = original_sample_rdd.filter(
        lambda larr: len(larr) > 102 and float(larr[17]) == 0.0
    )
    neg_sample_rdd_count = neg_sample_rdd.count()
    print("neg_sample_rdd_count :%d" % neg_sample_rdd_count)

    neg_pos_ratio = neg_sample_rdd_count / pos_sample_rdd_count
    print("neg_pos_ratio:", neg_pos_ratio)

    neg_sample_ratio = sample_num / (neg_pos_ratio * 2)
    print("neg_sample_ratio:", neg_sample_ratio)

    neg_sample_rdd = (
        neg_sample_rdd.map(lambda line: neg_sample_with_ratio(line, neg_sample_ratio))
        .filter(lambda x: x != None)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    print("neg_sample_rdd count:", neg_sample_rdd.count())

    sampled_rdd = pos_sample_rdd.flatMap(
        lambda larr: sample_item(int(sample_num / 2), larr)
    ).persist(StorageLevel.MEMORY_AND_DISK)
    print("sampled_rdd count:%d" % sampled_rdd.count())
    for x in sampled_rdd.take(5):
        print(x)
        # for x1 in x:
        x1 = x.split("\t")
        print(x1)
        print_line = ""
        for key, value in iterm_key_values.items():
            print_line += "{}:{}:feature_index: {}".format(
                key, x1[value["feature_index"]], value["feature_index"]
            )
        print(print_line)
    sampled_all = sampled_rdd.union(neg_sample_rdd)
    print("sample_all count:", sampled_all.count())
    # add freq feature for training
    ID = config_realtime["item_newsid"]["feature_index"]  # index of cmsid
    item_ids = sampled_all.map(lambda x: x.strip().split("\t")[ID])  # cmsid
    for x in item_ids.take(5):
        print(x)
    total_items = item_ids.count()
    print(f"Total items: {total_items}")
    item_clicks = item_ids.countByValue()  # cmsid: clicks
    for id_ in item_clicks.keys():
        item_clicks[id_] = item_clicks[id_] / total_items

    sorted_clicks = sorted(list(item_clicks.items()), key=lambda x: x[1], reverse=True)
    print("Item clicks: ", sorted_clicks[:50])

    g_item_clicks = sc.broadcast(item_clicks)

    sampled_all_with_freq = sampled_all.map(lambda x: add_freq_to_data(x)).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    for x in sampled_all_with_freq.take(10):
        print(x)

    FREQ = config_realtime["item_freq"][
        "feature_index"
    ]  # index of item frequency (position to fill in)
    debug_sample = sampled_all_with_freq.map(lambda x: x.split("\t")).map(
        lambda x: (x[ID], x[FREQ])
    )
    for x in debug_sample.take(10):
        print(x)

    sampled_all_with_freq.repartition(200).saveAsTextFile(output_file)

    sc.stop()
