# coding=utf-8

from pyspark.sql import SparkSession
from pyspark import SparkContext
import sys
from optparse import OptionParser
from pyspark.storagelevel import StorageLevel
import datetime
from datetime import datetime, timedelta
import json
import random
import hashlib
import os

from dssm_v2.configs.config_realtime import config_realtime, placeholder_str

ITEM_FIELDS_DICT = {
    key: value for key, value in config_realtime.items() if value["type"] == "item"
}
print("iterm features dict: {}".format(ITEM_FIELDS_DICT))
# get news info from news index (without QA news and short video news)


def getNewsInfo_index(line):
    try:
        feature_line = [
            placeholder_str
        ] * 300  # 定义一个超长的feature line 和训练的时候要对齐。item 预测的时候要替换这个占位符
        newsinfo = json.loads(line.strip())
        cmsid = newsinfo.get("ID", None)
        if cmsid.endswith("00"):
            cmsid = cmsid[:-2]

        for key, value in ITEM_FIELDS_DICT.items():
            feature_index = value["feature_index"]
            index_name = value["index_name"]
            if key == "item_news_tagterm":
                tags = []
                index_value = newsinfo.get(index_name, [])
                for tag in index_value:
                    if tag.get("k") != None:
                        tags.append(tag.get("k"))
                tag = ",".join([t.split("|")[0] for t in tags])
                feature_line[feature_index] = tag
            elif key == "item_news_category1":
                cat = newsinfo.get("CATEGORY", [])
                cat1 = cat[0].split("|")[0] if len(cat) > 0 and "|" in cat[0] else ""
                feature_line[feature_index] = cat1
            elif key == "item_news_category2":
                cat = newsinfo.get("CATEGORY", [])
                cat2 = cat[1].split("|")[0] if len(cat) > 1 and "|" in cat[1] else ""
                feature_line[feature_index] = cat2
            elif key == "item_newsid":
                cmsid = newsinfo.get(index_name, "")
                if cmsid.endswith("00"):
                    cmsid = cmsid[:-2]
                feature_line[feature_index] = cmsid
            else:
                index_value = newsinfo.get(index_name, "")
                feature_line[feature_index] = index_value
        feature_line = [str(x).strip() for x in feature_line]
        line = "\t".join(feature_line)
        return (cmsid, line)
    except:
        pass
    return None


def getNewsInfo(line):
    try:
        res = getNewsInfo_index(line)
        if res == None:
            return None
        cmsid, newsinfo = res
        if cmsid == None:
            return None
        return (cmsid, newsinfo)
    except:
        pass
    return None


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


def get_filelist_by_patten(sc, input_path):
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    FileStatus = sc._gateway.jvm.org.apache.hadoop.fs.FileStatus
    fs = FileSystem.get(URI(input_path), sc._jsc.hadoopConfiguration())

    listdir = fs.globStatus(Path(input_path))
    if len(listdir) > 0:
        return listdir[-1].getPath().toString()
    else:
        return None


if __name__ == "__main__":
    index_root = sys.argv[1]
    output_dir = sys.argv[2]
    # mdfs://cloudhdfs/newspluginalgo/data/shueli/bottom/indexB/ mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/bottom_realtime/only_item_feature/%YYYYMMDDHH%
    # mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/video_bottom_half_realneg/only_item_feature/%YYYYMMDDHH%
    ss = SparkSession.builder.appName("modelrec_item_featuremaker").getOrCreate()

    sc = ss.sparkContext

    try:
        print("try to delete ", output_dir)
        deletPath_b(sc, output_dir)
    except:
        print("No file to remove")

    index_root_list = index_root.split(",")
    indexfile = get_newest_file(sc, index_root_list[0])
    for one_index in index_root_list[1:]:
        indexfile = indexfile + "," + get_newest_file(sc, one_index)
    print("using indexfile: %s" % indexfile)

    newsinfo_rdd = (
        sc.textFile(indexfile)
        .map(lambda line: getNewsInfo(line))
        .filter(lambda x: x is not None)
        .reduceByKey(lambda x, y: x)
        .map(lambda item: item[1])
    )
    print("newsinfo_rdd size: %d" % newsinfo_rdd.count())

    newsinfo_rdd.repartition(80).saveAsTextFile(output_dir)
