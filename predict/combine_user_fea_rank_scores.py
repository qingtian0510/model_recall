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
from dssm_v2.configs.config_realtime import placeholder_str, config_realtime
config2user_report_name = {
    "user_age": "user_age",
    "user_category1" : "user_cat1",
    "user_category1_score": "user_cat1",
    "user_category2": "user_cat2",
    "user_category2_score": "user_cat2",
    "user_city": "user_city",
    "user_history": "user_click_news",
    "user_realtime_pos_video": "realtime_pos_video",
    "user_realtime_neg_video": "realtime_neg_video",
    "user_id": "user_id",
    "user_sex": "user_gender",
    "user_tag_terms": "user_tags",
    "user_tag_terms_score": "user_tag_terms_score",
    "user_top_newscat1": "topnews_video_cat1",
    "user_top_newscat2": "topnews_video_cat2",
    "user_top_newsid": "topnews_id",
    "user_top_newstags": "topnews_video_tags",
    "user_video_category1": "user_video_cat1",
    "user_video_category1_score": "user_video_cat1",
    "user_video_category2": "user_video_cat2",
    "user_video_category2_score": "user_video_cat2",
    "user_video_tag": "user_video_tags",
    "user_video_tag_score": "user_video_tags",
    'user_PAGE_START': "page_start",
}

user_features = {key: value for key, value in config_realtime.items() if value['type'] == 'user' and value["using"] == 1}
user_features["user_PAGE_START"] = config_realtime["user_PAGE_START"]
user_features2report_name = {key: config2user_report_name[key] for key, value in user_features.items()}

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


def convert_report_to_features(row):
#     print(row)
    line = [placeholder_str] * 300
    line[299] = row['sorted_docs']
    line[298] = row['flowid']
    for key, value in user_features.items():
        feature_index = value['feature_index']
        line[feature_index] = row[user_features2report_name[key]]
#     print(line)
    return line


if __name__ == "__main__":
    date_str = sys.argv[1]
    output_root = sys.argv[2]
    page_start = sys.argv[3]
    print("date str: {}, output root: {}, page start: {}".format(date_str, output_root, page_start))
    ss = SparkSession.builder.appName("conbine_user_feature_and_rank_scores").getOrCreate()
    sc = ss.sparkContext
    # %YYYYMMDDHH% mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/landing_bottom_list_realtime/combine_user_feat_rank_scores/%YYYYMMDDHH%
    try:
        deletPath_b(sc, output_root)
    except:
        print("No file to remove")

    print('start to read ranking scores')
    sql = 'select flowid, user_id, topnews_id, sorted_docs from szrecone_generalrecalgo_interface.t_sh_atta_v2_03200056125 where ds in ({})'.format(date_str)
    ranking_scored_docs = ss.sql(sql)
    print(ranking_scored_docs.take(1))

    # hive表的读取，对于spark.sql操作时使用oms场景表全名进行操作
    print('start to read user and top news feature')
    sql = "select * from szrecone_generalrecalgo.t_sh_atta_v1_08100046780 where ds in ({})".format(date_str)
    ranking_user_features = ss.sql(sql)
    print(ranking_user_features.take(1))

    user_feature_scores = ranking_user_features.join(ranking_scored_docs,
                                                     ranking_scored_docs.flowid == ranking_user_features.flowid)
    print('user_feature_scores: {}, ranking_user_features: {}, ranking_scored_docs: {}'.format(
        user_feature_scores.count(), ranking_user_features.count(), ranking_scored_docs.count()))
    print('user_feature_scores.take(1)[0].__fields__: {}'.format(user_feature_scores.take(1)[0].__fields__))

    user_feature_scores_rdd = user_feature_scores.rdd.map(lambda x: convert_report_to_features(x))
    for x in user_feature_scores_rdd.take(5):
        print(x)
    user_feature_scores_rdd = user_feature_scores_rdd.filter(lambda x: x[config_realtime["user_PAGE_START"]["feature_index"]] == page_start)
    print("user_feature_scores_rdd count: {}".format(user_feature_scores_rdd.count()))
    print('save user, top news features and ranking scores in 299th faeture to {}'.format(output_root))
    user_feature_scores_rdd.repartition(50).saveAsTextFile(output_root)

    sc.stop()