#-*- coding=utf-8 -*-

import sys
import os
import time
import json
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, Row
from pyspark.sql import HiveContext

'''
%YYYYMMDDHH%
'''

if __name__ == "__main__":
    date_str = sys.argv[1]
    output_hdfs = sys.argv[2]
#     %YYYYMMDDHH% mdfs://cloudhdfs/newspluginalgo/user/pinnie/model_recall/bottom_add_feature/boss_5890_vtime

    print (output_hdfs)

    ss = SparkSession \
        .builder \
        .appName("guangjupeng_user_action_5890") \
        .getOrCreate()

    sc = ss.sparkContext

    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    URI = sc._gateway.jvm.java.net.URI
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fs = FileSystem.get(URI("mdfs://cloudhdfs/newsclient"),sc._jsc.hadoopConfiguration())

    output_hdfs = os.path.join(output_hdfs, date_str)

    if fs.exists(Path(output_hdfs)):
        fs.delete(Path(output_hdfs))


    def parse_5890_line(row):
        appid = row["id"]
        time_int = -1

        try:
            time_int = int(time.mktime(time.strptime(row["ftime"][:-1], "%Y-%m-%d %H:%M:%S")))
        except:
            time_int = -1
        if time_int < 0:
            try:
                time_int = int(time.mktime(time.strptime(row["ftime"], "%Y-%m-%d %H:%M:%S")))
            except:
                time_int = -1
        if time_int < 0:
            return None

        try:
            kv_dict = json.loads(row["kv"].replace('\\\\,', ','))
        except:
            kv_dict = {}

        if not kv_dict:
            try:
                kv_dict = json.loads(row["kv"].replace('\\\\,', ',').replace('\\', '').replace("\"{", "{").replace("}\"", "}"))
            except:
                kv_dict = {}

        if not kv_dict:
            return None

        extraInfo = kv_dict.get("extraInfo", {})
        if not isinstance(extraInfo, dict):
            try:
                extraInfo = json.loads(extraInfo)
            except:
                extraInfo = {}
        if not extraInfo:
            extraInfo = kv_dict.get("76", {})
        if not isinstance(extraInfo, dict):
            try:
                extraInfo = json.loads(extraInfo)
            except:
                extraInfo = {}


        report_item_params = extraInfo.get("report_item_params", {})

        if not isinstance(report_item_params, dict):
            try:
                report_item_params = json.loads(report_item_params)
            except:
                report_item_params = {}

        imei = ""
        nIMEI = ""
        idfv = ""
        idfa = ""

        userid = ""

        if appid == "1100678685": # android
            imei = kv_dict.get("imei", "")
            if len(imei) < 5:
                imei = extraInfo.get("imei", "")
            nIMEI = kv_dict.get("IMEI", "")
            if len(nIMEI) < 5:
                nIMEI = extraInfo.get("IMEI", "")
            imsi = kv_dict.get("imsi", "")
            if len(imsi) < 5:
                imsi = extraInfo.get("imsi", "")

            if imei.lower() in ["none", "null"]:
                imei = ""
            if nIMEI.lower() in ["none", "null"]:
                nIMEI = ""
            if imsi.lower() in ["none", "null"]:
                imsi = ""

            if len(nIMEI) > 4:
                userid = nIMEI + "_" + imsi
            elif len(imei) > 4:
                userid = imei + "_" + imsi
            else:
                userid = "_" + imsi

        else:
            idfv = kv_dict.get("idfv", "")
            if len(idfv) < 5:
                idfv = extraInfo.get("idfv", "")
            idfa = kv_dict.get("idfa", "")
            if len(idfa) < 5:
                idfa = extraInfo.get("idfa", "")

            if idfv.lower() in ["none", "null"]:
                idfv = ""
            if idfa.lower() in ["none", "null"]:
                idfa = ""
            userid = idfv

        plugin_openid = ''
        try:
            extinfo_str = report_item_params.get('extinfo', '')
            if not extinfo_str:
                extinfo_str = extraInfo.get('extinfo', '')
            if extinfo_str.startswith('openid='):
                plugin_openid = extinfo_str.split('=')[1]
        except:
            plugin_openid = ''

        if plugin_openid[:6] != "o04IBA":
            plugin_openid = ''

        if (not userid or len(userid) < 10) and (not plugin_openid or len(plugin_openid) < 10):
            return None

        newsid = kv_dict.get("newsId", "")
        if len(newsid) < 5:
            newsid = extraInfo.get("newsId", "")
        if len(newsid) < 5:
            newsid = extraInfo.get("newsID", "")
        if len(newsid) < 5:
            newsid = extraInfo.get("articleId", "")
        if len(newsid) < 5:
            newsid = report_item_params.get("newsID", "")

        if not newsid or len(newsid) < 10:
            return None

        userqq = kv_dict.get("qq", "")
        if len(userqq) < 5 or len(userqq) > 15:
            userqq = extraInfo.get("qq", "")

        if len(userqq) < 5 or len(userqq) > 15:
            userqq = ""

        userwx = kv_dict.get("openid", "")
        if len(userwx) < 10 or userwx[:6] != "oI6CFj":
            userwx = kv_dict.get("wx_openid", "")
        if len(userwx) < 10 or userwx[:6] != "oI6CFj":
            userwx = extraInfo.get("OpenId", "")
        if len(userwx) < 10 or userwx[:6] != "oI6CFj":
            userwx = ""

        newstype = "U"
        if newsid[8] == "V" or newsid[8] == "S":
            newstype = "V"
        if not newstype and kv_dict.get("articletype", "") in ["4", "56", "101", "118"]:
            newstype = "V"
        if not newstype and kv_dict.get("articleType", "") in ["4", "56", "101", "118"]:
            newstype = "V"
        if not newstype and extraInfo.get("articletype", "") in ["4", "56", "101", "118"]:
            newstype = "V"
        if not newstype and extraInfo.get("articleType", "") in ["4", "56", "101", "118"]:
            newstype = "V"
        if not newstype and report_item_params.get("articletype", "") in ["4", "56", "101", "118"]:
            newstype = "V"
        if not newstype and report_item_params.get("articleType", "") in ["4", "56", "101", "118"]:
            newstype = "V"
        # if not newstype:
        #     return None

        try:
            duration = float(kv_dict.get('play', '-1'))
            if duration == -1:
                duration = float(kv_dict.get('209', '-1'))
            if duration < 0:
                duration = 0
            if appid == '1100678685':
                duration = duration / 1000
        except:
            duration = 0

        try:
            vtime = float(kv_dict.get('vtime', '-1'))
            if vtime == -1:
                vtime = float(kv_dict.get('226', '-1'))
            if vtime == -1:
                vtime = float(report_item_params.get('videoTimeLen', '-1'))
        except:
            vtime = -1

        action = 2

        sr = ""
        sr = kv_dict.get("pagestartfrom", "")
        if not sr:
            sr = extraInfo.get("pagestartfrom", "")
        if not sr:
            sr = report_item_params.get("pagestartfrom", "")

        if sr.find("$") != -1 or sr.find(",") != -1 or sr.find(":") != -1 or sr.find(" ") != -1 or sr.find("\n") != -1 or sr.find("{") != -1 or sr.find("|") != -1 or sr.find("<") != -1:
            sr = ""

        if len(sr) > 50:
            sr = ""

        if not sr:
            sr = "unk"

        ch = ""
        ch = kv_dict.get("channelID", "")
        if not ch:
            ch = kv_dict.get("currentChannelId", "")
        if not ch:
            ch = kv_dict.get("channel_id", "")
        if not ch:
            ch = kv_dict.get("chlid", "")
        if not ch:
            ch = kv_dict.get("channel", "")

        if not ch:
            ch = extraInfo.get("channelID", "")
        if not ch:
            ch = extraInfo.get("currentChannelId", "")
        if not ch:
            ch = extraInfo.get("channel_id", "")
        if not ch:
            ch = extraInfo.get("chlid", "")
        if not ch:
            ch = extraInfo.get("channel", "")

        if not ch:
            ch = report_item_params.get("channelID", "")
        if not ch:
            ch = report_item_params.get("currentChannelId", "")
        if not ch:
            ch = report_item_params.get("channel_id", "")
        if not ch:
            ch = report_item_params.get("chlid", "")
        if not ch:
            ch = report_item_params.get("channel", "")

        if len(ch) > 50:
            ch = ""

        if not ch:
            ch = "unk"



        return ','.join([userid, str(time_int), newsid, sr, ch, str(action), str(duration) + "|" + str(vtime), newstype, idfa, userqq, userwx, plugin_openid])


    app_rdd = ss.sql("select * from sz1_newsclient_interface.t_sh_atta_v2_zaf00005890 where ds=%s" % date_str).rdd

    app_rdd = app_rdd.map(lambda x: parse_5890_line(x))\
        .filter(lambda x: x!=None)\
        .repartition(200)\
        .saveAsTextFile(output_hdfs, "org.apache.hadoop.io.compress.GzipCodec")
