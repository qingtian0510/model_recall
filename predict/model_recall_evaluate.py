#!/usr/bin/env python
# coding: utf-8

# # 模型召回离线T+N 时长/召回率评测

# In[ ]:

#
# get_ipython().system('pip install tensorflow-serving-api==1.15')
# get_ipython().system('pip install tensorflow==1.15')
# get_ipython().system('pip install requests')


# In[65]:

import sys
from tensorflow_serving.apis import predict_pb2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
import requests
from tensorflow_serving.apis import predict_pb2
import os

import json
import requests
import subprocess
import re
from tqdm import tqdm


sys.path.append("../")
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
print(sys.path)

from dssm_v2.configs.config_realtime import (
    string_padding,
    float_padding,
    feature_id_padding,
    config_realtime,
    format_string,
    format_float,
    format_features,
)

user_key_values = {
    key: value
    for key, value in config_realtime.items()
    if value["type"] == "user" and value["using"] == 1
}
print("user_key_values: {}".format(user_key_values))
input_keys = user_key_values
print("user key and values: {}".format(input_keys))
value_type_to_dtype_dict = {
    "string": dtypes.string,
    "dense": dtypes.float32,
    "sparse": dtypes.int64,
}
value_type_dict = {
    "string": string_padding,
    "dense": float_padding,
    "sparse": feature_id_padding,
}

feature_path = sys.argv[1]
evaluate_rate = 0.05
print("sys.argv: {}".format(sys.argv))
# feature_path = "/cfs/cfs-3cde0a5bc/pinnie/dsssm/data/neg_sample_user_statis/2021053110"
# model_name_version = "newsPluginAlgo_bottom_model_recall_pinnie9G.default"
model_name_version = sys.argv[2]
# faiss_index = 'ef_model_recall_pinnie_9G'
faiss_index = sys.argv[3]
faiss_query_num = 200
# user_model_taf_ip = '10.60.20.185'
user_model_taf_ip = sys.argv[4]
# user_model_taf_port = '10370'
user_model_taf_port = sys.argv[5]
result_save_path = sys.argv[6]
signature_name = "user_embedding"
l5_ip_port = "64956993:65536"
out_name = "user_norm"
ef_cli_path = "/cfs/cfs-3cde0a5bc/pinnie/dsssm/faiss/ef_cli"

print(
    "feature_path: {}, model_name_version: {}, faiss_index:{}, faiss_query_num: {}, user_model_taf_ip: {}, user_model_taf_port: {}, result_save_path: {}".format(
        feature_path,
        model_name_version,
        faiss_index,
        faiss_query_num,
        user_model_taf_ip,
        user_model_taf_port,
        result_save_path,
    )
)


def get_faiss_address(
    ef_cli="/ceph/szgpu/10929/pinnie/dssm/faiss_search/ef_cli", l5="64956993:65536"
):
    #     ./ef_cli --l5=64956993:65536 info_cluster
    status, output = subprocess.getstatusoutput(
        "{} --l5={} info_cluster".format(ef_cli, l5)
    )
    # print(status,output)
    ips = []
    if status == 0:
        for x in output.split("\n"):
            if "nodes" not in x:
                continue
            ips.extend(re.findall(r"[0-9]+(?:\.[0-9]+){3}:\d+", x))
    return ips


def call_faiss_vec(host, vecs, index, size):
    url = "http://{}/SearchEngine/search".format(host)
    payload = {
        "index": index,
        "size": size,
        "offset": 0,
        "conds": [{"query_vec": vec} for vec in vecs],
    }
    r = requests.post(url, data=json.dumps(payload))
    return json.loads(r.text)


def change_output_to_array(output_proto, out_name="user_norm"):
    # 获取返回信息，并转成array，根据业务具体情况配置，与模型保存时配置相关
    embedding_array = tf.make_ndarray(output_proto[out_name])
    return embedding_array


def get_embedding(request, venus_url, id_features, out_name="user_norm"):
    for field_name in id_features.keys():
        id_features[field_name] = np.array(id_features[field_name])
        request.inputs[field_name].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                id_features[field_name],
                dtype=value_type_to_dtype_dict[input_keys[field_name]["value_type"]],
                shape=[
                    id_features[field_name].shape[0],
                    id_features[field_name].shape[1],
                ],
            )
        )

    msg = request.SerializeToString()
    header = {"Content-Length": str(len(msg))}
    req = requests.post(url=venus_url, data=msg, headers=header, timeout=3)
    if req.status_code != requests.codes.ok:
        print("status code error: {}".format(req.status_code))
    content = req.content
    #     print("test content: {}".format(content))
    # 将结果反序列化为tensorProto
    resp = predict_pb2.PredictResponse()
    resp.ParseFromString(content)
    output_proto = resp.outputs
    #     print(output_proto)
    # 从tensorProto中获取embedding信息
    embedding_array = change_output_to_array(output_proto, out_name=out_name)
    #     print(embedding_array)
    id_features[out_name] = embedding_array.tolist()


def get_embedding_from_taf(id_features, out_name="user_norm"):
    request = predict_pb2.PredictRequest()
    # 模型名称，根据业务具体情况配置
    request.model_spec.name = model_name_version
    # 请求时签名的名称，如没有额外设置，默认参数是'serving_default'
    # request.model_spec.signature_name = 'serving_default'
    # 请求时签名的名称，根据业务具体情况配置
    request.model_spec.signature_name = signature_name
    host = user_model_taf_ip
    port = user_model_taf_port
    venus_url = "http://%s:%s/service/ner.json" % (host, port)
    get_embedding(request, venus_url, id_features, out_name=out_name)


def gen_fea(input_fea_str, value_type, fea_len, split_index, field_name=""):
    padding = value_type_dict[value_type]

    if value_type == "string":
        input_list = format_string(
            input_fea_str, fea_len, split_index, default_val=padding
        )
        return np.asarray(input_list)
    elif value_type == "dense":
        input_list = format_float(
            input_fea_str, fea_len, split_index, default_val=padding
        )
        return np.asarray(input_list)
    elif value_type == "sparse":
        input_list = format_features(
            input_fea_str, fea_len, split_index, default_val=padding
        )
        return np.asarray(input_list)
    else:
        assert ValueError("value type error!")


def process_features_batch(features_batch):
    id_features = {key: [] for key in input_keys.keys()}
    userid_ori = []
    for sample in tqdm(
        features_batch, desc="process_features_batch", total=len(features_batch)
    ):
        field_list = sample.strip("\n").split("\t")
        # assert len(field_list) == 152
        userid_ori.append(field_list[input_keys["user_id"]["feature_index"]])
        for field_name, value in input_keys.items():
            id_features[field_name].append(
                gen_fea(
                    field_list[value["feature_index"]],
                    value["value_type"],
                    value["field_size"],
                    value.get("split_index", 0),
                )
            )
    #     print("id_features num: {}".format(len(id_features)))
    get_embedding_from_taf(id_features, out_name=out_name)
    vecs = id_features[out_name]

    ips = get_faiss_address(ef_cli=ef_cli_path, l5=l5_ip_port)
    #     print(ips)
    faiss_ip = ips[0]
    print("faiss_ip: {}".format(faiss_ip))
    results = call_faiss_vec(faiss_ip, vecs, faiss_index, faiss_query_num)
    #     print("test: results {}".format(results))
    #     print(results["results"][0]["items"])

    recall_items = [x["items"] for x in results["results"] if results["success"]]
    if not results["success"]:
        print(
            "failed to get recall items from faiss: {}".format(
                id_features["userid_ori"]
            )
        )
    id_features["recall_items"] = recall_items
    user_recall_items = {x: y for x, y in zip(userid_ori, id_features["recall_items"])}
    print("user_recall_items num: {}".format(len(user_recall_items)))
    return user_recall_items


# In[77]:


import random


def get_features_labels_from_data():
    features = []
    user_labels = dict()
    batch_size = 2000
    for part_file in tqdm(
        os.listdir(feature_path),
        total=len(os.listdir(feature_path)),
        desc="get_features_labels_from_data",
    ):
        features_batch = []
        if "part-" not in part_file:
            continue
        print("evaluate file: {}".format(part_file))
        with open(os.path.join(feature_path, part_file), "r", encoding="utf-8") as f:
            for line in f:
                field_list = line.strip("\n").split("\t")
                user_id = field_list[input_keys["user_id"]["feature_index"]]
                label = field_list[17]
                watch_time = float(field_list[53])
                news_id = field_list[1]
                if not news_id.endswith("00"):
                    news_id = news_id + "00"

                if user_id not in user_labels:
                    user_labels[user_id] = [[news_id, label, watch_time]]
                    features_batch.append(line)
                    if len(features_batch) == batch_size:
                        if random.random() < evaluate_rate:
                            features.append(features_batch[:])
                        features_batch = []
                else:
                    user_labels[user_id].append([news_id, label, watch_time])
        if features_batch:
            if random.random() < evaluate_rate:
                features.append(features_batch[:])
            features_batch = []
    #     break
    print("features batches num: {}".format(len(features)))
    return features, user_labels


def get_recall_predictions(features):
    user_predictions = dict()
    user_features = dict()
    for features_batch in features:
        # try:
        user_features.update({x.split("\t")[2]: x for x in features_batch})
        user_recall_items = process_features_batch(features_batch)
        user_predictions.update(user_recall_items)
        # except Exception as e:
        #     print(e)

    len(features[0][0])
    print(len(user_predictions))
    return user_features, user_predictions


# In[80]:


def evaluate_recall(
    user_predictions,
    user_labels,
    user_features,
    filter_no_exp_overlap=False,
    score_th=0,
    save_path="./model_recall_evaluate.txt",
):
    total_time_rate = 0.0
    total_exp_rate = 0.0
    total_recall_rate = 0.0
    total_label_avg_cnt = 0.0
    total_label_time_avg = 0.0
    total_pre_time_avg = 0.0
    total_lc_rate = 0.0
    total_overlap_cnt = 0.0
    cnt = 0
    no_prediction = dict()
    for user_key in user_predictions.keys():
        predictions = user_predictions[user_key]
        predictions = [x for x in predictions if x["score"] > score_th]
        labels = user_labels[user_key]
        label_time = {x[0]: x[2] for x in labels}

        pred_ids = set(x["id"] for x in predictions)
        label_ids = set([x[0] for x in labels if x[1] == "1"])
        exp_ids = set([x[0] for x in labels])

        overlap_ids = pred_ids & label_ids
        overlap_time = sum([label_time[x] for x in overlap_ids])
        overlap_exp_ids = pred_ids & exp_ids

        total_time = sum([x[2] for x in labels if x[2] > 0])
        time_rate = overlap_time / total_time if total_time > 0 else 0.0

        exp_rate = len(overlap_exp_ids) / len(exp_ids) if len(exp_ids) > 0 else 0.0
        recall_rate = len(overlap_ids) / len(label_ids) if len(label_ids) > 0 else 0.0
        lc_rate = (
            len(overlap_ids) / len(overlap_exp_ids) if len(overlap_exp_ids) > 0 else 0.0
        )
        if len(overlap_exp_ids) == 0 and filter_no_exp_overlap:
            no_prediction[user_key] = user_predictions[user_key]
            continue
        cnt += 1

        total_label_avg_cnt += len(label_ids)
        total_time_rate += time_rate
        total_exp_rate += exp_rate
        total_recall_rate += recall_rate
        total_label_time_avg += total_time
        total_pre_time_avg += overlap_time
        total_lc_rate += lc_rate
        total_overlap_cnt += len(overlap_ids)
    print(
        "data: {}, score_th: {}, filter_no_overlap: {}, average time rate: {}, exp_rate: {}, recall_rate: {}, avg_label_cnt: {}, total_label_time_avg: {}, total_pre_time_avg: {}, total_lc_rate: {}, total_overlap_cnt: {}".format(
            feature_path,
            score_th,
            filter_no_exp_overlap,
            total_time_rate / cnt,
            total_exp_rate / cnt,
            total_recall_rate / cnt,
            total_label_avg_cnt / cnt,
            total_label_time_avg / cnt,
            total_pre_time_avg / cnt,
            total_lc_rate / cnt,
            total_overlap_cnt / cnt,
        )
    )
    print("total cnt: {}, cnt: {}".format(len(user_predictions), cnt))
    total_time_rate = total_time_rate / cnt
    total_exp_rate = total_exp_rate / cnt
    total_recall_rate = total_recall_rate / cnt
    total_label_avg_cnt = total_label_avg_cnt / cnt
    total_label_time_avg = total_label_time_avg / cnt
    total_pre_time_avg = total_pre_time_avg / cnt
    total_lc_rate = total_lc_rate / cnt
    total_overlap_cnt = total_overlap_cnt / cnt
    result = {
        "total_time_rate": total_time_rate,
        "total_exp_rate": total_exp_rate,
        "total_recall_rate": total_recall_rate,
        "total_label_avg_cnt": total_label_avg_cnt,
        "total_label_time_avg": total_label_time_avg,
        "total_pre_time_avg": total_pre_time_avg,
        "total_lc_rate": total_lc_rate,
        "total_overlap_cnt": total_overlap_cnt,
        "prediction_cnt": len(user_predictions),
        "evaluate_cnt": cnt,
        "model": model_name_version,
        "faiss_index": faiss_index,
        "evaluate_data": feature_path,
        "filter_no_exp_overlap": filter_no_exp_overlap,
        "score_th": score_th,
    }
    with open(save_path, "a+") as f:
        f.write(json.dumps(result) + "\n")
    for user_id, pred in list(no_prediction.items())[:5]:
        print(
            "user_id: {}, predictions: {}, labels: {}, features: {}".format(
                user_id, pred, user_labels[user_id], user_features[user_id]
            )
        )
        break
    return result


features, user_labels = get_features_labels_from_data()
user_features, user_predictions = get_recall_predictions(features)
for score_th in [0.4]:
    result = evaluate_recall(
        user_predictions,
        user_labels,
        user_features,
        filter_no_exp_overlap=False,
        score_th=score_th,
        save_path=result_save_path,
    )
    result = evaluate_recall(
        user_predictions,
        user_labels,
        user_features,
        filter_no_exp_overlap=True,
        score_th=score_th,
        save_path=result_save_path,
    )
