#!/usr/bin/env python
# coding: utf-8

# # 模型召回离线T+N 时长/召回率评测

# In[ ]:

#
# get_ipython().system('pip install tensorflow-serving-api==1.15')
# get_ipython().system('pip install tensorflow==1.15')
# get_ipython().system('pip install requests')


# In[65]:
import json
from tqdm import tqdm
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
from ast import literal_eval as make_tuple

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
evaluate_rate = 1.0
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

def call_faiss_ids_sort_vec(host, sort_vec, ids, index, size):
    url = 'http://{}/SearchEngine/sort'.format(host)
    payload = {
        "index":index,
        "topk": size,
        "metric": 0,
        "conds":{"sort_vec": sort_vec,
                "ids": ids}
    }
#     print("url: {}".format(url))
#     print("payload: {}".format(payload))
    r=requests.post(url, data=json.dumps(payload))
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
    flowid_ori = []
    for sample in tqdm(
        features_batch, desc="process_features_batch", total=len(features_batch)
    ):
        field_list = sample.strip("\n").split("\t")
        # assert len(field_list) == 152
        flowid_ori.append(field_list[298])
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
    user_recall_items = {x: y for x, y in zip(flowid_ori, id_features["recall_items"])}
    flowids_embeddings = {x: y for x, y in zip(flowid_ori, vecs)}
    print("user_recall_items num: {}".format(len(user_recall_items)))
    return user_recall_items, flowids_embeddings


# In[77]:


import random


def get_features_labels_from_data():
    features = []
    flowid_labels = dict()
    batch_size = 200
    part_file = feature_path
    features_batch = []
    print("evaluate file: {}".format(part_file))
    with open(os.path.join(feature_path, part_file), "r", encoding="utf-8") as f:
        for line in f:
            field_list = line.strip("\n")
            field_list = make_tuple(field_list)
            flowid = field_list[298]
            user_id = field_list[input_keys["user_id"]["feature_index"]]
            ranking_scored = field_list[299].split(",")
            ranking_scored = [x.split(":") for x in ranking_scored if x]
            ranking_scored = [[x[0] + "00", x[1]] for x in ranking_scored]
            news_id = field_list[1]
            if not news_id.endswith("00"):
                news_id = news_id + "00"

            if flowid not in flowid_labels:
                flowid_labels[flowid] = [flowid, user_id, ranking_scored]
                features_batch.append("\t".join(field_list[:299]))
                if len(features_batch) == batch_size:
                    if random.random() < evaluate_rate:
                        features.append(features_batch[:])
                    features_batch = []

            if len(features) > 5:
                break
    if features_batch:
        features.append(features_batch[:])
    print("features batches num: {}".format(len(features)))
    return features, flowid_labels

def get_recall_predictions(features):
    flowid_predictions = dict()
    flowid_features = dict()
    flowid_embeddings = dict()
    for features_batch in features:

        flowid_features.update({x.split("\t")[298]: x for x in features_batch})
        flowid_recall_items, flowid_embedding = process_features_batch(features_batch)
        flowid_predictions.update(flowid_recall_items)
        flowid_embeddings.update(flowid_embedding)


    len(features[0][0])
    print(len(flowid_predictions))
    return flowid_features, flowid_predictions, flowid_embeddings


def sort_flowid_items(flowid_embeddings, flowid_labels, topk=100):
    rate = dict()
    eval_cnt = 0
    eval_every_n = 100
    for key, vec in tqdm(flowid_embeddings.items(), total=len(flowid_embeddings), desc='sort with embeddings'):
        try:
            items = [x[0] for x in flowid_labels[key][2]]
            topk_labels = set([x[0] for x in flowid_labels[key][2][:topk]])
            ips = get_faiss_address(ef_cli=ef_cli_path, l5=l5_ip_port)
            faiss_ip = ips[0]
            # print("faiss_ip: {}".format(faiss_ip))
            result = call_faiss_ids_sort_vec(faiss_ip, vec, items, faiss_index, 1000)
            # print(result)
            if not result['result']['success']:
                print('sort failed for flowid: {}'.format(key))
            else:
                items = [x['id'] for x in result['items'][0]['docs']]
                for eval_topn in [100, 200, 400, 600, 800, 1000]:
                    overlap_cnt = len(set(items[:eval_topn]) & topk_labels)
                    eval_topn_rate = overlap_cnt / topk
                    rate_key = "eval_top_{}".format(eval_topn)
                    if rate_key in rate:
                        rate[rate_key] += eval_topn_rate
                    else:
                        rate[rate_key] = eval_topn_rate
            if eval_cnt % eval_every_n == 10:
                for key, value in rate.items():
                    print("{}:{}".format(key, value / eval_cnt))
            eval_cnt += 1
        except Exception as e:
            print(e)
    for key, value in rate.items():
        print("{}:{}".format(key, value / eval_cnt))





# In[80]:


def evaluate_recall(
    flowid_predictions,
    flowid_labels,
    flowid_features,
    filter_no_exp_overlap=False,
    score_th=0,
    save_path="./model_recall_evaluate.txt",
    rank_topk=100,
):
    total_recall_rate = 0.0
    total_label_avg_cnt = 0.0
    total_overlap_cnt = 0.0
    cnt = 0
    no_prediction = dict()
    for flowid in flowid_predictions.keys():
        predictions = flowid_predictions[flowid]
        predictions = [x for x in predictions if x["score"] > score_th]
        items_scores = flowid_labels[flowid][2]

        pred_ids = set(x["id"] for x in predictions)
        label_ids = set([x[0] for x in items_scores[:rank_topk]])


        overlap_ids = pred_ids & label_ids
        recall_rate = len(overlap_ids) / len(label_ids) if len(label_ids) > 0 else 0.0
        cnt += 1

        total_label_avg_cnt += len(label_ids)
        total_recall_rate += recall_rate
        total_overlap_cnt += len(overlap_ids)
    print(
        "data: {}, score_th: {}, filter_no_overlap: {}, recall_rate: {}, avg_label_cnt: {}, total_overlap_cnt: {}".format(
            feature_path,
            score_th,
            filter_no_exp_overlap,
            total_recall_rate / cnt,
            total_label_avg_cnt / cnt,
            total_overlap_cnt / cnt,
        )
    )
    print("total cnt: {}, cnt: {}".format(len(flowid_predictions), cnt))

    total_recall_rate = total_recall_rate / cnt
    total_label_avg_cnt = total_label_avg_cnt / cnt

    total_overlap_cnt = total_overlap_cnt / cnt
    result = {
        "total_recall_rate": total_recall_rate,
        "total_label_avg_cnt": total_label_avg_cnt,
        "total_overlap_cnt": total_overlap_cnt,
        "prediction_cnt": len(flowid_predictions),
        "evaluate_cnt": cnt,
        "model": model_name_version,
        "faiss_index": faiss_index,
        "evaluate_data": feature_path,
        "filter_no_exp_overlap": filter_no_exp_overlap,
        "score_th": score_th,
    }
    with open(save_path, "a+") as f:
        f.write(json.dumps(result) + "\n")
    return result


features, flowid_labels = get_features_labels_from_data()
flowid_features, flowid_predictions, flowid_embeddings = get_recall_predictions(features)

for rank_topk in [10, 100]:
    result = evaluate_recall(
        flowid_predictions,
        flowid_labels,
        flowid_features,
        filter_no_exp_overlap=False,
        score_th=0.2,
        save_path=result_save_path,
        rank_topk= rank_topk
    )
    result = evaluate_recall(
        flowid_predictions,
        flowid_labels,
        flowid_features,
        filter_no_exp_overlap=True,
        score_th=0.2,
        save_path=result_save_path,
        rank_topk=rank_topk
    )

sort_flowid_items(flowid_embeddings=flowid_embeddings, flowid_labels=flowid_labels,topk=100)
