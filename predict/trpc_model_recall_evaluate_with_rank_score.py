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
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
import requests

sys.path.append("./ktrecall")
from rondaserving_interface.pb import predict_pb2
from rondaserving_interface.pb import types_pb2
from common.predict_proxy import TensorflowTrpcProxy

import os

import json
import requests
import subprocess
import re
from tqdm import tqdm
from ast import literal_eval as make_tuple
import time

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
faiss_index = sys.argv[2] #ef_bottom_dssm1
faiss_query_num = 200
user_model_trpc = sys.argv[3]
result_save_path = sys.argv[4]
signature_name = "user_embedding"
l5_ip_port = "64956993:65536"
out_name = "user_norm"
ef_cli_path = "/cfs/cfs-3cde0a5bc/pinnie/dsssm/faiss/ef_cli"

proxy = TensorflowTrpcProxy("{}".format(user_model_trpc))
batch_size = 200
vector_dimension = 64

print(
    "feature_path: {}, faiss_index:{}, faiss_query_num: {}, user_model_trpc: {}, result_save_path: {}".format(
        feature_path,
        faiss_index,
        faiss_query_num,
        user_model_trpc,
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

def gen_fea(input_fea_str, value_type, fea_len, split_index, field_name=""):
    padding = value_type_dict[value_type]

    if value_type == "string":
        input_list = format_string(
            input_fea_str, fea_len, split_index, default_val=padding
        )
        return input_list
        # return np.asarray(input_list)
    elif value_type == "dense":
        input_list = format_float(
            input_fea_str, fea_len, split_index, default_val=padding
        )
        return input_list
        # return np.asarray(input_list)
    elif value_type == "sparse":
        input_list = format_features(
            input_fea_str, fea_len, split_index, default_val=padding
        )
        return input_list
        # return np.asarray(input_list)
    else:
        assert ValueError("value type error!")

# 纯业务代码，根据业务具体情况配置
def get_id_features_from_sample(sample_batch):
    tf_service_input = {key: [] for key in input_keys.keys()}
    input_batch_flowid = []
    for sample in sample_batch:
        field_list = sample.strip("\n").split("\t")
        assert len(field_list) == 299, "field_list: len is: {}".format(len(field_list))
        for key, value in input_keys.items():
            tf_service_input[key].append(
                gen_fea(
                    field_list[input_keys[key]["feature_index"]],
                    input_keys[key]["value_type"],
                    input_keys[key]["field_size"],
                    input_keys[key].get("split_index", 0),
                )
            )
        input_batch_flowid.append(field_list[298])
    # print("input_batch_newsid: {}".format(input_batch_newsid))
    # print("tf_service_input: {}".format(tf_service_input))
    return (input_batch_flowid, tf_service_input)

    # TODO： 处理输入， 必须实现
def build_input(sample_batch, req_type):
    # 请求包
    input_batch_flowid, tf_service_input = get_id_features_from_sample(sample_batch)
    req = predict_pb2.DensePredictRequest()
    req.req_id = str(time.time())
    req.output_filter.append(req_type)
    req.model_spec.name = ""
    req.model_spec.version = ""
    for key, value_list in tf_service_input.items():
        if input_keys[key]["value_type"] == "string":
            tf_type = types_pb2.DataType.DT_STRING
        elif input_keys[key]["value_type"] == "dense":
            tf_type = types_pb2.DataType.DT_FLOAT
        elif input_keys[key]["value_type"] == "sparse":
            tf_type = types_pb2.DataType.DT_INT64

        #     print("key: {}, tf_type: {}, value_list: {}".format(key, tf_type, value_list))
        req.inputs[key].dtype = tf_type

        dim = req.inputs[key].tensor_shape.dim.add()
        dim.size = len(value_list)
        dim = req.inputs[key].tensor_shape.dim.add()
        dim.size = input_keys[key]['field_size']

        if tf_type == dtypes.int64:
            req.inputs[key].int64_val.extend(sum(value_list, []))
        elif tf_type == dtypes.float32:
            req.inputs[key].float_val.extend(sum(value_list, []))
        elif tf_type == dtypes.string:
            req_input = sum(value_list, [])
            req_input = [bytes(x, 'utf-8') for x in req_input]
            req.inputs[key].string_val.extend(req_input)
    # print("req: {}".format(req))
    return input_batch_flowid[:], req

    # TODO： 处理输出， 必须实现
def build_output(ids, rsp, build_vector_req_type, build_vector_batch_size, build_vector_dimension):
    print("rsp: {}".format(rsp))
    res = []
    res_size = len(rsp.outputs[build_vector_req_type].float_val)
    vector_size = res_size // build_vector_batch_size
    mod = res_size % build_vector_batch_size

    if vector_size != build_vector_dimension:
        print("向量维度异常， 设置={}, 返回={}".format(build_vector_dimension, vector_size))
        return None, None

    if mod != 0:
        print("返回数据异常， 返回数据不能被")
        return None, None

    for i in range(0, build_vector_batch_size):
        res.append(rsp.outputs[build_vector_req_type].float_val[i * vector_size: (i + 1) * vector_size])
    print("batch size: {}, result number: {}".format(build_vector_batch_size, len(res)))
    return vector_size, res

def process_features_batch(features_batch):
    input_batch_flowid, req = build_input(features_batch, req_type=out_name)
    rsp = proxy.predict(req)
    vector_size, vecs = build_output(ids=input_batch_flowid, rsp=rsp,
                                     build_vector_req_type=out_name,
                                     build_vector_batch_size=batch_size, build_vector_dimension=vector_dimension)
    ips = get_faiss_address(ef_cli=ef_cli_path, l5=l5_ip_port)
    #     print(ips)
    faiss_ip = ips[0]
    print("faiss_ip: {}".format(faiss_ip))
    results = call_faiss_vec(faiss_ip, vecs, faiss_index, faiss_query_num)
    #     print("test: results {}".format(results))
    #     print(results["results"][0]["items"])

    recall_items = [x["items"] for x in results["results"] if results["success"]]
    if not results["success"]:
        print("failed to get recall items from faiss")
    # id_features["recall_items"] = recall_items
    user_recall_items = {x: y for x, y in zip(input_batch_flowid, recall_items)}
    flowids_embeddings = {x: y for x, y in zip(input_batch_flowid, vecs)}
    print("user_recall_items num: {}".format(len(user_recall_items)))
    return user_recall_items, flowids_embeddings


# In[77]:


import random


def get_features_labels_from_data():
    features = []
    flowid_labels = dict()
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
        "model": user_model_trpc,
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
