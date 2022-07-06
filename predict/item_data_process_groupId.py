# -*- coding: UTF-8 -*-
import sys, os, re

from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import socket
import time
import requests
from dssm_v2.configs.config_realtime_groupId import (
    config_realtime,
    string_padding,
    float_padding,
    feature_id_padding,
    placeholder_str,
    format_string,
    format_features,
    format_float,
)

value_type_dict = {
    "string": string_padding,
    "dense": float_padding,
    "sparse": feature_id_padding,
}
input_keys = {
    key: value
    for key, value in config_realtime.items()
    if value["type"] == "item" and value["using"] == 1
}
value_type_to_dtype_dict = {
    "string": dtypes.string,
    "dense": dtypes.float32,
    "sparse": dtypes.int64,
}


def gen_fea(input_fea_str, value_type, fea_len, split_index, field_name=""):
    padding = value_type_dict[value_type]
    assert input_fea_str != placeholder_str
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


# 将输入样本数据转换成请求打分的tensor格式，并加入到request中
# 该方法必须保留
def change_sample_add_request(sample_batch, request):
    # 将批量样本转成input_batch_features，根据业务具体情况配置

    input_batch_newsid, tf_service_input = get_id_features_from_sample(sample_batch)
    for key, value_list in tf_service_input.items():
        value_nparray = np.array(value_list)
        request.inputs[key].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                value_nparray,
                dtype=value_type_to_dtype_dict[input_keys[key]["value_type"]],
                shape=[value_nparray.shape[0], value_nparray.shape[1]],
            )
        )

    return (input_batch_newsid[:], request)


# 将请求结果转换成numpy.ndarray， result.shape：(n,m) 其中n为batch，m为向量维度
# 该方法必须保留
def change_output_to_array(output_proto):
    # 获取返回信息，并转成array，根据业务具体情况配置，与模型保存时配置相关
    embedding_array = tf.make_ndarray(output_proto["item_norm"])
    return embedding_array


# 建立读取数据迭代器，根据业务具体情况配置
def read_dateset_iter(data_path):
    dataset = []
    if tf.io.gfile.isdir(data_path):
        # 输入为一个目录
        # 则读入目录下所有的文件
        for filename in tf.io.gfile.listdir(data_path):
            if filename.startswith("."):
                continue
            file_path = os.path.join(data_path, filename)
            with tf.io.gfile.GFile(file_path, "rb") as input:
                for line in input:
                    try:
                        line = line.decode(encoding="utf8")
                    except UnicodeDecodeError:
                        continue
                    line = line.strip()
                    if line == "":
                        continue
                    yield line
    else:
        # 输入为一个文件
        with tf.io.gfile.GFile(data_path, "rb") as input:
            for line in input:
                try:
                    line = line.decode(encoding="utf8")
                except UnicodeDecodeError:
                    continue
                line = line.strip()
                if line == "":
                    continue
                yield line


# 纯业务代码，根据业务具体情况配置
def get_id_features_from_sample(sample_batch):
    tf_service_input = {key: [] for key in input_keys.keys()}
    input_batch_newsid = []
    for sample in sample_batch:
        field_list = sample.strip("\n").split("\t")
        assert len(field_list) == 300
        for key, value in input_keys.items():
            tf_service_input[key].append(
                gen_fea(
                    field_list[input_keys[key]["feature_index"]],
                    input_keys[key]["value_type"],
                    input_keys[key]["field_size"],
                    input_keys[key].get("split_index", 0),
                )
            )
        cmsid = field_list[input_keys["item_newsid"]["feature_index"]]
        if cmsid.endswith("00") == False:
            cmsid = cmsid + "00"
        input_batch_newsid.append(cmsid)
    # print("input_batch_newsid: {}".format(input_batch_newsid))
    # print("tf_service_input: {}".format(tf_service_input))
    return (input_batch_newsid, tf_service_input)


# 用于业务接入时本地测试
def test_predict(host, port, model_spec_name, model_spec_signature_name, data_path):
    # 初始化请求
    request = predict_pb2.PredictRequest()
    # 模型名称，根据业务具体情况配置
    request.model_spec.name = model_spec_name
    # 请求时签名的名称，如没有额外设置，默认参数是'serving_default'
    # request.model_spec.signature_name = 'serving_default'
    # 请求时签名的名称，根据业务具体情况配置
    request.model_spec.signature_name = model_spec_signature_name
    venus_url = "http://%s:%s/service/ner.json" % (host, port)
    print("test")
    # 从文件中批量读取样本数据，并请求打分，根据业务具体情况配置
    sample_batch = []
    batch_size = 2
    for sample in read_dateset_iter(data_path):
        sample_batch.append(sample)
        if len(sample_batch) == batch_size:
            input_batch_ids, request = change_sample_add_request(sample_batch, request)
            msg = request.SerializeToString()
            header = {"Content-Length": str(len(msg))}
            req = requests.post(url=venus_url, data=msg, headers=header, timeout=1)
            if req.status_code != requests.codes.ok:
                print("status code error: {}".format(req.status_code))
                continue
            content = req.content
            # 将结果反序列化为tensorProto
            resp = predict_pb2.PredictResponse()
            resp.ParseFromString(content)
            output_proto = resp.outputs

            # 从tensorProto中获取embedding信息
            embedding_array = change_output_to_array(output_proto)
            print(embedding_array)
            sample_batch = []
            # break


if __name__ == "__main__":
    # data_path = "/cfs/cfs-3cde0a5bc/pinnie/dsssm/data/item_data/ft_local/part-00000"
    data_path = "/data/model_recall/dssm/data/dssm_with_statis_half_realneg/iterm_data/ft_local/part-00000-item"
    host = "9.44.33.72"
    port = 10188
    model_spec_name = "newsPluginAlgo_bottom_model_recall_pinnie9G.default"
    model_spec_signature_name = "item_embedding"
    test_predict(host, port, model_spec_name, model_spec_signature_name, data_path)
