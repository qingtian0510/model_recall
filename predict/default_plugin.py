# -*- coding: UTF-8 -*-
import sys


sys.path.append("../")

from common.shell_tools import ShellTools

shell_tools = ShellTools()
cmd = "pip uninstall tensorflow -y; pip uninstall tensorflow-serving-api -y; pip install -U tensorflow==2.1.0 tensorflow-serving-api==2.1.0 --upgrade -i https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple --extra-index-url https://mirrors.tencent.com/pypi/simple/"
shell_tools.exec_shell(cmd)

import logging
import time

import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import socket
import time

from rondaserving_interface.pb import feature_pb2
from rondaserving_interface.pb import predict_pb2
from rondaserving_interface.pb import types_pb2
from common.predict_proxy import TensorflowTrpcProxy

GLOBAL_nowtimestamp = time.time()

import sys, os, re

# coding=utf-8
# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
import sys, os, re

from plugins.dssm_v2.configs.config_realtime import (
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




class Plugin:
    def gen_fea(self, input_fea_str, value_type, fea_len, split_index, field_name=""):
        padding = value_type_dict[value_type]
        assert input_fea_str != placeholder_str
        if value_type == "string":
            input_list = format_string(
                input_fea_str, fea_len, split_index, default_val=padding
            )
            return input_list
        elif value_type == "dense":
            input_list = format_float(
                input_fea_str, fea_len, split_index, default_val=padding
            )
            return input_list
        elif value_type == "sparse":
            input_list = format_features(
                input_fea_str, fea_len, split_index, default_val=padding
            )
            return input_list
        else:
            assert ValueError("value type error!")

    # 将请求结果转换成numpy.ndarray， result.shape：(n,m) 其中n为batch，m为向量维度
    # 该方法必须保留
    def change_output_to_array(self, output_proto):
        # 获取返回信息，并转成array，根据业务具体情况配置，与模型保存时配置相关
        embedding_array = tf.make_ndarray(output_proto["item_norm"])
        return embedding_array

    # 纯业务代码，根据业务具体情况配置
    def get_id_features_from_sample(self, sample_batch):
        tf_service_input = {key: [] for key in input_keys.keys()}
        input_batch_newsid = []
        for sample in sample_batch:
            field_list = sample.strip("\n").split("\t")
            assert len(field_list) == 300
            for key, value in input_keys.items():
                tf_service_input[key].append(
                    self.gen_fea(
                        field_list[input_keys[key]["feature_index"]],
                        input_keys[key]["value_type"],
                        input_keys[key]["field_size"],
                        input_keys[key].get("split_index", 0),
                    )
                )
            cmsid = field_list[input_keys["item_newsid"]["feature_index"]]
            if cmsid.endswith("00") == False:
                cmsid = cmsid + "00"
            cat2 = field_list[input_keys["item_news_category2"]["feature_index"]]
            input_batch_newsid.append([cmsid, cat2])
        # print("input_batch_newsid: {}".format(input_batch_newsid))
        # print("tf_service_input: {}".format(tf_service_input))
        return (input_batch_newsid, tf_service_input)

    # TODO: 建立读取数据迭代器， 必须实现
    # 建立读取数据迭代器，根据业务具体情况配置
    def read_dateset_iter(self, data_path):
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

    # TODO： 处理输入， 必须实现
    def build_input(self, sample_batch, req_type):
        # 请求包
        input_batch_newsid, tf_service_input = self.get_id_features_from_sample(sample_batch)
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
        return input_batch_newsid[:], req

    # TODO： 处理输出， 必须实现
    def build_output(self, ids, rsp, build_vector_req_type, build_vector_batch_size, build_vector_dimension):
        # print("rsp: {}".format(rsp))
        res = []
        res_size = len(rsp.outputs[build_vector_req_type].float_val)
        vector_size = res_size // build_vector_batch_size
        mod = res_size % build_vector_batch_size

        if vector_size != build_vector_dimension:
            logging.error("向量维度异常， 设置={}, 返回={}".format(build_vector_dimension, vector_size))
            return None, None

        if mod != 0:
            logging.error("返回数据异常， 返回数据不能被")
            return None, None

        for i in range(0, build_vector_batch_size):
            assert len(ids[i]) == 2, "ids[i] must contain id, cat2"
            res.append(
                "{}|{}|{}".format(ids[i][0], ids[i][1], ",".join(
                    [str(x) for x in
                     rsp.outputs[build_vector_req_type].float_val[i * vector_size:(i + 1) * vector_size]])))

        return vector_size, "\n".join(res)


if __name__ == '__main__':
    fh = logging.FileHandler("./plug.log", mode='a', encoding='utf-8', delay=False)
    logging.basicConfig(handlers=[fh],
                        format='[%(asctime)s %(levelname)s]<%(process)d> %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    proxy = TensorflowTrpcProxy("trpc.Serving.BottomDssmItem1ServerTF.PredictInterfaceObj")
    plugin = Plugin()

    input_file_name = "../data/trpc.Serving.TFRecallDemoItemServerTF.log"
    build_vector_batch_size = 10
    build_vector_req_type = "item_norm"
    build_vector_dimension = 32

    samples = []
    cnt = 0
    res_success = 0
    res_fail = 0
    for line in plugin.read_dateset_iter(input_file_name):
        samples.append(line.strip())
        cnt = cnt + 1
        if cnt >= build_vector_batch_size:
            ids, req = plugin.build_input(samples, build_vector_req_type)
            rsp = proxy.predict(req)
            if rsp is None:
                res_fail = res_fail + build_vector_batch_size
            else:
                # print("rsp: {}".format(rsp))
                vector_size, res = plugin.build_output(ids, rsp, build_vector_batch_size, build_vector_dimension)
                # print(res)
                res_success = res_success + build_vector_batch_size
            samples = []
            cnt = 0

    if cnt > 0:
        tmp_batch_size = len(samples)
        ids, req = plugin.build_input(samples, build_vector_req_type)
        rsp = proxy.predict(req)
        if rsp is None:
            res_fail = res_fail + tmp_batch_size
        else:
            vector_size, res = plugin.build_output(ids, rsp, tmp_batch_size, build_vector_dimension)
            # print(res)
            res_success = res_success + tmp_batch_size
        samples = []
        cnt = 0
    # print(res_success, res_fail)
