import sys, os, re

from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import socket
import traceback
from tensorflow_serving.apis import prediction_service_pb2_grpc
import requests


def load_and_predict_pbmodel(model_path, t_sample, location_table):
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], model_path)
        signature = meta_graph_def.signature_def["user_embedding"]
        feed_dict = {
            tf.saved_model.utils.get_tensor_from_tensor_info(
                signature.inputs[k]
            ): t_sample[v["ret_index"]]
            for k, v in location_table.items()
        }
        ret = sess.run(
            {
                k: tf.saved_model.utils.get_tensor_from_tensor_info(x)
                for k, x in signature.outputs.items()
            },
            feed_dict=feed_dict,
        )

    return ret


def call_tfserving(
    host, model_spec_name, model_signature, sample_batch, location_table
):
    headers = {"content-type": "application/json"}
    tf_req = predict_pb2.PredictRequest()
    tf_req.model_spec.signature_name = model_signature
    tf_req.model_spec.name = model_spec_name
    for k, v in location_table.items():
        tf_req.inputs[k].CopyFrom(
            tf.make_tensor_proto(
                sample_batch[v["ret_index"]],
                dtype=dtypes.int64,
                shape=sample_batch[v["ret_index"]].shape,
            )
        )
    json_response = requests.post(
        "http://{}/service/ner.json".format(host),
        data=tf_req.SerializePartialToString(),
        headers=headers,
    )

    resp = predict_pb2.PredictResponse()
    resp.ParseFromString(json_response.content)

    output_proto = resp.outputs
    ret = {k: tf.make_ndarray(v) for k, v in output_proto.items()}
    return ret
