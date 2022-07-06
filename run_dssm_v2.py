import glob
import json
import os
import uuid

import tensorflow as tf
import argparse

from dssm_v2.utils.train_utils import input_fn, LOG, load_module_class, add_log_file
from datetime import datetime

import numpy as np
import shutil

from dssm_v2.evaluate.generate_test_data import generate_test_data


def run(args):
    gpu_info = {}
    log_steps = args.log_steps
    estimator_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count=gpu_info),
        log_step_count_steps=log_steps,
        save_summary_steps=log_steps,
        keep_checkpoint_max=3,
    )

    """
    data and model version deprecated;
    we only use v1
    """

    input_fn_class = load_module_class(
        name=args.data_version, module_name="dssm_v2.data"
    )
    LOG("input_fn_class", input_fn_class)
    (
        data_format,
        user_ids,
        item_ids,
        label_name,
        export_placeholders,
    ) = input_fn_class.get_data_format_fn_and_ids(args)
    user_item_ids = {"user_ids": user_ids, "item_ids": item_ids}
    with open(os.path.join(args.model_dir, "user_item_ids.json"), "w") as f:
        f.write(json.dumps(user_item_ids))
    model_class = load_module_class(
        name=args.model_version, module_name="dssm_v2.models"
    )
    LOG("input_fn_class", model_class)
    model_fn = model_class.get_model_fn(
        user_ids=user_ids,
        item_ids=item_ids,
        filter_default=args.filter_default.lower() == "true",
    )

    LOG("user keys:", user_ids.keys())
    LOG("item keys:", item_ids.keys())

    data_files = glob.glob(args.data_dir)
    LOG("data_files", data_files)
    dssm_input_fn = lambda: input_fn(
        file_names=data_files,
        label_name=label_name,
        data_format=data_format,
        batch_size=int(args.batch_size),
        num_epochs=args.num_epoch,
        num_threads=int(args.num_threads),
        prefetch=args.prefetch,
        perform_shuffle=bool(args.perform_shuffle),
    )

    DSSM = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params=args,
        config=estimator_config,
    )

    if args.estimator_mode == "export":
        from dssm_v2.utils import model_utils

        LOG("START EXPORT MODEL")
        # model_fn_export = dssm_v1.get_model_fn(user_ids=user_ids, item_ids=item_ids, filter_default=True)
        DSSM_export = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=args.model_dir,
            params=args,
            config=estimator_config,
        )
        raw_serving_input_fn = (
            lambda: tf.estimator.export.build_raw_serving_input_receiver_fn(
                export_placeholders
            )()
        )
        export_dict = {
            "user_embedding": {
                "inputs": list(user_ids.keys()),
                "outputs": ["user_norm"],
            },
            "item_embedding": {
                "inputs": list(item_ids.keys()),
                "outputs": ["item_norm"],
            },
        }
        export_dict_keys = {"user_embedding": user_ids, "item_embedding": item_ids}
        print("export_dict: {}".format(export_dict_keys))
        model_dir = model_utils.export_all_saved_models(
            classifier=DSSM_export,
            model_dir=args.model_dir,
            export_dir_base=os.path.join(args.model_dir, "export_test"),
            input_receiver_fn_map={tf.estimator.ModeKeys.PREDICT: raw_serving_input_fn},
            assets_extra=None,
            as_text=False,
            checkpoint_path=None,
            strip_default_attrs=True,
            export_dict=export_dict
            # inputs_keys=user_ids.keys(),
            # outputs_keys=['user_norm']
        )
        LOG("exported dir", model_dir)
        if not os.path.exists(os.path.join(args.model_dir, "online")):
            os.makedirs(os.path.join(args.model_dir, "online"), exist_ok=True)

        target_dir = os.path.join(args.model_dir, "online/{}".format(args.date))
        if args.date:
            LOG("need move data", target_dir)
            shutil.move(model_dir, target_dir)

    elif args.estimator_mode == "exported_model_check":
        LOG("EXPORTED_MODEL_CHECK")
        os.mkdir(os.path.join(args.model_dir, "test_data"))
        pickle_file = os.path.join(args.model_dir, "test_data/pickle_file")
        tfrecord_file = os.path.join(args.model_dir, "test_data/tfrecord_file")
        results = os.path.join(args.model_dir, "test_data/results.npy")
        evaluate_data_fn = lambda: input_fn(
            file_names=tfrecord_file,
            label_name=label_name,
            data_format=data_format,
            batch_size=int(args.batch_size),
            num_epochs=1,
            num_threads=int(args.num_threads),
            prefetch=args.prefetch,
            perform_shuffle=False,
        )
        generate_test_data(user_ids, item_ids, label_name, pickle_file, tfrecord_file)
        user_norms = []
        for x in DSSM.predict(input_fn=evaluate_data_fn):
            # print(x)
            user_norms.append(np.expand_dims(x["user_norm"], axis=0))
        np.save(results, np.concatenate(user_norms, axis=0))

    elif args.estimator_mode == "training" or args.estimator_mode == "train":
        LOG("START TRAIN")
        DSSM.train(input_fn=dssm_input_fn)
        LOG("FINISH TRAINING")
    elif args.estimator_mode == "eval":
        LOG("START EVAL")
        eval_results = DSSM.evaluate(input_fn=dssm_input_fn, steps=1000)
        if args.eval_records:
            with open(args.eval_records, "a") as fout:
                fout.writelines(
                    "TASK_ID:{}\nDate:{}\nFile:{}\nAUC:{}\n".format(
                        args.task_id, datetime.now(), args.data_dir, eval_results
                    )
                )
    elif args.estimator_mode == "predict":
        LOG("START PREDICT")
        with open(args.predict_file, "w") as fout:
            for x in DSSM.predict(input_fn=dssm_input_fn):
                sx = ",".join(x.keys())
                sx = "\t".join(
                    [sx] + [",".join([str(z) for z in x[y]]) for y in x.keys()]
                )
                fout.write(sx + "\n")
        LOG("DONE PREDICT")
    else:
        LOG("Unknown operation.")

    # BACKUP CHECKPOINT
    if args.backup_model.lower() == "true":
        LOG("BACK UP MODEL")
        backup_dir = os.path.join(args.model_dir, "ckpt_backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)
        target_ckpt_dir = os.path.join(backup_dir, datetime.now().strftime("%Y%m%d%H"))
        if not os.path.exists(target_ckpt_dir):
            os.makedirs(target_ckpt_dir, exist_ok=True)
        LOG("TARGET DIR", target_ckpt_dir)

        checkpoint_cfg_path = os.path.join(args.model_dir, "checkpoint")
        if os.path.exists(checkpoint_cfg_path):
            model_ckpt_version = ""
            for line in open(checkpoint_cfg_path, "r"):
                line_arr = line.strip().split(":")
                if len(line_arr) > 1 and line_arr[0].strip() == "model_checkpoint_path":
                    model_ckpt_version = line_arr[1].strip().replace('"', "")
                    break
            if model_ckpt_version == "":
                LOG("CAN NOT FIND VALID MODEL CHECKPOINT")
            else:
                model_ckpt_version = os.path.join(args.model_dir, model_ckpt_version)
                LOG("START COPY NEWEST MODEL CHECKPOINT:", model_ckpt_version)
                for fn in glob.glob(model_ckpt_version + ".*", recursive=False):
                    target_ckpt_fn = os.path.join(target_ckpt_dir, os.path.basename(fn))
                    LOG("MOVE FROM  TO ", fn, target_ckpt_fn)
                    shutil.copy(fn, target_ckpt_fn)

                shutil.copy(
                    os.path.join(args.model_dir, "graph.pbtxt"),
                    os.path.join(target_ckpt_dir, "graph.pbtxt"),
                )
                shutil.copy(
                    os.path.join(args.model_dir, "checkpoint"),
                    os.path.join(target_ckpt_dir, "checkpoint"),
                )
                LOG("FINISH BACKUP CHECKPOINT")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    # config = parse_parameter()
    parser = argparse.ArgumentParser(description="test memory")
    parser.add_argument(
        "--estimator-mode", type=str, default="training", help="training,eval,predict"
    )
    parser.add_argument(
        "--model-version", type=str, default="dssm_v1", help="model version: dssm_v0"
    )
    parser.add_argument(
        "--data-version", type=str, default="ranking_v0", help="data version: ranking"
    )
    parser.add_argument("--model-dir", type=str, default="", help="model saved dir")
    parser.add_argument("--data-dir", type=str, default="", help="data dir")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--num-threads", type=int, default=64, help="num threads")
    parser.add_argument("--prefetch", type=int, default=4, help="prefetch")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-2, help="learning rate"
    )
    parser.add_argument(
        "--embedding-size-dnn", type=int, default=-1, help="embedding size"
    )
    parser.add_argument("--drop-rate", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--output-size", type=int, default=64, help="emb output size")
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="Adam,Adamgrad,Momentum,ftrl,sgd"
    )
    parser.add_argument(
        "--user-dnn-layers",
        type=str,
        default="512,256",
        help="user side dnn layer dims",
    )
    parser.add_argument(
        "--item-dnn-layers",
        type=str,
        default="512,256",
        help="item side dnn layer dims",
    )
    parser.add_argument(
        "--eval-records", type=str, default="", help="evaluation results file"
    )
    parser.add_argument(
        "--batch-norm", type=str, default="false", help="need batch normalization"
    )
    parser.add_argument(
        "--embedding-combiner", type=str, default="sum", help="embedding-combiner"
    )
    parser.add_argument("--log-tau", type=float, default="2.0", help="log tau")
    parser.add_argument(
        "--sim-type", type=str, default="cosine", help="similarity type: cosine,"
    )
    parser.add_argument(
        "--loss-type", type=str, default="dot", help="loss type: dot,softmax"
    )
    parser.add_argument(
        "--predict-file", type=str, default="predict.out", help="predict output file"
    )
    parser.add_argument("--date", type=str, default="", help="predict output file")
    parser.add_argument(
        "--l2-reg", type=float, default=0.001, help="predict output file"
    )
    parser.add_argument(
        "--add-time-weight", type=str, default="false", help="user watchtime as weight"
    )
    parser.add_argument(
        "--backup-model", type=str, default="false", help="user watchtime as weight"
    )
    parser.add_argument(
        "--attention-heads", type=int, default=3, help="user attention heads"
    )
    parser.add_argument(
        "--skip-features", type=str, default="", help="ignored features"
    )
    parser.add_argument(
        "--filter-default", type=str, default="true", help="ignored features"
    )
    parser.add_argument(
        "--enable-attention", type=str, default="false", help="enable attention net"
    )
    parser.add_argument(
        "--recall-heads",
        type=int,
        default=1,
        help="recall heads for multi-interest head",
    )
    parser.add_argument("--num_epoch", type=int, default=1, help="number of epochs")
    parser.add_argument(
        "--perform_shuffle", type=int, default=0, help="if shuffle input examples"
    )
    parser.add_argument(
        "--use_features", type=str, default="", help="use features or using in config"
    )
    parser.add_argument(
        "--use_feature_mha", type=int, default=1, help="use feature mha or not"
    )
    parser.add_argument(
        "--log_steps", type=int, default=10, help="log steps for loss and steps"
    )
    parser.add_argument(
        "--mask_false_negative",
        type=int,
        default=0,
        help="mask false negative in softmax loss",
    )
    parser.add_argument(
        "--add_doc2vec_loss", type=int, default=0, help="add_doc2vec_loss"
    )
    parser.add_argument(
        "--doc2vec_loss_weight", type=float, default=0.0, help="add_doc2vec_loss"
    )
    parser.add_argument(
        "--add_more_his", type=int, default=0, help="if to add more history items"
    )
    parser.add_argument(
        "--freq_weight", type=float, default=0.0, help="weight to item frequency"
    )

    args = parser.parse_args()

    args.user_dnn_layers = [int(x) for x in args.user_dnn_layers.split(",")]
    args.item_dnn_layers = [int(x) for x in args.item_dnn_layers.split(",")]
    args.skip_features = set(args.skip_features.split(","))
    args.use_features = set([x for x in args.use_features.split(",") if x])

    args.batch_norm = args.batch_norm.lower() == "true"

    if not args.eval_records:
        args.eval_records = os.path.join(args.model_dir, "EVAL_RECORDS.info")

    # add log to file
    if not os.path.exists(os.path.join(args.model_dir, "log")):
        os.makedirs(os.path.join(args.model_dir, "log"), exist_ok=True)
    add_log_file(
        os.path.join(
            args.model_dir,
            "log/{}-{}.log".format(
                args.estimator_mode, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            ),
        )
    )
    args.task_id = str(uuid.uuid4())

    LOG("args", args)
    with open(os.path.join(args.model_dir, "CONFIG.ini"), "a") as fout:
        fout.writelines(
            "TASK_ID:{}\nDate:{}\nCONFIG:{}\n".format(
                args.task_id, datetime.now(), str(args)
            )
        )

    run(args)
