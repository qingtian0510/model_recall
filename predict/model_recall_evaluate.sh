#!/usr/bin/env bash
set -x
# define python
python="python"
pip list | grep tensorflow
pip install tensorflow-serving-api==1.15
pip install requests
pip install tqdm
sdate=`date '+%Y_%m_%d_%H_%M_%S'`
pwd
ls /
base_path="/cfs/cfs-3cde0a5bc/pinnie/dsssm/models_test/rec_sys/bottom_page_video/recall/tensorflow_dssm/predict"
cd $base_path
eval_dir=/cfs/cfs-3cde0a5bc/pinnie/dsssm/data/neg_sample_user_statis/$1
python model_recall_evaluate.py ${eval_dir}
find /cfs/cfs-3cde0a5bc/pinnie/dsssm/data/neg_sample_user_statis -type d -name "202*" |sort | head -n -24 | xargs rm -rf

