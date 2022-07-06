from dssm_v2.data import bottom_page_data
from dssm_v2.utils.train_utils import LOG
from dssm_v2.configs.config_realtime_groupId import config_realtime


def get_data_format_fn_and_ids(args):
    LOG("get_data_format_fn_and_ids", "focus_page_data_attention_v1")
    feature_dict = config_realtime
    return bottom_page_data.get_data_format_fn_and_ids(args, feature_dict)