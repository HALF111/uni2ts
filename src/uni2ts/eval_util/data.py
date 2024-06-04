from functools import partial
from typing import NamedTuple

import gluonts
from gluonts.dataset.common import _FileDataset
from gluonts.dataset.split import TestData, split

from uni2ts.data.builder.lotsa_v1.gluonts import get_dataset

from ._hf_dataset import HFDataset
from ._lsf_dataset import LSFDataset
from ._pf_dataset import generate_pf_dataset, pf_load_func_map

gluonts.dataset.repository.dataset_recipes |= {
    k: partial(generate_pf_dataset, dataset_name=k) for k in pf_load_func_map.keys()
}


# 这个就是数据集的metadata
# 包括：freq:频率、target_dim:目标所在的维度、prediction_length:预测长度等等。
# past_feat_dynamic_real_dim:大概是过去的协变量需要几维？
# split:需要哪部分数据？可以为train或val或test。
class MetaData(NamedTuple):
    freq: str
    target_dim: int
    prediction_length: int
    feat_dynamic_real_dim: int = 0
    past_feat_dynamic_real_dim: int = 0
    split: str = "test"


# 获得gluonts上的验证数据集
def get_gluonts_val_dataset(
    dataset_name: str,
    prediction_length: int = None,
    mode: str = None,
    regenerate: bool = False,
) -> tuple[TestData, MetaData]:
    default_prediction_lengths = {
        "australian_electricity_demand": 336,
        "pedestrian_counts": 24,
    }
    if prediction_length is None and dataset_name in default_prediction_lengths:
        prediction_length = default_prediction_lengths[dataset_name]

    dataset = get_dataset(
        dataset_name, prediction_length=prediction_length, regenerate=regenerate
    )

    prediction_length = prediction_length or dataset.metadata.prediction_length
    _, test_template = split(dataset.train, offset=-prediction_length)
    test_data = test_template.generate_instances(prediction_length)
    metadata = MetaData(
        freq=dataset.metadata.freq,
        target_dim=1,
        prediction_length=prediction_length,
        split="val",
    )
    return test_data, metadata


# 获得gluonts上的测试数据集
def get_gluonts_test_dataset(
    dataset_name: str,
    prediction_length: int = None,
    mode: str = None,
    regenerate: bool = False,
) -> tuple[TestData, MetaData]:
    default_prediction_lengths = {
        "australian_electricity_demand": 336,
        "pedestrian_counts": 24,
    }
    if prediction_length is None and dataset_name in default_prediction_lengths:
        prediction_length = default_prediction_lengths[dataset_name]

    dataset = get_dataset(
        dataset_name, prediction_length=prediction_length, regenerate=regenerate
    )

    prediction_length = prediction_length or dataset.metadata.prediction_length
    _, test_template = split(dataset.test, offset=-prediction_length)
    test_data = test_template.generate_instances(prediction_length)
    metadata = MetaData(
        freq=dataset.metadata.freq,
        target_dim=1,
        prediction_length=prediction_length,
        split="test",
    )
    return test_data, metadata


# 获取LSF的验证集
def get_lsf_val_dataset(
    dataset_name: str,
    prediction_length: int = 96,
    mode: str = "S",
) -> tuple[TestData, MetaData]:
    # 获取数据集，其中split为val
    lsf_dataset = LSFDataset(dataset_name, mode=mode, split="val")
    dataset = _FileDataset(
        lsf_dataset, freq=lsf_dataset.freq, one_dim_target=lsf_dataset.target_dim == 1
    )
    # 做测试集/验证集？的切割
    _, test_template = split(dataset, offset=-lsf_dataset.length)
    # 测试数据
    test_data = test_template.generate_instances(
        prediction_length,
        windows=lsf_dataset.length - prediction_length + 1,
        distance=1,
    )
    metadata = MetaData(
        freq=lsf_dataset.freq,
        target_dim=lsf_dataset.target_dim,
        prediction_length=prediction_length,
        past_feat_dynamic_real_dim=lsf_dataset.past_feat_dynamic_real_dim,
        split="val",
    )
    return test_data, metadata


# 获取LSF的测试集
def get_lsf_test_dataset(
    dataset_name: str,
    prediction_length: int = 96,
    mode: str = "S",
) -> tuple[TestData, MetaData]:
    lsf_dataset = LSFDataset(dataset_name, mode=mode, split="test")
    dataset = _FileDataset(
        lsf_dataset, freq=lsf_dataset.freq, one_dim_target=lsf_dataset.target_dim == 1
    )
    _, test_template = split(dataset, offset=-lsf_dataset.length)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=lsf_dataset.length - prediction_length + 1,
        distance=1,
    )
    metadata = MetaData(
        freq=lsf_dataset.freq,
        target_dim=lsf_dataset.target_dim,
        prediction_length=prediction_length,
        past_feat_dynamic_real_dim=lsf_dataset.past_feat_dynamic_real_dim,
        split="test",  # 这里也是test。和上面的区别应该就是把split从val改成test吧
    )
    return test_data, metadata


# HF_dataset?
def get_custom_eval_dataset(
    dataset_name: str,
    offset: int,
    windows: int,
    distance: int,
    prediction_length: int,
    mode: None = None,
) -> tuple[TestData, MetaData]:
    hf_dataset = HFDataset(dataset_name)
    dataset = _FileDataset(
        hf_dataset, freq=hf_dataset.freq, one_dim_target=hf_dataset.target_dim == 1
    )
    _, test_template = split(dataset, offset=offset)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=windows,
        distance=distance,
    )
    metadata = MetaData(
        freq=hf_dataset.freq,
        target_dim=hf_dataset.target_dim,
        prediction_length=prediction_length,
        split="test",
    )
    return test_data, metadata
