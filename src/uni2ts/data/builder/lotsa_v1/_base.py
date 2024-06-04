#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import abc
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from datasets import load_from_disk
from torch.utils.data import ConcatDataset, Dataset

from uni2ts.common.core import abstract_class_property
from uni2ts.common.env import env
from uni2ts.data.builder._base import DatasetBuilder
from uni2ts.data.dataset import SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Identity, Transformation


# LOTSA的builder，继承自DatasetBuilder。
@abstract_class_property("dataset_list", "dataset_type_map", "dataset_load_func_map")
class LOTSADatasetBuilder(DatasetBuilder, abc.ABC):
    # 例子：[f"era5_{year}" for year in range(1989, 2018 + 1)]或["buildings_900k"]
    dataset_list: list[str] = NotImplemented
    # 例子：defaultdict(lambda: TimeSeriesDataset)
    dataset_type_map: dict[str, type[TimeSeriesDataset]] = NotImplemented
    # 例子：defaultdict(lambda: partial(TimeSeriesDataset))
    dataset_load_func_map: dict[str, Callable[..., TimeSeriesDataset]] = NotImplemented
    uniform: bool = False

    # 初始化：包括当前的数据集、以及对应的权重参数等
    def __init__(
        self,
        datasets: list[str],
        weight_map: Optional[dict[str, float]] = None,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        storage_path: Path = env.LOTSA_V1_PATH,
    ):
        # 要求datcasets参数中的所有数据集都在预定义的dataset_list中
        assert all(
            dataset in self.dataset_list for dataset in datasets
        ), f"Invalid datasets {set(datasets).difference(self.dataset_list)}, must be one of {self.dataset_list}"
        weight_map = weight_map or dict()
        self.datasets = datasets
        # 获得各个数据集的具体权重，如果不存在则默认为1.0。
        self.weights = [weight_map.get(dataset, 1.0) for dataset in datasets]
        # 采样的方式，可以为None、uniform（给所有样本相同的权重）、proportional（根据样本长度分配权重）
        self.sample_time_series = sample_time_series
        self.storage_path = storage_path

    # # 初步的做法就是直接在dataset层面上做筛选？
    # def data_filter(self, raw_data):
    #     def detect_anomalies(time_series):
    #         import numpy as np
    #         # 计算四分位数
    #         q1 = np.percentile(time_series, 25)
    #         q3 = np.percentile(time_series, 75)
    #         iqr = q3 - q1
    #         # 计算上下界Q1-1.5IQR和Q3+1.5IQR
    #         lower_bound = q1 - 1.5 * iqr
    #         upper_bound = q3 + 1.5 * iqr
    #         # 如果存在值在上下界之外，返回True
    #         return any(not (lower_bound <= x <= upper_bound) for x in time_series)
        
    #     print(raw_data)
    #     features = [item for item in raw_data.features]
    #     data = raw_data['target']  # 二维数组，形状为即[channel, timesteps]？
    #     print(len(data))
    #     print([len(item) for item in data])
        
    #     channels, timesteps = data.shape
    #     filtered_data = []
    #     for channel in channels:
    #         time_series = data[channel]
    #         if not detect_anomalies(time_series):
    #             filtered_data.append(time_series)
    #     raw_data['target'] = filtered_data
    #     return raw_data
        

    def load_dataset(self, transform_map: dict[str | type, Transformation]) -> Dataset:
        # datasets = []
        # for dataset, weight in zip(self.datasets, self.weights):
        #     # load_from_disk从arrow文件中读取数据，返回的为一个dict？
        #     raw_data = load_from_disk(self.storage_path / dataset)
        #     # * 做一个数据的筛选？
        #     raw_data = self.data_filter(raw_data)
            
        #     indexer = HuggingFaceDatasetIndexer(raw_data, uniform=self.uniform)
        #     dataset_processed = self.dataset_load_func_map[dataset](
        #         indexer,
        #         self._get_transform(transform_map, dataset),
        #         sample_time_series=self.sample_time_series,
        #         dataset_weight=weight,
        #     )
        #     datasets.append(dataset_processed)
        
        # # 如果只有一个dataset那么直接返回之，否则返回ConcatDataset将他们合并在一起
        # return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        
        
        # 此时开始加载数据？可以以第一个buildings_900k为例子。
        datasets = [
            # * 根据当前的dataset_name，使用如TimeSeriesDataset来生成数据集
            self.dataset_load_func_map[dataset](
                # 1、从arrow文件中取出数据，并以indexer进行包装
                HuggingFaceDatasetIndexer(
                    # load_from_disk从arrow文件中读取数据，返回的为一个dict？
                    # PS：就是在这里会生成一个读取数据的进度条
                    load_from_disk(self.storage_path / dataset), uniform=self.uniform
                ),
                # 2、对当前数据做trasform
                self._get_transform(transform_map, dataset),
                # 3、采样的方式，可以为None、uniform（给所有样本相同的权重）、proportional（根据样本长度分配权重）
                sample_time_series=self.sample_time_series,
                # 4、各个数据集的具体权重，如果不存在则默认均为1.0。
                dataset_weight=weight,
            )
            # 遍历各个dataset和其权重
            for dataset, weight in zip(self.datasets, self.weights)
        ]
        
        # 如果只有一个dataset那么直接返回之，否则返回ConcatDataset将他们合并在一起
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


    def _get_transform(
        self,
        transform_map: dict[str | type, Callable[..., Transformation]],
        dataset: str,
    ) -> Transformation:
        # 如果有对当前dataset或dataset_type的transform方法，则使用之
        # 否则使用默认的transform方法？
        if dataset in transform_map:
            transform = transform_map[dataset]
        elif (dataset_type := self.dataset_type_map[dataset]) in transform_map:
            transform = transform_map[dataset_type]
        else:
            try:  # defaultdict
                transform = transform_map[dataset]
            except KeyError:
                transform = transform_map.get("default", Identity)
        return transform()
