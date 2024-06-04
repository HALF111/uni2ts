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

from collections.abc import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset
from datasets.features import Sequence
from datasets.formatting import query_table

from uni2ts.common.typing import BatchedData, Data, MultivarTimeSeries, UnivarTimeSeries

from ._base import Indexer


<<<<<<< HEAD
class HuggingFaceDatasetIndexer(Indexer):
    def __init__(self, dataset: Dataset, uniform: bool = False):
        super().__init__(uniform=uniform)
        self.dataset = dataset
        self.features = dict(self.dataset.features)
        self.non_seq_cols = [
            name
            for name, feat in self.features.items()
            if not isinstance(feat, Sequence)
        ]
        self.seq_cols = [
            name for name, feat in self.features.items() if isinstance(feat, Sequence)
        ]
        self.dataset.set_format("numpy", columns=self.non_seq_cols)

=======
# indexer类
class HuggingFaceDatasetIndexer(Indexer):
    def __init__(self, dataset: Dataset, uniform: bool = False):
        super().__init__(uniform=uniform)
        # 这里dataset是用load_from_disk函数从arrow文件中读取出的数据，类型为一个dict？
        # 其包含item_id、syart、freq和target四项，其中target包含时间序列的值
        self.dataset = dataset
        # dataset的features，如['item_id', 'start', 'freq', 'target']等
        self.features = dict(self.dataset.features)
        
        # # TODO:
        # # * 能否在这里加一个filtering？
        # target_dataset = self.dataset["target"]
        # for data in target_dataset:
        #     has_anomaly = self.detect_anomaly(data)
        #     print(has_anomaly)
        
        # 分别汇总是&不是Sequence类型的列：['item_id', 'start', 'freq']
        self.non_seq_cols = [
            name for name, feat in self.features.items() if not isinstance(feat, Sequence)
        ]
        # 是Sequence类型的列：['target']
        self.seq_cols = [
            name for name, feat in self.features.items() if isinstance(feat, Sequence)
        ]
        # 这里的意思是：在输出时（如使用__getitem__时）会让columns的格式为numpy
        self.dataset.set_format("numpy", columns=self.non_seq_cols)
    
    # def detect_anomaly(self, data, threshold=5):
    #     print(type(data))
    #     print(data)
    #     print(data["target"])
    #     print(type(data["target"]))
    #     print(len(data["target"]))
    #     target_data = np.array(data["target"])
    #     # print(type(target_data))
    #     # print(target_data.shape)
    #     if target_data.ndim == 1:
    #         mean = np.mean(target_data)
    #         std = np.std(target_data)
    #         std = np.where(std == 0, 1.0, std)  # 使用np.where来避免除以0
            
    #         z_scores = (target_data - mean) / std
            
    #         # 判断异常值，Z-score绝对值大于阈值的点标记为True
    #         anomalies_mask = np.abs(z_scores) > threshold
    #         has_anomaly = np.any(anomalies_mask)
            
    #         # print("has_anomaly:", has_anomaly)
    #         return has_anomaly
    #     elif target_data.ndim == 2:
    #         has_anomaly_per_channel = []
    #         for channel_data in target_data:
    #             mean = np.mean(channel_data)
    #             std = np.std(channel_data)
    #             std = np.where(std == 0, 1.0, std)  # 使用np.where来避免除以0
                
    #             z_scores = (channel_data - mean) / std
                
    #             # 判断异常值，Z-score绝对值大于阈值的点标记为True
    #             anomalies_mask = np.abs(z_scores) > threshold
    #             has_anomaly_cur = np.any(anomalies_mask)
                
    #             # print("has_anomaly_cur:", has_anomaly_cur)
    #             has_anomaly_per_channel.append(has_anomaly_cur)
            
    #         # 如果需要知道至少有一个channel有异常
    #         has_anomaly_overall = np.any(has_anomaly_per_channel)
    #         return has_anomaly_overall
    
    # 长度就是数据集的长度
>>>>>>> 754bd6b (add comments on src code)
    def __len__(self) -> int:
        return len(self.dataset)

    def _getitem_int(self, idx: int) -> dict[str, Data]:
<<<<<<< HEAD
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
=======
        # 非sequence的直接按照下标取就可以
        non_seqs = self.dataset[idx]
        # query_table：从table中根据指定的key取出对应的subtable
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        # sequence列（即target）则会以dict形式？
>>>>>>> 754bd6b (add comments on src code)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col)[0] for col in self.seq_cols
        }
        return non_seqs | seqs

    def _getitem_iterable(self, idx: Iterable[int]) -> dict[str, BatchedData]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
<<<<<<< HEAD
=======
            # * 区别1：去掉了[0]？？
            # * 区别2：输入的idx为Iterable[int]
>>>>>>> 754bd6b (add comments on src code)
            col: self._pa_column_to_numpy(pa_subtable, col) for col in self.seq_cols
        }
        return non_seqs | seqs

    def _getitem_slice(self, idx: slice) -> dict[str, BatchedData]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
<<<<<<< HEAD
=======
            # * 区别1：idx变成slice格式了
>>>>>>> 754bd6b (add comments on src code)
            col: self._pa_column_to_numpy(pa_subtable, col) for col in self.seq_cols
        }
        return non_seqs | seqs

<<<<<<< HEAD
    def _pa_column_to_numpy(
        self, pa_table: pa.Table, column_name: str
    ) -> list[UnivarTimeSeries] | list[MultivarTimeSeries]:
        pa_array: pa.Array = pa_table.column(column_name)
        feature = self.features[column_name]

        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(feature.feature, Sequence):
                array = [
                    flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                    if (flat_slice := chunk.slice(i, 1).flatten())
=======
    # 将pyarrow列转成numpy？
    def _pa_column_to_numpy(
        self, pa_table: pa.Table, column_name: str
    ) -> list[UnivarTimeSeries] | list[MultivarTimeSeries]:
        # * 按照某一列从pa.Table取出pa.Array！以及对应的feature。
        pa_array: pa.Array = pa_table.column(column_name)
        feature = self.features[column_name]

        # 1、pa.ChunkedArray
        if isinstance(pa_array, pa.ChunkedArray):
            # 1.1 sequence列（即target）
            if isinstance(feature.feature, Sequence):
                array = [
                    # 将每个chunk里的每个slice拼接起来？
                    flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)
                    for chunk in pa_array.chunks  # 外循环
                    for i in range(len(chunk))  # 内循环
                    if (flat_slice := chunk.slice(i, 1).flatten())  # 内循环里的条件判断
>>>>>>> 754bd6b (add comments on src code)
                    and (
                        feat_length := (
                            feature.length if feature.length != -1 else len(flat_slice)
                        )
                    )
                ]
<<<<<<< HEAD
=======
            # 1.2 非sequence列，
>>>>>>> 754bd6b (add comments on src code)
            else:
                array = [
                    chunk.slice(i, 1).flatten().to_numpy(False)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                ]
<<<<<<< HEAD
        elif isinstance(pa_array, pa.ListArray):
            if isinstance(feature.feature, Sequence):
=======
        # 2、pa.ListArray
        elif isinstance(pa_array, pa.ListArray):
            # 2.1 sequence列（即target）
            if isinstance(feature.feature, Sequence):
                # 无需遍历，直接把整个flatten就可以了！
>>>>>>> 754bd6b (add comments on src code)
                flat_slice = pa_array.flatten()
                feat_length = (
                    feature.length if feature.length != -1 else len(flat_slice)
                )
                array = [flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)]
<<<<<<< HEAD
=======
            # 2.2 sequence列
>>>>>>> 754bd6b (add comments on src code)
            else:
                array = [pa_array.flatten().to_numpy(False)]
        else:
            raise NotImplementedError

        return array

    def get_proportional_probabilities(self, field: str = "target") -> np.ndarray:
        if self.uniform:
            return self.get_uniform_probabilities()

<<<<<<< HEAD
=======
        # 按照样本长度比例分配概率？
>>>>>>> 754bd6b (add comments on src code)
        if self[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(pc.list_slice(self.dataset.data.column(field), 0, 1))
            )
        else:
            lengths = pc.list_value_length(self.dataset.data.column(field))
        lengths = lengths.to_numpy()
        probs = lengths / lengths.sum()
        return probs
