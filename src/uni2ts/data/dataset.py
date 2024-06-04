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

from enum import Enum
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from uni2ts.common.sampler import Sampler, get_sampler
from uni2ts.common.typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    FlattenedData,
    MultivarTimeSeries,
    UnivarTimeSeries,
)
from uni2ts.data.indexer import Indexer
from uni2ts.transform import Transformation

from ..global_var import get_value, set_value, lock

import timeout_decorator
import concurrent.futures
import signal


# 自定义异常
class TimeoutException(Exception):
    pass

# 超时处理函数
def timeout_handler(signum, frame):
    raise TimeoutException



# 一般的数据集（如训练集）
class SampleTimeSeriesType(Enum):
    NONE = "none"
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],  # 读取数据的indexer
        transform: Transformation,  # 对数据做的transformation
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE, # 采样方式
        dataset_weight: float = 1.0,  # 数据集权重
    ):
        self.indexer = indexer
        self.transform = transform
        self.sample_time_series = sample_time_series
        self.dataset_weight = dataset_weight
        
        # self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # # TODO:
        # # * 能否在init的时候就对dataset做filtering？
        # indexer_len = len(indexer)
        # for i in range(indexer_len):
        #     cur_data = 

        # 做概率预测？默认为None
        if sample_time_series == SampleTimeSeriesType.NONE:
            self.probabilities = None
        elif sample_time_series == SampleTimeSeriesType.UNIFORM:
            # 给所有样本相同的权重
            self.probabilities = indexer.get_uniform_probabilities()
        elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
            # 根据样本长度分配权重
            self.probabilities = indexer.get_proportional_probabilities()
        else:
            raise ValueError(f"Unknown sample type {sample_time_series}")

    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        # 以buildings_900k为例，其len(self)为1795256，idx可以为535375
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        # 如果不是None，则idx会做一个随机采样？（如uniform或proportional）；否则的话就直接用当前的这个idx
        # PS：numpy.random.choice(a, size=None, replace=True, p=None) - https://blog.csdn.net/ImwaterP/article/details/96282230
        # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组。
        # replace：True可以取相同数字，False不可以取相同数字。
        # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        if self.sample_time_series != SampleTimeSeriesType.NONE:
            idx = np.random.choice(len(self.probabilities), p=self.probabilities)

        # (1)根据下标idx获得数据，(2)然后展平，(3)最后做一份transform。
        data = self._get_data(idx)  # data["target"].shape：(8761,)
        
        
        # TODO:
        # * 1.1 特别地，我们希望在这里做一个数据筛选，过滤掉有异常值的数据
        # * 为了防止数据维度不对，我们随机选一个新的下标idx，直到其通过了筛选条件为止
        # # lock.acquire()
        
        # CALC_SAMPLE_NUM = False
        CALC_SAMPLE_NUM = True
        
        if CALC_SAMPLE_NUM:
            total_samples = get_value("total_samples"); select_samples = get_value("select_samples")
            total_samples += 1; select_samples += 1
        
        # 加一个判断，如果超出指定MAX_DETECT_ITER轮数就结束
        cnt = 0; MAX_DETECT_ITER = 30
        while cnt < MAX_DETECT_ITER and self.detect_anomaly(data):
            cnt += 1
            idx = np.random.choice(self.__len__())
            data = self._get_data(idx)
            if CALC_SAMPLE_NUM:
                total_samples += 1
        
        # # try:
        # while self.detect_anomaly(data):
        #     idx = np.random.choice(self.__len__())
        #     data = self._get_data(idx)
        #     total_samples += 1
        # # except:
        # #     print("detecting anomaly timeout!")
        # # print(f"idx:{idx} success")
        
        # # 使用线程池执行检查函数并设置超时时间
        # # future = self.executor.submit(self.detect_anomaly, data)
        # TIMEOUT = 3
        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        #     try:
        #         select_samples += 1
        #         while True:
        #             total_samples += 1
        #             future = executor.submit(self.detect_anomaly, data)
        #             result = future.result(timeout=TIMEOUT)  # 设置超时时间
        #             # print("result:", result)
        #             if result is None:
        #                 raise ValueError(f"Data check returned None for item {idx}")
        #             if result == False:
        #                 break
        #             elif result == True:
        #                 idx = np.random.choice(self.__len__())
        #                 data = self._get_data(idx)
        #     except concurrent.futures.TimeoutError:
        #         # raise ValueError(f"Data check timed out for item {idx}")
        #         print(f"detecting anomaly timeout on idx {idx}!")
        
        
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(TIMEOUT)  # 设置超时时间
        # try:
        #     while True:
        #         total_samples += 1
        #         has_anomaly = self.detect_anomaly(data)
        #         signal.alarm(0)
        #         # print("has_anomaly in __getitem__:", has_anomaly)
        #         if has_anomaly is None:
        #             # breakpoint()
        #             # has_anomaly = self.detect_anomaly(data)
        #             exit(1)
        #         if has_anomaly == False:
        #             break
        #         elif has_anomaly == True:
        #             idx = np.random.choice(self.__len__())
        #             data = self._get_data(idx)
        #             signal.alarm(TIMEOUT)
        # except TimeoutException:
        #     signal.alarm(0)
        #     raise ValueError(f"Data check timed out for item {idx}")
        # except Exception as e:
        #     signal.alarm(0)  # 取消定时器
        #     print(f"Exception in check_data for item {idx}: {e}")
        #     raise
        
        if CALC_SAMPLE_NUM:
            set_value("total_samples", total_samples); set_value("select_samples", select_samples)
            if total_samples % 1000 == 0:
                print(f"total_samples: {total_samples}, select_samples: {select_samples}, ratio = {select_samples / total_samples}")
        
        # # lock.release()
        
        data_flattened = self._flatten_data(data)  # data_flattened["target"][0].shape：(8761,)
        
        data_transformed = self.transform(data_flattened)  # data_transformed["target"].shape：(71, 128)
        
        # PS：这个时候还没有做packing，所以其形状可能是如(71, 128)，表示当前样本包含71个patch，每个patch长度被扩展到128。
        return data_transformed
        # return self.transform(self._flatten_data(self._get_data(idx)))
    
    
    # 如果读取数据超过指定时间，则跳过
    # @timeout_decorator.timeout(60, use_signals=False)
    def detect_anomaly(self, data, threshold=3):
        # print("type(data):", type(data))
        # print("data:", data)
        # print("data['target']:", data["target"])
        # print("type(data['target']):", type(data["target"]))
        # print("len(data['target']:", len(data["target"]))
        
        target_data = np.array(data["target"])  # (channel, seq_len)
        # print("type(target_data):", type(target_data))
        # print("target_data.shape:", target_data.shape)
        
        
        # # * 方法一：Z-score
        # ANOMALY_RATIO = 0.01
        # if target_data.ndim == 1:
        #     mean = np.mean(target_data)
        #     std = np.std(target_data)
        #     std = np.where(std == 0, 1.0, std)  # 使用np.where来避免除以0
            
        #     z_scores = (target_data - mean) / std
            
        #     # 判断异常值，Z-score绝对值大于阈值的点标记为True
        #     anomalies_mask = np.abs(z_scores) > threshold
            
        #     # has_anomaly = np.any(anomalies_mask)
        #     anomaly_cnt = np.sum(anomalies_mask)
        #     total_cnt = anomalies_mask.size  # size是返回总元素的大小，相当于seq_len
            
        #     import math
        #     anomaly_max = math.ceil(ANOMALY_RATIO * total_cnt)
        #     has_anomaly = (anomaly_cnt > anomaly_max)
            
        #     # print("anomaly_cnt:", anomaly_cnt)
        #     # print("total_cnt:", total_cnt)
        #     # print("anomaly_max:", anomaly_max)
        #     # print("has_anomaly:", has_anomaly)
        #     # print("-"*30)
                
        #     # print("has_anomaly:", has_anomaly)
        #     return has_anomaly
        # elif target_data.ndim == 2:
        #     # has_anomaly_per_channel = []
            
        #     mean = np.mean(target_data, axis=-1)  # (channel, 1)
        #     mean = np.expand_dims(mean, axis=-1)
        #     std = np.std(target_data, axis=-1)  # (channel, 1)
        #     std = np.expand_dims(std, axis=-1)
        #     std = np.where(std == 0, 1.0, std)  # 使用np.where来避免除以0
            
        #     z_scores = (target_data - mean) / std  # shape: (channel, seq_len)
        #     # 判断异常值，Z-score绝对值大于阈值的点标记为True
        #     anomalies_mask = np.abs(z_scores) > threshold
            
        #     # has_anomaly = np.any(anomalies_mask)
        #     anomaly_cnt = np.sum(anomalies_mask)
        #     total_cnt = anomalies_mask.size  # size是返回总元素的大小，相当于seq_len
            
        #     import math
        #     anomaly_max = math.ceil(ANOMALY_RATIO * total_cnt)
        #     has_anomaly = (anomaly_cnt > anomaly_max)
            
        #     # print(mean.shape, std.shape, z_scores.shape, anomalies_mask.shape)
            
        #     # for channel_data in target_data:
        #     #     mean = np.mean(channel_data)
        #     #     std = np.std(channel_data)
        #     #     std = np.where(std == 0, 1.0, std)  # 使用np.where来避免除以0
                
        #     #     z_scores = (channel_data - mean) / std
                
        #     #     # 判断异常值，Z-score绝对值大于阈值的点标记为True
        #     #     anomalies_mask = np.abs(z_scores) > threshold
        #     #     has_anomaly_cur = np.any(anomalies_mask)
                
        #     #     # print("has_anomaly_cur:", has_anomaly_cur)
        #     #     has_anomaly_per_channel.append(has_anomaly_cur)
            
        #     # # 如果需要知道至少有一个channel有异常
        #     # has_anomaly_overall = np.any(has_anomaly_per_channel)
            
        #     # return has_anomaly_overall
        #     # print("has_anomaly:", has_anomaly)
        #     return has_anomaly
        # elif target_data.ndim == 3:
        #     # print("target_data.shape:", target_data.shape)
        #     # # 方法1：直接将前面的展平
        #     # target_data = target_data.reshape(-1, target_data.shape[-1])
        #     # 方法2：只取一个batch？
        #     target_data = target_data[0, :, :]
        #     # print("target_data.shape:", target_data.shape)
            
        #     mean = np.mean(target_data, axis=-1)  # (channel, 1)
        #     mean = np.expand_dims(mean, axis=-1)
        #     std = np.std(target_data, axis=-1)  # (channel, 1)
        #     std = np.expand_dims(std, axis=-1)
        #     std = np.where(std == 0, 1.0, std)  # 使用np.where来避免除以0
            
        #     z_scores = (target_data - mean) / std  # shape: (channel, seq_len)
        #     # 判断异常值，Z-score绝对值大于阈值的点标记为True
        #     anomalies_mask = np.abs(z_scores) > threshold
            
        #     # has_anomaly = np.any(anomalies_mask)
        #     anomaly_cnt = np.sum(anomalies_mask)
        #     total_cnt = anomalies_mask.size  # size是返回总元素的大小，相当于seq_len
            
        #     import math
        #     anomaly_max = math.ceil(ANOMALY_RATIO * total_cnt)
        #     has_anomaly = (anomaly_cnt > anomaly_max)
            
        #     return has_anomaly
        # else:
        #     return False
        
        
        # * 方法二：IQN
        RATIO = 1.5
        # ANOMALY_RATIO = 0.008
        ANOMALY_RATIO = 0.07
        if target_data.ndim == 1:
            q1, q3 = np.percentile(target_data, (25, 75))  # (seq_len), (seq_len)
            # assert (q1 <= q3).all()
            iqn = q3 - q1
            lower_bound = q1 - RATIO * iqn
            upper_bound = q3 + RATIO * iqn
            
            has_lower = (target_data < lower_bound)  # (seq_len)
            has_upper = (target_data > upper_bound)  # (seq_len)
            
            has_lower_cnt = np.sum(has_lower)
            has_upper_cnt = np.sum(has_upper)
            anomaly_cnt = has_lower_cnt + has_upper_cnt
            total_cnt = has_lower.size  # size是返回总元素的大小，相当于seq_len
            
            import math
            anomaly_max = math.ceil(ANOMALY_RATIO * total_cnt)
            has_anomaly = (anomaly_cnt > anomaly_max)
            # print("has_anomaly:", has_anomaly)
            
            return has_anomaly
        
        elif target_data.ndim == 2:
            # has_anomaly_per_channel = []
            
            q1, q3 = np.percentile(target_data, (25, 75), axis=-1)  # (channel,), (channel,)
            iqn = q3 - q1
            lower_bound = q1 - RATIO * iqn
            upper_bound = q3 + RATIO * iqn
            lower_bound = np.expand_dims(lower_bound, axis=-1)  # (channel, 1)
            upper_bound = np.expand_dims(upper_bound, axis=-1)  # (channel, 1)
            
            # print("lower_bound:", lower_bound)
            # print("upper_bound:", upper_bound)
            # print("lower_bound.shape:", lower_bound.shape)
            # print("upper_bound.shape:", upper_bound.shape)
            
            has_lower = (target_data < lower_bound)  # (channel, seq_len)
            has_upper = (target_data > upper_bound)  # (channel, seq_len)
            # print("has_lower", has_lower)
            # print("has_upper", has_upper)
            
            has_lower_cnt = np.sum(has_lower)
            has_upper_cnt = np.sum(has_upper)
            anomaly_cnt = has_lower_cnt + has_upper_cnt
            total_cnt = has_lower.size  # size是返回总元素的大小，相当于channel*seq_len
            
            import math
            anomaly_max = math.ceil(ANOMALY_RATIO * total_cnt)
            has_anomaly = (anomaly_cnt > anomaly_max)
            # print(f"anomaly_cnt: {anomaly_cnt}, anomaly_max: {anomaly_max}, total_cnt: {total_cnt}")
            # print("has_anomaly:", has_anomaly)
            
            return has_anomaly
        # elif target_data.ndim == 3:
        #     # print("target_data.shape:", target_data.shape)
        #     # # 方法1：直接将前面的展平
        #     # target_data = target_data.reshape(-1, target_data.shape[-1])
        #     # 方法2：只取一个batch？
        #     target_data = target_data[0, :, :]
        #     # print("target_data.shape:", target_data.shape)
            
        #     q1, q3 = np.percentile(target_data, (25, 75), axis=-1)  # (channel,), (channel,)
        #     iqn = q3 - q1
        #     lower_bound = q1 - RATIO * iqn
        #     upper_bound = q3 + RATIO * iqn
        #     lower_bound = np.expand_dims(lower_bound, axis=-1)  # (channel, 1)
        #     upper_bound = np.expand_dims(upper_bound, axis=-1)  # (channel, 1)
            
        #     # print("lower_bound:", lower_bound)
        #     # print("upper_bound:", upper_bound)
        #     # print("lower_bound.shape:", lower_bound.shape)
        #     # print("upper_bound.shape:", upper_bound.shape)
            
        #     has_lower = (target_data < lower_bound)  # (channel, seq_len)
        #     has_upper = (target_data > upper_bound)  # (channel, seq_len)
        #     # print("has_lower", has_lower)
        #     # print("has_upper", has_upper)
            
        #     has_lower_cnt = np.sum(has_lower)
        #     has_upper_cnt = np.sum(has_upper)
        #     anomaly_cnt = has_lower_cnt + has_upper_cnt
        #     total_cnt = has_lower.size  # size是返回总元素的大小，相当于channel*seq_len
            
        #     import math
        #     anomaly_max = math.ceil(ANOMALY_RATIO * total_cnt)
        #     has_anomaly = (anomaly_cnt > anomaly_max)
        #     # print(f"anomaly_cnt: {anomaly_cnt}, anomaly_max: {anomaly_max}, total_cnt: {total_cnt}")
        #     # print("has_anomaly:", has_anomaly)
            
        #     return has_anomaly
        else:
            # print("target_data.shape:", target_data.shape)
            return False
        
        

    @property
    def num_ts(self) -> int:
        # indexer的长度事实上就是数据集的长度
        return len(self.indexer)

    def __len__(self) -> int:
        # 但为了保证数据集间的平衡，对于每个数据集我们事实上只采样一小部分
        # 所以这里要乘上一个dataset_weight
        return int(np.ceil(self.num_ts * self.dataset_weight))

    # 根据下标idx从数据集中获得数据点，数据为Data或BatchedData
    # 例如buildings_900k取出后该序列的形状为(8761,)
    def _get_data(self, idx: int) -> dict[str, Data | BatchedData]:
        return self.indexer[idx % self.num_ts]

    # 对data做展平
    @staticmethod
    def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
        return {
            k: (
                # 如果v是单变量序列，则变成[v]；如果是多变量序列，则变成list(v)；否则则仍保留v
                [v]
                if isinstance(v, UnivarTimeSeries)
                else list(v) if isinstance(v, MultivarTimeSeries) else v
            )
            for k, v in data.items()
        }


# 多样本的TimeSeriesDataset？
class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        self.sampler = sampler

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        n_series = self.sampler(min(self.num_ts, self.max_ts))
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        others = np.random.choice(choices, n_series - 1, replace=False)
        samples = self.indexer[np.concatenate([[idx], others])]
        return samples

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        for field in samples.keys():
            if field in self.combine_fields:
                item = samples[field]
                if isinstance(item, list) and isinstance(item[0], MultivarTimeSeries):
                    samples[field] = [
                        univar for sample in samples[field] for univar in sample
                    ]
            elif isinstance(samples[field], BatchedDateTime):
                samples[field] = np.asarray(samples[field][0])
            elif isinstance(samples[field], BatchedString):
                samples[field] = samples[field][0]
            else:
                raise AssertionError(
                    f"Field {field} not accounted for in {self.indexer} MultiSampleTimeSeriesDataset"
                )
        return samples


# 验证集的数据集
# 继承自TimeSeriesDataset类
class EvalDataset(TimeSeriesDataset):
    def __init__(
        self,
        windows: int,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,  # windows就是数据集的权重？
    ):
        super().__init__(
            indexer,
            transform,
            SampleTimeSeriesType.NONE,
            dataset_weight=windows,
        )

    # 原来是直接self.indexer[idx % self.num_ts]
    # 现在还额外需要一个第几个窗口中的第几个数字？
    def _get_data(self, idx: int) -> dict[str, Data]:
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item

    # PS：由于__getitem__函数没有修改，所以还是取出数据后先flatten再做transform。
