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

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from uni2ts.common.sampler import Sampler, get_sampler
from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin


# 预测任务
# 论文中有三个点需要采样：
# 1、均匀采样一个总窗口长度，长度范围为[2, 512]，同时划分出预测窗口的比例范围为[0.15, 0.5]。
# 2、在变量维度上使用二项分布采样（n=128，a=2，b=5），最多支持128个变量，平均约为37个变量。
# 3、将多个单变量时间序列随机组合成多变量。
# 其中2和3在这里实现，1参考task.py！
@dataclass
class SampleDimension(
    CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin, Transformation
):
    # 变量采样的最大的个数
    max_dim: int  # 如128
    fields: tuple[str, ...]  # 如["target"]
    optional_fields: tuple[str, ...] = tuple()
    # 这里sampler默认是一个uniform的，代码为np.random.randint(1, n+1)
    # 此外，也可以是binomial，代码为np.random.binomial(n - 1, p) + 1
    # 或者beta_binomial，代码为：先计算p=np.random.beta(a, b, size=n.shape)，再返回np.random.binomial(n - 1, p) + 1
    # 详见common/sample.py文件！
    sampler: Sampler = get_sampler("uniform")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        total_field_dim = sum(
            self.collect_func_list(
                self._get_dim,
                data_entry,
                self.fields,
                optional_fields=self.optional_fields,
            )
        )
        self.map_func(
            partial(self._process, total_field_dim=total_field_dim),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _get_dim(self, data_entry: dict[str, Any], field: str) -> int:
        self.check_ndim(field, data_entry[field], 2)
        return len(data_entry[field])

    def _process(
        self, data_entry: dict[str, Any], field: str, total_field_dim: int
    ) -> list[UnivarTimeSeries]:
        arr: list[UnivarTimeSeries] = data_entry[field]
        # 将arr下标随机排序，得到rand_idx
        rand_idx = np.random.permutation(len(arr))
        field_max_dim = (self.max_dim * len(arr)) // total_field_dim
        # 采样一个需要的变量个数n，n为arr长度和field_max_dim中的较小值。
        # 这里sampler默认是一个uniform的，代码为np.random.randint(1, n+1)
        # 此外，也可以是binomial，代码为np.random.binomial(n - 1, p) + 1
        # 或者beta_binomial，代码为：先计算p=np.random.beta(a, b, size=n.shape)，再返回np.random.binomial(n - 1, p) + 1
        # 详见common/sample.py文件！
        n = self.sampler(min(len(arr), field_max_dim))
        # 采样出这些变量，长度为n？
        return [arr[idx] for idx in rand_idx[:n]]


# Subsample也是Transformation的一种
# 每隔n个element采样一次
@dataclass
class Subsample(Transformation):  # just take every n-th element
    fields: tuple[str, ...] = ("target", "past_feat_dynamic_real")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


# subsampling做一个gaussian filter的blur？
class GaussianFilterSubsample(
    Subsample
):  # blur using gaussian filter before subsampling
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # gaussian filter
        return super()(data_entry)


# 下采样
class Downsample(Transformation):  # aggregate
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


# 上采样
class Upsample(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass
