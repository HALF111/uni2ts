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

from functools import reduce
from typing import Callable, Optional

import torch
from jaxtyping import PyTree
from torch.distributions import Categorical, Distribution, constraints

from uni2ts.common.torch_util import unsqueeze_trailing_dims

from ._base import DistributionOutput


class Mixture(Distribution):
    arg_constraints = dict()
    has_rsample = False

    def __init__(
        self,
        weights: Categorical,  # Categorical：创建由probs或logits（但不同时包含两者）参数化的分类分布。
        components: list[Distribution],
        validate_args: Optional[bool] = None,
    ):
        # Mxiture可以包含多个子分布
        for comp in components:
            comp._validate_args = False

        # 各个分布的权重？，是Categorical类
        self.weights = weights
        self.components = components

        if not isinstance(weights, Categorical):
            raise TypeError("weights must be a Categorical distribution")

        if not all(isinstance(comp, Distribution) for comp in components):
            raise TypeError("components must all be instances of Distribution")

        # event_shape、batch_shape和sample_shape：
        # event_shape：描述了从分布中单次抽样的形状；抽样可能依赖于不同维度。
        # batch_shape：描述了独立但并非同分布的抽样，也称为分布的“批次”
        # sample_shape：描述了来自分布系列的独立同分布批次抽样
        # ref：https://tensorflow.google.cn/probability/examples/Understanding_TensorFlow_Distributions_Shapes?hl=zh-cn
        # ref2：https://bochang.me/blog/posts/pytorch-distributions/
        batch_shape = weights.batch_shape
        event_shape = components[0].event_shape
        if validate_args:
            if not all(comp.batch_shape == batch_shape for comp in components):
                raise ValueError("components must have the same batch_shape as weights")
            if not all(comp.event_shape == event_shape for comp in components):
                raise ValueError("components must have the same event_shape")
            if weights.logits.shape[-1] != len(components):
                raise ValueError(
                    "number of logits of weights must be equal to number of components, "
                    f"got {weights.logits.shape[-1]} and {len(components)} respectively"
                )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def expand(self, batch_shape: torch.Size, _instance=None) -> "Mixture":
        new = self._get_checked_instance(Mixture, _instance)
        batch_shape = torch.Size(batch_shape)
        new.weights = self.weights.expand(batch_shape)
        new.components = [comp.expand(batch_shape) for comp in self.components]
        super(Mixture, new).__init__(
            batch_shape=batch_shape,
            event_shape=new.components[0].event_shape,
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

            # Check at least in 1 support
            valid = reduce(
                torch.logical_or,
                (comp.support.check(value) for comp in self.components),
            )
            if not valid.all():
                raise ValueError(
                    "Expected value argument "
                    f"({type(value).__name__} of shape {tuple(value.shape)}) "
                    f"to be within the support (one of {[repr(comp.support) for comp in self.components]}) "
                    f"of the distribution {repr(self)}, "
                    f"but found invalid values:\n{value}"
                )

        weights_log_probs = self.weights.logits.expand(
            value.shape + (len(self.components),)
        )
        weights_log_probs = torch.stack(weights_log_probs.unbind(dim=-1))
        # avoid nan grads, https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        components_log_probs = torch.stack(
            [
                torch.where(
                    comp.support.check(value),
                    comp.log_prob(
                        torch.where(
                            comp.support.check(value),
                            value,
                            comp.sample(),
                        )
                    ),
                    float("-inf"),
                )
                for comp in self.components
            ]
        )
        weights_log_probs = torch.where(
            torch.isinf(components_log_probs),
            0.0,
            weights_log_probs,
        )
        return (weights_log_probs + components_log_probs).logsumexp(dim=0)

    # 根据样本形状，从分布中进行采样？
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            # 先从各个component中采样出sample
            components_samples = torch.stack(
                [comp.sample(sample_shape) for comp in self.components], dim=-1
            )
            # 再对权重做采样？？
            weights_sample = unsqueeze_trailing_dims(
                self.weights.sample(sample_shape), components_samples.shape
            )
            # 然后使用gather函数沿着指定的轴收集数值
            # torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor：
            # input：目标变量，输入；dim：需要沿着取值的坐标轴；index：需要取值的索引矩阵。
            # 能否理解为根据权重比例从各个component的samples中的各个维做挑选？？？
            samples = torch.gather(
                components_samples,
                dim=-1,
                index=weights_sample,
            ).squeeze(-1)
        return samples

    @constraints.dependent_property
    def support(self) -> constraints.Constraint:
        return constraints.real

    # 计算新的mean和var？
    @property
    def mean(self) -> torch.Tensor:
        weights_probs = torch.stack(self.weights.probs.unbind(dim=-1))
        components_means = torch.stack([comp.mean for comp in self.components])
        return (weights_probs * components_means).sum(dim=0)

    @property
    def variance(self) -> torch.Tensor:
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        weights_probs = torch.stack(self.weights.probs.unbind(dim=-1))
        components_var = torch.stack([comp.variance for comp in self.components])
        expected_cond_var = (weights_probs * components_var).sum(dim=0)
        components_means = torch.stack([comp.mean for comp in self.components])
        var_cond_expectation = (weights_probs * components_means.pow(2.0)).sum(
            dim=0
        ) - self.mean.pow(2.0)
        return expected_cond_var + var_cond_expectation

    # 返回在value处的cdf函数的值？
    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        weights_prob = self.weights.probs
        components_cdf = torch.stack([comp.cdf(value) for comp in self.components])
        return (weights_prob * components_cdf).sum(dim=0)


# 混合的输出分布
# 该模块会在初始化模型之前被调用？？
class MixtureOutput(DistributionOutput):
    distr_cls = Mixture

    def __init__(self, components: list[DistributionOutput]):
        self.components = components

    # 相较于基类，这里的distr_cls就是Mxiture；并且可以将多个component的分布给混合起来。
    def _distribution(
        self,
        distr_params: PyTree[torch.Tensor, "T"],  # pytree是指用python容器（如list、dict、tuple、OrderedDict、None、namedtuple等）储存的树状结构的数据。大白话：就是嵌套式的list/dict/tuple结构
        validate_args: Optional[bool] = None,
    ) -> Distribution:
        return self.distr_cls(
            weights=Categorical(
                logits=distr_params["weights_logits"], validate_args=validate_args  # distr_params["weights_logits"]的shape为(128, 512, 128, 4)
            ),
            components=[
                component._distribution(comp_params, validate_args=validate_args)
                for component, comp_params in zip(
                    self.components, distr_params["components"]
                )  # 这里共有4个分布，分别是StudentT、NormalFixedScale、NegativeBinomial、LogNormal。以及各自对应的参数。
            ],
            validate_args=validate_args,
        )

    @property
    def args_dim(self) -> PyTree[int, "T"]:
        return dict(
            weights_logits=len(self.components),
            components=[comp.args_dim for comp in self.components],
        )

    @property
    def domain_map(self) -> PyTree[Callable[[torch.Tensor], torch.Tensor], "T"]:
        return dict(
            weights_logits=lambda x: x,
            components=[comp.domain_map for comp in self.components],
        )
