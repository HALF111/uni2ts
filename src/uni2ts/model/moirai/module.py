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

from functools import partial

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
from torch.utils._pytree import tree_map

from uni2ts.common.torch_util import mask_fill, packed_attention_mask
from uni2ts.distribution import DistributionOutput
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from uni2ts.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.ts_embed import MultiInSizeLinear


def encode_distr_output(
    distr_output: DistributionOutput,
) -> dict[str, str | float | int]:
    def _encode(val):
        if not isinstance(val, DistributionOutput):
            return val

        return {
            "_target_": f"{val.__class__.__module__}.{val.__class__.__name__}",
            **tree_map(_encode, val.__dict__),
        }

    return _encode(distr_output)


def decode_distr_output(config: dict[str, str | float | int]) -> DistributionOutput:
    return instantiate(config, _convert_="all")


# Moirai的模型！！
# 继承自nn.Module和PyTorchModelHubMixin
class MoiraiModule(
    nn.Module,
    PyTorchModelHubMixin,
    coders={DistributionOutput: (encode_distr_output, decode_distr_output)},
):
    """Contains components of Moirai to ensure implementation is identical across models"""

    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        num_layers: int,
        patch_sizes: tuple[int, ...],  # tuple[int, ...] | list[int]
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        # * num_embeddings代表词典大小，而embedding_dim表示嵌入向量的维度；
        # * 这里只需要一个全局共享的mask，故前者为1；后者则需要和模型的d_model一致。
        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        # scaling表示做归一化，否则后者表示不做归一化。
        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        # 核心的encoder层！！
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,  # 使用RMSNorm！！
            activation=F.silu,  # 使用silu激活函数？
            use_glu=True,
            use_qk_norm=True,  # 做qk-norm
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )
        # 这个是控制输出按照预定的分布来？
        # 默认是一个mixture.MixtureOutput的模块？
        # PS：并且其由StudentTOutput、NormalFixedScaleOutput、NegativeBinomialOutput、LogNormalOutput共4个分布混合而来
        self.distr_output = distr_output
        # 多patch的输出层
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        # 1、先对数据(target)以及mask和id做归一化？  # 这里target的输入可能为(128, 512, 128)，第一个128表示batch大小为128个bins，512表示每个bins可以装512个patch，最后的128表示每个patch的长度为128。
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )  # 二者的shape均为(128, 512, 1)，说明是在各个patch内部做的scale。
        scaled_target = (target - loc) / scale
        # 2、然后经过多patch的输入线性层
        reprs = self.in_proj(scaled_target, patch_size)  # 此时reprs为(128, 512, 384)，表示patch长度从128维映射到了384维。
        # 3、将预测窗口部分标记上mask，也即用self.mask_encoding来做masking
        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)
        # 4、经过encoder层
        reprs = self.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )
        # 5、然后是输出部分的参数映射层
        # PS：但本质上还是MultiOutSizeLinear这个类  # PS：此时reprs为(128, 512, 384)，表示patch长度从128维映射到了中间向量的384维。# 最后得到的distr_param为一个dict，包含"weights_lofits"和"components"，前者的shape为(128,512,128,4)，4表示有4种分布混合（这里分别是StudentT、NormalFixedScale、NegativeBinomial、LogNormal）。后者则包含4个分布对应的如df、loc、scale、logits等信息。
        distr_param = self.param_proj(reprs, patch_size)
        # 6、最后做一个分布的转换？
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        # 返回该分布！！
        return distr
