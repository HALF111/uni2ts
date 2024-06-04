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

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution

from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
    PackedPointLoss,
)
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
)

from .module import MoiraiModule


# 做预训练！！
# 不同于模型定义，这里继承自lightning中的module。
class MoiraiPretrain(L.LightningModule):
    # 数据包含的各个列？
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    # 对各个列做的padding函数（基本都是np.zeros？）
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }

    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiModule] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        # warmup的步骤必须比总步骤少
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        # 保存超参数？
        # 也即将init函数的所有参数都保存到self.hparams中？
        self.save_hyperparameters(ignore=["module"])
        # moirai模型
        self.module = MoiraiModule(**module_kwargs) if module is None else module

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],  # 如[128, 512, 128]
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        # 经过模型后，会得到一个输出的分布！
        distr = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return distr

        # # PS：原来的uni2ts_main里在这里就做了采样？
        # # 然后从输出分布中采样出num_samples个样本，作为最后的预测输出
        # # PS：在init函数中num_samples被默认设置为100。
        # preds = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))
        # return rearrange(preds, "n b ... -> b n ...")

    # uni2ts_main中单独的loss函数
    # def loss(
    #     self,
    #     target: Float[torch.Tensor, "*batch seq_len max_patch"],
    #     observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
    #     sample_id: Int[torch.Tensor, "*batch seq_len"],
    #     time_id: Int[torch.Tensor, "*batch seq_len"],
    #     variate_id: Int[torch.Tensor, "*batch seq_len"],
    #     prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
    #     patch_size: Int[torch.Tensor, "*batch seq_len"],
    # ) -> Float[torch.Tensor, ""]:
    #     # 经过模型得到输出的分布！
    #     distr = self.module(
    #         target,
    #         observed_mask,
    #         sample_id,
    #         time_id,
    #         variate_id,
    #         prediction_mask,
    #         patch_size,
    #     )
    #     # 从超参中获得损失函数，其中默认为PackedNLLLoss？
    #     loss = self.hparams.loss_func(
    #         pred=distr,
    #         target=target,
    #         prediction_mask=prediction_mask,
    #         observed_mask=observed_mask,
    #         sample_id=sample_id,
    #         variate_id=variate_id,
    #     )
    #     return loss
    

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # PS：这的输入batch是经过各种Tranformation之后的数据，例如patch切割、sequence packing等等
        # batch为dict，包含以下这些项：
        # target: Float[torch.Tensor, "*batch seq_len max_patch"],  # * 如[128, 512, 128]
        # observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        # sample_id: Int[torch.Tensor, "*batch seq_len"],
        # time_id: Int[torch.Tensor, "*batch seq_len"],
        # variate_id: Int[torch.Tensor, "*batch seq_len"],
        # prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        # patch_size: Int[torch.Tensor, "*batch seq_len"],
        
        # 先运行模型得到输出分布，这里为经过AffineTransformed包装后的Mixture分布
        distr = self(
            # 这里在输入额外加上sample_id
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )
        # 计算loss，得到的为标量。
        loss = self.hparams.loss_func(
            pred=distr,
            **{
                field: batch[field]
                for field in [
                    "target",
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        # 处理一下batch中的内容？得到当前这个batch实际包含的样本数（如490）
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None  # batch["sample_id"]的shape为(128,512)，batch["sample_id"].max(dim=1)统计每个bins中的最大值，其实也就是统计每个bins中装了几个样本？再求sum，就得到这128个bins中实际有多少个样本。
        )
        # Log a key, value pair。例如self.log('train_loss', loss)，用于存储下当前的loss。# 这里会将loss记录在tensorboard中，方便后续可视化。
        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        # 返回loss？
        return loss

    # validation的step和training的step的一样的
    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        distr = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )
        val_loss = self.hparams.loss_func(
            pred=distr,
            **{
                field: batch[field]
                for field in [
                    "target",
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        if self.hparams.val_metric is not None:
            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                if isinstance(metric_func, PackedPointLoss):
                    pred = distr.sample(torch.Size((self.hparams.num_samples,)))
                    pred = torch.median(pred, dim=0).values
                elif isinstance(metric_func, PackedDistributionLoss):
                    pred = distr
                else:
                    raise ValueError(f"Unsupported loss function: {metric_func}")

                metric = metric_func(
                    pred=pred,
                    **{
                        field: batch[field]
                        for field in [
                            "target",
                            "prediction_mask",
                            "observed_mask",
                            "sample_id",
                            "variate_id",
                        ]
                    },
                )

                self.log(
                    f"val/{metric_func.__class__.__name__}",
                    metric,
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                )

        return val_loss

    # 关于优化器的一些设置
    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        # 设置黑白名单，表示是否需要做decay？
        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                # bias统一不做decay（PS：原文是不是说所有layer都不包含bias项了来着）
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                # 是weight并且在白名单里：做decay
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                # 是weight但是在黑名单里，也不做decay。
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # 确保decay和非decay之间没有重叠
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        # 确保二者加起来和总参数一样多
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        # optim的参数
        # 按照decay和no_dacy分别设置
        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        # 使用AdamW的optimizer
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        # 这里默认使用COSINE_WITH_RESTARTS的策略
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }

    # 预训练时默认使用的transformation函数？
    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        # 不妨假设单变量输入的item_id为resstock_tmy3_south_G40000400_280620，形状为[(8761,)]，频率为"H"，start为"2018-01-01T00:00"。
        # 多变量输入的形状为(23, 102)
        def default_train_transform():
            return (
                # 为当前任务采样变量个数
                SampleDimension(
                    max_dim=self.hparams.max_dim,  # 最大的采样维度不超过128
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )  # 单变量：由于只有1个变量，所以仍为[(8761,)]。多变量：23个变量采样后可能只剩18个变量了。
                # 获得patch大小？
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=DefaultPatchSizeConstraints(),
                    offset=True,
                )  # 当前单变量例子的patch大小为64，多变量例子的为32。
                # 做patch的裁剪
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                # 打包各个特征列？
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=False,
                )
                # 生成一个针对NaN的mask
                # 其中NaN的值会变成False，正常值的地方为True
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                # 将NaN的地方用指定的imputation方法来填充，默认是用0来填充
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                # 真正将数据切割成patch
                + Patchify(
                    max_patch_size=max(self.module.patch_sizes),
                    fields=("target", "observed_mask"),
                    optional_fields=("past_feat_dynamic_real",),
                )
                # 添加variate_id
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=True,
                    collection_type=dict,
                )
                # 添加time_id
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                # 对当前任务按一定比例切割出预测窗口？
                # 相当于做一个总窗口长度的采样，以及回溯窗口和预测窗口比例的采样
                + MaskedPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    truncate_fields=("variate_id", "time_id", "observed_mask"),
                    optional_truncate_fields=("past_feat_dynamic_real",),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                # 延长mask？
                # 在mask_field基础上再加一个aux的mask；
                # 但这里的mask全为0，所以相当于没有加？
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                # 对variate_id、time_id、prediction_mask、observed_mask、target等维度分别做packing
                # 也就是沿着某个维度做合并
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                # target要特别合并？
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=True,
                )
                # 将field列的axis维、按照target_field列中的target_axis维的大小做repeat！！
                # 这里就是将patch_size列的第0维、按照target列中的第0维的大小做repeat！！
                + SequencifyField(
                    field="patch_size", 
                    target_field="target"
                )
                # 按照预定义好的fields，从调用时输入的data_entry中取出这些fields！
                + SelectFields(
                    fields=list(self.seq_fields)
                )
            )

        # 使用我们定义好的默认的transformation
        return defaultdict(lambda: default_train_transform)
