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
from typing import Callable, Optional

# Hydra:通过组合动态创建分层设置&多任务处理。ref：https://zhuanlan.zhihu.com/p/662221581
import hydra
# lightning：轻量级的pytorch，支持分布式开发。ref：https://zhuanlan.zhihu.com/p/353985363
import lightning as L
import torch
from hydra.utils import instantiate
# omegaconf：高效的配置。ref：https://zhuanlan.zhihu.com/p/638713472
from omegaconf import DictConfig
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, DistributedSampler

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.data.loader import DataLoader
from uni2ts.data.builder import DatasetBuilder


# LightningDataModule是封装了pytorch中涉及数据处理的5个步骤：
# 1、下载/标记化/处理。
# 2、清理并（可能）保存到磁盘。
# 3、加载内部数据集。
# 4、应用变换（旋转、标记化等）。
# 5、包装在 DataLoader 内。
# 因此这里DataModule继承该类，需要实现其中的
class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset  # 训练数据集

        # 验证数据集
        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        # DistributedSampler用于分布式单机多卡/多机多卡的网络训练。
        # 以单机多卡为例，若当前环境有N张显卡，整个数据集会被分割为N份，每张卡会获取到属于自己的那一份数据。
        # ref1：https://blog.csdn.net/chanbo8205/article/details/115242635
        # ref2：https://zhuanlan.zhihu.com/p/660485845
        # 假设4张卡，那么数据就是按照0,1,2,3,0,1,2,3,...这样的顺序分给四张卡的么
        sampler = (
            DistributedSampler(
                dataset,  # 类型为torch.utils.data.Dataset，是采样器需要处理的对象
                num_replicas=None,  # 将数据集划分为几块，默认为None
                rank=None,  # 此sampler要处理的环境的rank号，在单机多卡环境下就是第几张显卡，默认为None
                shuffle=shuffle,  # 是否要打乱数据。这里为true
                seed=0,  # 随机数种子，用于打乱顺序
                drop_last=False,  # 是否丢弃最后一组数据
            )
            # WORLD_SIZE：使用os.environ["world_size"]获取当前启动的所有的进程的数量（所有机器进程的和），
            # 一般world_size = gpus_per_node * nnodes
            if world_size > 1
            else None
        )
        # 实例化data_loader：
        # 根据yaml配置，这里dataloader_func用的是uni2ts.data.loader.DataLoader类
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    # 调用get_dataloader得到loader
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            instantiate(self.cfg.train_dataloader, _partial_=True),
            self.cfg.train_dataloader.shuffle,
            self.trainer.world_size,
            self.train_batch_size,
            num_batches_per_epoch=self.train_num_batches_per_epoch,
        )

    # 验证集的loader？
    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        # tree_map(f, tree) maps f to every leaf of tree
        # partial相当于生成一个部分参数被固定的get_dataloader函数
        return tree_map(
            partial(
                self.get_dataloader,
                dataloader_func=instantiate(self.cfg.val_dataloader, _partial_=True),  # val的
                shuffle=self.cfg.val_dataloader.shuffle,  # val的
                world_size=self.trainer.world_size,
                batch_size=self.val_batch_size,
                num_batches_per_epoch=None,
            ),
            self.val_dataset,
        )

    # 将这几个重定义为成员变量
    @property
    def train_batch_size(self) -> int:
        return self.cfg.train_dataloader.batch_size // (
            self.trainer.world_size 
            * self.trainer.accumulate_grad_batches  # 后者这里为1。
        )

    @property
    def val_batch_size(self) -> int:
        return self.cfg.val_dataloader.batch_size // (
            self.trainer.world_size 
            * self.trainer.accumulate_grad_batches  # 后者这里为1。
        )

    @property
    def train_num_batches_per_epoch(self) -> int:
        return (
            self.cfg.train_dataloader.num_batches_per_epoch
            * self.trainer.accumulate_grad_batches  # 前者为100，后者为1？
        )


# version_base用于选择Hydra在不同版本下的表现，不是很重要，具体请查阅https://hydra.cc/docs/upgrades/version_base/
# config_path表示配置文件所在路径
# config_name表示配置文件文件名，不包含后缀
# 所以这里会加载./conf/pretrain/default.yaml作为配置文件的地址。
@hydra.main(version_base="1.3", config_path="conf/pretrain", config_name="default.yaml")
def main(cfg: DictConfig):
    # 默认为True，float精度为32
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # model是用lightning定义的模型。
    # call也即：hydra的instantiate函数。
    # 它用于实例化python类或函数。主要目标为通过配置文件来完成自动实例化。
    # 事实上，配置文件中的_target_选项便用于指定对象的类或者函数。
    # ref：https://zhuanlan.zhihu.com/p/673818044
    model: L.LightningModule = instantiate(cfg.model, _convert_="all")

    # 是否做compile，这里为false
    if cfg.compile:
        model.module.compile(mode=cfg.compile)
        
    # 定义训练器、数据构建器、数据集等。使用default.yaml文件中的trainer和data设置
    # trainer是lightning.Trainer类
    # 参考：https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
    trainer: L.Trainer = instantiate(cfg.trainer)
    train_dataset_builder: DatasetBuilder = instantiate(cfg.data)
    train_dataset: Dataset = train_dataset_builder.load_dataset(
        # 这里利用model/moirai/pretrain.py中定义的train_transform_map，来在处理数据时间做转换！！
        model.train_transform_map
    )
    # 验证数据集
    val_dataset: Optional[Dataset | list[Dataset]] = (
        tree_map(
            lambda ds: ds.load_dataset(model.val_transform_map),
            instantiate(cfg.val_data, _convert_="all"),
        )
        if "val_data" in cfg
        else None
    )
    
    # 设置seed
    L.seed_everything(cfg.seed + trainer.logger.version, workers=True)
    
    # 开始用当前数据训练模型。
    trainer.fit(
        model,
        datamodule=DataModule(cfg, train_dataset, val_dataset),  # 用cfg、train_dataset和val_dataset定义的data_loader
    )


if __name__ == "__main__":
    main()
