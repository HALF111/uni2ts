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

import itertools
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
import torch
from jaxtyping import Bool, Int
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler, default_collate, default_convert

from uni2ts.common.typing import BatchedSample, Sample


# collate_fn的用处：自定义数据堆叠过程；自定义batch数据的输出形式。（官方解释：合并一系列samples来组成一个Tensor格式的mini-batch）
# collate_fn的使用：定义一个以data为输入的函数；注意, 输入输出分别域getitem函数和loader调用时对应。
@dataclass
class Collate:
    # max_length表示填充后序列的最大长度？
    max_length: Optional[int]
    # seq_fields包含需要做padding的列的名称
    seq_fields: tuple[str, ...]
    # pad_func_map是一个字典，从列名映射到对应的padding函数
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = field(
        default_factory=dict
    )
    # 默认的目标列是“target”
    target_field: str = "target"

    def __post_init__(self):
        # 如果用户自定义了pad_func_map函数，则会覆盖掉原来的。
        self.pad_func_map = defaultdict(self._default_pad_func) | self.pad_func_map

    # 默认的paddng函数：为np.zeros？
    @staticmethod
    def _default_pad_func() -> Callable[[Sequence[int], np.dtype], np.ndarray]:
        return np.zeros

    # 接受一些列sapmles的输入，返回BatchedSample的输出
    def __call__(self, batch: list[Sample]) -> BatchedSample:
        raise NotImplementedError


class PadCollate(Collate):
    def __call__(self, batch: list[Sample]) -> BatchedSample:
        assert all(
            [
                len(sample[self.target_field]) == len(sample[key])
                for sample in batch
                for key in self.seq_fields
            ]
        ), "All fields must have the same length."
        assert all(
            [len(sample[self.target_field]) <= self.max_length for sample in batch]
        ), f"Sample length must be less than or equal to max_length ({self.max_length})"

        sample_id = self.get_sample_id(batch)
        padded_batch = self.pad_samples(batch)
        merged_batch = padded_batch | dict(sample_id=sample_id)
        return merged_batch

    def pad_samples(self, batch: list[Sample]) -> BatchedSample:
        for sample in batch:
            length = len(sample[self.target_field])
            for key in self.seq_fields:
                sample[key] = torch.cat(
                    [
                        default_convert(sample[key]),
                        default_convert(
                            self.pad_func_map[key](
                                (self.max_length - length,) + sample[key].shape[1:],
                                sample[key].dtype,
                            )
                        ),
                    ]
                )
        return default_collate(batch)

    def get_sample_id(self, batch: list[Sample]) -> Int[torch.Tensor, "batch seq"]:
        sample_id = torch.stack(
            [
                torch.cat([torch.ones(length), torch.zeros(self.max_length - length)])
                for sample in batch
                if (length := len(sample[self.target_field]))
            ]
        ).to(torch.long)
        return sample_id


# 预训练时使用PackCollate
class PackCollate(Collate):
    # 由于batch_size为256，这里会先调用256次__getitem__函数构造256个样本！
    # 因此batch中每个样本的长度可能都不相同，如(54,128), (37,128), (163,128)等。做collate_fn的目的就是要对这种不等长的数据做补齐
    # 后面的128都相同是因为max_patch_size都是128。而无论当前patch_size是多少，都需要后面补0填充到最大长度的128。
    def __call__(self, batch: list[Sample]) -> BatchedSample:
        # 1、确保每个样本的目标字段和序列字段具有相同的长度，并且每个样本的目标字段长度不超过最大长度。
        assert all(
            [
                len(sample[self.target_field]) == len(sample[key])
                for sample in batch
                for key in self.seq_fields
            ]
        ), "All fields must have the same length."
        assert all(
            [len(sample[self.target_field]) <= self.max_length for sample in batch]
        ), f"Sample length must be less than or equal to max_length ({self.max_length})"

        # 2、使用首次递减装箱算法对样本进行分组，使得每个组中的样本长度尽可能接近但不超过最大长度。
        packed_batch, bin_spaces = self.first_fit_decreasing_bin_packing(batch)
        # 3、获取样本的ID，其中ID由每个样本在其所在组中的位置确定。
        sample_id = self.get_sample_id(packed_batch, bin_spaces)
        # 4、合并分组中的样本，并根据需要进行填充，以便每个分组中的样本具有相同的长度。
        merged_batch = self.merge_batch(packed_batch, bin_spaces) | dict(
            sample_id=sample_id
        )
        return merged_batch  # 例如这里的merged_batch["target"]可能会变成(66, 512, 128)，也就是将所有bins中样本合并成一整个数据。# PS：66表示前66个bins有数据，512表示每个bins长度为512（包含512个patch的数据），128表示每个patch的长度为128个数据点。

    # 首次递减装箱算法！
    # * 感觉很像内存分配中的First Fit算法？？？
    def first_fit_decreasing_bin_packing(
        self,
        batch: list[Sample],
    ) -> tuple[list[list[Sample]], Int[np.ndarray, "batch"]]:
        # 首先将输入的样本按照各个样本的target_field的长度，进行降序排序。最长的样本在最前面
        batch = sorted(
            batch, key=lambda sample: len(sample[self.target_field]), reverse=True
        )
        # 大小为batch的数组，用于存储每个bins剩余的空间大小。这里batch大小为256，所以有256个bins。
        # 其中每个bin的初始值为max_length，这里为512，表示最大能存储的patch个数！！
        # # 所以，bin_spaces的大小为(256,)，而其中的值均初始化为512。
        bin_spaces: Int[np.ndarray, "batch"] = np.full(len(batch), self.max_length)
        # packed_batch用于存储做了packing之后的样本
        # 也即存储的是：每个bin里面分别存里哪些sample。（如bin-0放了哪几个sample，bin-1放了哪几个sample）
        packed_batch: list[list[Sample]] = [[]]  # 其初始长度为1！！

        for sample in batch:
            length = len(sample[self.target_field])  # 假设第一个为(510, 128)
            # 检查当前样本是否能放入现有的bins中
            criterion: Bool[np.ndarray, "batch"] = bin_spaces - length >= 0
            # 由于criterion为bool类型，值非0即1；而np.argmax在多个最大值时会返回第一个。所以这里是找到第一个有足够空间的bin。
            bin_id: int = criterion.argmax()
            # 如果当前bin_id大于packed_batch的长度，说明这个bin_id在当前的存储做样本的数组中已经装不下了，需要在盛放结果的packed_batch中添加新的且为空的bin。
            # 并且此时应当保证bin_id和packed_batch长度相等，才应当添加新的空bin；否则应该报错！
            if len(packed_batch) <= bin_id:
                if len(packed_batch) != bin_id:
                    raise ValueError
                packed_batch.append([])

            # 将样本添加到盛放结果的packed_batch对应的bin_id中，并更新盛放空间大小的bin_spaces的剩余空间大小。
            packed_batch[bin_id].append(sample)
            bin_spaces[bin_id] -= length

        # 返回pack后的样本列表pakced_batch、和每个bin的剩余空间数组（只保留已使用的bin）。
        # * PS：通过将各个列装到bins，充分利用空间，避免了为了对齐各个样本而需要过长的padding的需求。
        # 以及为了保证长度相等，只返回bin_spaces的前len(packed_batch)部分
        
        # * 由于大部分batch长度都远小于bin的容量；所以256个桶最后可能只有前66个包含内容。
        return packed_batch, bin_spaces[: len(packed_batch)]

    # 获得样本的id
    # 输入为first_fit_decreasing_bin_packing函数处理好的样本列表(packed_batch)和剩余空间数组(bin_spaces)
    # 这里应该是给每个sample一个id，以防止从sequence packing还原后无法区分样本间的边界了。
    def get_sample_id(
        self, batch: list[list[Sample]], bin_spaces: Int[np.ndarray, "batch"]
    ) -> Int[torch.Tensor, "batch seq"]:
        # 对于所有bin，将其stack在一起，形成二维张量。
        sample_id = torch.stack(
            [
                torch.cat(
                    [
                        # 这一步不是原论文的添加variate_id的那一步？！虽然有点像
                        # 这里应该是给每个sample一个id，以防止从sequence packing还原后无法区分样本间的边界了。
                        # * 构建一个和当前样本等长的张量，其值全为idx+1。
                        # * 例如当前bin包含3个样本，长度分别为200，150，100。那么会生成长为200的全为1的向量、长为150的全为2的向量，以及长为100的全为3的向量。
                        torch.ones(len(sample[self.target_field])) * (idx + 1)
                        for idx, sample in enumerate(bin_)  # 遍历当前bin中的样本？
                    ]
                    # padding，将剩余空间作为idx=0给拼在最后。也即假设最后还剩30的space，那么最后还会添上一个长为30的全为0的向量。
                    + [torch.zeros(space)],
                )
                # 由于二者长度相同，遍历获得当前bin和对应的空间
                for bin_, space in zip(batch, bin_spaces)
            ]
        ).to(torch.long)
        return sample_id  # 假设有66个bins，那么其大小恰为(66, 512)

    # 输入同样为first_fit_decreasing_bin_packing函数处理好的样本列表(packed_batch)和剩余空间数组(bin_spaces)
    # 感觉像是将batch中的数据做stack后整理为统一的张量格式
    def merge_batch(
        self, batch: list[list[Sample]], bin_spaces: Int[np.ndarray, "batch"]
    ) -> BatchedSample:
        batch = {
            key: torch.stack(
                [
                    torch.cat(
                        [default_convert(sample[key]) for sample in bin_]  # defualt_convert尝试将所有np.array的内部元素转成tensor。
                        + [
                            # PS：default_convert函数将numpy.array转成torch.tensor
                            default_convert(
                                # pad_func_map是一个字典，从列名映射到对应的padding函数，这些padding函数大多默认为np.zeros。
                                self.pad_func_map[key](
                                    # 长度是剩余的长度（如0）+样本的长度（如128）？
                                    (space,) + bin_[0][key].shape[1:],
                                    bin_[0][key].dtype,
                                )
                            )
                        ]
                    )
                    # 遍历获得当前bin和对应的空间
                    for bin_, space in zip(batch, bin_spaces)
                ],
            )
            for key in self.seq_fields  # 先访问所有列？seq_fields包含需要做padding的列的名称
        }
        # 假设是前66个bins包含样本，这里的batch["target"]会变成(66, 512, 128)，也就是将所有bins中样本合并成一整个数据。
        # PS：66表示前66个bins有数据，512表示每个bins长度为512（包含512个patch的数据），128表示每个patch的长度为128个数据点。
        return batch


@dataclass
class SliceableBatchedSample:
    data: BatchedSample

    def __post_init__(self):
        assert all(
            [
                len(self.data[key]) == len(self.data[next(iter(self.data))])
                for key in self.data.keys()
            ]
        )

    def __len__(self) -> int:
        return len(self.data[next(iter(self.data))])

    def __getitem__(self, item: slice) -> "SliceableBatchedSample":
        # 从data的各个keys中获取item的这一项？
        return SliceableBatchedSample(
            {key: self.data[key][item] for key in self.data.keys()}
        )


# Metadata的类？
class Metadata(NamedTuple):
    shape: tuple[int, ...]
    dtype: torch.dtype


# 以batch为单位的样本队列？
@dataclass
class BatchedSampleQueue:
    container: deque[SliceableBatchedSample] = field(default_factory=deque)
    schema: Optional[dict[str, Metadata]] = None

    def _check_schema(self, batch: SliceableBatchedSample):
        if self.schema is None:
            self.schema = {
                key: Metadata(
                    shape=tuple(batch.data[key].shape[1:]), dtype=batch.data[key].dtype
                )
                for key in batch.data.keys()
            }
        else:
            assert all(
                [
                    (key in batch.data)
                    and (metadata.shape == tuple(batch.data[key].shape[1:]))
                    and (metadata.dtype == batch.data[key].dtype)
                    for key, metadata in self.schema.items()
                ]
            ), "batch must have the same schema as the first batch"

    def append(self, batch: SliceableBatchedSample | BatchedSample):
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.append(batch)

    def appendleft(self, batch: SliceableBatchedSample | BatchedSample):
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.appendleft(batch)

    def popleft(self, size: int) -> BatchedSample:
        if size > len(self):
            raise ValueError(
                f"pop size ({size}) must be less than or equal to queue size ({len(self)})"
            )

        out = BatchedSampleQueue()
        # 大概是不断pop、直到长度和size相等为止？
        while len(out) < size:
            curr = self.container.popleft()
            if len(out) + len(curr) > size:
                self.appendleft(curr[size - len(out) :])
                curr = curr[: size - len(out)]
            out.append(curr)
        return out.as_batched_data()

    def as_batched_data(self) -> BatchedSample:
        return {
            key: torch.cat([batch.data[key] for batch in self.container], dim=0)
            for key in self.schema.keys()
        }

    def __len__(self) -> int:
        # 长度是所有batch的长度乘以batch的个数？
        return sum(len(batch) for batch in self.container)


# 以一个batch的样本为单位的iterator
@dataclass
class _BatchedSampleIterator:
    dataloader_iter: Iterator[BatchedSample]
    batch_size: int
    drop_last: bool
    fill_last: bool
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]]

    def __post_init__(self):
        # 先定义一个队列
        self.queue = BatchedSampleQueue()

    def __iter__(self):
        return self

    def __next__(self) -> BatchedSample:
        while (data := self._next_batch()) is None:
            continue
        return data

    def _next_batch(self) -> Optional[BatchedSample]:
        if len(self.queue) < self.batch_size:
            try:
                # 如果当前queue的长度小于batch_size，则先从当前dataloader中取出一个数据、放到queue中
                data = next(self.dataloader_iter)
                self.queue.append(data)
                return None
            except StopIteration:
                # 如果dataloader没数据了，如何处理呢？
                if self.drop_last or len(self.queue) == 0:
                    raise StopIteration
                elif self.fill_last:
                    # 如果是fill_last，那么在最后做padding，直到其大小和batch_size相等
                    self._pad_queue(self.batch_size - len(self.queue))

        # popleft大概就是不断pop直到长度和batch_size相等为止？
        # 如果剩余的长度小于batch，则全部pop出来
        batch = self.queue.popleft(min(self.batch_size, len(self.queue)))
        return batch

    def _pad_queue(self, size: int):
        if self.queue.schema is None:
            raise ValueError("schema must be set before padding")
        padding = {
            key: default_convert(
                self.pad_func_map[key]((size,) + metadata.shape, np.dtype(np.float32))
            ).to(metadata.dtype)
            for key, metadata in self.queue.schema.items()
        }
        self.queue.append(padding)

    def has_next(self) -> bool:
        if len(self.queue) < self.batch_size:
            try:
                next_batch = next(self)
                self.queue.appendleft(next_batch)
            except StopIteration:
                return False
        return True


# 用于控制加载数据的类！！
class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        batch_size_factor: float = 1.0,
        cycle: bool = False,
        num_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Collate] = None,  # * 这里会去调用TorchDataLoader中的collate_fn，事实上用的是PackCollate类！！
        pin_memory: bool = False,
        drop_last: bool = True,
        fill_last: bool = False,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        if num_batches_per_epoch is not None:
            assert cycle, "can only set 'num_batches_per_epoch' when 'cycle=True'"

        # 就是torch库下的dataloader
        self.dataloader = TorchDataLoader(
            dataset=dataset,
            # 实际dataloader取batch的时候是batch_size和batch_size_factor的乘积？
            batch_size=int(batch_size * batch_size_factor),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            # * 这里会去调用TorchDataLoader中的collate_fn，事实上用的是PackCollate类！！
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
        )
        self.batch_size = batch_size
        self.cycle = cycle
        self.num_batches_per_epoch = num_batches_per_epoch
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.fill_last = fill_last
        self.iterator: Optional[_BatchedSampleIterator] = None

    def __iter__(self) -> Iterator:
        if self.iterator is None or not self.iterator.has_next():
            dataloader_iter = (
                # 如果cycle为False，则正常用torch的取出数据
                # 否则：使用itertools.chain.from_iterable将多个迭代器连接成一个统一的迭代器。
                # 参考：https://docs.python.org/zh-cn/3/library/itertools.html#itertools.chain.from_iterable
                # 以及repeat函数用于持续地返回self.dataloader的迭代器；如果不指定times参数，则会被无限次的返回？
                iter(self.dataloader)
                if not self.cycle
                else itertools.chain.from_iterable(itertools.repeat(self.dataloader))
            )
            # 这里是新定义一个以batch为单位的样本iterator！！
            self.iterator = _BatchedSampleIterator(
                dataloader_iter=dataloader_iter,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                fill_last=self.fill_last,
                # 用的是PackCollate的pad_func_map，其为一个字典，包含从列名映射到对应的padding函数
                pad_func_map=self.collate_fn.pad_func_map,
            )
        # https://docs.python.org/zh-cn/3/library/itertools.html#itertools.islice
        # itertools.islice(iterable, stop)
        # itertools.islice(iterable, start, stop[, step])
        # 一个从可迭代对象返回选定元素的迭代器。
        # 如果start为非零值，则会跳过可迭代对象中的部分元素直至到达start位置。在此之后，将连续返回元素除非step被设为大于1的值而会间隔跳过部分结果。
        # 如果stop为None，则迭代将持续进行直至迭代器中的元素耗尽；在其他情况下，它将在指定的位置上停止。
        # 如果start为None，迭代从0开始。如果step为None，步长缺省为1。
        # 与常规的切片不同，islice()不支持start, stop或step为负值。可被用来从内部结构已被展平的数据中提取相关字段（例如，一个多行报告可以每三行列出一个名称字段）。
        # * 所以这里只设置了一个num_batches_per_epoch=100
        # * 那么也就是每个epoch/iter的数据，只有前面100个batch的数据会被取出来用于训练？
        return itertools.islice(self.iterator, self.num_batches_per_epoch)

    @property
    def worker_init_fn(self) -> Optional[Callable[[int], None]]:
        return self.dataloader.worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, worker_init_fn: Optional[Callable[[int], None]]):
        self.dataloader.worker_init_fn = worker_init_fn
