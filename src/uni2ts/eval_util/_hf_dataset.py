from pathlib import Path

import datasets

from uni2ts.common.env import env


class HFDataset:
    def __init__(self, dataset_name: str, storage_path: Path = env.CUSTOM_DATA_PATH):
        # 加载数据并转成numpy格式
        self.hf_dataset = datasets.load_from_disk(
            str(storage_path / dataset_name)
        ).with_format("numpy")
        # 获取freq
        self.freq = self.hf_dataset[0]["freq"]
        # 根据target维度来判断单变量/多变量预测？
        self.target_dim = (
            target.shape[-1]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    def __iter__(self):
        # 简单遍历即可？
        for sample in self.hf_dataset:
            sample["start"] = sample["start"].item()
            yield sample
