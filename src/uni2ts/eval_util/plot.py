from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts import maybe
from gluonts.model import Forecast


def plot_single(
    inp: dict,
    label: dict,
    forecast: Forecast,
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    ax: Optional[plt.axis] = None,
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    ax = maybe.unwrap_or_else(ax, plt.gca)

    # 将inpute和label的长度拼接在一起？
    target = np.concatenate([inp["target"], label["target"]], axis=-1)
    start = inp["start"]
    if dim is not None:
        target = target[dim]
        forecast = forecast.copy_dim(dim)

    # 长度和上面的target相同，并且从start对应的时间点开始
    index = pd.period_range(start, periods=len(target), freq=start.freq)
    # 这个相当于画ground truth（包括输入窗口和预测窗口）
    ax.plot(
        index.to_timestamp()[-context_length - forecast.prediction_length :],
        target[-context_length - forecast.prediction_length :],
        label="target",
        color="black",
    )
    # 由于intervals默认设置为[0.5, 0.9]
    # 因此其会画出median的预测，50%范围内的预测，以及90%范围内的预测。
    forecast.plot(
        intervals=intervals,
        ax=ax,
        color="blue",
        name=name,
        show_label=show_label,
    )
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left")


def plot_next_multi(
    axes: np.ndarray,
    input_it: Iterator[dict],
    label_it: Iterator[dict],
    forecast_it: Iterator[Forecast],
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax, inp, label, forecast in zip(axes, input_it, label_it, forecast_it):
        plot_single(
            inp,
            label,
            forecast,
            context_length,
            intervals=intervals,
            ax=ax,
            dim=dim,
            name=name,
            show_label=show_label,
        )
