import numpy as np
import torch
from typing import Union
def convert_any_to_numpy(
    x: Union[torch.Tensor, np.ndarray, int, float, list, tuple],
    accept_none: bool=True) -> np.ndarray:
    if x is None and accept_none: return None
    if x is None and not accept_none: raise ValueError('Trying to convert an empty value.')
    if isinstance(x, np.ndarray): return x
    elif isinstance(x, int) or isinstance(x, float): return np.array([x, ])
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accept_none: return None
        if x.numel() == 0 and not accept_none: raise ValueError('Trying to convert an empty value.')
        if x.numel() >= 1: return x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as numpy type.')


def compute_mse_loss(histogram: list, start: int, step: int, end: int):
    # 如果你觉得 mse 太慢，想办法加速这段代码就可以了
    # 求解 mse 时，我们假设每一个 bin 里面的数据都是均匀分布的
    # 我们需要给一个直方图，并用 start, end, step 给出量化表示的范围
    # losses = [0 for _ in histogram]  debug
    num_of_elements = sum(histogram)
    loss = 0
    for idx, bin in enumerate(histogram):
        if idx < start:
            # 如果所选的 bin 已经超出了起点，那从 bin 的中心到起点的距离即
            # ((idx 到 起点的距离) + 0.5)
            # 注意 hist 统计时是直接取 floor 的，因此会在这里额外 - 1
            error = ((start - idx - 1) + 0.5)
        elif idx > end:
            # 注意 hist 统计时是直接取 floor 的
            error = ((idx - end) + 0.5)
        else:
            # 分别计算左右两侧的 err
            l_idx = (idx - start) % step
            r_idx = step - l_idx - 1
            if l_idx == r_idx:
                error = (l_idx + 0.25)
            else:
                l_err = (l_idx + 0.5)
                r_err = (r_idx + 0.5)
                error = min(l_err, r_err)
        loss += (bin * error * error) / num_of_elements
        # losses[idx] = bin * error * error
    return loss