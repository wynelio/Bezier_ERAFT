import math
from typing import Optional

import torch
torch.set_num_threads(1) # intraop parallelism (this can be a good option)
torch.set_num_interop_threads(1) # interop parallelism


def norm_voxel_grid(voxel_grid: torch.Tensor):
    mask = torch.nonzero(voxel_grid, as_tuple=True)
    if mask[0].size()[0] > 0:
        mean = voxel_grid[mask].mean()
        std = voxel_grid[mask].std()
        if std > 0:
            voxel_grid[mask] = (voxel_grid[mask] - mean) / std
        else:
            voxel_grid[mask] = voxel_grid[mask] - mean
    return voxel_grid


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor, t_from: Optional[int]=None, t_to: Optional[int]=None):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    # 定义初始化方法，接受三个参数:channels\height\width，进行断言检查并保存在类的实例变量中
    def __init__(self, channels: int, height: int, width: int):
        assert channels > 1
        assert height > 1
        assert width > 1
        self.nb_channels = channels
        self.height = height
        self.width = width

    # 根据中心时间计算时间窗口中的起始时间和结束时间
    def get_extended_time_window(self, t0_center: int, t1_center: int):
        dt = self._get_dt(t0_center, t1_center)
        t_start = math.floor(t0_center - dt)
        t_end = math.ceil(t1_center + dt)
        return t_start, t_end

    # 定义私有方法，构建空体素网格，返回一个形状为（self.nb_channels, self.height, self.width）的全0张量
    def _construct_empty_voxel_grid(self):
        return torch.zeros(
            (self.nb_channels, self.height, self.width),
            dtype=torch.float,
            requires_grad=False,
            device=torch.device('cpu'))

    # 定义私有方法，计算时间间隔dt
    def _get_dt(self, t0_center: int, t1_center: int):
        assert t1_center > t0_center
        return (t1_center - t0_center)/(self.nb_channels - 1)

    # 对时间归一化，范围【0, self.nb_channels - 1】
    def _normalize_time(self, time: torch.Tensor, t0_center: int, t1_center: int):
        # time_norm < t0_center will be negative
        # time_norm == t0_center is 0
        # time_norm > t0_center is positive
        # time_norm == t1_center is (nb_channels - 1)
        # time_norm > t1_center is greater than (nb_channels - 1)
        return (time - t0_center)/(t1_center - t0_center)*(self.nb_channels - 1)

    # 定义静态方法，判断张量是否为整数
    @staticmethod
    def _is_int_tensor(tensor: torch.Tensor) -> bool:
        return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)

    # 父类中抽象方法的实现
    # 根据输入的时间窗口中心和时间值，使用一些计算和插值操作构建体素网格
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor, t0_center: Optional[int]=None, t1_center: Optional[int]=None):
        assert x.device == y.device == pol.device == time.device == torch.device('cpu')
        assert type(t0_center) == type(t1_center)
        assert x.shape == y.shape == pol.shape == time.shape
        # 确保x是一个一维张量
        assert x.ndim == 1
        # 确保time是离散的时间值
        assert self._is_int_tensor(time)

        # 判断x是否为整数类型
        is_int_xy = self._is_int_tensor(x)
        if is_int_xy:
            assert self._is_int_tensor(y)

        voxel_grid = self._construct_empty_voxel_grid()
        ch, ht, wd = self.nb_channels, self.height, self.width
        # 禁用梯度计算，提高计算效率
        with torch.no_grad():
            t0_center = t0_center if t0_center is not None else time[0]
            t1_center = t1_center if t1_center is not None else time[-1]
            t_norm = self._normalize_time(time, t0_center, t1_center)

            t0 = t_norm.floor().int()
            value = 2*pol.float()-1

            if is_int_xy:
                for tlim in [t0,t0+1]:
                    mask = (tlim >= 0) & (tlim < ch)
                    interp_weights = value * (1 - (tlim - t_norm).abs())

                    # 计算体素在一维张量中的索引
                    index = ht * wd * tlim.long() + \
                            wd * y.long() + \
                            x.long()

                    # 将插值权重 根据布尔掩码mask放入体素网格中的指定索引位置
                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)
            else:
                x0 = x.floor().int()
                y0 = y.floor().int()
                for xlim in [x0,x0+1]:
                    for ylim in [y0,y0+1]:
                        for tlim in [t0,t0+1]:

                            mask = (xlim < wd) & (xlim >= 0) & (ylim < ht) & (ylim >= 0) & (tlim >= 0) & (tlim < ch)
                            interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                            index = ht * wd * tlim.long() + \
                                    wd * ylim.long() + \
                                    xlim.long()

                            voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid
