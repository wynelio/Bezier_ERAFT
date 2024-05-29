from __future__ import annotations

import math
from typing import Union, List

import numpy as np
import torch as th
from numba import jit

has_scipy_special = True
try:
    from scipy import special
except ImportError:
    has_scipy_special = False

from models.raft_utils.utils import cvx_upsample

class BezierCurves:
    # Each ctrl point lives in R^2
    CTRL_DIM: int = 2

    def __init__(self, bezier_params: th.Tensor):
        # bezier_params: batch, ctrl_dim*(n_ctrl_pts - 1), height, width
        assert bezier_params.ndim == 4
        self._params = bezier_params

        # some helpful meta-data:
        self.batch, channels, self.ht, self.wd = self._params.shape
        assert channels % 2 == 0
        # P0 is always zeros as it corresponds to the pixel locations.
        # Consequently, we only compute P1, P2, ...
        self.n_ctrl_pts = channels // self.CTRL_DIM + 1
        assert self.n_ctrl_pts > 0

        # math.comb is only available in python 3.8 or higher
        self.use_math_comb = hasattr(math, 'comb')
        if not self.use_math_comb:
            assert has_scipy_special
            assert hasattr(special, 'comb')

    def comb(self, n: int, k:int):
        if self.use_math_comb:
            return math.comb(n, k)
        return special.comb(n, k)

    @classmethod
    def create_from_specification(cls, batch_size: int, n_ctrl_pts: int, height: int, width: int, device: th.device) -> BezierCurves:
        assert batch_size > 0
        assert n_ctrl_pts > 1
        assert height > 0
        assert width > 0
        params = th.zeros(batch_size, cls.CTRL_DIM * (n_ctrl_pts - 1), height, width, device=device)
        return cls(params)

    @classmethod
    def from_2view(cls, flow_tensor: th.Tensor) -> BezierCurves:
        # This function has been written to visualize 2-view predictions for our paper.
        batch_size, channel_size, height, width = flow_tensor.shape
        assert channel_size == 2 == cls.CTRL_DIM
        return cls(flow_tensor)

    @classmethod
    def create_from_voxel_grid(cls, voxel_grid: th.Tensor, downsample_factor: int=8, bezier_degree: int=2) -> BezierCurves:
        assert isinstance(downsample_factor, int)
        assert downsample_factor >= 1
        batch, _, ht, wd = voxel_grid.shape
        assert ht % 8 == 0
        assert wd % 8 == 0
        ht, wd = ht//downsample_factor, wd//downsample_factor
        n_ctrl_pts = bezier_degree + 1
        return cls.create_from_specification(batch_size=batch, n_ctrl_pts=n_ctrl_pts, height=ht, width=wd, device=voxel_grid.device)

    @property
    def device(self):
        return self._params.device

    @property
    def dtype(self):
        return self._params.dtype

    def create_upsampled(self, mask: th.Tensor) -> BezierCurves:
        """ Upsample params [N, dim, H/8, W/8] -> [N, dim, H, W] using convex combination """
        up_params = cvx_upsample(self._params, mask)
        return BezierCurves(up_params)

    def detach(self, clone: bool=False, cpu: bool=False) -> BezierCurves:
        params = self._params.detach()
        if cpu:
            return BezierCurves(params.cpu())
        if clone:
            params = params.clone()
        return BezierCurves(params)

    def detach_(self, cpu: bool=False) -> None:
        # Detaches the bezier parameters in-place!
        self._params = self._params.detach()
        if cpu:
            self._params = self._params.cpu()

    def cpu(self) -> BezierCurves:
        return BezierCurves(self._params.cpu())

    def cpu_(self) -> None:
        # Puts the bezier parameters to CPU in-place!
        self._params = self._params.cpu()

    @property
    def requires_grad(self):
        return self._params.requires_grad

    @property
    def batch_size(self):
        return self._params.shape[0]

    @property
    def degree(self):
        return self.n_ctrl_pts - 1

    @property
    def dim(self):
        return self._params.shape[1]

    @property
    def height(self):
        return self._params.shape[-2]

    @property
    def width(self):
        return self._params.shape[-1]

    def get_params(self) -> th.Tensor:
        return self._params

    def _param_view(self) -> th.Tensor:
        return self._params.view(self.batch, self.CTRL_DIM, self.degree, self.ht, self.wd)

    def delta_update_params(self, delta_bezier: th.Tensor) -> None:
        assert delta_bezier.shape == self._params.shape
        self._params = self._params + delta_bezier

    @staticmethod
    def _get_binom_coeffs(degree: int):
        n = degree
        k = np.arange(degree) + 1
        return special.binom(n, k)

    @staticmethod
    @jit(nopython=True)
    def _get_time_coeffs(timestamps: np.ndarray, degree: int):
        assert timestamps.min() >= 0
        assert timestamps.max() <= 1
        assert timestamps.ndim == 1
        # I would like to check ensure float64 dtype but have not found a way to check in jit
        #assert timestamps.dtype == np.dtype('float64')

        num_ts = timestamps.size
        out = np.zeros((num_ts, degree))
        for t_idx in range(num_ts):
            for d_idx in range(degree):
                time = timestamps[t_idx]
                i = d_idx + 1
                out[t_idx, d_idx] = (1 - time)**(degree - i)*time**i
        return out

    def _compute_flow_from_timestamps(self, timestamps: Union[List[float], np.ndarray]):
        if isinstance(timestamps, list):
            timestamps = np.asarray(timestamps)
        else:
            assert isinstance(timestamps, np.ndarray)
        assert timestamps.dtype == 'float64'
        assert timestamps.size > 0
        assert np.min(timestamps) >= 0
        assert np.max(timestamps) <= 1

        degree = self.degree
        # 根据阶数计算二项式系数和时间系数
        binom_coeffs = self._get_binom_coeffs(degree)
        time_coeffs = self._get_time_coeffs(timestamps, degree)
        # poly coeffs: time, degree
        # 将二项式系数和时间系数相乘，得到多项式系数，形状为【degree+1, timestamp.size】
        polynomial_coeffs = np.einsum('j,ij->ij', binom_coeffs, time_coeffs)
        polynomial_coeffs = th.from_numpy(polynomial_coeffs).float().to(device=self.device)

        # params: batch, dim, degree, height, width
        params = self._param_view()
        # flow: timestamps, batch, dim, height, width
        # 参数params和多项式系数polynomial_coeffs相乘，得到光流，维度顺序是timestamps, batch, dim, height, width
        flow = th.einsum('bdphw,tp->tbdhw', params, polynomial_coeffs)
        return flow

    def get_flow_from_reference(self, time: Union[float, int, List[float], np.ndarray]) -> th.Tensor:
        params = self._param_view()
        batch, dim, degree, height, width = params.shape
        time_is_scalar = isinstance(time, int) or isinstance(time, float)
        if time_is_scalar:
            assert time >= 0.0
            assert time <= 1.0
            if time == 1:
                P_end = params[:, :, -1, ...]
                return P_end
            if time == 0:
                return th.zeros((batch, dim, height, width), dtype=self.dtype, device=self.device)
            time = np.array([time], dtype='float64')
        elif isinstance(time, list):
            time = np.asarray(time, dtype='float64')
        else:
            assert isinstance(time, np.ndarray)
        assert time.dtype == 'float64'
        assert time.size > 0
        assert np.min(time) >= 0
        assert np.max(time) <= 1

        # flow is coords1 - coords0
        # flows: timestamps, batch, dim, height, width
        flows = self._compute_flow_from_timestamps(timestamps=time)
        if time_is_scalar:
            assert flows.shape[0] == 1
            return flows[0]
        return flows