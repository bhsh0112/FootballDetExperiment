"""
简单常速度（CV）卡尔曼滤波器：用于 2D 点轨迹平滑与补点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KalmanState:
    """
    卡尔曼滤波内部状态。

    @param {np.ndarray} x - 状态向量 (4,1): [x, y, vx, vy]^T
    @param {np.ndarray} P - 状态协方差 (4,4)
    """

    x: np.ndarray
    P: np.ndarray


class ConstantVelocityKalman:
    """
    2D 常速度模型卡尔曼滤波器。

    状态： [x, y, vx, vy]
    观测： [x, y]
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_var: float = 50.0,
        meas_var: float = 25.0,
        init_var: float = 1e4,
    ) -> None:
        """
        @param {float} dt - 时间间隔（秒）；若按帧更新，可设为 1/fps
        @param {float} process_var - 过程噪声强度（越大越“跟得上”突变）
        @param {float} meas_var - 观测噪声方差（越大越不信观测）
        @param {float} init_var - 初始协方差对角值（越大越不确定）
        """

        self.dt = float(dt)
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)
        self.init_var = float(init_var)

        self._state: Optional[KalmanState] = None

    def reset(self) -> None:
        """
        重置滤波器。

        @returns {None}
        """

        self._state = None

    def _build_mats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构造 F/Q/H/R 矩阵。

        @returns {Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]} - (F,Q,H,R)
        """

        dt = self.dt
        # 状态转移
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # 过程噪声（简化：对位置/速度加噪）
        q = self.process_var
        Q = np.array(
            [
                [dt**4 / 4, 0, dt**3 / 2, 0],
                [0, dt**4 / 4, 0, dt**3 / 2],
                [dt**3 / 2, 0, dt**2, 0],
                [0, dt**3 / 2, 0, dt**2],
            ],
            dtype=np.float64,
        ) * q

        # 观测矩阵
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

        # 观测噪声
        R = np.eye(2, dtype=np.float64) * float(self.meas_var)

        return F, Q, H, R

    def init(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """
        初始化滤波器状态。

        @param {float} x - 初始 x
        @param {float} y - 初始 y
        @param {float} vx - 初始 vx
        @param {float} vy - 初始 vy
        @returns {None}
        """

        x_vec = np.array([[x], [y], [vx], [vy]], dtype=np.float64)
        P = np.eye(4, dtype=np.float64) * float(self.init_var)
        self._state = KalmanState(x=x_vec, P=P)

    def predict(self) -> Tuple[float, float]:
        """
        预测一步。

        @returns {Tuple[float,float]} - 预测位置 (x,y)
        """

        if self._state is None:
            raise RuntimeError("Kalman: 未初始化，无法 predict()")

        F, Q, _, _ = self._build_mats()
        x = F @ self._state.x
        P = F @ self._state.P @ F.T + Q
        self._state = KalmanState(x=x, P=P)
        return float(x[0, 0]), float(x[1, 0])

    def update(self, meas_x: float, meas_y: float) -> Tuple[float, float]:
        """
        用观测更新一步。

        @param {float} meas_x - 观测 x
        @param {float} meas_y - 观测 y
        @returns {Tuple[float,float]} - 更新后位置 (x,y)
        """

        if self._state is None:
            self.init(meas_x, meas_y, 0.0, 0.0)
            return float(meas_x), float(meas_y)

        _, _, H, R = self._build_mats()

        z = np.array([[meas_x], [meas_y]], dtype=np.float64)
        x = self._state.x
        P = self._state.P

        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x_new = x + K @ y
        I = np.eye(4, dtype=np.float64)
        P_new = (I - K @ H) @ P

        self._state = KalmanState(x=x_new, P=P_new)
        return float(x_new[0, 0]), float(x_new[1, 0])

    def step(self, measurement: Optional[Tuple[float, float]]) -> Tuple[float, float, bool]:
        """
        一步滤波：先 predict，再按是否有观测决定 update 或仅返回预测。

        @param {Optional[Tuple[float,float]]} measurement - (x,y) 或 None
        @returns {Tuple[float,float,bool]} - (x,y,used_measurement)
        """

        if self._state is None:
            if measurement is None:
                # 没有观测无法初始化：返回占位
                return 0.0, 0.0, False
            self.init(measurement[0], measurement[1], 0.0, 0.0)
            return float(measurement[0]), float(measurement[1]), True

        px, py = self.predict()
        if measurement is None:
            return float(px), float(py), False
        ux, uy = self.update(measurement[0], measurement[1])
        return float(ux), float(uy), True


