
import numpy as np
from typing import Sequence, Tuple

Array = np.ndarray

def _linspace(n: int) -> Array:
    if n <= 1: return np.array([0.0])
    return np.linspace(0.0, 1.0, n)

def ensure_T(x: Array, T: int) -> Array:
    C,t,V,M = x.shape
    if t == T: return x
    if t > T:
        s = np.random.randint(0, t - T + 1)
        return x[:, s:s+T, :, :]
    pad = T - t
    left = pad // 2; right = pad - left
    return np.pad(x, ((0,0),(left,right),(0,0),(0,0)), mode='edge')

class Compose:
    def __init__(self, ops): self.ops = [op for op in ops if op is not None]
    def __call__(self, x: Array) -> Array:
        for op in self.ops: x = op(x)
        return x

# ---- Time ----
class RandomTimeCropPad:
    def __init__(self, T: int = 300): self.T = T
    def __call__(self, x: Array) -> Array: return ensure_T(x, self.T)

# ---- Spatial ----
class RandomRotateXYZ:
    def __init__(self, yaw_deg=15.0, pitch_deg=8.0, roll_deg=8.0):
        self.yaw_deg, self.pitch_deg, self.roll_deg = yaw_deg, pitch_deg, roll_deg
    def __call__(self, x: Array) -> Array:
        C,t,V,M = x.shape
        if C < 3: return x
        yaw  = np.deg2rad(np.random.uniform(-self.yaw_deg, self.yaw_deg))
        pit  = np.deg2rad(np.random.uniform(-self.pitch_deg, self.pitch_deg))
        rol  = np.deg2rad(np.random.uniform(-self.roll_deg, self.roll_deg))
        cz, sz = np.cos(yaw),  np.sin(yaw)
        cy, sy = np.cos(pit),  np.sin(pit)
        cx, sx = np.cos(rol),  np.sin(rol)
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=x.dtype)
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=x.dtype)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=x.dtype)
        R = (Rz @ Ry @ Rx).astype(x.dtype)
        y = x.copy()
        for m in range(M):
            for tt in range(t):
                P = y[0:3, tt, :, m]
                y[0:3, tt, :, m] = R @ P
        return y

class RandomScale:
    def __init__(self, smin=0.95, smax=1.05): self.smin, self.smax = smin, smax
    def __call__(self, x: Array) -> Array:
        y = x.copy(); s = np.random.uniform(self.smin, self.smax); y[0:3] *= s; return y

class RandomTranslateXY:
    def __init__(self, max_shift=0.05): self.max_shift = max_shift
    def __call__(self, x: Array) -> Array:
        dx = np.random.uniform(-self.max_shift, self.max_shift)
        dy = np.random.uniform(-self.max_shift, self.max_shift)
        y = x.copy()
        if x.shape[0] >= 1: y[0] += dx
        if x.shape[0] >= 2: y[1] += dy
        return y

class RandomFlipLR:
    def __init__(self, swap_idx_pairs: Sequence[Tuple[int,int]], p: float = 0.5):
        self.swap_idx_pairs = list(swap_idx_pairs); self.p = p
    def __call__(self, x: Array) -> Array:
        if np.random.rand() > self.p: return x
        y = x.copy()
        if x.shape[0] >= 1: y[0] *= -1.0  # mirror X
        for a,b in self.swap_idx_pairs:
            y[:, :, [a,b], :] = y[:, :, [b,a], :]
        return y

class RandomFlipAxis:
    """Mirror a single coordinate axis: axis=0 (X), 1 (Y), 2 (Z)."""
    def __init__(self, axis: int = 1, p: float = 1.0):
        assert axis in (0,1,2)
        self.axis, self.p = axis, p
    def __call__(self, x: Array) -> Array:
        if np.random.rand() > self.p: return x
        if self.axis >= x.shape[0]: return x
        y = x.copy(); y[self.axis] *= -1.0; return y

# ---- Bone/Motion ----
def compute_bone_from_joint(joint: Array, bone_pairs: Sequence[Tuple[int,int]]) -> Array:
    C,T,V,M = joint.shape
    bone = np.zeros_like(joint)
    for (v1,v2) in bone_pairs:
        bone[:, :, v1, :] = joint[:, :, v1, :] - joint[:, :, v2, :]
    return bone

def compute_motion(x: Array) -> Array:
    mot = np.zeros_like(x); mot[:,1:,:,:] = x[:,1:,:,:] - x[:,:-1,:,:]; return mot
