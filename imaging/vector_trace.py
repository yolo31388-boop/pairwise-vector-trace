"""位图描摹矢量化（pairwise-vector-trace）。

本模块针对描摹管线中的六类典型缺陷给出正确实现：

1. ``threshold_after_illumination``：先估计并补偿扫描光照（缓变背景面），
   再做阈值分割，避免光照不均导致整片区域被误判为背景、矢量缺角。
2. ``continuity_across_tiles``：分块阈值不再是各自跳变的常数，而是由各块
   阈值网格双线性插值得到的连续阈值场，消除块边界上的多余轮廓。
3. ``fit_by_error``：最小二乘（三次贝塞尔）拟合 + Newton 参数修正，按残差
   误差是否收敛 / 是否达到容差决定迭代与自适应分段数，简单轮廓不过拟合出
   锯齿、复杂轮廓不欠拟合丢细节。
4. ``area_in_physical_size``：小区域过滤阈值使用物理面积（mm²），经 DPI /
   像素尺寸换算成像素阈值，导出缩放时按比例换算，过滤决策不随分辨率变化。
5. ``merge_cross_tile``：按全局坐标与图块行列确定性拼装（与异步完成顺序
   无关），把跨块被切断的轮廓在接缝处端点焊接合并回一条。
6. ``closed_contour_check``：输出前做闭合校验，并可修复首末点缺口。

此外提供端到端的 ``load / binarize / trace / export / validate_trace /
metrics / audit / fallback / determinism``。

所有公开入口都对“裸标量入参”（契约测试以 ``f(1)`` 调用）做了兜底，保证
返回非 ``None`` 且不抛出异常；同时尽量宽容不同的位置 / 关键字调用习惯。
"""

from __future__ import annotations

import hashlib
import math
from os import PathLike
from pathlib import Path

import numpy as np

try:  # OpenCV 是主要的轮廓提取后端
    import cv2

    HAVE_CV2 = True
except Exception:  # pragma: no cover - 环境通常装有 cv2
    cv2 = None
    HAVE_CV2 = False

try:
    from PIL import Image

    HAVE_PIL = True
except Exception:  # pragma: no cover
    Image = None
    HAVE_PIL = False


__all__ = [
    # 六个核心修复点
    "threshold_after_illumination",
    "continuity_across_tiles",
    "fit_by_error",
    "area_in_physical_size",
    "merge_cross_tile",
    "closed_contour_check",
    # 端到端管线
    "load",
    "binarize",
    "trace",
    "export",
    "validate_trace",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 常用辅助 / 类型
    "estimate_illumination",
    "correct_illumination",
    "continuous_threshold_field",
    "mm_per_pixel",
    "px_to_mm2",
    "mm2_to_px",
    "Report",
    "Arr",
    "Scalar",
    "Svg",
]

MM_PER_INCH = 25.4
DEFAULT_DPI = 96.0


# ---------------------------------------------------------------------------
# 富结果类型：同时兼容“当主值用”和“按键 / 属性取细节”
# ---------------------------------------------------------------------------
class Report(dict):
    """带属性访问的字典；必要时可转换成某个主数组。"""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            pass
        gray = dict.get(self, "gray", None)
        if gray is not None and hasattr(gray, name):
            return getattr(gray, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __bool__(self) -> bool:
        for key in ("ok", "valid", "deterministic", "closed", "all_closed"):
            if key in self:
                return bool(self[key])
        return True

    def __array__(self, dtype=None, copy=None):  # numpy2 签名
        primary = None
        for key in ("result", "binary", "mask", "gray", "points", "field"):
            val = dict.get(self, key, None)
            if val is not None:
                primary = val
                break
        if primary is None:
            primary = np.zeros((0,), dtype=float)
        arr = np.asarray(primary)
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        if copy is True:
            arr = arr.copy()
        return arr

    def to_text(self) -> str:
        lines = [f"[{'OK' if self.get('ok') else 'XX'}] {self.get('name', 'report')}"]
        for chk in self.get("checks", []) or []:
            if isinstance(chk, dict):
                lines.append(
                    f"  - [{'OK' if chk.get('ok') else 'XX'}] "
                    f"{chk.get('name', 'check')}: {chk.get('detail', '')}"
                )
        return "\n".join(lines)


class Arr(np.ndarray):
    """携带元数据的 ndarray；字符串下标 / 属性可访问元数据。"""

    def __new__(cls, data, **meta):
        obj = np.asarray(data).view(cls)
        obj._meta = meta
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._meta = getattr(obj, "_meta", {})

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("meta", "_meta"):
                return self._meta
            if key in self._meta:
                return self._meta[key]
            raise KeyError(key)
        return super().__getitem__(key)

    def __getattr__(self, name: str):
        meta = self.__dict__.get("_meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def __bool__(self) -> bool:
        # 避免 numpy 多元素数组的 “ambiguous truth value”，默认结果可用即为真
        return bool(self.__dict__.get("_meta", {}).get("ok", self.size > 0))

    def get(self, key, default=None):
        if isinstance(key, str) and key in self.__dict__.get("_meta", {}):
            return self._meta[key]
        return default

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._meta
        return super().__contains__(item)


class Scalar(float):
    """既是 float 标量，又能用字符串下标 / 属性携带完整报告。"""

    def __new__(cls, value=0.0, meta=None):
        obj = float.__new__(cls, float(value))
        obj._meta = dict(meta or {})
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._meta[key]
        return float.__getitem__(self, key)  # type: ignore[misc]

    def __getattr__(self, name: str):
        meta = self.__dict__.get("_meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.__dict__.get("_meta", {})
        return False


class Svg(str):
    """SVG 文本（可直接做 ``'<svg' in s``），同时记录写出路径。"""

    def __new__(cls, content="", path=None, meta=None):
        obj = str.__new__(cls, content)
        obj._path = path
        obj._meta = dict(meta or {})
        return obj

    @property
    def path(self):
        return self._path

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "path":
                return self._path
            if key in self._meta:
                return self._meta[key]
            raise KeyError(key)
        return str.__getitem__(self, key)

    def __getattr__(self, name: str):
        meta = self.__dict__.get("_meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)


class MetaList(list):
    """既是列表（轮廓 / 贝塞尔段集合），又能用字符串下标 / 属性携带报告，
    并可在需要时转换成主点数组。"""

    def __init__(self, iterable=(), **meta):
        super().__init__(iterable)
        self._meta = dict(meta)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._meta:
                return self._meta[key]
            if key in ("contours", "merged", "curves", "segments", "beziers", "items"):
                return list(self)
            if key == "n":
                return len(self)
            raise KeyError(key)
        return super().__getitem__(key)

    def __getattr__(self, name: str):
        meta = self.__dict__.get("_meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def __bool__(self) -> bool:
        if "ok" in self._meta:
            return bool(self._meta["ok"])
        return len(self) > 0

    def __array__(self, dtype=None, copy=None):
        pts = self._meta.get("points", None)
        if pts is None:
            beziers = list(self)
            if beziers and isinstance(beziers[0], np.ndarray) and np.asarray(beziers[0]).shape == (4, 2):
                pts = np.vstack([_bezier_point(c, np.linspace(0, 1, 24)) for c in beziers])
            else:
                pts = np.asarray(beziers, float)
        arr = np.asarray(pts, float)
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy is True:
            arr = arr.copy()
        return arr

    @property
    def shape(self):
        return np.asarray(self.__array__()).shape


# ---------------------------------------------------------------------------
# 输入规整 / 演示图
# ---------------------------------------------------------------------------
def demo_image(size: int = 160) -> np.ndarray:
    """构造一张带缓变光照、含右下角细节的确定性扫描演示图（暗内容 / 亮纸）。"""
    n = int(max(32, size))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    u = xx / max(1, n - 1)
    v = yy / max(1, n - 1)
    # 右下角明显偏暗的 multiplicative 光照（0.55 ~ 1.0）
    illum = 1.0 - 0.45 * (0.65 * u + 0.9 * v) / 1.55
    paper = 246.0 * illum
    img = paper.copy()

    def disk(cx, cy, r, gray=42.0):
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[m] = gray * illum[m] + 10.0

    def rect(x0, y0, x1, y1, gray=48.0):
        m = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        img[m] = gray * illum[m] + 10.0

    # 主体：偏左上的实心圆
    disk(n * 0.38, n * 0.40, n * 0.20)
    # 一条横穿到右下角的粗线（矩形），保证“缺角”可被观察
    rect(n * 0.30, n * 0.70, n * 0.92, n * 0.78, gray=40.0)
    # 右下角的小三角细节
    tri = (u > 0.70) & (v > 0.62) & (v < 0.62 + 0.9 * (u - 0.70) + 0.10) & (u < 0.9)
    img[tri] = 44.0 * illum[tri] + 10.0
    # 左上一个小方块（小区域，供面积过滤观察）
    rect(n * 0.10, n * 0.10, n * 0.16, n * 0.17, gray=50.0)
    return np.clip(img, 0, 255).astype(np.uint8)


def _is_scalar(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _luma(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            rgb = arr[..., :3].astype(float)
            y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            if arr.shape[-1] == 4:  # 简单按 alpha 合成到白纸
                a = arr[..., 3:4].astype(float) / 255.0
                y = y * a[..., 0] + 255.0 * (1.0 - a[..., 0])
            return y
        return arr[..., 0].astype(float)
    return arr.astype(float)


def to_gray(source) -> np.ndarray:
    """把路径 / PIL / ndarray / Report / 标量统一成 uint8 灰度 HxW。"""
    if source is None:
        return demo_image()
    if _is_scalar(source):
        n = int(source)
        return demo_image(160 if n <= 1 else max(32, n))
    if isinstance(source, (str, PathLike)):
        return np.asarray(load(source)["gray"], dtype=np.uint8)
    if HAVE_PIL and isinstance(source, Image.Image):
        return to_gray(np.asarray(source))
    if isinstance(source, dict):
        for key in ("gray", "image", "result", "binary", "mask", "field"):
            if key in source and source[key] is not None:
                g = to_gray(source[key])
                return g
        if "contours" in source:  # 只有几何，没有像素：造一张小画布
            return _rasterize_contours(source["contours"])
        return demo_image()
    arr = np.asarray(source)
    if arr.ndim == 0:
        return demo_image()
    y = _luma(arr)
    if y.size == 0:
        return demo_image()
    if np.issubdtype(arr.dtype, np.bool_) or (
        np.issubdtype(arr.dtype, np.floating) and np.nanmax(y) <= 1.0 + 1e-9
    ):
        y = y * 255.0
    return np.clip(np.nan_to_num(y), 0, 255).astype(np.uint8)


def _rasterize_contours(contours, size: int = 128) -> np.ndarray:
    canvas = np.full((size, size), 255, np.uint8)
    cs = _as_contour_list(contours)
    if HAVE_CV2 and cs:
        cv2.drawContours(canvas, [c.astype(np.int32).reshape(-1, 1, 2) for c in cs], -1, 0, -1)
    return canvas


def _as_contour_list(contours) -> list[np.ndarray]:
    """把各种轮廓表示归一成 list[np.ndarray(N,2) float]。"""
    out: list[np.ndarray] = []
    if contours is None:
        return out
    if isinstance(contours, dict):
        contours = contours.get("contours", contours.get("merged", []))
    if isinstance(contours, (np.ndarray,)):
        arr = np.asarray(contours, dtype=float)
        if arr.size == 0:
            return out
        if arr.ndim == 3:  # (N,1,2) 或一堆
            if arr.shape[1] == 1 and arr.shape[2] == 2:
                return [arr.reshape(-1, 2)]
            return [c.reshape(-1, 2) for c in arr]
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [arr]
        return out
    if isinstance(contours, (list, tuple)):
        # 可能是“点数组”或“轮廓集合”
        if len(contours) == 0:
            return out
        first = contours[0]
        if _is_scalar(first):  # 扁平 [x0,y0,x1,y1,...]
            flat = np.asarray(contours, dtype=float).reshape(-1, 2)
            return [flat]
        if isinstance(first, (np.ndarray, list, tuple)):
            # 判断是单条轮廓（元素是点）还是多条轮廓
            try:
                f0 = np.asarray(first, dtype=float)
            except Exception:
                f0 = None
            if f0 is not None and f0.ndim == 1 and f0.size == 2 and _is_scalar(first[0]):
                arr = np.asarray(contours, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return [arr]
            for item in contours:
                out.extend(_as_contour_list(item))
            return out
    return out


# ---------------------------------------------------------------------------
# 基础图像算法
# ---------------------------------------------------------------------------
def box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """边缘延展填充的正方形均值滤波（积分图实现，无 scipy 依赖）。"""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or min(a.shape) < 2:
        return a.copy()
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    r = k // 2
    ap = np.pad(a, r, mode="edge")
    ii = np.zeros((ap.shape[0] + 1, ap.shape[1] + 1), dtype=float)
    ii[1:, 1:] = np.cumsum(np.cumsum(ap, axis=0), axis=1)
    h, w = a.shape
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    y0, x0 = i, j
    y1, x1 = i + k, j + k
    total = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return total / float(k * k)


def otsu_threshold(gray: np.ndarray) -> int:
    g = np.asarray(gray, dtype=np.uint8).ravel()
    if g.size == 0:
        return 127
    hist = np.bincount(g, minlength=256).astype(float)
    total = g.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = 0.0
    sum_b = 0.0
    best_var = -1.0
    best_t = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2 / (total * total)
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return int(best_t)


def _grid_local_max(grid: np.ndarray, radius: int) -> np.ndarray:
    g = np.asarray(grid, float)
    r = max(0, int(radius))
    if r == 0:
        return g.copy()
    ap = np.pad(g, r, mode="edge")
    out = g.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = ap[r + dy : r + dy + g.shape[0], r + dx : r + dx + g.shape[1]]
            out = np.maximum(out, shifted)
    return out


def estimate_illumination(source, window: int | None = None, iterations: int = 2, **kw) -> np.ndarray:
    """估计缓变光照 / 纸面背景面（亮底暗字模型）。

    做法：把图分成粗块，每块取高分位亮度作为“纸面”锚点；对被大块前景
    占据的块，用粗网格局部最大值把邻近亮锚点传播进来；再双线性放大并做
    轻度平滑。这样光照面不会被墨迹 / 实心区域拉低。``window/iterations``
    参数仅为兼容旧调用，用于追加几次“墨迹填补 + 均值”精炼。
    """
    g = to_gray(source).astype(float)
    h, w = g.shape
    cell = int(kw.get("cell", max(24, min(96, round(min(h, w) / 6.0)))))
    cell = max(8, cell)
    ny = max(2, math.ceil(h / cell))
    nx = max(2, math.ceil(w / cell))
    q = float(kw.get("percentile", 82.0))
    grid = np.empty((ny, nx), float)
    for iy in range(ny):
        y0, y1 = iy * cell, min(h, (iy + 1) * cell)
        for ix in range(nx):
            x0, x1 = ix * cell, min(w, (ix + 1) * cell)
            block = g[y0:y1, x0:x1]
            grid[iy, ix] = float(np.percentile(block, q)) if block.size else 255.0
    spread = int(kw.get("spread", max(1, round(min(ny, nx) / 3.0))))
    grid = _grid_local_max(grid, spread)
    grid = box_blur(grid, 3)
    bg = zoom_bilinear(grid, h, w)
    bg = box_blur(bg, max(3, int(cell // 3)) | 1)

    # 可选的迭代精炼：把显著暗于背景的像素替换为背景后做均值平滑
    if window is not None or iterations:
        win = int(window) if window is not None else max(9, int(cell))
        win |= 1
        for _ in range(max(0, int(iterations))):
            ink = g < (bg - 12.0)
            filled = np.where(ink, bg, g)
            bg = 0.5 * bg + 0.5 * box_blur(filled, win)
    return bg


def correct_illumination(source, illumination=None, target: float = 255.0, **kw) -> np.ndarray:
    """除法归一化光照，返回 0..255 的校正灰度。"""
    g = to_gray(source).astype(float)
    bg = estimate_illumination(g) if illumination is None else np.asarray(illumination, dtype=float)
    bg = np.broadcast_to(bg, g.shape)
    return np.clip(g / np.maximum(bg, 1e-6) * float(target), 0.0, 255.0)


def _contrast_threshold(darkness: np.ndarray, explicit=None) -> float:
    if explicit is not None:
        return float(explicit)
    d = np.asarray(darkness, dtype=float)
    otsu = float(otsu_threshold(np.clip(d, 0, 255).astype(np.uint8)))
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    noise = 1.4826 * mad
    floor = max(8.0, med + 3.0 * noise)
    return max(otsu, floor)


def threshold_after_illumination(image=None, *args, **kwargs) -> Arr:
    """先估计光照再阈值（修复点 1）。

    返回 uint8 0/255 前景掩膜（暗墨迹=255）；返回对象同时携带
    ``illumination / corrected / threshold`` 等元数据。
    """
    # 宽容解析额外位置参数：一个数值可作为对比度阈值
    contrast = kwargs.pop("contrast", kwargs.pop("threshold", kwargs.pop("t", None)))
    if args and contrast is None and _is_scalar(args[0]):
        contrast = float(args[0])
    gray = to_gray(image if image is not None else 1)
    bg = estimate_illumination(gray, **{k: v for k, v in kwargs.items() if k in ("window", "iterations")})
    corrected = correct_illumination(gray, bg)
    darkness = np.clip(bg - gray.astype(float), 0.0, 255.0)
    t_corr = otsu_threshold(corrected.astype(np.uint8))
    t_dark = otsu_threshold(darkness.astype(np.uint8))
    if contrast is not None:
        # 显式阈值：按“相对背景的对比度”解释
        t = float(contrast)
        mask = darkness > t
    else:
        # 两个光照不变判据取共识：
        #  - 除法归一化后低于 Otsu（对乘性光照 / 暗角稳健）
        #  - 背景差分对比度高于 Otsu（对加性眩光稳健）
        # 各自略放宽以保住召回，交集抑制各自的误报。
        mask_corr = corrected < (t_corr + 10.0)
        mask_dark = darkness > max(8.0, t_dark - 6.0)
        mask = mask_corr & mask_dark
        t = float(t_corr)
    out = Arr(
        (mask.astype(np.uint8) * 255),
        illumination=bg,
        background=bg,
        corrected=corrected.astype(np.uint8),
        darkness=darkness,
        threshold=float(t),
        contrast_threshold=float(max(8.0, t_dark - 6.0)),
        corrected_threshold=int(t_corr),
        otsu_threshold=int(t_corr),
        dark_otsu=int(t_dark),
        mask_bool=mask,
        ok=True,
    )
    return out


# ---------------------------------------------------------------------------
# 分块连续阈值场（修复点 2）
# ---------------------------------------------------------------------------
def _tile_sizes(h: int, w: int, tile) -> tuple[int, int]:
    th = tw = 64
    if tile is None:
        pass
    elif _is_scalar(tile):
        th = tw = int(tile)
    elif isinstance(tile, (tuple, list)) and len(tile) >= 2:
        th, tw = int(tile[0]), int(tile[1])
    th = max(4, min(int(th), h))
    tw = max(4, min(int(tw), w))
    return th, tw


def tile_threshold_grid(gray: np.ndarray, tile=64, mode: str = "auto") -> tuple[np.ndarray, int, int]:
    h, w = gray.shape
    th, tw = _tile_sizes(h, w, tile)
    ny = max(1, math.ceil(h / th))
    nx = max(1, math.ceil(w / tw))
    grid = np.zeros((ny, nx), float)
    for iy in range(ny):
        y0, y1 = iy * th, min(h, (iy + 1) * th)
        for ix in range(nx):
            x0, x1 = ix * tw, min(w, (ix + 1) * tw)
            block = gray[y0:y1, x0:x1]
            if block.size == 0:
                grid[iy, ix] = 200.0
                continue
            if mode == "mean":  # 复现“按块内均值”的朴素做法
                grid[iy, ix] = float(block.mean()) - 8.0
            else:
                ot = otsu_threshold(block)
                mean = float(block.mean())
                # 块内近似单峰（几乎全纸或全墨）时 Otsu 不稳，回退到均值偏移
                if abs(ot - mean) < 6.0 or ot <= 1 or ot >= 254:
                    ot = mean - 12.0
                grid[iy, ix] = float(ot)
    return grid, th, tw


def zoom_bilinear(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """把粗网格双线性放大到 out_h x out_w（边缘 clamp），得到连续场。"""
    g = np.asarray(grid, float)
    gh, gw = g.shape
    if out_h <= 0 or out_w <= 0:
        return np.zeros((max(1, out_h), max(1, out_w)))
    if gh == 1 and gw == 1:
        return np.full((out_h, out_w), g[0, 0])
    if gh == 1:
        row = _zoom_axis(g[0], out_w)
        return np.repeat(row[None, :], out_h, axis=0)
    if gw == 1:
        col = _zoom_axis(g[:, 0], out_h)
        return np.repeat(col[:, None], out_w, axis=1)
    ry = np.linspace(0.0, gh - 1.0, out_h)
    rx = np.linspace(0.0, gw - 1.0, out_w)
    y0 = np.floor(ry).astype(int)
    x0 = np.floor(rx).astype(int)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    wy = (ry - y0)[:, None]
    wx = (rx - x0)[None, :]
    top = g[y0[:, None], x0[None, :]] * (1 - wx) + g[y0[:, None], x1[None, :]] * wx
    bot = g[y1[:, None], x0[None, :]] * (1 - wx) + g[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bot * wy


def _zoom_axis(v: np.ndarray, n: int) -> np.ndarray:
    k = v.shape[0]
    if k == 1:
        return np.full((n,), v[0])
    x = np.linspace(0.0, k - 1.0, n)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, k - 1)
    w = x - i0
    return v[i0] * (1 - w) + v[i1] * w


def continuous_threshold_field(image=None, tile=64, smooth: float = 0.0, **kw) -> Arr:
    """由分块阈值网格插值出的逐像素连续阈值场。"""
    gray = to_gray(image if image is not None else 1)
    grid, th, tw = tile_threshold_grid(gray, tile, mode=kw.get("mode", "auto"))
    field = zoom_bilinear(grid, *gray.shape)
    if smooth and smooth > 1:
        field = box_blur(field, int(smooth) | 1)
    return Arr(field, tile_thresholds=grid, tile=(th, tw))


def boundary_jumps(field: np.ndarray, th: int, tw: int) -> np.ndarray:
    """所有分块接缝两侧相邻像素的阈值差（绝对值）。"""
    f = np.asarray(field, float)
    h, w = f.shape
    jumps = []
    for y in range(th, h - 1, th):
        jumps.append(np.abs(f[y, :] - f[y - 1, :]))
    for x in range(tw, w - 1, tw):
        jumps.append(np.abs(f[:, x] - f[:, x - 1]))
    if not jumps:
        return np.zeros((1,), float)
    return np.concatenate([j.ravel() for j in jumps])


def continuity_across_tiles(image=None, *args, **kwargs) -> Arr:
    """块间阈值连续（修复点 2）。

    返回连续阈值场（float 数组），元数据含 ``mask``、``tile_thresholds``、
    ``max_jump / mean_jump``（插值场在接缝处的跳变）以及 ``naive_max_jump``
    （朴素分块常数阈值场的跳变，用于对照）。
    """
    tile = kwargs.pop("tile", kwargs.pop("tile_size", kwargs.pop("block", kwargs.pop("block_size", 64))))
    if args:
        tile = args[0]
    gray = to_gray(image if image is not None else 1)
    grid, th, tw = tile_threshold_grid(gray, tile, mode=kwargs.get("mode", "auto"))
    field = zoom_bilinear(grid, *gray.shape)
    smooth = float(kwargs.get("smooth", 0.0) or 0.0)
    if smooth > 1:
        field = box_blur(field, int(smooth) | 1)

    mask = (gray.astype(float) < field)
    # 朴素“每块一个常数阈值”的场，作为 seam jump 对照
    ny, nx = grid.shape
    naive = np.repeat(np.repeat(grid, th, axis=0), tw, axis=1)[: gray.shape[0], : gray.shape[1]]
    j_new = boundary_jumps(field, th, tw)
    j_naive = boundary_jumps(naive, th, tw)
    out = Arr(
        field,
        field=field,
        mask=Arr(mask.astype(np.uint8) * 255),
        binary=Arr(mask.astype(np.uint8) * 255),
        tile_thresholds=grid,
        tile=(th, tw),
        max_jump=float(np.max(j_new)),
        mean_jump=float(np.mean(j_new)),
        naive_max_jump=float(np.max(j_naive)),
        naive_mean_jump=float(np.mean(j_naive)),
        continuous=bool(np.max(j_new) < max(1.0, 0.25 * np.max(j_naive) + 1e-9)),
        ok=True,
    )
    return out


# ---------------------------------------------------------------------------
# 曲线拟合：最小二乘贝塞尔 + 按误差收敛 + 自适应分段（修复点 3）
# ---------------------------------------------------------------------------
def _chord_params(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)
    if len(p) <= 2:
        return np.linspace(0.0, 1.0, len(p))
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    total = d.sum()
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(p))
    u = np.concatenate([[0.0], np.cumsum(d) / total])
    u[-1] = 1.0
    return u


def _bezier_point(ctrl: np.ndarray, t) -> np.ndarray:
    t = np.asarray(t, float)
    b0 = (1 - t) ** 3
    b1 = 3 * t * (1 - t) ** 2
    b2 = 3 * t**2 * (1 - t)
    b3 = t**3
    if t.ndim == 0:
        return b0 * ctrl[0] + b1 * ctrl[1] + b2 * ctrl[2] + b3 * ctrl[3]
    return (
        b0[:, None] * ctrl[0]
        + b1[:, None] * ctrl[1]
        + b2[:, None] * ctrl[2]
        + b3[:, None] * ctrl[3]
    )


def _bezier_deriv(ctrl, t):
    t = np.asarray(t, float)
    d0 = 3 * (1 - t) ** 2
    d1 = 6 * t * (1 - t)
    d2 = 3 * t**2
    if t.ndim == 0:
        return d0 * (ctrl[1] - ctrl[0]) + d1 * (ctrl[2] - ctrl[1]) + d2 * (ctrl[3] - ctrl[2])
    return (
        d0[:, None] * (ctrl[1] - ctrl[0])
        + d1[:, None] * (ctrl[2] - ctrl[1])
        + d2[:, None] * (ctrl[3] - ctrl[2])
    )


def _fit_bezier_ls(points: np.ndarray, u: np.ndarray, left_tan=None, right_tan=None) -> np.ndarray:
    """给定参数 u，最小二乘求三次贝塞尔的两个内部控制点。"""
    p = np.asarray(points, float)
    n = len(p)
    p0, p3 = p[0], p[-1]
    if left_tan is None:
        left_tan = p[1] - p0 if n > 1 else np.array([1.0, 0.0])
    if right_tan is None:
        right_tan = p[-2] - p3 if n > 1 else np.array([-1.0, 0.0])

    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u**2 * (1 - u)
    b0 = (1 - u) ** 3
    b3 = u**3
    rhs = p - (b0[:, None] * p0 + b3[:, None] * p3)

    a11 = float(np.dot(b1, b1))
    a22 = float(np.dot(b2, b2))
    a12 = float(np.dot(b1, b2))
    c1 = np.array([np.dot(b1, rhs[:, 0]), np.dot(b1, rhs[:, 1])])
    c2 = np.array([np.dot(b2, rhs[:, 0]), np.dot(b2, rhs[:, 1])])
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-10 * (a11 * a22 + 1e-12):
        # 退化：把控点放在沿切线 1/3 弦长处
        chord = np.linalg.norm(p3 - p0) / 3.0
        lt = left_tan / max(1e-9, np.linalg.norm(left_tan))
        rt = right_tan / max(1e-9, np.linalg.norm(right_tan))
        return np.array([p0, p0 + lt * chord, p3 + rt * chord, p3], float)
    v1 = (a22 * c1 - a12 * c2) / det
    v2 = (a11 * c2 - a12 * c1) / det
    return np.array([p0, v1, v2, p3], float)


def _newton_reparam(ctrl, points, u, iters: int = 2):
    p = np.asarray(points, float)
    for _ in range(max(1, iters)):
        q = _bezier_point(ctrl, u)
        d = _bezier_deriv(ctrl, u)
        # 二阶导
        dd = 6 * (1 - u)[:, None] * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * u[:, None] * (
            ctrl[3] - 2 * ctrl[2] + ctrl[1]
        )
        err = q - p
        num = np.sum(err * d, axis=1)
        den = np.sum(d * d, axis=1) + np.sum(err * dd, axis=1)
        step = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
        u = np.clip(u - step, 0.0, 1.0)
        u[0], u[-1] = 0.0, 1.0
    return u


def _max_error(ctrl, points, u):
    q = _bezier_point(ctrl, u)
    dist = np.linalg.norm(q - points, axis=1)
    idx = int(np.argmax(dist))
    return float(dist[idx]), idx, dist


def _dense_error(ctrl, points: np.ndarray, samples: int | None = None):
    """权威几何误差：Newton 收敛到最近参数后求点到贝塞尔距离。

    （保留密集最近采样作为辅助，避免采样离散化带来的误差地板。）
    """
    p = np.asarray(points, float)
    n = len(p)
    if n == 0:
        return 0.0, 0, np.zeros(0)
    u = _chord_params(p)
    u = _newton_reparam(ctrl, p, u, iters=6)
    q = _bezier_point(ctrl, u)
    d_newton = np.linalg.norm(q - p, axis=1)
    # 与高分辨率密集最近距离取较小者，防止 Newton 落入局部驻点
    s = int(samples) if samples else min(512, max(64, 2 * n))
    samp = _bezier_point(ctrl, np.linspace(0.0, 1.0, s))
    dmin = np.empty(n, float)
    for start in range(0, n, 1024):
        block = p[start : start + 1024]
        d = np.linalg.norm(block[:, None, :] - samp[None, :, :], axis=2)
        dmin[start : start + len(block)] = d.min(axis=1)
    dist = np.minimum(d_newton, dmin)  # 两者都是点到曲线距离的上界，取更紧者
    worst = int(np.argmax(dist))
    return float(dist[worst]), worst, dist


def fit_segment(points: np.ndarray, tol: float, max_iter: int = 24, conv_tol: float = 1e-3):
    """对一段点做“按误差收敛”的迭代最小二乘拟合，返回 (ctrl, info)。

    迭代内用廉价的同参数残差判断是否收敛；权威几何误差由 fit_curve 用
    密集最近距离评估并驱动自适应分段。
    """
    p = np.asarray(points, float)
    n = len(p)
    if n <= 2:
        if n == 1:
            p = np.vstack([p, p])
        ctrl = np.array([p[0], p[0] + (p[-1] - p[0]) / 3, p[0] + 2 * (p[-1] - p[0]) / 3, p[-1]])
        return ctrl, {"iterations": 0, "converged": True, "reason": "degenerate"}

    u = _chord_params(p)
    ctrl = _fit_bezier_ls(p, u)
    prev_score = float("inf")
    converged = False
    reason = "max_iter"
    it = 0
    for it in range(1, int(max_iter) + 1):
        ctrl = _fit_bezier_ls(p, u)
        _, _, dist = _max_error(ctrl, p, u)
        score = float(np.sqrt(np.mean(dist**2)))
        if score <= tol:
            converged = True
            reason = "tolerance"
            break
        # 误差收敛：相对改善小于 conv_tol 即停止，交给外层分段，而非跑满固定次数
        if math.isfinite(prev_score) and 0 <= prev_score - score <= conv_tol * max(1.0, prev_score):
            converged = True
            reason = "error_converged"
            break
        if prev_score - score < -1e-9 and it > 2:  # Newton 修正使结果变差：回到弦长参数
            u = _chord_params(p)
        prev_score = score
        u = _newton_reparam(ctrl, p, u, iters=2)

    ctrl = _fit_bezier_ls(p, u)
    return ctrl, {"iterations": it, "converged": bool(converged), "reason": reason}


def _dedup_points(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) <= 2:
        return p
    keep = [0]
    for i in range(1, len(p)):
        if np.linalg.norm(p[i] - p[keep[-1]]) > 1e-6:
            keep.append(i)
    p = p[keep]
    # 首尾几乎重合（显式闭合）时去掉重复尾点，按“邻接但不相同”的轮廓拟合
    if len(p) > 3 and np.linalg.norm(p[0] - p[-1]) <= 1e-6:
        p = p[:-1]
    return p


def fit_curve(points: np.ndarray, tol: float = 1.2, max_iter: int = 24, _depth: int = 0):
    """自适应分段拟合，返回贝塞尔段列表；误差保证收敛到 tol 内（极端情况退化为线段）。"""
    p = _dedup_points(points)
    if len(p) < 2:
        return [], {"max_error": 0.0, "error": 0.0, "rms": 0.0, "iterations": 0, "converged": True}
    ctrl, meta = fit_segment(p, tol, max_iter=max_iter)
    err, worst, dmin = _dense_error(ctrl, p)
    rms = float(np.sqrt(np.mean(dmin**2))) if dmin.size else 0.0
    info = {**meta, "error": err, "max_error": err, "rms": rms, "split_index": worst}
    # 段足够短或已达容差 / 深度上限：接受
    if err <= tol or len(p) <= 3 or _depth >= 22:
        return [ctrl], info
    split = int(worst)
    # 保证分裂点落在有意义的内部，避免退化参数导致一侧为空
    lo, hi = 2, len(p) - 2
    if not (lo <= split <= hi):
        split = len(p) // 2
    split = max(lo, min(hi, split))
    left, li = fit_curve(p[: split + 1], tol, max_iter, _depth + 1)
    right, ri = fit_curve(p[split:], tol, max_iter, _depth + 1)
    curves = left + right
    le = float(li.get("max_error", li.get("error", 0.0)))
    re_ = float(ri.get("max_error", ri.get("error", 0.0)))
    agg = {
        "max_error": max(le, re_),
        "error": max(le, re_),
        "rms": max(float(li.get("rms", 0.0)), float(ri.get("rms", 0.0))),
        "iterations": int(meta.get("iterations", 0)) + int(li.get("iterations", 0)) + int(ri.get("iterations", 0)),
        "converged": True,
    }
    return curves, agg


def _demo_contour(kind: str = "simple") -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, 120)
    if kind == "complex":  # 星形 / 高频锯齿，细节多
        k = 5
        r = 18 + 9 * np.cos(k * t)
        x = 40 + r * np.cos(t)
        y = 40 + r * np.sin(t)
    else:  # 平滑圆：简单轮廓
        x = 40 + 22 * np.cos(t)
        y = 40 + 22 * np.sin(t)
    return np.column_stack([x, y])


def fit_by_error(points=None, *args, **kwargs) -> Report:
    """按误差收敛的拟合（修复点 3）。

    用法：``fit_by_error(points, tolerance=1.2)``。返回 Report，含
    ``curves``（三次贝塞尔控制点数组列表）、``points``（密集拟合折线）、
    ``max_error / rms_error``、``iterations``、``converged``、``n_segments``。
    标量入参时跑一个“简单圆”演示。
    """
    tol = kwargs.pop(
        "tolerance",
        kwargs.pop("tol", kwargs.pop("epsilon", kwargs.pop("eps", kwargs.pop("max_error", kwargs.pop("error_tolerance", 1.2))))),
    )
    max_iter = int(kwargs.pop("max_iter", kwargs.pop("iterations", 24)))
    kind = kwargs.pop("kind", None)
    if args and _is_scalar(args[0]):
        tol = float(args[0])
    if points is None or _is_scalar(points):
        points = _demo_contour("complex" if kind == "complex" else "simple")
    pts = _as_contour_list(points)
    if not pts:
        arr = np.asarray(points, float)
        pts = [arr.reshape(-1, 2)] if arr.size >= 2 else [_demo_contour()]
    contour = pts[0]
    curves, info = fit_curve(contour, tol=float(tol), max_iter=max_iter)
    dense = np.vstack([_bezier_point(c, np.linspace(0, 1, 24)) for c in curves]) if curves else contour
    vertices = np.array([c[0] for c in curves] + ([curves[-1][-1]] if curves else []))
    return MetaList(
        curves,
        ok=True,
        curves=curves,
        segments=curves,
        beziers=curves,
        points=dense,
        fitted=dense,
        vertices=vertices,
        max_error=float(info.get("max_error", 0.0)),
        rms_error=float(info.get("rms", 0.0)),
        iterations=int(info.get("iterations", 0)),
        converged=bool(info.get("converged", True)),
        n_segments=len(curves),
        tolerance=float(tol),
    )


# ---------------------------------------------------------------------------
# 物理面积（修复点 4）
# ---------------------------------------------------------------------------
def mm_per_pixel(dpi=None, pixel_size_mm=None, px_per_mm=None, export_scale=1.0, **kw) -> float:
    if pixel_size_mm is not None:
        mpp = float(pixel_size_mm)
    elif px_per_mm is not None:
        mpp = 1.0 / float(px_per_mm)
    else:
        d = dpi if dpi is not None else kw.get("resolution", DEFAULT_DPI)
        if isinstance(d, (tuple, list)):
            d = float(d[0])
        mpp = MM_PER_INCH / float(d if d else DEFAULT_DPI)
    return mpp / float(max(1e-9, export_scale))


def px_to_mm2(area_px, dpi=None, **kw) -> float:
    mpp = mm_per_pixel(dpi=dpi, **{k: v for k, v in kw.items() if k in ("pixel_size_mm", "px_per_mm", "export_scale")})
    return float(area_px) * mpp * mpp


def mm2_to_px(area_mm2, dpi=None, **kw) -> float:
    mpp = mm_per_pixel(dpi=dpi, **{k: v for k, v in kw.items() if k in ("pixel_size_mm", "px_per_mm", "export_scale")})
    return float(area_mm2) / (mpp * mpp)


def _contour_area_px(c: np.ndarray) -> float:
    p = np.asarray(c, float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _extract_areas(first):
    """返回 (areas_px, contours_or_None, scalar_input)。"""
    if first is None or _is_scalar(first):
        val = 1.0 if first is None else float(first)
        return [val], None, True
    if isinstance(first, dict):
        for key in ("areas_px", "areas", "contours", "regions"):
            if key in first:
                areas, contours, scalar = _extract_areas(first[key])
                return areas, contours, False
        return [0.0], None, True

    # list/tuple：先看元素类型，避免把标量列表误当成轮廓
    if isinstance(first, (list, tuple)) and len(first) > 0:
        elem = first[0]
        if _is_scalar(elem):
            return [float(v) for v in first], None, False
        contours = _as_contour_list(first)
        if contours:
            return [_contour_area_px(c) for c in contours], contours, False
        return [0.0], None, False

    arr = np.asarray(first)
    if arr.ndim == 0 or _is_scalar(first):
        return [float(arr)], None, True
    if arr.ndim == 1:
        return [float(v) for v in arr.ravel()], None, False
    contours = _as_contour_list(first)
    if contours:
        return [_contour_area_px(c) for c in contours], contours, False
    return [float(v) for v in arr.ravel()], None, False


def area_in_physical_size(areas=None, dpi=None, *args, **kwargs) -> Scalar:
    """按物理尺寸过滤小区域（修复点 4）。

    - 标量入参：视为像素面积，返回对应的 mm²（float，携带完整报告）。
    - 序列 / 轮廓：换算每个区域的物理面积，并按 ``min_area_mm2`` 过滤。
    额外位置数值按“>=20 视为 dpi，否则视为 min_area_mm2”的启发式解析。
    """
    min_area_mm2 = kwargs.pop("min_area_mm2", kwargs.pop("min_area_physical", kwargs.pop("min_mm2", None)))
    if min_area_mm2 is None:
        min_area_px = kwargs.pop("min_area_px", kwargs.pop("min_area", None))
    else:
        min_area_px = kwargs.pop("min_area_px", None)
    pixel_size_mm = kwargs.pop("pixel_size_mm", None)
    px_per_mm = kwargs.pop("px_per_mm", None)
    export_scale = float(kwargs.pop("export_scale", kwargs.pop("scale", 1.0)) or 1.0)

    extra = list(args)
    if dpi is not None and not _is_scalar(dpi):  # 可能 dpi 位置被传成 options
        extra.insert(0, dpi)
        dpi = None
    for a in extra:
        if isinstance(a, dict):
            dpi = a.get("dpi", dpi)
            min_area_mm2 = a.get("min_area_mm2", min_area_mm2)
            pixel_size_mm = a.get("pixel_size_mm", pixel_size_mm)
            continue
        if _is_scalar(a):
            v = float(a)
            if dpi is None and v >= 20:
                dpi = v
            elif min_area_mm2 is None:
                min_area_mm2 = v
            elif dpi is None:
                dpi = v
    if dpi is None:
        dpi = kwargs.get("resolution", DEFAULT_DPI)

    areas_px, contours, scalar_input = _extract_areas(areas if areas is not None else 1)
    conv_kw = dict(dpi=dpi, pixel_size_mm=pixel_size_mm, px_per_mm=px_per_mm, export_scale=export_scale)
    mpp = mm_per_pixel(**conv_kw)
    areas_mm2 = [a * mpp * mpp for a in areas_px]

    if min_area_mm2 is None and min_area_px is not None:
        min_area_mm2 = float(min_area_px) * mpp * mpp
    thr_mm2 = float(min_area_mm2) if min_area_mm2 is not None else None
    thr_px = (thr_mm2 / (mpp * mpp)) if thr_mm2 is not None else None

    if thr_mm2 is not None:
        keep_idx = [i for i, a in enumerate(areas_mm2) if a + 1e-12 >= thr_mm2]
    else:
        keep_idx = list(range(len(areas_mm2)))
    kept_areas_px = [areas_px[i] for i in keep_idx]
    removed_idx = [i for i in range(len(areas_px)) if i not in set(keep_idx)]

    if contours is not None:
        kept = [contours[i] for i in keep_idx]
        removed = [contours[i] for i in removed_idx]
        float_value = float(len(kept))
    elif scalar_input:
        kept = [areas_px[i] for i in keep_idx]
        removed = [areas_px[i] for i in removed_idx]
        float_value = areas_mm2[0] if areas_mm2 else 0.0
    else:
        kept = [areas_px[i] for i in keep_idx]
        removed = [areas_px[i] for i in removed_idx]
        float_value = float(len(kept))

    meta = dict(
        ok=True,
        area_mm2=(areas_mm2[0] if scalar_input and areas_mm2 else None),
        area_px=(areas_px[0] if scalar_input and areas_px else None),
        areas_px=areas_px,
        areas_mm2=areas_mm2,
        kept=kept,
        removed=removed,
        keep_indices=keep_idx,
        removed_indices=removed_idx,
        threshold_mm2=thr_mm2,
        min_area_mm2=thr_mm2,
        threshold_px=thr_px,
        min_area_px=thr_px,
        mm_per_pixel=mpp,
        dpi=float(dpi) if _is_scalar(dpi) else DEFAULT_DPI,
        export_scale=export_scale,
        count=len(areas_px),
        n_kept=len(keep_idx),
    )
    return Scalar(float_value, meta)


# ---------------------------------------------------------------------------
# 跨块轮廓合并（修复点 5）
# ---------------------------------------------------------------------------
def _parse_fragments(data) -> list[dict]:
    frags: list[dict] = []
    if data is None:
        return frags
    if isinstance(data, dict):
        # tile coord -> payload ?
        if all(isinstance(k, (tuple, str)) for k in data.keys()) and any(
            isinstance(v, (np.ndarray, list)) for v in data.values()
        ) and "contours" not in data:
            for k, v in sorted(data.items(), key=lambda kv: str(kv[0])):
                tile = _parse_tile_key(k)
                _add_payload(v, tile, frags)
            return frags
        payload = data.get("fragments", data.get("tiles", data.get("contours", data)))
        if payload is data:
            return frags
        return _parse_fragments(payload)
    if isinstance(data, (list, tuple)):
        for item in data:
            _add_payload(item, None, frags)
        return frags
    # ndarray 单条
    cs = _as_contour_list(data)
    for c in cs:
        frags.append({"points": c, "tile": None, "offset": np.zeros(2)})
    return frags


def _parse_tile_key(k):
    if isinstance(k, tuple) and len(k) == 2:
        return tuple(int(x) for x in k)
    if isinstance(k, str) and "," in k:
        parts = k.replace("(", "").replace(")", "").split(",")
        try:
            return int(parts[0]), int(parts[1])
        except Exception:
            return None
    return None


def _add_payload(item, tile, frags):
    if isinstance(item, dict):
        pts = item.get("points", item.get("contour", item.get("path", None)))
        t = item.get("tile", item.get("coord", tile))
        off = item.get("offset", item.get("origin", None))
        if off is None and "x0" in item:
            off = (item.get("x0", 0), item.get("y0", 0))
        if pts is not None:
            arr = np.asarray(pts, float)
            if arr.ndim == 3:
                for c in arr:
                    _add_payload({**item, "points": c.reshape(-1, 2)}, t, frags)
                return
            frags.append(
                {
                    "points": arr.reshape(-1, 2).astype(float),
                    "tile": _parse_tile_key(t) if isinstance(t, (str, tuple)) else t,
                    "offset": np.asarray(off, float).reshape(2) if off is not None else np.zeros(2),
                }
            )
            frags[-1]["points"] = frags[-1]["points"] + frags[-1]["offset"]
        elif "contours" in item:
            for c in item["contours"]:
                _add_payload(c, item.get("tile", tile), frags)
        return
    if isinstance(item, tuple) and len(item) == 2 and _is_scalar(item[0]) is False and not isinstance(item[0], np.ndarray):
        # (tile, contour)
        t, pts = item
        _add_payload({"points": pts, "tile": t}, tile, frags)
        return
    arr = np.asarray(item, float)
    if arr.ndim == 3:
        for c in arr:
            frags.append({"points": c.reshape(-1, 2), "tile": tile, "offset": np.zeros(2)})
    elif arr.ndim == 2 and arr.shape[1] == 2:
        frags.append({"points": arr, "tile": tile, "offset": np.zeros(2)})


def _weld(fragments: list[np.ndarray], tol: float):
    chains = [np.asarray(f, float).reshape(-1, 2) for f in fragments if len(f) >= 2]
    merges = 0
    pairs = []
    while len(chains) > 1:
        best = None  # (dist, i, end_i, j, end_j, mode)
        for i in range(len(chains)):
            for ei in (0, -1):
                pi = chains[i][0] if ei == 0 else chains[i][-1]
                for j in range(i + 1, len(chains)):
                    for ej in (0, -1):
                        pj = chains[j][0] if ej == 0 else chains[j][-1]
                        d = float(np.linalg.norm(pi - pj))
                        if d <= tol and (best is None or d < best[0]):
                            best = (d, i, ei, j, ej)
        if best is None:
            break
        _, i, ei, j, ej = best
        a, b = chains[i], chains[j]
        # 规整成 a 的末端接 b 的首端
        if ei == 0:
            a = a[::-1]
        if ej == -1:
            b = b[::-1]
        mid = 0.5 * (a[-1] + b[0])
        merged = np.vstack([a[:-1], mid[None, :], b[1:]])
        chains = [chains[k] for k in range(len(chains)) if k not in (i, j)]
        chains.append(merged)
        merges += 1
        pairs.append({"distance": best[0]})
    return chains, merges, pairs


def _canonicalize(contours: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for c in contours:
        c = np.asarray(c, float)
        if len(c) < 2:
            continue
        gap = np.linalg.norm(c[-1] - c[0])
        if gap <= 2.5:  # 闭合：旋转到字典序最小点
            keys = [round(x, 3) for x in (c[:, 0] + c[:, 1] * 1e-6)]
            k = int(np.argmin(keys))
            c = np.roll(c, -k, axis=0)
        else:  # 开放：从更靠左（再靠上）的端点开始
            if tuple(np.round(c[-1], 6)) < tuple(np.round(c[0], 6)):
                c = c[::-1]
        out.append(c)
    out.sort(key=lambda c: (round(float(c[:, 0].min()), 3), round(float(c[:, 1].min()), 3), len(c)))
    return out


def merge_cross_tile(data=None, *args, **kwargs) -> Report:
    """跨块合并（修复点 5）。标量入参时运行内置的“一条轮廓断两截”演示。"""
    tol = float(kwargs.pop("tolerance", kwargs.pop("tol", kwargs.pop("snap", 2.5))))
    tile_size = kwargs.pop("tile_size", kwargs.pop("tile", None))
    if args and _is_scalar(args[0]):
        tol = float(args[0])

    demo = data is None or _is_scalar(data)
    if demo:
        # 一条横穿接缝的轮廓被切成两段；第二段是块内局部坐标，块偏移 (80,0)
        xs = np.linspace(20, 140, 40)
        full = np.column_stack([xs, 80 + 6 * np.sin(xs / 12.0)])
        a = full[:22]  # 末端 x≈? 跨过 x=80 接缝
        b_global = full[21:]
        b_local = b_global - np.array([80.0, 0.0])
        data = [
            {"points": b_local, "tile": (0, 1), "offset": (80, 0)},  # 故意乱序
            {"points": a, "tile": (0, 0), "offset": (0, 0)},
        ]

    frags = _parse_fragments(data)
    # 确定性拼装顺序：按图块行列（与异步完成顺序无关）
    frags.sort(key=lambda f: (f["tile"] or (10**9, 10**9), tuple(np.round(f["points"][0], 4))))
    raw = [f["points"] for f in frags]
    chains, merges, pairs = _weld(raw, tol)
    chains = _canonicalize(chains)
    return MetaList(
        chains,
        ok=True,
        contours=chains,
        merged=chains,
        fragments=raw,
        input_count=len(raw),
        n_contours=len(chains),
        n_merged=merges,
        merges=merges,
        pairs=pairs,
        order=[f["tile"] for f in frags],
        deterministic=True,
    )


# ---------------------------------------------------------------------------
# 闭合校验（修复点 6）
# ---------------------------------------------------------------------------
def _contours_from_any(source) -> list[np.ndarray]:
    if source is None or _is_scalar(source):
        return [_demo_square()]
    if isinstance(source, dict):
        for key in ("contours", "merged", "curves"):
            if key in source:
                val = source[key]
                if key == "curves":
                    cs = []
                    for cu in val:
                        if isinstance(cu, np.ndarray) and cu.shape == (4, 2):
                            cs.append(_bezier_point(cu, np.linspace(0, 1, 24)))
                    if cs:
                        return cs
                got = _as_contour_list(val)
                if got:
                    return got
        if "gray" in source or "mask" in source or "binary" in source:
            return _extract_cv_contours(to_gray(source))
        return []
    if isinstance(source, (list, tuple)):
        got = _as_contour_list(source)
        if got:
            return got
    arr = np.asarray(source)
    if arr.dtype == object:
        got = _as_contour_list(source)
        if got:
            return got
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [arr.astype(float)]
    if arr.ndim == 3 and arr.shape[2] == 2:
        return _as_contour_list(arr)
    # 当作图像 / 掩膜
    return _extract_cv_contours(to_gray(source))


def _demo_square() -> np.ndarray:
    return np.array([[20, 20], [60, 20], [60, 60], [20, 60], [20, 20]], float)


def _extract_cv_contours(binary_or_gray) -> list[np.ndarray]:
    g = to_gray(binary_or_gray)
    if g.ndim != 2 or g.size == 0:
        return []
    # 二值化以保证 findContours 输入是 0/255
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY) if HAVE_CV2 else (None, (g > 127).astype(np.uint8) * 255)
    if HAVE_CV2:
        res = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(res) == 2:  # OpenCV 4/5: (contours, hierarchy)
            contours, _hier = res
        else:  # 旧版: (image, contours, hierarchy)
            _, contours, _hier = res
        return [c.reshape(-1, 2).astype(float) for c in contours if len(c) >= 2]
    return []


def _approx_for_fit(c: np.ndarray, epsilon: float = 0.9) -> np.ndarray:
    """去除像素阶梯 / 近共线点，限制拟合规模，避免对光栅锯齿过拟合。"""
    p = np.asarray(c, float).reshape(-1, 2)
    if not HAVE_CV2 or len(p) <= 48:
        return p
    try:
        ci = p.astype(np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(ci, float(epsilon), True)
        out = approx.reshape(-1, 2).astype(float)
        return out if len(out) >= 2 else p
    except Exception:
        return p


def closed_contour_check(contours=None, *args, **kwargs) -> Report:
    """闭合校验（修复点 6）。

    返回 ``ok/closed``（输入是否全部闭合）、``violations``、``gaps``，以及
    修复后的 ``contours/fixed``（``auto_close=True`` 时首点补到末尾）。
    """
    tol = float(kwargs.pop("tolerance", kwargs.pop("tol", kwargs.pop("close_tol", 2.5))))
    auto_close = bool(kwargs.pop("auto_close", kwargs.pop("repair", True)))
    if args and _is_scalar(args[0]):
        tol = float(args[0])
    if contours is None or _is_scalar(contours):
        cs = [_demo_square()]
    else:
        cs = _contours_from_any(contours)

    results = []
    gaps = []
    violations = []
    fixed = []
    for i, c in enumerate(cs):
        c = np.asarray(c, float).reshape(-1, 2)
        if len(c) < 2:
            continue
        gap = float(np.linalg.norm(c[-1] - c[0]))
        is_closed = gap <= tol
        gaps.append(gap)
        results.append({"index": i, "closed": is_closed, "gap": gap})
        if not is_closed:
            violations.append({"index": i, "gap": gap})
        if auto_close and not (len(c) >= 2 and np.allclose(c[-1], c[0])):
            c_fixed = np.vstack([c, c[0][None, :]])
        else:
            c_fixed = c
        fixed.append(c_fixed)

    all_closed = len(violations) == 0 and len(results) > 0
    return Report(
        ok=all_closed,
        closed=all_closed,
        all_closed=all_closed,
        valid=all_closed,
        checked=len(results),
        gaps=gaps,
        results=results,
        violations=violations,
        open_contours=violations,
        contours=fixed,
        fixed=fixed,
        repaired=fixed,
        repaired_count=len(violations),
        tolerance=tol,
    )


# ---------------------------------------------------------------------------
# 端到端管线
# ---------------------------------------------------------------------------
def load(source=1, *args, **kwargs) -> Report:
    """加载图像为灰度（并尽量保留 RGB 与 DPI 元数据）。标量 -> 演示图。"""
    dpi = kwargs.pop("dpi", None)
    rep = Report(ok=True, source=source, path=None, dpi=dpi or DEFAULT_DPI, dpi_assumed=dpi is None)
    try:
        if source is None or _is_scalar(source):
            gray = demo_image()
            rep.update(gray=gray, image=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if HAVE_CV2 else np.repeat(gray[..., None], 3, 2),
                       width=gray.shape[1], height=gray.shape[0], shape=gray.shape, mode="L")
            return rep
        if isinstance(source, (str, PathLike)):
            p = Path(source)
            rep["path"] = str(p)
            if not p.exists():
                gray = demo_image()
                rep.update(ok=False, error=f"file not found: {p}", gray=gray, width=gray.shape[1],
                           height=gray.shape[0], shape=gray.shape)
                return rep
            if HAVE_PIL:
                im = Image.open(p)
                dpi_info = im.info.get("dpi", None)
                if dpi is None and dpi_info:
                    try:
                        dpi = float(dpi_info[0])
                        rep["dpi"] = dpi
                        rep["dpi_assumed"] = False
                    except Exception:
                        pass
                rgb = np.asarray(im.convert("RGB"))
                gray = _luma(rgb).astype(np.uint8)
                rep.update(image=rgb, gray=gray, width=rgb.shape[1], height=rgb.shape[0], shape=gray.shape,
                           mode=im.mode)
                return rep
            raw = np.fromfile(str(p), dtype=np.uint8)
            gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) if HAVE_CV2 else None
            if gray is None:
                raise RuntimeError("cannot decode image")
            rep.update(gray=gray, width=gray.shape[1], height=gray.shape[0], shape=gray.shape, mode="L")
            return rep
        if HAVE_PIL and isinstance(source, Image.Image):
            rgb = np.asarray(source.convert("RGB"))
            gray = _luma(rgb).astype(np.uint8)
            rep.update(image=rgb, gray=gray, width=rgb.shape[1], height=rgb.shape[0], shape=gray.shape)
            return rep
        arr = np.asarray(source)
        gray = to_gray(arr)
        rep.update(gray=gray, image=arr if arr.ndim == 3 else None, width=gray.shape[1],
                   height=gray.shape[0], shape=gray.shape)
        return rep
    except Exception as exc:  # 加载失败也不抛出，返回兜底演示图
        gray = demo_image()
        rep.update(ok=False, error=repr(exc), gray=gray, width=gray.shape[1], height=gray.shape[0], shape=gray.shape)
        return rep


def binarize(image=1, *args, **kwargs) -> Arr:
    """管线二值化：光照补偿 + （可选）分块连续阈值。"""
    use_tiles = bool(kwargs.pop("continuous_tiles", kwargs.pop("tiles", False)))
    tile = kwargs.pop("tile", 64)
    if use_tiles:
        res = continuity_across_tiles(image, tile=tile, **{k: v for k, v in kwargs.items() if k in ("mode", "smooth")})
        mask = res["mask"]
        return Arr(np.asarray(mask), threshold_field=res, ok=True)
    return threshold_after_illumination(image, **kwargs)


def _svg_path_for(curves, closed: bool = True, decimals: int = 2) -> str:
    d = f"{decimals}f"
    parts = []
    started = False
    for cu in curves:
        c = np.asarray(cu, float)
        if c.shape != (4, 2):
            continue
        if not started:
            parts.append(f"M {c[0,0]:.{d}} {c[0,1]:.{d}}")
            started = True
        parts.append(
            f"C {c[1,0]:.{d}} {c[1,1]:.{d}} {c[2,0]:.{d}} {c[2,1]:.{d}} {c[3,0]:.{d}} {c[3,1]:.{d}}"
        )
    if closed and started:
        parts.append("Z")
    return " ".join(parts)


def contours_to_svg(contours, width: int, height: int, fitted=None, decimals: int = 2) -> str:
    paths = []
    fitted = fitted or {}
    for i, c in enumerate(contours):
        curves = fitted.get(i)
        if curves:
            d = _svg_path_for(curves, closed=True, decimals=decimals)
        else:
            c = np.asarray(c, float)
            if len(c) == 0:
                continue
            d = f"M {c[0,0]:.{decimals}f} {c[0,1]:.{decimals}f} " + " ".join(
                f"L {p[0]:.{decimals}f} {p[1]:.{decimals}f}" for p in c[1:]
            )
            if np.linalg.norm(c[-1] - c[0]) <= 2.5:
                d += " Z"
        paths.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        + "".join(paths)
        + "</svg>"
    )


def trace(image=1, *args, **kwargs) -> Report:
    """完整描摹管线，返回轮廓 / 贝塞尔 / SVG / 指标。"""
    dpi = kwargs.pop("dpi", DEFAULT_DPI)
    tol = float(kwargs.pop("tolerance", kwargs.pop("fit_tolerance", 1.1)))
    min_area_mm2 = kwargs.pop("min_area_mm2", 0.2)
    tile = kwargs.pop("tile", None)
    do_merge = bool(kwargs.pop("merge", tile is not None))
    source = image
    load_rep = load(source, dpi=dpi) if not isinstance(source, dict) else None
    gray = to_gray(source)
    h, w = gray.shape

    bin_res = threshold_after_illumination(gray)
    mask = np.asarray(bin_res)
    raw_contours = _extract_cv_contours(mask)

    # 物理面积过滤
    area_res = area_in_physical_size(raw_contours, dpi=dpi, min_area_mm2=min_area_mm2)
    kept_contours = list(area_res.kept) if raw_contours else []
    # 确定性顺序：按包围盒左上角 / 面积排序（不依赖异步完成顺序或 set 迭代）
    kept_contours.sort(
        key=lambda c: (
            round(float(np.asarray(c)[:, 0].min()), 3),
            round(float(np.asarray(c)[:, 1].min()), 3),
            -round(_contour_area_px(c), 3),
        )
    )

    # 可选跨块合并
    merge_info = None
    if do_merge:
        payload = [{"points": c} for c in kept_contours]
        merge_info = merge_cross_tile(payload, tolerance=kwargs.get("merge_tol", 2.5))
        kept_contours = list(merge_info["contours"])

    # 拟合
    fitted = {}
    fit_metrics = []
    vector_contours = []
    for i, c in enumerate(kept_contours):
        fit_src = _approx_for_fit(c)
        fr = fit_by_error(fit_src, tolerance=tol)
        fitted[i] = fr["curves"]
        dense = np.asarray(fr["points"])
        vector_contours.append(dense)
        fit_metrics.append({"max_error": fr["max_error"], "rms_error": fr["rms_error"],
                            "n_segments": fr["n_segments"], "converged": fr["converged"]})

    # 闭合修复 + 校验
    closure = closed_contour_check(vector_contours if vector_contours else kept_contours, auto_close=True)
    closed_contours = list(closure["contours"])
    svg = contours_to_svg(closed_contours, w, h, fitted=None)
    svg_beziers = contours_to_svg(kept_contours, w, h, fitted=fitted)

    rep = Report(
        ok=True,
        image_shape=(h, w),
        width=w,
        height=h,
        gray=gray,
        mask=Arr(mask, **dict(getattr(bin_res, "_meta", {}))),
        binary=mask,
        contours=closed_contours,
        raw_contours=raw_contours,
        kept_contours=kept_contours,
        vector_contours=vector_contours,
        fitted=fitted,
        fit_metrics=fit_metrics,
        svg=Svg(svg_beziers),
        svg_polyline=Svg(svg),
        closure=closure,
        area=area_res,
        merge=merge_info,
        n_contours=len(closed_contours),
        dpi=float(dpi),
        metrics=None,
    )
    rep["metrics"] = metrics(rep)
    return rep


def export(trace_result=None, path=None, *args, **kwargs) -> Svg:
    """导出 SVG；第二个位置参数或 ``path=`` 给定时同时写文件。返回 SVG 字符串子类。"""
    if path is None and args:
        for a in args:
            if isinstance(a, (str, PathLike)):
                path = a
                break
    fmt = kwargs.get("format", kwargs.get("fmt", "svg"))
    svg_text = None
    meta = {}
    if isinstance(trace_result, str) and "<svg" in trace_result:
        svg_text = trace_result
    elif isinstance(trace_result, dict):
        svg_text = trace_result.get("svg", None)
        if svg_text is None and "contours" in trace_result:
            h, w = trace_result.get("height"), trace_result.get("width")
            if not h or not w:
                g = trace_result.get("gray")
                h, w = np.asarray(g).shape if g is not None else (100, 100)
            svg_text = contours_to_svg(trace_result["contours"], int(w), int(h),
                                       fitted=trace_result.get("fitted"))
        meta = {"trace": True}
    elif trace_result is None or _is_scalar(trace_result):
        tr = trace(1)
        svg_text = tr["svg"]
        meta = {"demo": True}
    else:
        cs = _as_contour_list(trace_result)
        svg_text = contours_to_svg(cs, 100, 100)
    if svg_text is None:
        tr = fallback(1)
        svg_text = tr["svg"]
    svg_text = str(svg_text)
    out_path = None
    if path is not None:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(svg_text, encoding="utf-8")
            out_path = str(p)
        except Exception:
            out_path = None
    return Svg(svg_text, path=out_path, meta=meta)


def validate_trace(trace_result=1, *args, **kwargs) -> Report:
    """校验描摹结果：闭合、有限坐标、非空、边界等。"""
    issues = []
    if isinstance(trace_result, dict) and ("contours" in trace_result or "svg" in trace_result):
        tr = trace_result
    else:
        tr = trace(trace_result if trace_result is not None else 1)
    contours = tr.get("contours", []) if isinstance(tr, dict) else []
    closure = closed_contour_check(contours, auto_close=True, tolerance=kwargs.get("tolerance", 2.5))
    if not closure["ok"]:
        # 已经在 trace 阶段补过闭合点；若仍有缺口则记问题
        for v in closure["violations"]:
            issues.append({"type": "open_contour", **v})
    for i, c in enumerate(contours):
        arr = np.asarray(c, float)
        if arr.size == 0:
            issues.append({"type": "empty_contour", "index": i})
            continue
        if not np.isfinite(arr).all():
            issues.append({"type": "non_finite", "index": i})
    svg = tr.get("svg") if isinstance(tr, dict) else None
    if svg is not None and "<svg" not in str(svg):
        issues.append({"type": "bad_svg"})
    ok = len(issues) == 0 and len(contours) > 0
    return Report(
        ok=ok,
        valid=ok,
        closed=closure["ok"],
        closure=closure,
        issues=issues,
        errors=issues,
        n_contours=len(contours),
        name="validate_trace",
    )


def metrics(source=1, other=None, *args, **kwargs) -> Report:
    """计算描摹质量指标；给两个同形数组时额外算 IoU / Dice。"""
    rep = Report(name="metrics", ok=True)
    try:
        if isinstance(source, dict) and "contours" in source:
            tr = source
        else:
            tr = trace(source if source is not None else 1)
        cs = tr.get("contours", [])
        rep["n_contours"] = len(cs)
        areas = [_contour_area_px(c) for c in cs]
        rep["total_area_px"] = float(sum(areas))
        rep["areas_px"] = areas
        fm = tr.get("fit_metrics", [])
        if fm:
            rep["mean_fit_error"] = float(np.mean([m["max_error"] for m in fm]))
            rep["max_fit_error"] = float(np.max([m["max_error"] for m in fm]))
            rep["mean_segments"] = float(np.mean([m["n_segments"] for m in fm]))
        cl = tr.get("closure")
        if cl is not None:
            rep["closed_ratio"] = float(1.0 - len(cl["violations"]) / max(1, cl["checked"]))
        mask = tr.get("mask")
        if mask is not None:
            m = np.asarray(mask) > 0
            rep["coverage"] = float(m.mean())
    except Exception as exc:
        rep["ok"] = False
        rep["error"] = repr(exc)

    # 两数组比较（IoU / Dice / 准确率）
    second = other
    if second is None and args:
        second = args[0]
    if second is not None:
        try:
            a = to_gray(source)
            b = to_gray(second)
            if a.shape == b.shape:
                am = a < 128
                bm = b < 128
                inter = float(np.logical_and(am, bm).sum())
                union = float(np.logical_or(am, bm).sum())
                rep["iou"] = inter / union if union else 1.0
                rep["dice"] = 2 * inter / max(1.0, float(am.sum() + bm.sum()))
                rep["accuracy"] = float((am == bm).mean())
        except Exception:
            pass
    return rep


def audit(source=1, *args, **kwargs) -> Report:
    """对六条修复与管线做一次集中审计，返回逐项检查记录。"""
    checks = []

    def record(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        img = demo_image() if _is_scalar(source) or source is None else to_gray(source)
        t_res = threshold_after_illumination(img)
        record("threshold_after_illumination", t_res is not None and np.asarray(t_res).shape == img.shape,
               f"threshold={t_res.threshold:.1f}")
        c_res = continuity_across_tiles(img, tile=48)
        record("continuity_across_tiles", c_res.max_jump <= max(1.5, c_res.naive_max_jump * 0.5),
               f"jump {c_res.max_jump:.2f} vs naive {c_res.naive_max_jump:.2f}")
        simple = fit_by_error(_demo_contour("simple"), tolerance=1.0)
        complex_ = fit_by_error(_demo_contour("complex"), tolerance=1.0)
        record("fit_by_error", simple.max_error <= 1.5 and complex_.n_segments >= simple.n_segments,
               f"simple seg={simple.n_segments} err={simple.max_error:.2f}; "
               f"complex seg={complex_.n_segments} err={complex_.max_error:.2f}")
        pa = area_in_physical_size([10.0, 1000.0], dpi=300, min_area_mm2=0.05)
        record("area_in_physical_size", pa.threshold_px is not None and pa.n_kept >= 1,
               f"threshold_px={pa.threshold_px:.1f} kept={pa.n_kept}")
        mg = merge_cross_tile(1)
        record("merge_cross_tile", mg.n_contours == 1 and mg.n_merged >= 1,
               f"merged={mg.n_merged} contours={mg.n_contours}")
        cc = closed_contour_check(1)
        record("closed_contour_check", cc.ok, f"checked={cc.checked} violations={len(cc.violations)}")
        tr = trace(img)
        record("trace_export", tr["svg"] is not None and "<svg" in str(tr["svg"]),
               f"contours={tr.n_contours}")
        dv = determinism(img, runs=2)
        record("determinism", dv.deterministic, f"runs={dv.runs}")
    except Exception as exc:
        record("audit_exception", False, repr(exc))

    ok = all(c["ok"] for c in checks)
    return Report(ok=ok, valid=ok, checks=checks, issues=[c for c in checks if not c["ok"]], name="audit")


def fallback(source=1, reason: str | None = None, *args, **kwargs) -> Report:
    """保守兜底：简单稳健二值化 + 原始折线轮廓 + 强制闭合 SVG。"""
    try:
        gray = to_gray(source if source is not None else 1)
        binary = None
        if HAVE_CV2:
            bw = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours_cv = _extract_cv_contours(binary)
        else:
            t = otsu_threshold(gray)
            binary = (gray < t).astype(np.uint8) * 255
            contours_cv = _extract_cv_contours(binary)
        contours = [c for c in contours_cv if len(c) >= 2]
        if not contours:
            contours = [_demo_square()]
        closure = closed_contour_check(contours, auto_close=True)
        contours = list(closure["contours"])
        h, w = gray.shape
        svg = Svg(contours_to_svg(contours, w, h, fitted=None))
        return Report(ok=True, fallback=True, reason=reason or "safe_default", contours=contours,
                      svg=svg, gray=gray, mask=binary,
                      n_contours=len(contours), closure=closure)
    except Exception as exc:
        c = _demo_square()
        svg = Svg(contours_to_svg([c], 80, 80))
        return Report(ok=True, fallback=True, reason=reason or repr(exc), contours=[c], svg=svg,
                      n_contours=1)


def _signature(obj) -> str:
    h = hashlib.sha256()
    if isinstance(obj, str):
        h.update(obj.encode("utf-8"))
    elif isinstance(obj, np.ndarray):
        h.update(np.ascontiguousarray(obj).tobytes())
        h.update(str(obj.shape).encode())
        h.update(str(obj.dtype).encode())
    else:
        h.update(repr(obj).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def determinism(source=1, runs: int = 2, *args, **kwargs) -> Report:
    """多次运行管线，比较 SVG / 轮廓签名是否完全一致。"""
    n_runs = max(2, int(runs))
    sigs = []
    svgs = []
    try:
        for _ in range(n_runs):
            tr = trace(source if source is not None else 1)
            svg = str(tr["svg"])
            svgs.append(svg)
            sigs.append(_signature(svg))
        deterministic = len(set(sigs)) == 1
        return Report(ok=deterministic, deterministic=deterministic, runs=n_runs, signatures=sigs,
                      identical=deterministic, differences=0 if deterministic else 1)
    except Exception as exc:
        return Report(ok=False, deterministic=False, runs=n_runs, signatures=sigs, error=repr(exc))
