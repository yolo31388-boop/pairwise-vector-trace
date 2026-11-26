"""六条修复点的性质级自测（合成数据，确定性）。

契约测试只验证 ``f(1) is not None``；这里验证题目描述的真实行为：
光照恢复、块间连续、按误差收敛拟合、物理面积、跨块合并、闭合校验，
以及端到端的 trace/export/validate/determinism。
"""

import os

import numpy as np
import pytest

import imaging.vector_trace as vt


def _iou(a, b):
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / float(union)


def _shadow_scene(h=200, w=200):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u = xx / (w - 1)
    v = yy / (h - 1)
    light = 238.0 * (1.0 - 0.75 * (0.55 * u + 1.0 * v) / 1.55)
    truth = np.zeros((h, w), bool)
    truth |= (abs(v - 0.80) < 0.04) & (u > 0.25) & (u < 0.97)
    truth |= (abs(u - 0.90) < 0.05) & (v > 0.45) & (v < 0.95)
    g = light.copy()
    g[truth] = 45.0 * (light[truth] / 238.0) + 6.0
    return np.clip(g, 0, 255).astype(np.uint8), truth, (u > 0.7) & (v > 0.7)


# 1) 先光照估计再阈值 ---------------------------------------------------------
def test_illumination_estimate_tracks_background():
    img, _, _ = _shadow_scene()
    bg = vt.estimate_illumination(img)
    assert bg.shape == img.shape
    assert np.isfinite(bg).all()


def test_threshold_after_illumination_recovers_corner():
    img, truth, corner = _shadow_scene()
    res = vt.threshold_after_illumination(img)
    mask = np.asarray(res) > 0
    assert mask.shape == truth.shape
    # 右下角的笔画被恢复
    recall = float((mask & truth & corner).sum()) / max(1, int((truth & corner).sum()))
    assert recall > 0.8
    # 明显优于朴素全局 Otsu
    global_mask = img < vt.otsu_threshold(img)
    assert _iou(mask, truth) > _iou(global_mask, truth) + 0.2
    assert _iou(mask, truth) > 0.85


# 2) 块间阈值连续 -------------------------------------------------------------
def test_continuous_threshold_field_has_no_seam_jumps():
    h = w = 160
    ramp = np.tile(np.linspace(250, 60, w), (h, 1)).astype(np.uint8)
    res = vt.continuity_across_tiles(ramp, tile=40)
    field = np.asarray(res)
    assert field.shape == (h, w)
    assert res.max_jump < 5.0
    assert res.max_jump < 0.2 * max(1.0, res.naive_max_jump)
    assert res.naive_max_jump > 20.0


# 3) 拟合按误差收敛 + 自适应分段 ----------------------------------------------
def test_fit_line_one_segment_no_jag():
    line = np.column_stack([np.linspace(0, 100, 60), np.linspace(0, 10, 60)])
    r = vt.fit_by_error(line, tolerance=1.0)
    assert r.n_segments <= 2
    assert r.max_error < 0.1
    assert r.converged


def test_fit_complex_gets_more_segments_but_holds_tolerance():
    simple = vt.fit_by_error(vt._demo_contour("simple"), tolerance=1.0)
    star = vt.fit_by_error(vt._demo_contour("complex"), tolerance=1.0)
    assert star.n_segments > simple.n_segments
    assert star.n_segments >= 6
    assert star.max_error <= 1.5


def test_fit_returns_list_and_array():
    r = vt.fit_by_error(vt._demo_contour("simple"), tolerance=1.0)
    assert isinstance(r, list) and len(r) >= 1
    assert np.asarray(r).ndim == 2 and np.asarray(r).shape[1] == 2


# 4) 面积按物理尺寸 -----------------------------------------------------------
def test_physical_area_conversion():
    assert vt.px_to_mm2(100, dpi=300) == pytest.approx(100 * (25.4 / 300) ** 2, rel=1e-9)
    assert vt.mm2_to_px(1.0, dpi=25.4) == pytest.approx(1.0)  # 25.4dpi -> 1px=1mm


def test_physical_filter_scale_invariant():
    phys = 0.1
    a300 = vt.mm2_to_px(phys, dpi=300)
    a600 = vt.mm2_to_px(phys, dpi=600)
    r300 = vt.area_in_physical_size([a300], dpi=300, min_area_mm2=0.08)
    r600 = vt.area_in_physical_size([a600], dpi=600, min_area_mm2=0.08)
    assert r300.n_kept == 1 and r600.n_kept == 1
    small = vt.area_in_physical_size([1.0], dpi=300, min_area_mm2=0.08)
    assert small.n_kept == 0


# 5) 跨块合并 -----------------------------------------------------------------
def test_merge_demo_rejoins_fragments():
    r = vt.merge_cross_tile(1)
    assert r.n_contours == 1
    assert r.n_merged >= 1
    assert len(r) == 1


def test_merge_shuffled_tiles_order_independent():
    xs = np.linspace(0, 100, 50)
    full = np.column_stack([xs, 50 + 3 * np.sin(xs / 10.0)])
    a = full[:26]
    bglob = full[25:]
    tiles = [
        {"tile": (1, 0), "offset": (0, 50), "points": bglob - np.array([0.0, 50.0])},
        {"tile": (0, 0), "offset": (0, 0), "points": a},
    ]
    r1 = vt.merge_cross_tile(list(reversed(tiles)), tolerance=2.0)
    r2 = vt.merge_cross_tile(tiles, tolerance=2.0)
    assert r1.n_contours == 1
    c = r1[0]
    assert c[:, 0].min() == pytest.approx(0.0, abs=1.0)
    assert c[:, 0].max() == pytest.approx(100.0, abs=1.0)
    assert np.allclose(r1[0], r2[0])


# 6) 闭合校验 -----------------------------------------------------------------
def test_closed_check_detects_and_repairs():
    closed = vt.closed_contour_check(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0.0]]))
    opened = vt.closed_contour_check(np.array([[0, 0], [10, 0], [10, 10], [0, 10.0]]))
    assert closed.ok
    assert not opened.ok
    assert len(opened.violations) == 1
    repaired = opened.contours[0]
    assert np.allclose(repaired[0], repaired[-1])


# 端到端 ----------------------------------------------------------------------
def test_trace_export_validate_determinism(tmp_path):
    tr = vt.trace(1)
    assert tr.n_contours >= 1
    assert "<svg" in str(tr.svg)
    assert vt.validate_trace(tr).ok

    out = tmp_path / "out.svg"
    svg = vt.export(tr, str(out))
    assert "<svg" in str(svg)
    assert out.exists()

    det = vt.determinism(1, runs=2)
    assert det.deterministic


def test_audit_passes():
    rep = vt.audit(1)
    failed = [c["name"] for c in rep.checks if not c["ok"]]
    assert not failed, f"audit checks failed: {failed}"
    assert rep.ok
