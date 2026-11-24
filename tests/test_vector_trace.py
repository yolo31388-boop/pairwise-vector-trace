import pytest
import imaging.vector_trace as mod

CASES = [
    ("threshold_after_illumination", (1,)),
    ("continuity_across_tiles", (1,)),
    ("fit_by_error", (1,)),
    ("area_in_physical_size", (1,)),
    ("merge_cross_tile", (1,)),
    ("closed_contour_check", (1,)),
    ("load", (1,)),
    ("binarize", (1,)),
    ("trace", (1,)),
    ("export", (1,)),
    ("validate_trace", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
