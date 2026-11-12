"""Tests for the construction of the index.

The risk these guard against is a silent inversion or a mis-scaling: an index
that runs the wrong way, or that changes when the same data arrives in a
different order, would still produce plausible tables. Reliability statistics
are checked against cases whose value is known analytically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adminplex.index.build import (  # noqa: E402
    _rescale_0_100,
    cronbach_alpha,
    equal_weight,
    expert_weight,
    mcdonald_omega,
    pca_index,
)
from adminplex.index.dimensions import BURDEN_FAMILY, DIMENSION_KEYS  # noqa: E402


def correlated_dimensions(n: int = 400, k: int = 6, strength: float = 0.8) -> pd.DataFrame:
    """Dimensions sharing one underlying factor, as the model assumes."""
    rng = np.random.default_rng(20260726)
    latent = rng.normal(size=n)
    data = {
        f"dim{i}": strength * latent + np.sqrt(1 - strength**2) * rng.normal(size=n)
        for i in range(k)
    }
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Scaling


def test_rescale_spans_zero_to_one_hundred():
    scaled = _rescale_0_100(np.array([2.0, 4.0, 6.0]))
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(100.0)
    assert scaled.iloc[1] == pytest.approx(50.0)


def test_rescale_of_a_constant_does_not_divide_by_zero():
    scaled = _rescale_0_100(np.array([3.0, 3.0, 3.0]))
    assert (scaled == 50.0).all()


# --------------------------------------------------------------------------
# Reliability


def test_alpha_is_high_when_dimensions_share_a_factor():
    assert cronbach_alpha(correlated_dimensions(strength=0.9)) > 0.9


def test_alpha_is_low_for_unrelated_dimensions():
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=(400, 6)), columns=[f"d{i}" for i in range(6)])
    assert cronbach_alpha(noise) < 0.3


def test_alpha_is_undefined_rather_than_wrong_with_one_item():
    assert np.isnan(cronbach_alpha(pd.DataFrame({"only": [1.0, 2.0, 3.0]})))


def test_omega_tracks_alpha_on_a_single_factor():
    frame = correlated_dimensions(strength=0.85)
    assert mcdonald_omega(frame) > 0.8


# --------------------------------------------------------------------------
# Orientation: the direction of the index must not depend on the data


def test_pca_index_points_the_same_way_as_the_dimensions():
    """Higher on every dimension must mean higher on the index.

    The first principal component is only defined up to sign, so the builder
    flips it when it comes out negative. If that logic were wrong the whole
    index would be inverted and every table would still look reasonable.
    """
    frame = correlated_dimensions(strength=0.85)
    result = pca_index(frame)
    scores = pd.Series(result.scores.values, index=frame.index)
    assert scores.corr(frame.mean(axis=1)) > 0.9


def test_pca_orientation_follows_the_data_rather_than_a_fixed_direction():
    frame = correlated_dimensions(strength=0.85)
    flipped = pca_index(-frame)
    upright = pca_index(frame)
    # Negating every dimension reverses the ordering; the orientation rule
    # should follow the data rather than lock to one arbitrary direction.
    assert pd.Series(flipped.scores.values).corr(pd.Series(upright.scores.values)) < 0


def test_equal_weight_is_the_mean_of_the_dimensions():
    frame = correlated_dimensions()
    result = equal_weight(frame)
    scores = pd.Series(result.scores.values)
    assert scores.corr(pd.Series(_rescale_0_100(frame.mean(axis=1)).values)) > 0.999
    assert sum(result.weights.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Determinism


def test_the_index_does_not_depend_on_the_order_of_the_documents():
    frame = correlated_dimensions()
    shuffled = frame.sample(frac=1.0, random_state=7)
    a = pd.Series(pca_index(frame).scores.values, index=frame.index)
    b = pd.Series(pca_index(shuffled).scores.values, index=shuffled.index)
    assert a.corr(b.reindex(a.index)) > 0.999


def test_expert_weights_sum_to_one_and_follow_the_families():
    dims = [k for k in DIMENSION_KEYS]
    frame = pd.DataFrame(
        np.random.default_rng(3).normal(size=(200, len(dims))), columns=dims
    )
    result = expert_weight(frame)
    assert sum(result.weights.values()) == pytest.approx(1.0)
    # Each family should carry a third of the total weight.
    for family, members in BURDEN_FAMILY.items():
        present = [m for m in members if m in result.weights]
        share = sum(result.weights[m] for m in present)
        assert share == pytest.approx(1 / 3, abs=0.02), family
