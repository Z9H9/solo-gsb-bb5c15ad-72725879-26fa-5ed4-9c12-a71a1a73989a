"""Centroiding of uniformly spaced mass/intensity profile data.

``POST /api/v1/profile-centroid`` converts a client-supplied continuous
profile — (mass, intensity) points on a uniform mass grid — into centroid
peaks:

1. Intensities are base-peak normalised (highest point = 100) and every
   point at or below ``relative_threshold`` percent of the base peak is
   discarded.
2. Each flat-top local-maximum interval of the remaining points — a maximal
   run of consecutive grid points sharing one intensity, flanked by strictly
   lower (discarded or absent) points — becomes one candidate peak
   positioned at the plateau midpoint.
3. Candidates are filtered by non-maximum suppression: strongest first
   (intensity ties go to the lower mass); a candidate lying closer than
   ``min_separation`` to an already retained candidate is dropped.
4. Every surviving (above-threshold) point is assigned to its nearest
   retained candidate by mass — equidistant points go to the lower-mass
   candidate — and contributes its *original* intensity to that cluster.

Each retained candidate yields the intensity-weighted centroid mass of its
cluster, the plateau's normalised apex intensity and the cluster's share of
the total assigned intensity in percent; peaks are returned sorted by
ascending centroid mass.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

# A profile grid must contain between MIN_POINTS and MAX_POINTS points.
MIN_POINTS = 3
MAX_POINTS = 20000
# Largest tolerated relative deviation of one grid step from the mean step.
STEP_REL_TOLERANCE = 1e-6


class ProfileGridError(ValueError):
    """Locatable validation error for a profile-centroid request.

    Carries a machine-readable ``error`` code, the offending request
    ``field`` and, when the problem is tied to a single profile point, that
    point's 0-based ``index`` (``None`` for whole-list problems).
    """

    def __init__(
        self, error: str, field: str, message: str, index: int | None = None
    ) -> None:
        super().__init__(message)
        self.error = error
        self.field = field
        self.message = message
        self.index = index


@dataclass
class CentroidPeak:
    mass: float  # intensity-weighted centroid mass of the cluster
    apex_intensity: float  # plateau intensity, base peak = 100
    intensity_percent: float  # cluster share of the total assigned intensity


@dataclass
class CentroidResult:
    point_count: int
    mass_step: float  # mean grid step
    candidate_count: int  # local maxima before non-maximum suppression
    peaks: list[CentroidPeak]  # retained after NMS, sorted by centroid mass


def _validate_grid(masses: list[float], intensities: list[float]) -> float:
    """Check the uniform profile grid and return the mean mass step."""
    n = len(masses)
    if not MIN_POINTS <= n <= MAX_POINTS:
        raise ProfileGridError(
            "invalid_grid",
            "points",
            f"Profile must contain between {MIN_POINTS} and {MAX_POINTS} "
            f"points, got {n}",
        )
    for i in range(1, n):
        if masses[i] <= masses[i - 1]:
            raise ProfileGridError(
                "invalid_grid",
                "points",
                f"Point masses must be strictly increasing; mass {masses[i]} "
                f"at index {i} does not exceed {masses[i - 1]}",
                i,
            )
    step = (masses[-1] - masses[0]) / (n - 1)
    for i in range(1, n):
        d = masses[i] - masses[i - 1]
        if abs(d - step) > STEP_REL_TOLERANCE * step:
            raise ProfileGridError(
                "invalid_grid",
                "points",
                f"Non-uniform mass grid: step {d:.6g} at index {i} deviates "
                f"from the mean step {step:.6g} by more than "
                f"{STEP_REL_TOLERANCE:.0e} relative",
                i,
            )
    if max(intensities) <= 0.0:
        raise ProfileGridError(
            "all_zero_profile",
            "points",
            "At least one profile point must have a positive intensity",
        )
    return step


def _find_candidates(
    masses: list[float], norm: list[float], above: list[bool]
) -> list[tuple[float, float]]:
    """Flat-top local maxima of the above-threshold points.

    A candidate is a maximal run of consecutive kept grid points sharing one
    normalised intensity, flanked by strictly lower points (a discarded or
    missing neighbour is automatically lower than the kept run). Returns
    ``(plateau midpoint mass, normalised plateau intensity)`` pairs.
    """
    n = len(masses)
    candidates: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1] and norm[j + 1] == norm[i]:
            j += 1
        if (i == 0 or norm[i - 1] < norm[i]) and (
            j == n - 1 or norm[j + 1] < norm[j]
        ):
            candidates.append(((masses[i] + masses[j]) / 2.0, norm[i]))
        i = j + 1
    return candidates


def _suppress_non_maxima(
    candidates: list[tuple[float, float]], min_separation: float
) -> list[tuple[float, float]]:
    """Greedy non-maximum suppression: strongest candidate first (intensity
    ties go to the lower mass); a candidate closer than ``min_separation``
    to an already retained one is dropped. Returns the retained candidates
    sorted by ascending mass."""
    kept: list[tuple[float, float]] = []
    for mass, intensity in sorted(candidates, key=lambda c: (-c[1], c[0])):
        if all(abs(mass - other) >= min_separation for other, _ in kept):
            kept.append((mass, intensity))
    kept.sort()
    return kept


def centroid_profile(
    masses: list[float],
    intensities: list[float],
    relative_threshold: float,
    min_separation: float,
) -> CentroidResult:
    """Centroid a uniform-grid mass/intensity profile.

    ``masses`` must be strictly increasing with every consecutive step
    within :data:`STEP_REL_TOLERANCE` (relative) of the mean step;
    ``intensities`` must be non-negative with at least one positive value.
    Points whose base-peak normalised intensity does not exceed
    ``relative_threshold`` (percent) take no further part. Raises
    :class:`ProfileGridError` for an invalid grid, an all-zero profile or
    when no above-threshold local maximum remains.
    """
    step = _validate_grid(masses, intensities)
    n = len(masses)
    base = max(intensities)
    norm = [100.0 * x / base for x in intensities]
    above = [nv > relative_threshold for nv in norm]

    candidates = _find_candidates(masses, norm, above)
    kept = _suppress_non_maxima(candidates, min_separation)
    if not kept:
        raise ProfileGridError(
            "no_candidates",
            "relative_threshold",
            "No above-threshold local maximum remains; lower "
            "relative_threshold",
        )

    # Assign each above-threshold point to its nearest retained candidate;
    # an equidistant point goes to the lower-mass candidate. Candidate
    # masses are strictly increasing, so a binary search suffices.
    cand_masses = [m for m, _ in kept]
    k = len(kept)
    weight_sum = [0.0] * k
    intensity_sum = [0.0] * k
    for idx in range(n):
        if not above[idx]:
            continue
        m = masses[idx]
        pos = bisect.bisect_left(cand_masses, m)
        if pos == 0:
            best = 0
        elif pos == k:
            best = k - 1
        else:
            left = m - cand_masses[pos - 1]
            right = cand_masses[pos] - m
            best = pos - 1 if left <= right else pos
        inten = intensities[idx]
        weight_sum[best] += m * inten
        intensity_sum[best] += inten

    # Every retained candidate keeps at least its plateau-centre point (the
    # nearest other candidate midpoint is provably farther), so no cluster
    # is empty and every intensity_sum is positive.
    total = sum(intensity_sum)
    peaks = [
        CentroidPeak(
            mass=weight_sum[c] / intensity_sum[c],
            apex_intensity=kept[c][1],
            intensity_percent=100.0 * intensity_sum[c] / total,
        )
        for c in range(k)
    ]
    peaks.sort(key=lambda p: p.mass)
    return CentroidResult(n, step, len(candidates), peaks)
