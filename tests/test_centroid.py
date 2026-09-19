"""Tests for profile-to-centroid conversion (unit + API)."""

import pytest
from fastapi.testclient import TestClient

from app.centroid import (
    MAX_POINTS,
    MIN_POINTS,
    ProfileGridError,
    centroid_profile,
)
from app.config import settings


def uniform_masses(start, step, n):
    return [start + k * step for k in range(n)]


def make_points(masses, intensities):
    return [{"mass": m, "intensity": i} for m, i in zip(masses, intensities)]


def test_single_triangle_peak():
    masses = uniform_masses(100.0, 0.1, 7)
    intensities = [0.0, 10.0, 50.0, 100.0, 50.0, 10.0, 0.0]
    res = centroid_profile(masses, intensities, 1.0, 0.05)
    assert res.point_count == 7
    assert res.mass_step == pytest.approx(0.1)
    assert res.candidate_count == 1
    assert len(res.peaks) == 1
    peak = res.peaks[0]
    assert peak.mass == pytest.approx(100.3)
    assert peak.apex_intensity == pytest.approx(100.0)
    assert peak.intensity_percent == pytest.approx(100.0)


def test_flat_top_plateau_uses_midpoint():
    masses = uniform_masses(500.0, 0.05, 7)
    intensities = [0.0, 20.0, 100.0, 100.0, 100.0, 20.0, 0.0]
    res = centroid_profile(masses, intensities, 1.0, 0.01)
    assert res.candidate_count == 1
    peak = res.peaks[0]
    # Plateau spans indices 2-4; candidate position is its midpoint.
    assert peak.mass == pytest.approx(500.15)
    assert peak.apex_intensity == pytest.approx(100.0)


def test_nms_keeps_distant_candidates():
    masses = uniform_masses(200.0, 0.1, 7)
    intensities = [10.0, 100.0, 10.0, 5.0, 40.0, 8.0, 2.0]
    res = centroid_profile(masses, intensities, 1.0, 0.05)
    assert res.candidate_count == 2
    assert len(res.peaks) == 2
    first, second = res.peaks
    assert first.mass == pytest.approx(200.1)
    assert first.apex_intensity == pytest.approx(100.0)
    assert first.intensity_percent == pytest.approx(100.0 * 120.0 / 175.0)
    assert second.mass == pytest.approx(11022.7 / 55.0)
    assert second.apex_intensity == pytest.approx(40.0)
    assert second.intensity_percent == pytest.approx(100.0 * 55.0 / 175.0)


def test_nms_suppresses_weaker_candidate_within_min_separation():
    masses = uniform_masses(200.0, 0.1, 7)
    intensities = [10.0, 100.0, 10.0, 5.0, 40.0, 8.0, 2.0]
    res = centroid_profile(masses, intensities, 1.0, 0.35)
    assert res.candidate_count == 2
    assert len(res.peaks) == 1
    peak = res.peaks[0]
    assert peak.mass == pytest.approx(35034.7 / 175.0)
    assert peak.apex_intensity == pytest.approx(100.0)
    assert peak.intensity_percent == pytest.approx(100.0)


def test_nms_intensity_tie_prefers_lower_mass():
    masses = uniform_masses(200.0, 0.1, 3)
    intensities = [100.0, 10.0, 100.0]
    res = centroid_profile(masses, intensities, 1.0, 0.25)
    assert res.candidate_count == 2
    assert len(res.peaks) == 1
    # Equal apex intensities: the lower-mass candidate wins and absorbs all
    # above-threshold points.
    assert res.peaks[0].mass == pytest.approx(42021.0 / 210.0)


def test_assignment_tie_goes_to_lower_mass_candidate():
    masses = uniform_masses(200.0, 0.1, 3)
    intensities = [100.0, 10.0, 100.0]
    res = centroid_profile(masses, intensities, 1.0, 0.15)
    assert len(res.peaks) == 2
    first, second = res.peaks
    # The middle point is equidistant and joins the lower-mass cluster.
    assert first.mass == pytest.approx(22001.0 / 110.0)
    assert first.intensity_percent == pytest.approx(100.0 * 110.0 / 210.0)
    assert second.mass == pytest.approx(200.2)
    assert second.intensity_percent == pytest.approx(100.0 * 100.0 / 210.0)


def test_original_intensities_drive_centroid_not_normalised():
    masses = uniform_masses(100.0, 0.1, 5)
    intensities = [4.0, 30.0, 100.0, 20.0, 3.0]
    scaled = [37.5 * i for i in intensities]
    base = centroid_profile(masses, intensities, 5.0, 0.01)
    rescaled = centroid_profile(masses, scaled, 5.0, 0.01)
    assert len(base.peaks) == len(rescaled.peaks) == 1
    peak, scaled_peak = base.peaks[0], rescaled.peaks[0]
    # Only the above-threshold points (indices 1-3) enter the centroid.
    assert peak.mass == pytest.approx(15029.0 / 150.0)
    assert scaled_peak.mass == pytest.approx(peak.mass)
    assert scaled_peak.apex_intensity == pytest.approx(peak.apex_intensity)
    assert scaled_peak.intensity_percent == pytest.approx(peak.intensity_percent)


def test_intensity_percents_sum_to_100_and_peaks_sorted():
    masses = uniform_masses(300.0, 0.02, 15)
    intensities = [
        2.0, 30.0, 90.0, 25.0, 3.0,
        4.0, 50.0, 70.0, 60.0, 5.0,
        10.0, 80.0, 40.0, 6.0, 1.0,
    ]
    res = centroid_profile(masses, intensities, 2.0, 0.03)
    assert len(res.peaks) >= 2
    assert sum(p.intensity_percent for p in res.peaks) == pytest.approx(100.0)
    centroid_masses = [p.mass for p in res.peaks]
    assert centroid_masses == sorted(centroid_masses)
    assert max(p.apex_intensity for p in res.peaks) == pytest.approx(100.0)


@pytest.mark.parametrize(
    "masses,index",
    [
        ([100.0, 100.0, 100.1], 1),  # duplicate
        ([100.0, 99.9, 100.1], 1),  # decreasing
        # One bad step in a 3-point grid shifts the mean step so both steps
        # deviate; the first offending point is reported.
        ([0.0, 1.0, 2.000003], 1),
        # In a larger grid only the deviating step exceeds the tolerance.
        ([0.0, 1.0, 2.0, 3.0, 4.000002], 4),
    ],
)
def test_invalid_grid_reports_index(masses, index):
    intensities = [1.0] * len(masses)
    with pytest.raises(ProfileGridError) as exc:
        centroid_profile(masses, intensities, 1.0, 0.01)
    assert exc.value.error == "invalid_grid"
    assert exc.value.field == "points"
    assert exc.value.index == index


def test_grid_within_step_tolerance_passes():
    masses = [0.0, 1.0, 1.9999999]
    res = centroid_profile(masses, [1.0, 2.0, 1.0], 0.0, 0.01)
    assert len(res.peaks) == 1


def test_point_count_bounds():
    with pytest.raises(ProfileGridError) as exc:
        centroid_profile([1.0, 2.0], [1.0, 1.0], 1.0, 0.01)
    assert exc.value.error == "invalid_grid"
    masses = uniform_masses(1.0, 0.001, MAX_POINTS + 1)
    with pytest.raises(ProfileGridError):
        centroid_profile(masses, [1.0] * (MAX_POINTS + 1), 1.0, 0.01)
    # Exactly MIN_POINTS / MAX_POINTS are accepted.
    res = centroid_profile(
        uniform_masses(1.0, 0.001, MIN_POINTS), [1.0, 2.0, 1.0], 0.0, 0.01
    )
    assert res.point_count == MIN_POINTS
    res = centroid_profile(
        uniform_masses(1.0, 0.001, MAX_POINTS), [1.0] * MAX_POINTS, 0.0, 0.01
    )
    assert res.point_count == MAX_POINTS


def test_all_zero_profile_rejected():
    with pytest.raises(ProfileGridError) as exc:
        centroid_profile(uniform_masses(1.0, 0.1, 5), [0.0] * 5, 1.0, 0.01)
    assert exc.value.error == "all_zero_profile"
    assert exc.value.field == "points"
    assert exc.value.index is None


def test_no_candidates_when_threshold_reaches_base_peak():
    masses = uniform_masses(100.0, 0.1, 5)
    with pytest.raises(ProfileGridError) as exc:
        centroid_profile(masses, [1.0, 10.0, 100.0, 10.0, 1.0], 100.0, 0.01)
    assert exc.value.error == "no_candidates"
    assert exc.value.field == "relative_threshold"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def _two_peak_payload(**overrides):
    payload = {
        "points": make_points(
            uniform_masses(100.0, 0.1, 11),
            [5.0, 40.0, 100.0, 42.0, 6.0, 3.0, 25.0, 80.0, 80.0, 22.0, 4.0],
        ),
        "relative_threshold": 2.0,
        "min_separation": 0.05,
    }
    payload.update(overrides)
    return payload


def test_api_profile_centroid_two_peaks(client):
    resp = client.post("/api/v1/profile-centroid", json=_two_peak_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["point_count"] == 11
    assert body["mass_step"] == pytest.approx(0.1)
    assert body["relative_threshold"] == 2.0
    assert body["min_separation"] == 0.05
    assert body["candidate_count"] == 2
    assert body["peak_count"] == 2
    first, second = body["peaks"]
    # Flat-top plateau at indices 7-8 reports the plateau midpoint cluster.
    assert first["mass"] == pytest.approx(19339.0 / 193.0, abs=1e-6)
    assert first["apex_intensity"] == pytest.approx(100.0)
    assert first["intensity_percent"] == pytest.approx(100.0 * 193.0 / 407.0)
    assert second["mass"] == pytest.approx(21560.3 / 214.0, abs=1e-6)
    assert second["apex_intensity"] == pytest.approx(80.0)
    assert second["intensity_percent"] == pytest.approx(100.0 * 214.0 / 407.0)
    assert first["intensity_percent"] + second["intensity_percent"] == pytest.approx(100.0)


def test_api_profile_centroid_flat_top_boundary_peak(client):
    # A plateau touching the upper grid edge is still a local maximum.
    points = make_points(
        uniform_masses(50.0, 0.01, 6), [1.0, 5.0, 20.0, 60.0, 60.0, 60.0]
    )
    resp = client.post(
        "/api/v1/profile-centroid",
        json={
            "points": points,
            "relative_threshold": 1.0,
            "min_separation": 0.005,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["peak_count"] == 1
    assert body["peaks"][0]["apex_intensity"] == pytest.approx(100.0)


def test_api_profile_centroid_invalid_grid_422(client):
    payload = _two_peak_payload()
    payload["points"] = make_points([100.0, 100.0, 100.1], [1.0, 2.0, 1.0])
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_grid"
    assert detail["field"] == "points"
    assert detail["index"] == 1

    payload = _two_peak_payload()
    payload["points"] = make_points([0.0, 1.0, 2.0, 3.000002], [1.0, 2.0, 3.0, 4.0])
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_grid"
    assert detail["index"] == 3


def test_api_profile_centroid_all_zero_422(client):
    payload = _two_peak_payload()
    payload["points"] = make_points(uniform_masses(1.0, 0.1, 5), [0.0] * 5)
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "all_zero_profile"
    assert detail["field"] == "points"
    assert detail["index"] is None


def test_api_profile_centroid_no_candidates_422(client):
    resp = client.post(
        "/api/v1/profile-centroid", json=_two_peak_payload(relative_threshold=100.0)
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "no_candidates"
    assert detail["field"] == "relative_threshold"


@pytest.mark.parametrize(
    "override",
    [
        {"relative_threshold": -0.5},
        {"relative_threshold": 100.5},
        {"relative_threshold": "nan"},
        {"min_separation": 0},
        {"min_separation": -0.1},
        {"min_separation": "inf"},
    ],
)
def test_api_profile_centroid_parameter_validation_422(client, override):
    resp = client.post(
        "/api/v1/profile-centroid", json=_two_peak_payload(**override)
    )
    assert resp.status_code == 422, resp.text


@pytest.mark.parametrize(
    "points",
    [
        make_points(uniform_masses(1.0, 0.1, 2), [1.0, 2.0]),  # too few
        make_points(uniform_masses(1.0, 0.1, 5), [1.0, 2.0, -3.0, 2.0, 1.0]),
        make_points(uniform_masses(1.0, 0.1, 3), [1.0, "nan", 1.0]),
        make_points([1.0, "inf", 3.0], [1.0, 2.0, 1.0]),
    ],
)
def test_api_profile_centroid_point_validation_422(client, points):
    resp = client.post(
        "/api/v1/profile-centroid", json=_two_peak_payload(points=points)
    )
    assert resp.status_code == 422, resp.text


def test_api_profile_centroid_missing_fields_422(client):
    assert client.post("/api/v1/profile-centroid", json={}).status_code == 422
    payload = _two_peak_payload()
    del payload["relative_threshold"]
    assert client.post("/api/v1/profile-centroid", json=payload).status_code == 422
