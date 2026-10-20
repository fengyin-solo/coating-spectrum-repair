"""镀膜车间的 s/p 光谱单与跨班次缓存。"""

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


APP = Path(os.environ.get("TASK_APP", "/app/coating_spectrum.py"))
CALL_TIMEOUT_SEC = 60
CLOSE = {"rel_tol": 1e-9, "abs_tol": 1e-12}


def _design(
    design_id="COAT-550-A",
    incident_n=1.0,
    substrate_n=1.52,
    layers=None,
    wavelengths_nm=None,
    angles_deg=None,
):
    return {
        "design_id": design_id,
        "incident_n": incident_n,
        "substrate_n": substrate_n,
        "layers": layers if layers is not None else [],
        "wavelengths_nm": wavelengths_nm if wavelengths_nm is not None else [550.0],
        "angles_deg": angles_deg if angles_deg is not None else [0.0],
    }


def _quarter_wave_stack(periods=4, high=2.3, low=1.45, center=550.0):
    layers = []
    for _ in range(periods):
        layers.append({"n": high, "thickness_nm": center / (4.0 * high)})
        layers.append({"n": low, "thickness_nm": center / (4.0 * low)})
    return layers


def _run(tmp_path, design, tag, spectrum=None, cache=None):
    design_path = tmp_path / ("%s-design.json" % tag)
    design_path.write_text(json.dumps(design, ensure_ascii=False), encoding="utf-8")
    spectrum_path = spectrum if spectrum is not None else tmp_path / ("%s-spectrum.json" % tag)
    cache_path = cache if cache is not None else tmp_path / ("%s-cache.jsonl" % tag)
    finished = subprocess.run(
        [sys.executable, str(APP), str(design_path), str(spectrum_path), str(cache_path)],
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT_SEC,
    )
    return finished, spectrum_path, cache_path


def _report(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _admittance(index, angle, polarization):
    if polarization == "s":
        return index * math.cos(angle)
    return index / math.cos(angle)


def _fresnel_r(index_a, index_b, angle_a, angle_b, polarization):
    eta_a = _admittance(index_a, angle_a, polarization)
    eta_b = _admittance(index_b, angle_b, polarization)
    return (eta_a - eta_b) / (eta_a + eta_b)


def _airy_single_r(index_0, index_1, index_s, angle_deg, thickness, wavelength, polarization):
    angle_0 = math.radians(angle_deg)
    transverse = index_0 * math.sin(angle_0)
    angle_1 = math.asin(transverse / index_1)
    angle_s = math.asin(transverse / index_s)
    r_01 = _fresnel_r(index_0, index_1, angle_0, angle_1, polarization)
    r_12 = _fresnel_r(index_1, index_s, angle_1, angle_s, polarization)
    beta = 2.0 * math.pi * index_1 * thickness * math.cos(angle_1) / wavelength
    turn = complex(math.cos(2.0 * beta), math.sin(2.0 * beta))
    reflected = (r_01 + r_12 * turn) / (1.0 + r_01 * r_12 * turn)
    return abs(reflected) ** 2


def _stack_r(design, wavelength, angle_deg, polarization):
    index_0 = design["incident_n"]
    index_s = design["substrate_n"]
    angle_0 = math.radians(angle_deg)
    transverse = index_0 * math.sin(angle_0)
    angle_s = math.asin(transverse / index_s)
    eta = _admittance(index_s, angle_s, polarization)
    for layer in reversed(design["layers"]):
        index = layer["n"]
        angle = math.asin(transverse / index)
        thickness = layer["thickness_nm"]
        eta_layer = _admittance(index, angle, polarization)
        beta = 2.0 * math.pi * index * thickness * math.cos(angle) / wavelength
        cosine = math.cos(beta)
        sine = math.sin(beta)
        eta = eta_layer * (eta * cosine + 1j * eta_layer * sine) / (
            eta_layer * cosine + 1j * eta * sine
        )
    eta_0 = _admittance(index_0, angle_0, polarization)
    reflected = (eta_0 - eta) / (eta_0 + eta)
    return abs(reflected) ** 2


def _point(report, wavelength, angle):
    for item in report["points"]:
        if item["wavelength_nm"] == wavelength and item["angle_deg"] == angle:
            return item
    raise AssertionError("missing point %s %s" % (wavelength, angle))


def _bad_designs():
    base = _design(layers=[{"n": 2.05, "thickness_nm": 120.0}], wavelengths_nm=[450.0, 550.0], angles_deg=[0.0, 45.0])
    cases = []

    missing = copy.deepcopy(base)
    missing.pop("layers")
    cases.append((missing, "MISSING-TOP"))

    extra = copy.deepcopy(base)
    extra["note"] = "x"
    cases.append((extra, "EXTRA-TOP"))

    empty_id = copy.deepcopy(base)
    empty_id["design_id"] = ""
    cases.append((empty_id, "EMPTY-ID"))

    long_id = copy.deepcopy(base)
    long_id["design_id"] = "D" * 65
    cases.append((long_id, "LONG-ID"))

    control_id = copy.deepcopy(base)
    control_id["design_id"] = "A\tB"
    cases.append((control_id, "CONTROL-ID"))

    low_incident = copy.deepcopy(base)
    low_incident["incident_n"] = 0.99
    cases.append((low_incident, "INCIDENT-LOW"))

    high_substrate = copy.deepcopy(base)
    high_substrate["substrate_n"] = 4.01
    cases.append((high_substrate, "SUBSTRATE-HIGH"))

    below_substrate = copy.deepcopy(base)
    below_substrate["incident_n"] = 2.0
    below_substrate["substrate_n"] = 1.5
    cases.append((below_substrate, "SUBSTRATE-BELOW"))

    thin_layer = copy.deepcopy(base)
    thin_layer["layers"][0]["thickness_nm"] = 0.99
    cases.append((thin_layer, "THICK-LOW"))

    thick_layer = copy.deepcopy(base)
    thick_layer["layers"][0]["thickness_nm"] = 2000.01
    cases.append((thick_layer, "THICK-HIGH"))

    low_layer = copy.deepcopy(base)
    low_layer["incident_n"] = 2.0
    low_layer["substrate_n"] = 2.5
    low_layer["layers"][0]["n"] = 1.8
    cases.append((low_layer, "LAYER-BELOW"))

    missing_layer_key = copy.deepcopy(base)
    missing_layer_key["layers"][0].pop("thickness_nm")
    cases.append((missing_layer_key, "LAYER-MISSING"))

    extra_layer_key = copy.deepcopy(base)
    extra_layer_key["layers"][0]["note"] = "x"
    cases.append((extra_layer_key, "LAYER-EXTRA"))

    descending_wave = copy.deepcopy(base)
    descending_wave["wavelengths_nm"] = [550.0, 450.0]
    cases.append((descending_wave, "WAVE-ORDER"))

    low_wave = copy.deepcopy(base)
    low_wave["wavelengths_nm"] = [299.9]
    cases.append((low_wave, "WAVE-LOW"))

    high_wave = copy.deepcopy(base)
    high_wave["wavelengths_nm"] = [1200.1]
    cases.append((high_wave, "WAVE-HIGH"))

    descending_angle = copy.deepcopy(base)
    descending_angle["angles_deg"] = [45.0, 0.0]
    cases.append((descending_angle, "ANGLE-ORDER"))

    low_angle = copy.deepcopy(base)
    low_angle["angles_deg"] = [-0.1]
    cases.append((low_angle, "ANGLE-LOW"))

    high_angle = copy.deepcopy(base)
    high_angle["angles_deg"] = [85.1]
    cases.append((high_angle, "ANGLE-HIGH"))

    bool_number = copy.deepcopy(base)
    bool_number["incident_n"] = True
    cases.append((bool_number, "BOOL-NUMBER"))

    too_many_layers = copy.deepcopy(base)
    too_many_layers["layers"] = [{"n": 2.0, "thickness_nm": 100.0} for _ in range(21)]
    cases.append((too_many_layers, "LAYER-COUNT"))

    too_many_waves = copy.deepcopy(base)
    too_many_waves["wavelengths_nm"] = [300.0 + 10.0 * index for index in range(13)]
    cases.append((too_many_waves, "WAVE-COUNT"))

    too_many_angles = copy.deepcopy(base)
    too_many_angles["angles_deg"] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    cases.append((too_many_angles, "ANGLE-COUNT"))

    empty_layers_list = copy.deepcopy(base)
    empty_layers_list["layers"] = {}
    cases.append((empty_layers_list, "LAYER-TYPE"))

    return [(design, label, index) for index, (design, label) in enumerate(cases, 1)]


def test_normal_incidence_bare_substrate_reflectance_matches_fresnel(tmp_path):
    """正入射裸基底的反射率应回到 Fresnel 结果；题面：「分光计会在正入射、斜入射、单层、多层和两种偏振上分别核对」。"""
    design = _design(substrate_n=1.52)
    finished, spectrum_path, _ = _run(tmp_path, design, "normal-bare")
    item = _point(_report(spectrum_path), 550.0, 0.0)
    expected = ((1.0 - 1.52) / (1.0 + 1.52)) ** 2
    assert finished.returncode == 0
    assert math.isclose(item["R_s"], expected, **CLOSE)
    assert math.isclose(item["R_p"], expected, **CLOSE)


def test_oblique_bare_substrate_matches_fresnel_for_both_polarizations(tmp_path):
    """六十度裸基底的 s/p 反射率应分别对上 Fresnel 结果；题面：「s 偏振和 p 偏振要分开算」。"""
    design = _design(substrate_n=1.52, angles_deg=[60.0])
    finished, spectrum_path, _ = _run(tmp_path, design, "oblique-bare")
    item = _point(_report(spectrum_path), 550.0, 60.0)
    angle_0 = math.radians(60.0)
    angle_s = math.asin(math.sin(angle_0) / 1.52)
    expected_s = _fresnel_r(1.0, 1.52, angle_0, angle_s, "s") ** 2
    expected_p = _fresnel_r(1.0, 1.52, angle_0, angle_s, "p") ** 2
    assert finished.returncode == 0
    assert math.isclose(item["R_s"], expected_s, **CLOSE)
    assert math.isclose(item["R_p"], expected_p, **CLOSE)


def test_oblique_single_layer_matches_independent_airy_sum(tmp_path):
    """题面：「斜入射时膜层里的相位厚度跟着折射角走」，不能拿入射角代替。"""
    design = _design(
        substrate_n=1.52,
        layers=[{"n": 2.1, "thickness_nm": 95.0}],
        wavelengths_nm=[532.0],
        angles_deg=[60.0],
    )
    finished, spectrum_path, _ = _run(tmp_path, design, "airy-single")
    item = _point(_report(spectrum_path), 532.0, 60.0)
    expected_s = _airy_single_r(1.0, 2.1, 1.52, 60.0, 95.0, 532.0, "s")
    expected_p = _airy_single_r(1.0, 2.1, 1.52, 60.0, 95.0, 532.0, "p")
    assert finished.returncode == 0
    assert math.isclose(item["R_s"], expected_s, **CLOSE)
    assert math.isclose(item["R_p"], expected_p, **CLOSE)


def test_quarter_wave_single_layer_at_normal_incidence(tmp_path):
    """正入射单层样品的反射率应与原有解析结果一致；题面：「分光计会在正入射、斜入射、单层、多层和两种偏振上分别核对」。"""
    wavelength = 550.0
    design = _design(
        substrate_n=1.5,
        layers=[{"n": 2.0, "thickness_nm": wavelength / 8.0}],
        wavelengths_nm=[wavelength],
    )
    finished, spectrum_path, _ = _run(tmp_path, design, "qw-single")
    item = _point(_report(spectrum_path), wavelength, 0.0)
    expected = ((1.0 * 1.5 - 2.0 * 2.0) / (1.0 * 1.5 + 2.0 * 2.0)) ** 2
    assert finished.returncode == 0
    assert math.isclose(item["R_s"], expected, **CLOSE)
    assert math.isclose(item["R_p"], expected, **CLOSE)


def test_quarter_wave_stack_matches_equivalent_admittance(tmp_path):
    """四周期四分之一波长堆在中心波长处应等于等效导纳的解析反射率；题面：「叠了多层以后，数值和分光计对不上」。"""
    wavelength = 550.0
    periods = 4
    design = _design(
        substrate_n=1.52,
        layers=_quarter_wave_stack(periods=periods),
        wavelengths_nm=[wavelength],
    )
    finished, spectrum_path, _ = _run(tmp_path, design, "qw-stack")
    item = _point(_report(spectrum_path), wavelength, 0.0)
    equivalent = 1.52 * (2.3 / 1.45) ** (2 * periods)
    expected = ((1.0 - equivalent) / (1.0 + equivalent)) ** 2
    assert finished.returncode == 0
    assert math.isclose(item["R_s"], expected, **CLOSE)
    assert math.isclose(item["R_p"], expected, **CLOSE)


def test_brewster_angle_suppresses_p_reflection(tmp_path):
    """裸基底在 Brewster 角上 p 偏振反射应趋近于零；题面：「s 偏振和 p 偏振要分开算」。"""
    substrate = 1.52
    angle = math.degrees(math.atan(substrate / 1.0))
    design = _design(substrate_n=substrate, angles_deg=[angle])
    finished, spectrum_path, _ = _run(tmp_path, design, "brewster")
    item = _point(_report(spectrum_path), 550.0, angle)
    assert finished.returncode == 0
    assert math.isclose(item["R_p"], 0.0, rel_tol=0.0, abs_tol=1e-10)
    assert item["R_s"] > 0.1


def test_s_and_p_polarizations_differ_at_oblique_single_layer(tmp_path):
    """同一个斜入射点不能把 s/p 两种偏振算成同一个数；题面：「s 偏振和 p 偏振要分开算」。"""
    design = _design(
        substrate_n=1.52,
        layers=[{"n": 2.1, "thickness_nm": 95.0}],
        wavelengths_nm=[532.0],
        angles_deg=[60.0],
    )
    finished, spectrum_path, _ = _run(tmp_path, design, "pol-diff")
    item = _point(_report(spectrum_path), 532.0, 60.0)
    assert finished.returncode == 0
    assert abs(item["R_s"] - item["R_p"]) > 1e-3


def test_lossless_energy_conservation_for_every_point(tmp_path):
    """题面：「所以每个点的反射率和透射率加起来是一」，逐点核对能量守恒。"""
    design = _design(
        substrate_n=1.52,
        layers=[
            {"n": 2.3, "thickness_nm": 59.8},
            {"n": 1.45, "thickness_nm": 94.8},
            {"n": 2.3, "thickness_nm": 59.8},
        ],
        wavelengths_nm=[420.0, 550.0, 700.0],
        angles_deg=[0.0, 35.0, 60.0],
    )
    finished, spectrum_path, _ = _run(tmp_path, design, "energy")
    report = _report(spectrum_path)
    assert finished.returncode == 0
    assert all(
        math.isclose(item["R_s"] + item["T_s"], 1.0, **CLOSE) for item in report["points"]
    )
    assert all(
        math.isclose(item["R_p"] + item["T_p"], 1.0, **CLOSE) for item in report["points"]
    )


def test_tilted_multilayer_p_polarization_matches_independent_stack_sum(tmp_path):
    """倾斜多层膜的 p 偏振反射率应独立复算一致；题面：「斜着测、偏振分开看、叠了多层以后，数值和分光计对不上」。"""
    design = _design(
        substrate_n=1.52,
        layers=[
            {"n": 2.3, "thickness_nm": 62.0},
            {"n": 1.45, "thickness_nm": 101.0},
            {"n": 2.05, "thickness_nm": 77.0},
        ],
        wavelengths_nm=[620.0],
        angles_deg=[55.0],
    )
    finished, spectrum_path, _ = _run(tmp_path, design, "tilted-stack")
    item = _point(_report(spectrum_path), 620.0, 55.0)
    expected = _stack_r(design, 620.0, 55.0, "p")
    assert finished.returncode == 0
    assert math.isclose(item["R_p"], expected, **CLOSE)
    assert math.isclose(item["T_p"], 1.0 - expected, **CLOSE)


def test_output_schema_and_point_order(tmp_path):
    """`points` 先按 `wavelengths_nm` 的顺序、再按 `angles_deg` 的顺序展开。"""
    design = _design(wavelengths_nm=[450.0, 550.0], angles_deg=[0.0, 45.0])
    finished, spectrum_path, _ = _run(tmp_path, design, "schema")
    report = _report(spectrum_path)
    assert finished.returncode == 0
    assert list(report) == ["design_id", "cache", "points", "summary"]
    assert list(report["summary"]) == ["min_R_s", "max_R_s", "min_R_p", "max_R_p"]
    assert [point["wavelength_nm"] for point in report["points"]] == [450.0, 450.0, 550.0, 550.0]
    assert [point["angle_deg"] for point in report["points"]] == [0.0, 45.0, 0.0, 45.0]
    assert list(report["points"][0]) == [
        "wavelength_nm",
        "angle_deg",
        "R_s",
        "R_p",
        "T_s",
        "T_p",
    ]


def test_output_is_compact_json_with_one_trailing_newline(tmp_path):
    """题面：「光谱单是 UTF-8 紧凑 JSON，分隔符后不留空格，末尾一个换行」。"""
    finished, spectrum_path, _ = _run(tmp_path, _design(), "compact")
    raw = spectrum_path.read_bytes()
    assert finished.returncode == 0
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b": " not in raw
    assert b", " not in raw


def test_summary_extrema_match_reported_points(tmp_path):
    """`summary` 四个键依次是 `min_R_s`、`max_R_s`、`min_R_p`、`max_R_p`。"""
    design = _design(
        substrate_n=1.52,
        layers=[{"n": 2.1, "thickness_nm": 95.0}],
        wavelengths_nm=[420.0, 550.0],
        angles_deg=[0.0, 60.0],
    )
    finished, spectrum_path, _ = _run(tmp_path, design, "summary")
    report = _report(spectrum_path)
    assert finished.returncode == 0
    assert report["summary"]["min_R_s"] == min(point["R_s"] for point in report["points"])
    assert report["summary"]["max_R_s"] == max(point["R_s"] for point in report["points"])
    assert report["summary"]["min_R_p"] == min(point["R_p"] for point in report["points"])
    assert report["summary"]["max_R_p"] == max(point["R_p"] for point in report["points"])


def test_cache_first_run_misses_and_second_run_hits_without_appending(tmp_path):
    """重复送检应命中缓存；题面：「点集和上一遍逐值相同，缓存行数不增加」。"""
    design = _design()
    first, spectrum_path, cache_path = _run(tmp_path, design, "cache-once")
    first_points = _report(spectrum_path)["points"]
    first_lines = cache_path.read_bytes()
    second, spectrum_path, cache_path = _run(tmp_path, design, "cache-once")
    assert first.returncode == 0
    assert second.returncode == 0
    assert _report(spectrum_path)["cache"] == "hit"
    assert _report(spectrum_path)["points"] == first_points
    assert cache_path.read_bytes() == first_lines
    assert len(_cache_lines(cache_path)) == 1


def test_changed_thickness_with_same_design_id_is_a_cache_miss(tmp_path):
    """题面：「设计里任何一个字段变了」，缓存键必须跟着变而不能套用旧点集。"""
    original = _design(design_id="FL-552-07", layers=[{"n": 2.1, "thickness_nm": 95.0}])
    changed = _design(design_id="FL-552-07", layers=[{"n": 2.1, "thickness_nm": 130.0}])
    first, spectrum_path, cache_path = _run(tmp_path, original, "cache-change")
    first_points = _report(spectrum_path)["points"]
    second, spectrum_path, cache_path = _run(tmp_path, changed, "cache-change")
    assert first.returncode == 0
    assert second.returncode == 0
    assert _report(spectrum_path)["cache"] == "miss"
    assert _report(spectrum_path)["points"] != first_points
    assert len(_cache_lines(cache_path)) == 2


def test_corrupt_cache_line_is_rejected_without_touching_outputs(tmp_path):
    """同一个 `key` 出现两次、某一行不是 JSON 对象、键集合不对，或者 `points` 与本次设计的波长/角度网格对不上，都按缓存损坏处理。"""
    design = _design()
    first, spectrum_path, cache_path = _run(tmp_path, design, "cache-check")
    spectrum_before = spectrum_path.read_bytes()
    cache_before = cache_path.read_bytes()
    cache_path.write_bytes(b"{not a json object}\n")
    second, spectrum_path, cache_path = _run(tmp_path, design, "cache-check")
    assert first.returncode == 0
    assert second.returncode == 2
    assert second.stderr.startswith("运行拒收: BAD_CACHE")
    assert spectrum_path.read_bytes() == spectrum_before
    assert cache_path.read_bytes() != cache_before


def test_duplicate_cache_key_is_rejected(tmp_path):
    """同一个 `key` 出现两次、某一行不是 JSON 对象、键集合不对，或者 `points` 与本次设计的波长/角度网格对不上，都按缓存损坏处理。"""
    design = _design()
    first, spectrum_path, cache_path = _run(tmp_path, design, "cache-dup")
    line = cache_path.read_text(encoding="utf-8")
    cache_path.write_text(line + line, encoding="utf-8")
    spectrum_before = spectrum_path.read_bytes()
    second, spectrum_path, cache_path = _run(tmp_path, design, "cache-dup")
    assert first.returncode == 0
    assert second.returncode == 2
    assert second.stderr.startswith("运行拒收: BAD_CACHE")
    assert spectrum_path.read_bytes() == spectrum_before


def test_cache_grid_mismatch_is_rejected(tmp_path):
    """`points` 与本次设计的波长/角度网格对不上。"""
    design = _design(wavelengths_nm=[450.0, 550.0], angles_deg=[0.0])
    first, spectrum_path, cache_path = _run(tmp_path, design, "cache-grid")
    record = _cache_lines(cache_path)[0]
    record["points"][0]["angle_deg"] = 10.0
    cache_path.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    second, spectrum_path, _ = _run(tmp_path, design, "cache-grid")
    assert first.returncode == 0
    assert second.returncode == 2
    assert second.stderr.startswith("运行拒收: BAD_CACHE")


@pytest.mark.parametrize("design,case_id,case_no", _bad_designs())
def test_invalid_design_is_rejected(tmp_path, design, case_id, case_no):
    """前面的输入规则有一条不满足时，往标准错误打一行以 `运行拒收: BAD_DESIGN` 开头的话，退出码 2。"""
    labels = {
        "MISSING-TOP": "顶层键缺失",
        "EXTRA-TOP": "顶层键多出",
        "EMPTY-ID": "设计编号为空",
        "LONG-ID": "设计编号超长",
        "CONTROL-ID": "设计编号带控制字符",
        "INCIDENT-LOW": "入射折射率越界",
        "SUBSTRATE-HIGH": "基底折射率越界",
        "SUBSTRATE-BELOW": "基底低于入射介质",
        "THICK-LOW": "层厚低于下限",
        "THICK-HIGH": "层厚高于上限",
        "LAYER-BELOW": "膜层折射率低于入射介质",
        "LAYER-MISSING": "膜层缺键",
        "LAYER-EXTRA": "膜层多键",
        "WAVE-ORDER": "波长不递增",
        "WAVE-LOW": "波长低于下限",
        "WAVE-HIGH": "波长高于上限",
        "ANGLE-ORDER": "角度不递增",
        "ANGLE-LOW": "角度低于下限",
        "ANGLE-HIGH": "角度高于上限",
        "BOOL-NUMBER": "布尔值冒充数字",
        "LAYER-COUNT": "膜层数量超限",
        "WAVE-COUNT": "波长数量超限",
        "ANGLE-COUNT": "角度数量超限",
        "LAYER-TYPE": "膜层容器类型错误",
    }
    print("CASE %s %s" % (case_id, labels[case_id]))
    finished, spectrum_path, cache_path = _run(tmp_path, design, "case-%02d" % case_no)
    assert finished.returncode == 2
    assert finished.stderr.startswith("运行拒收: BAD_DESIGN")
    assert not spectrum_path.exists()
    assert not cache_path.exists()


def test_duplicate_json_key_is_rejected(tmp_path):
    """设计文件结构不对、字段缺失或多出、数值越界、波长或角度不递增，都往标准错误打一行以 `运行拒收: BAD_DESIGN` 开头的话，退出码 2。"""
    raw = '{"design_id":"D","design_id":"E","incident_n":1.0,"substrate_n":1.5,"layers":[],"wavelengths_nm":[550.0],"angles_deg":[0.0]}'
    design_path = tmp_path / "duplicate.json"
    design_path.write_text(raw, encoding="utf-8")
    spectrum_path = tmp_path / "duplicate-spectrum.json"
    cache_path = tmp_path / "duplicate-cache.jsonl"
    finished = subprocess.run(
        [sys.executable, str(APP), str(design_path), str(spectrum_path), str(cache_path)],
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT_SEC,
    )
    assert finished.returncode == 2
    assert finished.stderr.startswith("运行拒收: BAD_DESIGN")
    assert not spectrum_path.exists()
    assert not cache_path.exists()


def test_minimum_design_boundaries_are_accepted(tmp_path):
    """`layers`、`wavelengths_nm`、`angles_deg` 的数量下限应被接受。"""
    design = _design(layers=[], wavelengths_nm=[300.0], angles_deg=[0.0])
    finished, spectrum_path, cache_path = _run(tmp_path, design, "min-bound")
    report = _report(spectrum_path)
    assert finished.returncode == 0
    assert len(report["points"]) == 1
    assert cache_path.is_file()


def test_maximum_design_boundaries_are_accepted(tmp_path):
    """`layers`、`wavelengths_nm`、`angles_deg` 的数量上限应被接受。"""
    layers = [{"n": 2.0, "thickness_nm": 100.0} for _ in range(20)]
    wavelengths = [300.0 + 80.0 * index for index in range(12)]
    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 85.0]
    design = _design(layers=layers, wavelengths_nm=wavelengths, angles_deg=angles)
    finished, spectrum_path, _ = _run(tmp_path, design, "max-bound")
    report = _report(spectrum_path)
    assert finished.returncode == 0
    assert len(report["points"]) == 72


def test_wrong_argument_count_writes_usage_to_stderr(tmp_path):
    """题面：「参数数目不对时」，标准错误上留一行以 `用法: ` 开头的话，退出码 2。"""
    design_path = tmp_path / "args.json"
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, str(APP), str(design_path)],
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT_SEC,
    )
    too_many = subprocess.run(
        [
            sys.executable,
            str(APP),
            str(design_path),
            str(tmp_path / "args-spectrum.json"),
            str(tmp_path / "args-cache.jsonl"),
            "extra",
        ],
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT_SEC,
    )
    assert finished.returncode == 2
    assert finished.stderr.startswith("用法: ")
    assert too_many.returncode == 2
    assert too_many.stderr.startswith("用法: ")


def test_spectrum_target_directory_is_rejected_without_touching_cache(tmp_path):
    """目标是个目录时，写 `运行拒收: IO_FAILURE`，两个目标文件都保持调用前的字节。"""
    design = _design()
    cache_path = tmp_path / "io-cache.jsonl"
    cache_path.write_bytes(b"")
    spectrum_path = tmp_path / "io-spectrum.json"
    spectrum_path.mkdir()
    finished, _, _ = _run(tmp_path, design, "io", spectrum=spectrum_path, cache=cache_path)
    assert finished.returncode == 2
    assert finished.stderr.startswith("运行拒收: IO_FAILURE")
    assert cache_path.read_bytes() == b""
    assert spectrum_path.is_dir()
