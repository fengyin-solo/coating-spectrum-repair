#!/usr/bin/env python3
import hashlib
import json
import math
import os
import re
import sys
import tempfile


HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOP_KEYS = {"design_id", "incident_n", "substrate_n", "layers", "wavelengths_nm", "angles_deg"}
LAYER_KEYS = {"n", "thickness_nm"}
POINT_KEYS = {"wavelength_nm", "angle_deg", "R_s", "R_p", "T_s", "T_p"}
CACHE_KEYS = {"key", "design_id", "points"}


class Reject(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Reject("BAD_DESIGN")
        result[key] = value
    return result


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_design(design):
    if not isinstance(design, dict) or set(design) != TOP_KEYS:
        raise Reject("BAD_DESIGN")
    design_id = design["design_id"]
    if not isinstance(design_id, str) or not design_id:
        raise Reject("BAD_DESIGN")
    if len(design_id.encode("utf-8")) > 64:
        raise Reject("BAD_DESIGN")
    if any(ord(char) < 32 or ord(char) == 127 for char in design_id):
        raise Reject("BAD_DESIGN")
    incident = design["incident_n"]
    substrate = design["substrate_n"]
    if not is_number(incident) or not 1.0 <= incident <= 4.0:
        raise Reject("BAD_DESIGN")
    if not is_number(substrate) or not 1.0 <= substrate <= 4.0 or substrate < incident:
        raise Reject("BAD_DESIGN")
    layers = design["layers"]
    if not isinstance(layers, list) or not 0 <= len(layers) <= 20:
        raise Reject("BAD_DESIGN")
    for layer in layers:
        if not isinstance(layer, dict) or set(layer) != LAYER_KEYS:
            raise Reject("BAD_DESIGN")
        index = layer["n"]
        thickness = layer["thickness_nm"]
        if not is_number(index) or not 1.0 <= index <= 4.0 or index < incident:
            raise Reject("BAD_DESIGN")
        if not is_number(thickness) or not 1.0 <= thickness <= 2000.0:
            raise Reject("BAD_DESIGN")
    wavelengths = design["wavelengths_nm"]
    if not isinstance(wavelengths, list) or not 1 <= len(wavelengths) <= 12:
        raise Reject("BAD_DESIGN")
    for wavelength in wavelengths:
        if not is_number(wavelength) or not 300.0 <= wavelength <= 1200.0:
            raise Reject("BAD_DESIGN")
    for left, right in zip(wavelengths, wavelengths[1:]):
        if right <= left:
            raise Reject("BAD_DESIGN")
    angles = design["angles_deg"]
    if not isinstance(angles, list) or not 1 <= len(angles) <= 6:
        raise Reject("BAD_DESIGN")
    for angle in angles:
        if not is_number(angle) or not 0.0 <= angle <= 85.0:
            raise Reject("BAD_DESIGN")
    for left, right in zip(angles, angles[1:]):
        if right <= left:
            raise Reject("BAD_DESIGN")


def load_design(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            design = json.load(handle, object_pairs_hook=no_duplicate_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise Reject("BAD_DESIGN")
    validate_design(design)
    return design


def admittance(index, cosine, polarization):
    if polarization == "s":
        return index * cosine
    return index


def layer_matrix(index, thickness, wavelength, transverse, polarization):
    sine = transverse / index
    cosine = math.sqrt(max(0.0, 1.0 - sine * sine))
    phase = 2.0 * math.pi * index * thickness / wavelength
    eta = admittance(index, cosine, polarization)
    cos_phase = math.cos(phase)
    sin_phase = math.sin(phase)
    return (
        (cos_phase + 0j, 1j * sin_phase / eta),
        (1j * eta * sin_phase, cos_phase + 0j),
    )


def multiply(left, right):
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def spectrum_point(design, wavelength, angle_deg, polarization):
    incident = design["incident_n"]
    substrate = design["substrate_n"]
    angle = math.radians(angle_deg)
    transverse = incident * math.sin(angle)
    incident_cosine = math.cos(angle)
    total = ((1 + 0j, 0j), (0j, 1 + 0j))
    for layer in design["layers"]:
        single = layer_matrix(layer["n"], layer["thickness_nm"], wavelength, transverse, polarization)
        total = multiply(single, total)
    substrate_sine = transverse / substrate
    substrate_cosine = math.sqrt(max(0.0, 1.0 - substrate_sine * substrate_sine))
    eta_substrate = admittance(substrate, substrate_cosine, polarization)
    eta_incident = admittance(incident, incident_cosine, polarization)
    b_value = total[0][0] + total[0][1] * eta_substrate
    c_value = total[1][0] + total[1][1] * eta_substrate
    reflection = (eta_incident * b_value - c_value) / (eta_incident * b_value + c_value)
    transmission = 2.0 * eta_incident / (eta_incident * b_value + c_value)
    reflectance = abs(reflection) ** 2
    transmittance = abs(transmission) ** 2
    return reflectance, transmittance


def compute_points(design):
    points = []
    for wavelength in design["wavelengths_nm"]:
        for angle in design["angles_deg"]:
            r_s, t_s = spectrum_point(design, wavelength, angle, "s")
            r_p, t_p = spectrum_point(design, wavelength, angle, "p")
            points.append(
                {
                    "wavelength_nm": wavelength,
                    "angle_deg": angle,
                    "R_s": r_s,
                    "R_p": r_p,
                    "T_s": t_s,
                    "T_p": t_p,
                }
            )
    return points


def canonical_design(design):
    return json.dumps(
        design, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def design_key(design):
    return hashlib.sha256(design["design_id"].encode("utf-8")).hexdigest()


def valid_point(point):
    if not isinstance(point, dict) or set(point) != POINT_KEYS:
        return False
    values = [point[name] for name in ("wavelength_nm", "angle_deg", "R_s", "R_p", "T_s", "T_p")]
    if not all(is_number(value) for value in values):
        return False
    if not 0.0 <= point["R_s"] <= 1.0 or not 0.0 <= point["R_p"] <= 1.0:
        return False
    if not 0.0 <= point["T_s"] <= 1.0 or not 0.0 <= point["T_p"] <= 1.0:
        return False
    return abs(point["R_s"] + point["T_s"] - 1.0) <= 1e-6 and abs(point["R_p"] + point["T_p"] - 1.0) <= 1e-6


def read_cache(path):
    if not os.path.exists(path):
        return [], b""
    if not os.path.isfile(path):
        raise Reject("IO_FAILURE")
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        raise Reject("IO_FAILURE")
    if not raw:
        return [], raw
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise Reject("BAD_CACHE")
    records = []
    seen = set()
    for line in text.splitlines():
        if not line:
            raise Reject("BAD_CACHE")
        try:
            record = json.loads(line, object_pairs_hook=no_duplicate_object)
        except (json.JSONDecodeError, Reject):
            raise Reject("BAD_CACHE")
        if not isinstance(record, dict) or set(record) != CACHE_KEYS:
            raise Reject("BAD_CACHE")
        key = record["key"]
        if not isinstance(key, str) or not HEX64.fullmatch(key):
            raise Reject("BAD_CACHE")
        if not isinstance(record["design_id"], str) or not record["design_id"]:
            raise Reject("BAD_CACHE")
        points = record["points"]
        if not isinstance(points, list) or len(points) == 0:
            raise Reject("BAD_CACHE")
        if not all(valid_point(point) for point in points):
            raise Reject("BAD_CACHE")
        if key in seen:
            raise Reject("BAD_CACHE")
        seen.add(key)
        records.append(record)
    return records, raw


def cached_points(design, records):
    key = design_key(design)
    expected = [(w, a) for w in design["wavelengths_nm"] for a in design["angles_deg"]]
    for record in records:
        if record["key"] != key:
            continue
        points = record["points"]
        if len(points) != len(expected):
            raise Reject("BAD_CACHE")
        for point, (wavelength, angle) in zip(points, expected):
            if point["wavelength_nm"] != wavelength or point["angle_deg"] != angle:
                raise Reject("BAD_CACHE")
        return points
    return None


def summary(points):
    return {
        "min_R_s": min(point["R_s"] for point in points),
        "max_R_s": max(point["R_s"] for point in points),
        "min_R_p": min(point["R_p"] for point in points),
        "max_R_p": max(point["R_p"] for point in points),
    }


def build_spectrum(design, records, raw_cache):
    hit = cached_points(design, records)
    if hit is not None:
        points = hit
        cache_state = "hit"
    else:
        points = compute_points(design)
        cache_state = "miss"
    report = {
        "design_id": design["design_id"],
        "cache": cache_state,
        "points": points,
        "summary": summary(points),
    }
    spectrum = (
        json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    cache = raw_cache
    if cache_state == "miss":
        record = {"key": design_key(design), "design_id": design["design_id"], "points": points}
        line = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if cache and not cache.endswith(b"\n"):
            cache += b"\n"
        cache += line
    return spectrum, cache


def old_state(path):
    if not os.path.exists(path):
        return None
    if not os.path.isfile(path):
        raise Reject("IO_FAILURE")
    try:
        with open(path, "rb") as handle:
            return handle.read(), os.stat(path).st_mode & 0o777
    except OSError:
        raise Reject("IO_FAILURE")


def check_target(path):
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise Reject("IO_FAILURE")
    if os.path.isdir(path):
        raise Reject("IO_FAILURE")
    if os.path.exists(path) and not os.path.isfile(path):
        raise Reject("IO_FAILURE")


def write_temp(directory, data, mode):
    handle, path = tempfile.mkstemp(prefix=".spectrum-", dir=directory)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, mode)
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise
    return path


def restore(path, state):
    if state is None:
        if os.path.exists(path):
            os.unlink(path)
        return
    data, mode = state
    replacement = write_temp(os.path.dirname(os.path.abspath(path)), data, mode)
    os.replace(replacement, path)


def publish(spectrum_path, cache_path, spectrum_bytes, cache_bytes):
    if os.path.abspath(spectrum_path) == os.path.abspath(cache_path):
        raise Reject("IO_FAILURE")
    check_target(spectrum_path)
    check_target(cache_path)
    spectrum_state = old_state(spectrum_path)
    cache_state = old_state(cache_path)
    spectrum_temp = None
    cache_temp = None
    try:
        spectrum_temp = write_temp(os.path.dirname(os.path.abspath(spectrum_path)), spectrum_bytes, 0o644)
        cache_temp = write_temp(os.path.dirname(os.path.abspath(cache_path)), cache_bytes, 0o644)
        os.replace(spectrum_temp, spectrum_path)
        spectrum_temp = None
        os.replace(cache_temp, cache_path)
        cache_temp = None
    except Exception:
        restore(spectrum_path, spectrum_state)
        restore(cache_path, cache_state)
        raise
    finally:
        for path in (spectrum_temp, cache_temp):
            if path and os.path.exists(path):
                os.unlink(path)


def main(argv):
    if len(argv) != 4:
        sys.stderr.write("用法: python3 /app/coating_spectrum.py <设计文件> <光谱单> <缓存>\n")
        return 2
    try:
        design = load_design(argv[1])
        records, raw_cache = read_cache(argv[3])
        spectrum_bytes, cache_bytes = build_spectrum(design, records, raw_cache)
        publish(argv[2], argv[3], spectrum_bytes, cache_bytes)
    except Reject as error:
        sys.stderr.write("运行拒收: %s\n" % error.code)
        return 2
    except Exception:
        sys.stderr.write("运行拒收: IO_FAILURE\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
