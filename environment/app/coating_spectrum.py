#!/usr/bin/env python3
"""夜班光谱单入口。

这份程序最早只给正入射单层样品用。
"""
import hashlib
import json
import math
import os
import sys


def load_design(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_design(design):
    if not isinstance(design, dict):
        raise ValueError("design")
    for key in (
        "design_id",
        "incident_n",
        "substrate_n",
        "layers",
        "wavelengths_nm",
        "angles_deg",
    ):
        if key not in design:
            raise ValueError(key)
    if not isinstance(design["design_id"], str) or not design["design_id"]:
        raise ValueError("design_id")
    if not isinstance(design["layers"], list):
        raise ValueError("layers")
    if not isinstance(design["wavelengths_nm"], list):
        raise ValueError("wavelengths_nm")
    if not isinstance(design["angles_deg"], list):
        raise ValueError("angles_deg")
    return design


def normal_reflectance(n0, n1, ns, thickness, wavelength):
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - ns) / (n1 + ns)
    phase = 4.0 * math.pi * n1 * thickness / wavelength
    turn = complex(math.cos(phase), math.sin(phase))
    reflected = (r01 + r12 * turn) / (1.0 + r01 * r12 * turn)
    return abs(reflected) ** 2


def point_value(design, wavelength):
    n0 = float(design["incident_n"])
    ns = float(design["substrate_n"])
    layers = design["layers"]
    if layers:
        first = layers[0]
        reflectance = normal_reflectance(
            n0,
            float(first["n"]),
            ns,
            float(first["thickness_nm"]),
            wavelength,
        )
    else:
        reflectance = ((n0 - ns) / (n0 + ns)) ** 2
    return reflectance, reflectance, 1.0 - reflectance, 1.0 - reflectance


def build_points(design):
    points = []
    for wavelength in design["wavelengths_nm"]:
        for angle in design["angles_deg"]:
            r_s, r_p, t_s, t_p = point_value(design, float(wavelength))
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


def summary(points):
    return {
        "min_R_s": min(point["R_s"] for point in points),
        "max_R_s": max(point["R_s"] for point in points),
        "min_R_p": min(point["R_p"] for point in points),
        "max_R_p": max(point["R_p"] for point in points),
    }


def load_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["design_id"]: row for row in rows}


def cache_line(design, points):
    key = hashlib.sha256(design["design_id"].encode("utf-8")).hexdigest()
    record = {"key": key, "design_id": design["design_id"], "points": points}
    return json.dumps(record, ensure_ascii=False) + "\n"


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv):
    if len(argv) != 4:
        sys.stderr.write("用法: python3 /app/coating_spectrum.py <设计文件> <光谱单> <缓存>\n")
        return 2
    try:
        design = check_design(load_design(argv[1]))
        cache = load_cache(argv[3])
        record = cache.get(design["design_id"])
        if record is not None:
            points = record["points"]
            state = "hit"
        else:
            points = build_points(design)
            state = "miss"
        report = {
            "design_id": design["design_id"],
            "cache": state,
            "points": points,
            "summary": summary(points),
        }
        write_report(argv[2], report)
        if state == "miss":
            with open(argv[3], "a", encoding="utf-8") as handle:
                handle.write(cache_line(design, points))
    except OSError:
        sys.stderr.write("运行拒收: IO_FAILURE\n")
        return 2
    except (ValueError, KeyError, TypeError):
        sys.stderr.write("运行拒收: BAD_DESIGN\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
