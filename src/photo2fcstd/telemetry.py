import json
import os
import time


def view_event(v):
    s = v["shape"]
    return {"source": os.path.basename(v["source"]), "length_px": round(v["length_px"], 1),
            "elongation": v["elongation"], "angle_deg": v["angle_deg"],
            "rectangularity": round(s["rectangularity"], 4), "solidity": round(s["solidity"], 4),
            "hole_frac": round(s["hole_frac"], 4), "holes": len(s["holes"]),
            "stroke_px": round(s["stroke_px"], 2), "bbox": list(s["bbox"]),
            "min_over_max": round(min(s["bbox"]) / max(s["bbox"]), 4),
            "ellipse_aspect": round(s["ellipse"]["aspect"], 4) if s.get("ellipse") else None,
            "ellipse_rms": round(s["ellipse_rms"], 4), "round": s["round"], "roundish": s["roundish"],
            "symmetric": v["symmetric"], "stations": len(v["stations"])}


def spec_event(doc):
    outline, revolve = doc.get("outline"), doc.get("revolve")
    counts = {"circle": 0, "arc": 0, "line": 0}
    if outline:
        for loop in outline["loops"]:
            if loop["type"] == "circle":
                counts["circle"] += 1
            else:
                for e in loop["elements"]:
                    counts[e["type"]] += 1
    if revolve:
        counts["circle"] += len(revolve["holes"])
    return {"mm_per_px": doc["mm_per_px"], "scale_note": doc["scale_note"], "primitives": counts,
            "depth_px": round(outline["depth_px"], 2) if outline else None,
            "depth_note": outline["depth_note"] if outline else (revolve["note"] if revolve else None),
            "revolve_generic": bool(revolve and revolve.get("generic")),
            "revolve_R": round(revolve["R"], 1) if revolve else None,
            "station_counts": {k: len(v["stations"]) for k, v in doc["views"].items()}}


def build_event(report):
    sketches = report.get("sketches", {})
    return {"valid": report.get("valid"), "solids": report.get("solids"), "bbox": report.get("bbox"),
            "volume": report.get("volume"), "sketches": len(sketches),
            "geometry": sum(s.get("geometry", 0) for s in sketches.values()),
            "constraints": sum(s.get("constraints", 0) for s in sketches.values()),
            "fully_constrained": all(s["solve"] == 0 and not s["redundant"] and not s["conflicting"] for s in sketches.values()) if sketches else False,
            "error": report.get("error")}


class Run:
    def __init__(self, path):
        self.path = path
        self.events = {}

    def record(self, part, **fields):
        self.events.setdefault(part, {"part": part, "t": time.time()}).update(fields)

    def write(self):
        with open(self.path, "w") as fh:
            for part in sorted(self.events):
                fh.write(json.dumps(self.events[part]) + "\n")
        return self.path


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]
