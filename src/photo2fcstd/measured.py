"""Dimensions read off a caliper, carried into the document as named, labelled entries.

A photograph of a part beside a caliper holds the one thing the pipeline cannot see: a real
length. Recording those readings against what they measure turns every dimension in the sheet
from pixels into millimetres, and leaves a note saying where the number came from.
"""
import os
import re

ENTRY = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([0-9]*\.?[0-9]+)\s*(?:mm)?\s*(?:@\s*(\S+))?\s*$")


def parse(text):
    """`name=value[mm][@photo]` into a labelled measurement, or None if it does not parse."""
    found = ENTRY.match(text or "")
    if not found:
        return None
    name, value, source = found.groups()
    return {"name": name, "mm": float(value), "source": source}


def parse_all(entries):
    out, bad = [], []
    for text in entries or []:
        got = parse(text)
        (out if got else bad).append(got or text)
    if bad:
        raise ValueError("could not read %s; expected name=value, for example body_width=12.53"
                         % ", ".join(repr(b) for b in bad))
    names = [m["name"] for m in out]
    duplicated = {n for n in names if names.count(n) > 1}
    if duplicated:
        raise ValueError("measured twice: %s" % ", ".join(sorted(duplicated)))
    return out


def rows(measurements):
    """Spreadsheet rows: the name, the value in millimetres, and where it was measured."""
    out = []
    for m in measurements:
        where = os.path.basename(m["source"]) if m.get("source") else "caliper"
        out.append((m["name"], round(m["mm"], 3), "measured, %s" % where))
    return out


def scale_from(measurements, name, length_px):
    """Millimetres per pixel implied by one measurement of a known pixel length."""
    for m in measurements:
        if m["name"] == name and length_px:
            return m["mm"] / float(length_px)
    return None
