"""Say once when a learned component gives up and the geometric rule takes over.

Every model here returns None on any failure and the caller quietly falls back. That is the right
behaviour and the wrong silence: `P2F_VIEW_MODEL=1` once set the model path to a file named "1", so
`load()` returned None, the view model fell back to the rules it was meant to replace, and the A/B
measured its control twice and reported +0.0000 on all 295 parts. One line at the first fallback
would have caught it immediately.
"""
import os
import sys

SILENT = os.environ.get("P2F_QUIET_FALLBACK") == "1"
_SEEN = set()
_LOG = []


def note(component, reason):
    """Warn the first time a component falls back, and record every occurrence for telemetry."""
    _LOG.append((component, reason))
    if component in _SEEN or SILENT:
        return
    _SEEN.add(component)
    sys.stderr.write("photo2fcstd: %s unavailable (%s); using the geometric rule instead\n"
                     % (component, reason))


def events():
    return list(_LOG)


def reset():
    _SEEN.clear()
    _LOG.clear()
