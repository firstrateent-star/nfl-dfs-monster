from __future__ import annotations

from pathlib import Path
import runpy

path = Path("scripts/apply_drive_trace_patch.py")
text = path.read_text()
old = '''loop = replace_nth(
    loop,
    "        _record(event, stats, defensive_stats)\\n",
    "        _record(event, stats, defensive_stats)\\n        drive_recorder.observe(before, event)\\n",
    occurrence=1,
    label="overtime event observation",
)
'''
new = '''loop = replace_once(
    loop,
    "        plays.append(event)\\n        _record(event, stats, defensive_stats)\\n",
    "        plays.append(event)\\n        _record(event, stats, defensive_stats)\\n        drive_recorder.observe(before, event)\\n",
    label="overtime event observation",
)
'''
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected one overtime observation patch block, found {count}")
path.write_text(text.replace(old, new, 1))
runpy.run_path(str(path), run_name="__main__")
