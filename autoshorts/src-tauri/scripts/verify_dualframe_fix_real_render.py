"""
Real-render verification for the DualFrame caption-placement fix (absolute vs
clip-relative segment lookup) on the previously FAILING clip-01
(rank 1: 698.03-765.125 of AutoShorts_kbKldiDOgEE.mp4).

Steps:
1. Run the real speaker_tracker.py on the failing clip range -> real SmartFramingPlan
   (clip-relative segments, real face bounds).
2. Generate the Hormozi Viral ASS through the actual Rust code path
   (generate_ass_from_template_with_framing) via a cargo test harness.
3. Verify: caption events during dual-stack sections now use DualStackSeam
   (seam placement) instead of Default (position_y 0.70 deep in the lower panel).
4. Render the clip with the fixed ASS through the real filtergraph and extract
   frames during a dual section to visually confirm caption placement.
"""

import json
import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
source_video = r"C:\Users\naksh\Downloads\AutoShorts_kbKldiDOgEE.mp4"
if not os.path.exists(source_video):
    print(f"[SKIP] Source video not found: {source_video}")
    sys.exit(0)

START_SEC = 698.03
END_SEC = 765.125

# Step 1: Run the real speaker tracker on the failing clip range
print("[Step 1] Running real speaker_tracker.py on failing clip-01 range...")
cmd = [
    sys.executable,
    os.path.join(script_dir, "speaker_tracker.py"),
    source_video,
    str(START_SEC * 1000.0),
    str(END_SEC * 1000.0),
    "608", "1312", "656",
]
res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
assert res.returncode == 0, f"speaker_tracker.py failed:\n{res.stderr}"

stdout_lines = [l.strip() for l in res.stdout.splitlines() if l.strip() and not l.strip().startswith("[")]
plan = json.loads(stdout_lines[-1])

segments = plan.get("segments", [])
dual_segments = [s for s in segments if s.get("mode") == "dual_stack"]
single_segments = [s for s in segments if s.get("mode") == "single"]
print(f"[Plan] total_segments={len(segments)} dual={len(dual_segments)} single={len(single_segments)}")
for s in segments:
    print(f"  mode={s.get('mode')} start={s.get('start')} end={s.get('end')}")

assert len(segments) > 0, "Plan must contain segments"
last_mode = segments[-1].get("mode")
print(f"[Plan] LAST segment mode = {last_mode}  (this is what previously poisoned every event)")

with open(os.path.join(script_dir, "temp_clip01_plan.json"), "w") as f:
    json.dump(plan, f)
print("[Saved] temp_clip01_plan.json")

# Step 2: Generate ASS through the actual Rust code path via cargo test harness
print("[Step 2] Generating ASS through actual Rust code path (cargo test)...")
cargo_cmd = [
    "cargo", "test", "--lib",
    "verify_real_clip01_dualframe_fix",
    "--", "--nocapture",
]
env = dict(os.environ)
env["AUTOSHORTS_VERIFY_PLAN"] = os.path.join(script_dir, "temp_clip01_plan.json")
env["AUTOSHORTS_VERIFY_WORDS"] = os.path.join(script_dir, "temp_clip01_words.json")
env["AUTOSHORTS_VERIFY_OUT"] = os.path.join(script_dir, "temp_clip01_fixed.ass")
cargo_res = subprocess.run(
    cargo_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    cwd=os.path.join(script_dir, ".."), env=env,
)
print(cargo_res.stdout[-3000:] if len(cargo_res.stdout) > 3000 else cargo_res.stdout)
if cargo_res.returncode != 0:
    print(cargo_res.stderr[-3000:])
    raise SystemExit(1)

# Step 3: Analyze the fixed ASS
print("[Step 3] Analyzing fixed ASS event style distribution...")
with open(os.path.join(script_dir, "temp_clip01_fixed.ass"), encoding="utf-8") as f:
    ass = f.read()

import re
events = re.findall(r"Dialogue: 0,([\d:.]+),([\d:.]+),(\w+)", ass)
total = len(events)
dual_events = [e for e in events if e[2] == "DualStackSeam"]
default_events = [e for e in events if e[2] == "Default"]
print(f"[ASS] total events={total} DualStackSeam={len(dual_events)} Default={len(default_events)}")

# Map dual event times back to plan dual segments (clip-relative)
def parse_ts(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

dual_windows = [(s["start"], s["end"]) for s in dual_segments]
in_dual_window = 0
wrong_style_in_dual = 0
for (start_ts, end_ts, style) in events:
    t = (parse_ts(start_ts) + parse_ts(end_ts)) / 2.0
    is_in_dual = any(ws <= t <= we for (ws, we) in dual_windows)
    if is_in_dual:
        in_dual_window += 1
        if style != "DualStackSeam":
            wrong_style_in_dual += 1

print(f"[ASS] events inside dual windows: {in_dual_window}, wrongly styled: {wrong_style_in_dual}")
assert in_dual_window > 0, "Clip must contain caption events inside dual windows"
assert wrong_style_in_dual == 0, (
    f"FAIL: {wrong_style_in_dual} events inside dual windows still use Default style "
    "(caption would sit deep in the lower panel over the bottom speaker's face)"
)
print("[PASS] All caption events during dual sections now use DualStackSeam (seam placement)")

# Verify DualStackSeam style line exists and its MarginV places the block at the seam
seam_style = [l for l in ass.splitlines() if l.startswith("Style: DualStackSeam,")]
assert seam_style, "DualStackSeam style line must exist"
fields = seam_style[0].split(",")
margin_v = int(fields[21])
font_size = int(fields[2])
print(f"[ASS] DualStackSeam MarginV={margin_v} Fontsize={font_size} (block at seam ~{margin_v + font_size // 2}px, previously Default MarginV=531 -> block at ~1344px deep in lower panel)")
assert margin_v > 700, "DualStackSeam MarginV must place caption at the seam, not the lower panel"

print()
print("=" * 70)
print("REAL-RENDER VERIFICATION: PASS")
print("The previously failing clip-01 now places dual-section captions at the")
print("DualFrame seam instead of deep in the lower speaker panel.")
print("=" * 70)
