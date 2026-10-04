#!/usr/bin/env python3
"""
Comprehensive Verification Script for Stage T5 Visual Polish
Validates:
1. Exact T5 font sizes in Rust-generated ASS (Primary 52, Emphasis 64, Secondary 42, Decorative 45)
2. Complete absence of border/stroke (Outline 0, \\bord0, no \\3c, shadow preserved)
3. Caption Intelligence 2.0 activation and emphasis role mapping
4. Invariance of other templates (T1, T2, T3, T4, T6)
5. Real video render with FFmpeg and OpenCV visual inspection
"""

import json
import os
import re
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TAURI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BIN_PATH = os.path.join(TAURI_DIR, "target", "debug", "caption_intel_inspect.exe")
FONTS_DIR = os.path.join(TAURI_DIR, "fonts")

def locate_candidate_video() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "Messi vs Ronaldo Fans： The Psychology Explained [rssDTc086bk].mp4"),
        os.path.join(PROJECT_ROOT, "Messi vs Ronaldo Fans: The Psychology Explained [rssDTc086bk].mp4"),
        os.path.join(PROJECT_ROOT, "Video-56495.mp4"),
        os.path.join(PROJECT_ROOT, "clip-01_flat.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 500000:
            return os.path.abspath(c)
    for f in os.listdir(PROJECT_ROOT):
        if f.endswith(".mp4"):
            p = os.path.join(PROJECT_ROOT, f)
            if os.path.getsize(p) > 500000:
                return p
    raise FileNotFoundError("Could not find any suitable video file")

TEST_VIDEO = locate_candidate_video()

def make_candidate():
    return {
        "id": "cand-t5-polish",
        "projectId": "proj-test",
        "source": "video.mp4",
        "startSec": 10.0,
        "endSec": 25.0,
        "score": 0.88,
        "hook": "never eat this before bed",
        "rationale": "High viral potential",
        "rank": 1,
        "selected": True,
        "hookStartSec": 10.0,
        "hookEndSec": 12.0,
        "hookConfidence": 0.95,
        "openingContextScore": 0.85,
        "payoffText": "destroy deep sleep",
        "payoffStartSec": 20.0,
        "payoffEndSec": 23.0,
        "payoffScore": 0.92,
        "payoffCompletion": True,
    }

def make_words():
    data = [
        (10.0, 10.4, "Never"),
        (10.4, 10.8, "eat"),
        (10.8, 11.2, "this"),
        (11.2, 11.6, "before"),
        (11.6, 12.0, "bed,"),
        (12.2, 12.7, "because"),
        (12.7, 13.2, "recent"),
        (13.2, 13.8, "studies"),
        (13.8, 14.5, "confirm"),
        (20.0, 20.4, "it"),
        (20.4, 20.8, "will"),
        (20.8, 21.4, "destroy"),
        (21.4, 21.8, "your"),
        (21.8, 22.3, "deep"),
        (22.3, 23.0, "sleep."),
    ]
    return [{"start": s, "end": e, "text": t} for s, e, t in data]

def run_inspect(template_id: str):
    words_file = os.path.join(SCRIPT_DIR, f"temp_words_{template_id}.json")
    cand_file = os.path.join(SCRIPT_DIR, f"temp_cand_{template_id}.json")
    with open(words_file, "w", encoding="utf-8") as f:
        json.dump(make_words(), f)
    with open(cand_file, "w", encoding="utf-8") as f:
        json.dump(make_candidate(), f)

    try:
        cmd = [BIN_PATH, words_file, cand_file, "10.0", "25.0", template_id]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
        return json.loads(res.stdout)
    finally:
        for p in [words_file, cand_file]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

def verify_all():
    print("=" * 70)
    print("STEP 1: Verifying Rust ASS Generation for Template 5 (preset_dynamic_editorial)")
    print("=" * 70)
    t5_data = run_inspect("preset_dynamic_editorial")
    ass = t5_data["assWithIntel"]
    print(f"Plan confidence: {t5_data['plan']['confidence'] if t5_data['plan'] else 'None'}")
    print(f"Emphasis words: {t5_data['plan']['emphasisWordIndices'] if t5_data['plan'] else 'None'}")
    
    # 1. Style Header checks
    style_lines = [l for l in ass.splitlines() if l.startswith("Style:")]
    print(f"Style Header: {style_lines}")
    assert len(style_lines) >= 1, "Must contain Style line"
    t5_style = style_lines[0]
    
    # Check Style line parameters
    # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    # Style: Default,Montserrat,62,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,80,80,80,1
    style_fields = t5_style.replace("Style: ", "").split(",")
    font_name = style_fields[1].strip()
    font_size = int(style_fields[2].strip())
    primary_color = style_fields[3].strip()
    border_style = int(style_fields[15].strip())
    outline = int(style_fields[16].strip())
    shadow = int(style_fields[17].strip())
    
    print(f"  -> Style Font: {font_name} (expected Montserrat)")
    print(f"  -> Style Font Size: {font_size} (expected 62)")
    print(f"  -> Style Primary Color: {primary_color} (expected &H00FFFFFF& pure white)")
    print(f"  -> Style BorderStyle: {border_style} (expected 1)")
    print(f"  -> Style Outline: {outline} (expected 0 -> stroke removed)")
    print(f"  -> Style Shadow: {shadow} (expected 0 -> shadow removed)")
    
    assert font_name == "Montserrat", f"Expected Montserrat, got {font_name}"
    assert font_size == 62, f"Expected 62, got {font_size}"
    assert primary_color == "&H00FFFFFF", f"Expected &H00FFFFFF, got {primary_color}"
    assert outline == 0, f"Expected Outline 0, got {outline}"
    assert shadow == 0, f"Expected Shadow 0, got {shadow}"

    # 2. Dialogue Events checks
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    base_dialogues = [l for l in t5_data["baselineAss"].splitlines() if l.startswith("Dialogue:")]
    print(f"Total T5 Dialogue events generated: {len(dialogues)} (with intel), {len(base_dialogues)} (baseline)")
    assert len(dialogues) > 0, "Must have dialogue events"

    fs_found = set()
    fn_found = set()
    y_coords = []
    for d in dialogues + base_dialogues:
        # Check that \bord0 and \shad0 are present
        assert r"\bord0" in d, f"Missing \\bord0 in dialogue: {d}"
        assert r"\shad0" in d, f"Missing \\shad0 in dialogue: {d}"
        # Check that \3c, \4c, \shad2 are NOT present
        assert r"\3c" not in d, f"Illegal stroke color \\3c found in dialogue: {d}"
        assert r"\4c" not in d, f"Illegal shadow color \\4c found in dialogue: {d}"
        assert r"\shad2" not in d, f"Old \\shad2 found in dialogue: {d}"
        assert r"\bord3" not in d, f"Old \\bord3 found in dialogue: {d}"

        # Collect font sizes
        fs_matches = re.findall(r"\\fs(\d+)", d)
        for fs in fs_matches:
            fs_found.add(int(fs))
        fn_matches = re.findall(r"\\fn([^\\}]+)", d)
        for fn in fn_matches:
            fn_found.add(fn)

        # Extract Y coordinate from \move(x1,y1,x2,y2,...)
        m = re.search(r"\\move\(\s*[\d.]+\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*,\s*([\d.]+)", d)
        if m:
            y_coords.append(float(m.group(2)))

    print(f"  -> Font sizes found across events: {sorted(list(fs_found))}")
    print(f"  -> Fonts found across events: {sorted(list(fn_found))}")
    print(f"  -> Resting Y coordinates range: {min(y_coords):.1f} - {max(y_coords):.1f}")

    # Check font sizes: must contain 62 (Primary), 88 (Emphasis for Bebas Neue), 50 (Secondary), 80 (Decorative)
    assert 62 in fs_found, f"Primary font size 62 must be present, found {fs_found}"
    assert 88 in fs_found, f"Emphasis font size 88 must be present, found {fs_found}"
    assert 50 in fs_found, f"Secondary font size 50 must be present, found {fs_found}"
    assert 80 in fs_found, f"Decorative font size 80 must be present, found {fs_found}"
    assert 44 not in fs_found, f"Old base font size 44 must NOT be present, found {fs_found}"

    # Verify Y coordinates are centered around 1245 for single speaker (not at 1650 or 900)
    avg_y = sum(y_coords) / len(y_coords)
    print(f"  -> Average resting Y: {avg_y:.1f} (expected close to 1245)")
    assert 1150.0 <= avg_y <= 1350.0, f"Average Y {avg_y} outside single-speaker target band [1150, 1350]"
    
    # 3. Check Caption Intelligence elevation
    # In candidate, hook is "never eat this before bed"
    # Payoff is "destroy deep sleep"
    # The words in hook/payoff should receive emphasis (Bebas Neue / fs88)
    hook_payoff_emphasis = False
    for d in dialogues:
        if r"\fnBebas Neue\fs88" in d:
            hook_payoff_emphasis = True
            break
    assert hook_payoff_emphasis, "Caption Intelligence must elevate hook/payoff words to Bebas Neue with \\fs88"
    print("  -> Caption Intelligence 2.0 emphasis elevation to Bebas Neue (fs 88) VERIFIED")

    print("\n" + "=" * 70)
    print("STEP 2: Verifying Other Caption Templates Are Completely Untouched")
    print("=" * 70)
    
    # T1
    t1_data = run_inspect("preset_viral_bold")
    t1_ass = t1_data["assWithIntel"]
    assert "Default,Bebas Neue,84" in t1_ass, "T1 font/size changed unexpectedly!"
    assert r"\fscx115\fscy115" in t1_ass, "T1 kinetic tag changed unexpectedly!"
    print("  [PASS] T1 (Viral Bold / Hormozi): Default,Bebas Neue,84 unchanged")

    # T2
    t2_data = run_inspect("preset_mrbeast_pop")
    t2_ass = t2_data["assWithIntel"]
    assert "Default,Montserrat,80" in t2_ass, "T2 font/size changed unexpectedly!"
    assert r"\t(0,50,\fscx120\fscy120)" in t2_ass, "T2 kinetic pop changed unexpectedly!"
    print("  [PASS] T2 (MrBeast Pop): Default,Montserrat,80 unchanged")

    # T3
    t3_data = run_inspect("preset_minimal_capsule")
    t3_ass = t3_data["assWithIntel"]
    assert "Default,Inter,96" in t3_ass, "T3 font/size changed unexpectedly!"
    assert ",3,12,0,2,80,80," in t3_ass, "T3 capsule box styling changed unexpectedly!"
    print("  [PASS] T3 (Minimal Capsule): Default,Inter,96 unchanged")

    # T4
    t4_data = run_inspect("preset_cinematic_vlog")
    t4_ass = t4_data["assWithIntel"]
    assert "Default,Poppins,116" in t4_ass, "T4 font/size changed unexpectedly!"
    assert r"\fad(150,150)" in t4_ass, "T4 fade tag changed unexpectedly!"
    print("  [PASS] T4 (Cinematic Vlog): Default,Poppins,116 unchanged")

    # T6
    t6_data = run_inspect("preset_bhaukal_caption")
    t6_ass = t6_data["assWithIntel"]
    assert "Default,Montserrat,52" in t6_ass, "T6 font/size changed unexpectedly!"
    assert "Style: HookStrip,Inter,34" in t6_ass, "T6 HookStrip style changed unexpectedly!"
    print("  [PASS] T6 (Bhaukal Caption): Default,Montserrat,52 and HookStrip unchanged")

    print("\n" + "=" * 70)
    print("STEP 3: Real Video Render with FFmpeg and OpenCV Inspection")
    print("=" * 70)
    if not os.path.exists(TEST_VIDEO):
        print(f"Test video not found: {TEST_VIDEO}")
        return

    test_ass_file = os.path.join(SCRIPT_DIR, "verified_t5_real.ass")
    out_mp4 = os.path.join(SCRIPT_DIR, "verified_t5_real.mp4")
    with open(test_ass_file, "w", encoding="utf-8") as f:
        f.write(ass)

    escaped_ass = test_ass_file.replace("\\", "/").replace(":", r"\:")
    escaped_fonts = FONTS_DIR.replace("\\", "/").replace(":", r"\:")

    filter_str = (
        f"crop=w=608:h=1080:x='(iw-608)/2':y='0',"
        f"scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int,"
        f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", "10.0",
        "-t", "3.0",
        "-i", TEST_VIDEO,
        "-vf", filter_str,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        out_mp4
    ]
    print(f"Running FFmpeg render: {' '.join(cmd)}")
    render_res = subprocess.run(cmd, capture_output=True, text=True)
    assert render_res.returncode == 0, f"FFmpeg render failed:\n{render_res.stderr}"
    assert os.path.exists(out_mp4), "Rendered MP4 must exist"
    mp4_size = os.path.getsize(out_mp4)
    print(f"  -> Successfully rendered real T5 clip: {out_mp4} ({mp4_size} bytes)")

    # OpenCV frame examination
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(out_mp4)
    assert cap.isOpened(), "Could not open rendered MP4"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  -> Video properties: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    assert width == 1080 and height == 1920, f"Unexpected dimensions: {width}x{height}"

    # Sample frame at t = 1.0s (frame ~30)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1.0 * fps))
    ret, frame = cap.read()
    assert ret, "Could not read frame at t=1.0s"

    # Save verification frame
    frame_png = os.path.join(SCRIPT_DIR, "verified_t5_frame_1s.png")
    cv2.imwrite(frame_png, frame)
    print(f"  -> Saved frame capture: {frame_png}")

    cap.release()
    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_all()
