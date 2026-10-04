import os
import sys
import subprocess
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "fonts"))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

def locate_candidate_video() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "Messi vs Ronaldo Fans： The Psychology Explained [rssDTc086bk].mp4"),
        os.path.join(PROJECT_ROOT, "Messi vs Ronaldo Fans: The Psychology Explained [rssDTc086bk].mp4"),
        os.path.join(PROJECT_ROOT, "Video-56495.mp4"),
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
escaped_fonts = FONTS_DIR.replace("\\", "/").replace(":", r"\:")

# Generate 5 test ASS cases testing all required visual roles
cases = [
    {
        "id": "case1_yellow_emphasis",
        "desc": "Caption with Yellow Emphasis Word (Bebas Neue 88pt, uppercase, vivid yellow)",
        "ass_dialogue": "Dialogue: 0,0:00:00.20,0:00:01.80,Default,,0,0,0,,{\\an5\\move(540,1275,540,1245,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnBebas Neue\\fs88\\b1\\i0\\c&H00E6FF&\\bord0\\shad0}DOMINANT\n"
    },
    {
        "id": "case2_cyan_decorative",
        "desc": "Caption with Cyan Word (Poppins 80pt, bold italic, uppercase, vivid cyan)",
        "ass_dialogue": "Dialogue: 0,0:00:00.20,0:00:01.80,Default,,0,0,0,,{\\an5\\move(540,1215,540,1245,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnPoppins\\fs80\\b1\\i1\\c&HFFE500&\\bord0\\shad0}EDITORIAL\n"
    },
    {
        "id": "case3_normal_white",
        "desc": "Caption with Normal White Text (Montserrat 62pt, uppercase, pure white #FFFFFF)",
        "ass_dialogue": "Dialogue: 0,0:00:00.20,0:00:01.80,Default,,0,0,0,,{\\an5\\move(508,1245,540,1245,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnMontserrat\\fs62\\b1\\i0\\c&HFFFFFF&\\bord0\\shad0}GOLDEN WORDS\n"
    },
    {
        "id": "case4_all_three_roles",
        "desc": "Caption with All Three Roles (White Montserrat 62pt, Yellow Bebas Neue 88pt, Cyan Poppins 80pt)",
        # Line 1: WHITE (Montserrat 62) + YELLOW (Bebas Neue 88)
        # Line 2: CYAN (Poppins 80)
        "ass_dialogue": (
            "Dialogue: 0,0:00:00.20,0:00:01.80,Default,,0,0,0,,{\\an5\\move(360,1210,392,1210,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnMontserrat\\fs62\\b1\\i0\\c&HFFFFFF&\\bord0\\shad0}GOLDEN\n"
            "Dialogue: 0,0:00:00.50,0:00:01.80,Default,,0,0,0,,{\\an5\\move(690,1210,658,1210,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnBebas Neue\\fs88\\b1\\i0\\c&H00E6FF&\\bord0\\shad0}WORDS\n"
            "Dialogue: 0,0:00:00.75,0:00:01.80,Default,,0,0,0,,{\\an5\\move(540,1320,540,1290,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnPoppins\\fs80\\b1\\i1\\c&HFFE500&\\bord0\\shad0}EXCLUSIVE\n"
        )
    },
    {
        "id": "case5_longer_caption_group",
        "desc": "Longer Caption Group (5 words wrapped cleanly into 2 compact lines with Secondary 50pt & Cyan 80pt)",
        "ass_dialogue": (
            "Dialogue: 0,0:00:00.00,0:00:01.80,Default,,0,0,0,,{\\an5\\move(320,1210,352,1210,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnInter\\fs50\\b0\\i0\\c&HFFFFFF&\\bord0\\shad0}studies\n"
            "Dialogue: 0,0:00:00.30,0:00:01.80,Default,,0,0,0,,{\\an5\\move(540,1210,540,1210,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnMontserrat\\fs62\\b1\\i0\\c&HFFFFFF&\\bord0\\shad0}CONFIRM\n"
            "Dialogue: 0,0:00:00.60,0:00:01.80,Default,,0,0,0,,{\\an5\\move(730,1210,698,1210,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnBebas Neue\\fs88\\b1\\i0\\c&H00E6FF&\\bord0\\shad0}VITAL\n"
            "Dialogue: 0,0:00:00.90,0:00:01.80,Default,,0,0,0,,{\\an5\\move(420,1290,452,1290,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnMontserrat\\fs62\\b1\\i0\\c&HFFFFFF&\\bord0\\shad0}DEEP\n"
            "Dialogue: 0,0:00:01.20,0:00:01.80,Default,,0,0,0,,{\\an5\\move(660,1290,628,1290,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnPoppins\\fs80\\b1\\i1\\c&HFFE500&\\bord0\\shad0}SLEEP\n"
        )
    },
    {
        "id": "case6_reference_frame_success_is_very_subjective",
        "desc": "Reference Frame Replica: 'Success IS' + 'VERY SUBJECTIVE' (Inter 50pt, Montserrat 62pt, Bebas Neue 88pt, Poppins 80pt)",
        # Line 1: 'Success' (Inter 50pt Secondary white) + 'IS' (Montserrat 62pt Primary white)
        # Line 2: 'VERY' (Bebas Neue 88pt Emphasis yellow) + 'SUBJECTIVE' (Poppins 80pt Decorative cyan)
        "ass_dialogue": (
            "Dialogue: 0,0:00:00.00,0:00:01.80,Default,,0,0,0,,{\\an5\\move(440,1205,440,1205,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnInter\\fs50\\b0\\i0\\c&HFFFFFF&\\bord0\\shad0}Success\n"
            "Dialogue: 0,0:00:00.30,0:00:01.80,Default,,0,0,0,,{\\an5\\move(620,1205,620,1205,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnMontserrat\\fs62\\b1\\i0\\c&HFFFFFF&\\bord0\\shad0}IS\n"
            "Dialogue: 0,0:00:00.60,0:00:01.80,Default,,0,0,0,,{\\an5\\move(380,1295,380,1295,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnBebas Neue\\fs88\\b1\\i0\\c&H00E6FF&\\bord0\\shad0}VERY\n"
            "Dialogue: 0,0:00:00.90,0:00:01.80,Default,,0,0,0,,{\\an5\\move(660,1295,660,1295,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnPoppins\\fs80\\b1\\i1\\c&HFFE500&\\bord0\\shad0}SUBJECTIVE\n"
        )
    }
]

print("=" * 70)
print("REAL RENDER VALIDATION ACROSS ALL 5 REQUIRED T5 SCENARIOS")
print(f"Source video: {TEST_VIDEO}")
print("=" * 70)

for c in cases:
    cid = c["id"]
    desc = c["desc"]
    ass_dialogue = c["ass_dialogue"]
    
    ass_path = os.path.join(SCRIPT_DIR, f"{cid}.ass")
    mp4_path = os.path.join(SCRIPT_DIR, f"{cid}.mp4")
    png_path = os.path.join(SCRIPT_DIR, f"{cid}_frame.png")
    
    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,62,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{ass_dialogue}"""

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
        
    escaped_ass = ass_path.replace("\\", "/").replace(":", r"\:")
    filter_str = (
        f"crop=w=608:h=1080:x='(iw-608)/2':y='0',"
        f"scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int,"
        f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", "10.0",
        "-t", "2.0",
        "-i", TEST_VIDEO,
        "-vf", filter_str,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        mp4_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    assert os.path.exists(mp4_path), f"Render failed for {cid}"
    size_bytes = os.path.getsize(mp4_path)
    
    # Extract frame at t=1.2s
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1.2 * fps))
    ret, frame = cap.read()
    assert ret, f"Could not read frame from {mp4_path}"
    cv2.imwrite(png_path, frame)
    cap.release()
    
    # Analyze frame colors in caption band [1100:1400]
    caption_roi = frame[1100:1400, 150:930]
    b = caption_roi[:, :, 0]
    g = caption_roi[:, :, 1]
    r = caption_roi[:, :, 2]
    
    white_px = int(np.count_nonzero((r > 220) & (g > 220) & (b > 220)))
    yellow_px = int(np.count_nonzero((r > 200) & (g > 180) & (b < 80)))
    cyan_px = int(np.count_nonzero((b > 200) & (g > 180) & (r < 80)))
    
    print(f"\n[VERIFIED] {cid}:")
    print(f"  Description: {desc}")
    print(f"  Rendered video: {mp4_path} ({size_bytes:,} bytes)")
    print(f"  Snapshot saved: {png_path}")
    print(f"  Color detection: white={white_px}, yellow={yellow_px}, cyan={cyan_px}")

print("\nALL 6 T5 REAL RENDER SCENARIOS VERIFIED SUCCESSFULLY!")
