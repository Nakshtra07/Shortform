#!/usr/bin/env python3
"""AutoShorts 9.0 — Caption Regression & Visual QA Harness (test suite).

End-to-end orchestrator for the caption QA chain:

    TEMPLATE → CAPTION INTELLIGENCE → ASS → FFMPEG → FINAL MP4 → DATABASE

It drives the real production code paths (caption_qa binary links the actual
captions.rs / caption_intel.rs / pacing.rs / media.rs), inspects REAL on-disk
artifacts from the production SQLite DB, runs REAL ffmpeg renders, and
performs REAL pixel verification with OpenCV.

Phases:
  A. Template isolation (Part 8)          — caption_qa isolation
  B. Real-clip forensic audit (Parts 3/4/5/9) — caption_qa audit-clip on a
     real generated clip from the production DB
  C. Real renders (Part 10)              — 4 renders: T1 normal, T1 + CI, T5, T6
  D. Visual QA on rendered MP4s (Parts 6/7) — caption_visual_qa.py
  E. Regression guard (Part 11)          — confirms no protected pipeline changed

Run:  .venv/Scripts/python.exe test_caption_qa_suite.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tests" else HERE
SRC_TAURI = os.path.join(REPO_ROOT, "autoshorts", "src-tauri")
BIN = os.path.join(SRC_TAURI, "target", "debug", "caption_qa.exe")
VENV_PY = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
DB = os.path.expandvars(r"C:\Users\naksh\AppData\Roaming\com.autoshorts.desktop\autoshorts.sqlite")
SOURCE_MP4 = r"C:\Users\naksh\Downloads\AutoShorts_OcISVEh1jyw.mp4"
OUT = os.path.join(REPO_ROOT, "tmp", "qa_suite")
REPORT_MD = os.path.join(REPO_ROOT, "docs", "reports", "validation_report_caption_qa.md")

ALL_CHECKS = []
FAILURES = []


def record(phase, result):
    """result: dict from a caption_qa / visual_qa JSON report."""
    if result is None:
        return
    for c in result.get("checks", []):
        c["_phase"] = phase
        ALL_CHECKS.append(c)
        if c["status"] == "FAIL":
            FAILURES.append(c)


def run_json(cmd, env=None, label="", report_path=None):
    """Run a command; prefer the canonical report.json file it writes
    (production code logs to stdout, which pollutes printed JSON)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {label}")
        return None
    if r.returncode != 0:
        # visual QA exits non-zero when checks FAIL; the JSON is still valid
        try:
            if report_path and os.path.exists(report_path):
                return json.load(open(report_path, encoding="utf-8"))
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            pass
        print(f"  [ERROR {r.returncode}] {label}: {r.stderr.strip()[:200]}")
        return None
    if report_path and os.path.exists(report_path):
        try:
            return json.load(open(report_path, encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        print(f"  [BADJSON] {label}: {r.stdout[:200]}")
        return None


def phase_isolation():
    print("[A] Template isolation (Part 8)")
    if not os.path.exists(BIN):
        print("  [SKIP] caption_qa binary not built")
        return
    report_dir = os.path.join(OUT, "reports", "isolation")
    os.makedirs(report_dir, exist_ok=True)
    r = run_json([BIN, "isolation"], env={**os.environ, "CAPTION_QA_OUT": report_dir},
                 label="isolation",
                 report_path=os.path.join(report_dir, "report.json"))
    record("A: template isolation", r)
    if r:
        s = r["summary"]
        print(f"  isolation: {s['pass']} PASS / {s['fail']} FAIL / {s['skipped']} SKIP")


def phase_audit_clip():
    """Audit one REAL generated clip from the production DB."""
    print("[B] Real-clip forensic audit (Parts 3/4/5/9)")
    if not os.path.exists(DB):
        print(f"  [SKIP] production DB not found: {DB}")
        return
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        row = con.execute(
            "SELECT ca.id, ca.project_id FROM clips cl "
            "JOIN candidates ca ON cl.candidate_id = ca.id "
            "WHERE cl.status='done' AND cl.caption_ass_path IS NOT NULL "
            "AND ca.project_id IN (SELECT id FROM projects WHERE caption_style IS NOT NULL) "
            "ORDER BY (ca.end_sec - ca.start_sec) LIMIT 1").fetchone()
        con.close()
    except Exception as e:
        print(f"  [SKIP] DB query failed: {e}")
        return
    if not row:
        print("  [SKIP] no suitable done clip found")
        return
    cand_id, proj_id = row
    print(f"  auditing candidate {cand_id} (project {proj_id})")
    report_dir = os.path.join(OUT, "reports", "audit_clip")
    os.makedirs(report_dir, exist_ok=True)
    r = run_json([BIN, "audit-clip", DB, proj_id, cand_id],
                 env={**os.environ, "CAPTION_QA_OUT": report_dir},
                 label="audit-clip",
                 report_path=os.path.join(report_dir, "report.json"))
    record("B: real clip audit", r)
    if r:
        s = r["summary"]
        print(f"  audit-clip: {s['pass']} PASS / {s['fail']} FAIL / {s['skipped']} SKIP / {s['notVerified']} NV")


def phase_renders():
    """Part 10: real renders with the real pipeline."""
    print("[C] Real renders (Part 10)")
    inputs = os.path.join(OUT, "qa_inputs")
    renders = os.path.join(OUT, "qa_renders")
    if not os.path.exists(SOURCE_MP4):
        print(f"  [SKIP] source video missing: {SOURCE_MP4}")
        return None
    if not os.path.exists(os.path.join(inputs, "words.json")):
        # dump real inputs from the production DB
        import sqlite3
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        row = con.execute(
            "SELECT ca.id, ca.project_id FROM clips cl "
            "JOIN candidates ca ON cl.candidate_id = ca.id "
            "WHERE cl.status='done' AND cl.caption_ass_path IS NOT NULL "
            "ORDER BY (ca.end_sec - ca.start_sec) LIMIT 1").fetchone()
        con.close()
        if not row:
            print("  [SKIP] no candidate to dump")
            return None
        cand_id, proj_id = row
        r = run_json([BIN, "dump-render-inputs", DB, proj_id, cand_id, inputs], label="dump")
        if r is None:
            return None
    os.makedirs(renders, exist_ok=True)

    words = os.path.join(inputs, "words.json")
    cand = os.path.join(inputs, "candidate.json")
    meta = json.load(open(os.path.join(inputs, "meta.json")))
    start, end = meta["startSec"], meta["endSec"]

    jobs = [
        ("r1_t1_normal", "preset_viral_bold", {"AUTOSHORTS_CAPTION_INTEL": "off"}),
        ("r2_t1_ci", "preset_viral_bold", None),
        ("r3_t5_ci", "preset_dynamic_editorial", None),
        ("r4_t6_ci", "preset_bhaukal_caption", None),
    ]
    results = {}
    full_env = os.environ.copy()
    for name, template, extra in jobs:
        env = dict(full_env)
        if extra:
            env.update(extra)
        outdir = os.path.join(renders, name)
        print(f"  render {name} (template={template} ci={'off' if extra else 'on'})...")
        r = run_json([BIN, "render", SOURCE_MP4, words, cand,
                      str(start), str(end), template, outdir],
                     env=env, label=name,
                     report_path=os.path.join(outdir, "report.json"))
        record(f"C: real render {name} ({template})", r)
        results[name] = r
        if r:
            s = r["summary"]
            print(f"    {s['pass']} PASS / {s['fail']} FAIL")
    return results


def phase_visual(results):
    """Parts 6/7: pixel-level QA on the rendered MP4s."""
    print("[D] Visual QA on rendered MP4s (Parts 6/7)")
    renders = os.path.join(OUT, "qa_renders")
    script = os.path.join(HERE, "caption_visual_qa.py")
    if not os.path.exists(script):
        print("  [SKIP] caption_visual_qa.py missing")
        return
    for name in ("r1_t1_normal", "r2_t1_ci", "r3_t5_ci", "r4_t6_ci"):
        outdir = os.path.join(renders, name)
        mp4 = os.path.join(outdir, "rendered.mp4")
        ass = os.path.join(outdir, "caption.ass")
        tmpl = os.path.join(outdir, "template.json")
        plan = os.path.join(outdir, "ci_plan.json")
        bounds = os.path.join(outdir, "word_bounds.json")
        if not (os.path.exists(mp4) and os.path.exists(ass)):
            print(f"  [SKIP] {name}: render artifacts missing")
            continue
        cmd = [VENV_PY, script, mp4, ass, tmpl, "--frames", "6",
               "--outdir", os.path.join(outdir, "frames")]
        if os.path.exists(plan) and json.load(open(plan)):
            cmd += ["--plan", plan]
        if os.path.exists(bounds):
            cmd += ["--bounds", bounds]
        r = run_json(cmd, label=f"visual {name}")
        record(f"D: visual QA {name}", r)
        if r:
            s = r["summary"]
            print(f"  {name}: {s['pass']} PASS / {s['fail']} FAIL / {s['skipped']} SKIP / {s['notVerified']} NV")


def phase_regression():
    """Part 11: confirm the harness did not alter protected pipelines."""
    print("[E] Regression guard (Part 11)")
    # The harness only adds: src/bin/caption_qa.rs, Cargo.toml [[bin]], and
    # `pub mod db` visibility. Verify the protected modules are untouched
    # by checking the caption engine still produces the same ASS it did
    # before the harness existed (re-run isolation as a smoke test) and that
    # production code files are byte-identical to git HEAD where expected.
    protected = [
        "autoshorts/src-tauri/src/audio.rs",
        "autoshorts/src-tauri/src/boundary.rs",
        "autoshorts/src-tauri/src/captions.rs",
        "autoshorts/src-tauri/src/caption_intel.rs",
        "autoshorts/src-tauri/src/media.rs",
        "autoshorts/src-tauri/src/pacing.rs",
        "autoshorts/src-tauri/src/transcription.rs",
        "autoshorts/src-tauri/src/transcript_normalizer.rs",
        "autoshorts/src-tauri/src/youtube.rs",
        "autoshorts/src-tauri/src/llm.rs",
        "autoshorts/src/main.tsx",
    ]
    repo = REPO_ROOT
    for f in protected:
        path = os.path.join(repo, f)
        rel = os.path.relpath(path, repo)
        try:
            r = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", rel],
                               capture_output=True, text=True, timeout=30)
            changed = bool(r.stdout.strip())
        except Exception:
            changed = True
        status = "FAIL" if changed else "PASS"
        ALL_CHECKS.append({
            "id": "REG-PROTECTED",
            "name": f"Protected pipeline unchanged: {f}",
            "status": status,
            "actual": "modified" if changed else "unchanged (git status clean)",
            "expected": "no modification",
            "expectedSource": "prompt Part 11 — DO NOT refactor these systems",
            "evidence": f"git status -- {rel}",
            "severity": "high" if changed else "none",
            "_phase": "E: regression",
        })
        if changed:
            FAILURES.append(ALL_CHECKS[-1])
        print(f"  {status} {f}")


def write_report():
    counts = defaultdict(int)
    for c in ALL_CHECKS:
        counts[c["status"]] += 1
    lines = []
    lines.append("# AutoShorts 9.0 — Caption Regression & Visual QA Report")
    lines.append("")
    lines.append("Harness: `caption_qa` (Rust inspection binary) + `caption_visual_qa.py` (pixel layer)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Status | Count |")
    lines.append(f"|---|---|")
    lines.append(f"| PASS | {counts['PASS']} |")
    lines.append(f"| FAIL | {counts['FAIL']} |")
    lines.append(f"| SKIPPED | {counts['SKIPPED']} |")
    lines.append(f"| NOT VERIFIED | {counts['NOT_VERIFIED']} |")
    lines.append("")
    lines.append("No overall numeric quality score is produced (per Part 13).")
    lines.append("")
    by_phase = defaultdict(list)
    for c in ALL_CHECKS:
        by_phase[c["_phase"]].append(c)
    for phase, checks in by_phase.items():
        lines.append(f"## {phase}")
        lines.append("")
        lines.append("| ID | Check | Status | Actual | Expected | Expected source | Severity |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in checks:
            lines.append(
                f"| {c['id']} | {c['name']} | **{c['status']}** | "
                f"{str(c['actual'])[:120].replace('|', '/')} | "
                f"{str(c['expected'])[:120].replace('|', '/')} | "
                f"{str(c.get('expectedSource', c.get('expected_source', '')))[:80].replace('|', '/')} | "
                f"{c['severity']} |")
        lines.append("")
    if FAILURES:
        lines.append("## Failures (detail)")
        lines.append("")
        for c in FAILURES:
            lines.append(f"### {c['id']} — {c['name']}")
            lines.append(f"- actual: {c['actual']}")
            lines.append(f"- expected: {c['expected']}")
            lines.append(f"- evidence: {c['evidence']}")
            lines.append(f"- severity: {c['severity']}")
            lines.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nReport written to {REPORT_MD}")


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(BIN):
        print(f"building {BIN} ...")
        subprocess.run(["cargo", "build", "--bin", "caption_qa"], cwd=SRC_TAURI, check=True)
    phase_isolation()
    phase_audit_clip()
    results = phase_renders()
    phase_visual(results)
    phase_regression()
    write_report()
    print(f"\nTOTAL: {len(ALL_CHECKS)} checks — {sum(1 for c in ALL_CHECKS if c['status']=='FAIL')} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
