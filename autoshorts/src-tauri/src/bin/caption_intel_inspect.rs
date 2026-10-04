//! caption_intel_inspect — AutoShorts 9.0 Caption Intelligence 2.0 CLI.
//!
//! Exposes Caption Intelligence 2.0 planning and subtitle rendering as a CLI tool
//! emitting JSON for test suites and forensic validation.
//!
//! Usage:
//!   caption_intel_inspect <words_json_path> <candidate_json_path> <start_sec> <end_sec> [style] [pacing_json_path]

use autoshorts_lib::caption_intel::{self, CaptionIntelPlan};
use autoshorts_lib::captions;
use autoshorts_lib::models::{Candidate, NormalizedTranscript, TranscriptWord};
use autoshorts_lib::pacing::SmartPacingPlan;
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputJson {
    plan: Option<CaptionIntelPlan>,
    ass_with_intel: String,
    baseline_ass: String,
    ass_matches_baseline: bool,
    kill_switch_active: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: caption_intel_inspect <words_json_path> <candidate_json_path> <start_sec> <end_sec> [style] [pacing_json_path]"
        );
        std::process::exit(2);
    }

    let words_path = &args[1];
    let candidate_path = &args[2];
    let start_sec: f64 = args[3].parse().expect("start_sec must be a float");
    let end_sec: f64 = args[4].parse().expect("end_sec must be a float");
    let style = args
        .get(5)
        .filter(|s| !s.is_empty())
        .map(|s| s.as_str())
        .unwrap_or("preset_viral_bold");
    let pacing_path = args.get(6).filter(|s| !s.is_empty());

    let words_json = std::fs::read_to_string(words_path).expect("failed to read words JSON");
    let words: Vec<TranscriptWord> = match serde_json::from_str::<NormalizedTranscript>(&words_json)
    {
        Ok(t) => t.words,
        Err(_) => serde_json::from_str(&words_json)
            .expect("words JSON must be NormalizedTranscript or TranscriptWord array"),
    };

    let candidate_json =
        std::fs::read_to_string(candidate_path).expect("failed to read candidate JSON");
    let candidate: Candidate =
        serde_json::from_str(&candidate_json).expect("failed to parse candidate JSON");

    let pacing_plan: Option<SmartPacingPlan> = pacing_path.map(|p| {
        let content = std::fs::read_to_string(p).expect("failed to read pacing JSON");
        serde_json::from_str(&content).expect("failed to parse pacing JSON")
    });

    let kill_switch_active = !caption_intel::caption_intelligence_enabled();

    let plan = caption_intel::plan_caption_intel(
        &words,
        &candidate,
        pacing_plan.as_ref(),
        start_sec,
        end_sec,
    );

    let template = captions::get_caption_template(style)
        .unwrap_or_else(|| captions::get_caption_template("preset_viral_bold").unwrap());

    let ass_with_intel = captions::generate_ass_from_template_with_framing_and_intel(
        &words,
        start_sec,
        end_sec,
        &template,
        None,
        plan.as_ref(),
        None,
    );

    let baseline_ass = captions::generate_ass_from_template_with_framing_and_intel(
        &words, start_sec, end_sec, &template, None, None, None,
    );

    let ass_matches_baseline = ass_with_intel == baseline_ass;

    let out = OutputJson {
        plan,
        ass_with_intel,
        baseline_ass,
        ass_matches_baseline,
        kill_switch_active,
    };

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
