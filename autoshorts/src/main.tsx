import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import {
  AlertCircle,
  AudioLines,
  BadgeCheck,
  Captions,
  Check,
  ChevronLeft,
  ChevronRight,
  Clapperboard,
  Download,
  FileVideo,
  Loader2,
  Play,
  RefreshCw,
  Scissors,
  SlidersHorizontal,
  Sparkles,
  Wand2,
  Copy,
  Database,
  Cloud,
  Youtube,
  AlertTriangle,
  Search,
  Clock,
  Calendar,
  ArrowUpDown,
  Upload,
  FolderOpen,
  Film,
  ArrowLeft,
  ArrowRight,
  Crop,
  Smartphone,
  ShieldCheck,
  Cpu,
  Key,
  Info,
  Lock,
  Zap,
  Layers,
  RotateCcw,
  X,
  Palette,
} from "lucide-react";
import "./styles.css";

type EnvironmentStatus = {
  dataDir: string;
  hasFfmpeg: boolean;
  hasFfprobe: boolean;
  hasDeepgramKey: boolean;
  hasAnthropicKey: boolean;
  hasDeepseekKey: boolean;
  hasGeminiKey: boolean;
  hasOpenaiKey: boolean;
  hasOpenrouterKey: boolean;
  hasGroqKey: boolean;
  hasNvidiaKey: boolean;
  llmProvider: string;
  hasLocalWhisperModel: boolean;
  hasOllama: boolean;
  hasYtdlp: boolean;
};

type Project = {
  id: string;
  name: string | null;
  sourcePath: string;
  sourceDuration: number | null;
  status: string;
  transcriptionMode: string;
  captionStyle?: string | null;
  framingMode?: string | null;
  presentationMode?: string | null;
  presentationConfig?: string | null;
  createdAt: string;
  updatedAt: string;
};

type PresentationEngineMode = "autoshorts" | "autoclip";
type AutoClipPlatformStrategy = "douyin" | "tiktok" | "youtube_shorts" | "bilibili" | "xiaohongshu";
type AutoClipSubtitleLook = "clean" | "bold" | "box" | "accent";
type AutoClipTitleStyle = "comic" | "neon" | "arena" | "editorial" | "pixel" | "frosted" | "plain";
type AutoClipTemplate = "interview_zh" | "podcast_en" | "none";

type Transcript = {
  id: string;
  projectId: string;
  engine: string;
  rawJson: string;
  language: string | null;
  createdAt: string;
};

type Candidate = {
  id: string;
  projectId: string;
  startSec: number;
  endSec: number;
  score: number;
  hook: string;
  rationale: string;
  rank: number;
  selected: boolean;
  metadataJson?: string | null;
};

type Clip = {
  id: string;
  candidateId: string;
  status: string;
  outputPath: string | null;
  faceTrackJson: string | null;
  captionAssPath: string | null;
  renderLog: string | null;
  appliedFeatures: string | null;
};

type SceneIntelStatus = {
  projectId: string;
  status: "idle" | "running" | "completed" | "error";
  progress: number;
  sceneCount: number;
};

interface AudioTrackInfo {
  formatId: string;
  language?: string | null;
  languageName: string;
  formatNote?: string | null;
  isOriginal: boolean;
  bitrateKbps?: number | null;
}

interface YoutubeProbeResult {
  isSafe: boolean;
  license?: string | null;
  title?: string | null;
  duration?: number | null;
  audioTracks: AudioTrackInfo[];
  defaultAudioFormatId?: string | null;
}

// Score provenance from the backend's candidate metadata. "llm" = a genuine
// NVIDIA AutoClip model score; "fallback" = explicitly unscored (score 0.0,
// never AutoClip-eligible). null = legacy composite scoring. The UI only
// READS this state — it never recomputes scores, ranks or eligibility.
function candidateScoreSource(candidate: Candidate): string | null {
  if (!candidate.metadataJson) return null;
  try {
    const meta = JSON.parse(candidate.metadataJson) as { scoreSource?: string | null };
    return typeof meta.scoreSource === "string" ? meta.scoreSource : null;
  } catch {
    return null;
  }
}

function formatTimestamp(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || isNaN(seconds)) return "00:00:00";
  const total = Math.max(0, Math.floor(seconds));
  const hrs = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return `${hrs.toString().padStart(2, "0")}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

type ProjectDetail = {
  project: Project;
  transcript: Transcript | null;
  candidates: Candidate[];
  clips: Clip[];
};

type NormalizedTranscript = {
  language: string;
  duration: number;
  speakers: string[];
  segments: Array<{
    start: number;
    end: number;
    speaker: string | null;
    text: string;
  }>;
};

type BusyState =
  | "idle"
  | "import"
  | "transcribe"
  | "demoTranscript"
  | "moments"
  | "clipCount"
  | "cut";

const CAPTION_PRESETS = [
  { id: "preset_minimal_capsule", name: "Minimal Capsule", family: "clean" },
  { id: "preset_viral_bold", name: "Hormozi Viral", family: "bold" },
  { id: "preset_mrbeast_pop", name: "Narrative Pop", family: "highlight" },
  { id: "preset_cinematic_vlog", name: "Cinematic Vlog", family: "clean" },
  { id: "preset_dynamic_editorial", name: "Dynamic Editorial", family: "highlight" },
  { id: "preset_t7", name: "T7 Rhythm", family: "highlight" },
  { id: "preset_bhaukal_caption", name: "Bhaukal Caption", family: "bold" },
] as const;

function App() {
  const [environment, setEnvironment] = useState<EnvironmentStatus | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [sceneIntelStatus, setSceneIntelStatus] = useState<SceneIntelStatus | null>(null);
  const [busy, setBusy] = useState<BusyState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [renderingCandidateId, setRenderingCandidateId] = useState<string | null>(null);
  const [showStyleModal, setShowStyleModal] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState<string>("preset_minimal_capsule");
  const [selectedFramingMode, setSelectedFramingMode] = useState<string>("adaptive");
  // Initial Setup Modal Presentation Engine & Mode 2 Configuration State
  const [selectedPresentationMode, setSelectedPresentationMode] = useState<PresentationEngineMode>("autoshorts");
  const [platformStrategy, setPlatformStrategy] = useState<AutoClipPlatformStrategy>("douyin");
  const [subtitleLook, setSubtitleLook] = useState<AutoClipSubtitleLook>("accent");
  const [titleStyle, setTitleStyle] = useState<AutoClipTitleStyle>("comic");
  const [titleMotion, setTitleMotion] = useState<boolean>(true);
  const [template, setTemplate] = useState<AutoClipTemplate>("interview_zh");
  // Candidate discovery engine for "Find Viral Moments": "timestamp_generation"
  // is the production default; "window_scoring" (REZE) is the experimental
  // opt-in. Per generation call — flip and re-run to compare on the same project.
  const [discoveryMode, setDiscoveryMode] = useState<string>("timestamp_generation");
  const [rezeProvider, setRezeProvider] = useState<string>("");
  // Scoring-mode selector (import/caption-style dialog + Find Viral Moments):
  // "legacy" (default) keeps the composite-score pipeline; "autoclip" makes the
  // NVIDIA holistic final_score authoritative (eligibility gate + deterministic
  // Top-N). Persisted like the other engine settings; changing captions or
  // framing never resets it. Passed explicitly to generate_candidates so the
  // selection overrides the backend's env default for the current operation.
  const [scoringMode, setScoringMode] = useState<"legacy" | "autoclip">(() => {
    return (localStorage.getItem("autoshorts_scoring_mode") as "legacy" | "autoclip") || "legacy";
  });
  const [mediaPathToImport, setMediaPathToImport] = useState<string | null>(null);
  const mediaFilename = useMemo(() => {
    if (!mediaPathToImport) return "";
    const parts = mediaPathToImport.split(/[\\/]/);
    return parts[parts.length - 1] || mediaPathToImport;
  }, [mediaPathToImport]);

  const [youtubeModalOpen, setYoutubeModalOpen] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [youtubeStatus, setYoutubeStatus] = useState<"idle" | "checking" | "warning" | "ready" | "downloading">("idle");
  const [youtubeWarningLicense, setYoutubeWarningLicense] = useState<string | null>(null);
  const [youtubeError, setYoutubeError] = useState<string | null>(null);
  const [youtubeAudioTracks, setYoutubeAudioTracks] = useState<AudioTrackInfo[]>([]);
  const [selectedAudioFormatId, setSelectedAudioFormatId] = useState<string | null>(null);
  const [youtubeVideoTitle, setYoutubeVideoTitle] = useState<string | null>(null);

  // Style Studio Drawer State (Screen D & E)
  const [styleStudioOpen, setStyleStudioOpen] = useState(false);
  const [styleStudioTargetCandidateId, setStyleStudioTargetCandidateId] = useState<string | null>(null);
  const [styleStudioFraming, setStyleStudioFraming] = useState<"adaptive" | "original">("adaptive");
  const [styleStudioCaption, setStyleStudioCaption] = useState<string>("preset_minimal_capsule");
  const [styleStudioPresentationMode, setStyleStudioPresentationMode] = useState<PresentationEngineMode>("autoshorts");
  const [styleStudioPlatformStrategy, setStyleStudioPlatformStrategy] = useState<AutoClipPlatformStrategy>("douyin");
  const [styleStudioSubtitleLook, setStyleStudioSubtitleLook] = useState<AutoClipSubtitleLook>("accent");
  const [styleStudioTitleStyle, setStyleStudioTitleStyle] = useState<AutoClipTitleStyle>("comic");
  const [styleStudioTitleMotion, setStyleStudioTitleMotion] = useState<boolean>(true);
  const [styleStudioTemplate, setStyleStudioTemplate] = useState<AutoClipTemplate>("interview_zh");
  const [styleStudioScope, setStyleStudioScope] = useState<"target" | "selected" | "all">("target");
  const [previousRenderHistory, setPreviousRenderHistory] = useState<Map<string, string>>(new Map());

  // Screen A Discovery & Dashboard Filter State
  const [projectSearch, setProjectSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "ready" | "processing" | "failed">("all");
  const [sortOption, setSortOption] = useState<"recent" | "date" | "duration">("recent");

  // Screen B Transcript Search State
  const [transcriptSearch, setTranscriptSearch] = useState("");

  // Synchronous in-flight guards against rapid duplicate clicks / concurrent triggers
  const isCuttingRef = useRef(false);
  const activeCuttingCandidatesRef = useRef(new Set<string>());

  // Persistence logic from localStorage
  const [isOnboarded, setIsOnboarded] = useState<boolean | null>(null);
  const [transcriptionEngine, setTranscriptionEngine] = useState<"deepgram" | "local">(() => {
    return (localStorage.getItem("autoshorts_transcription_engine") as "deepgram" | "local") || "local";
  });
  const [llmEngine, setLlmEngine] = useState<"claude" | "deepseek" | "local" | "gemini" | "openai" | "openrouter" | "groq" | "nvidia">(() => {
    return (localStorage.getItem("autoshorts_llm_engine") as "claude" | "deepseek" | "local" | "gemini" | "openai" | "openrouter" | "groq" | "nvidia") || "gemini";
  });
  const [localLlmModel, setLocalLlmModel] = useState(() => {
    return localStorage.getItem("autoshorts_local_llm_model") || "llama3.2";
  });
  const [deepgramKey, setDeepgramKey] = useState(() => {
    return localStorage.getItem("autoshorts_deepgram_key") || "";
  });
  const [anthropicKey, setAnthropicKey] = useState(() => {
    return localStorage.getItem("autoshorts_anthropic_key") || "";
  });
  const [deepseekKey, setDeepseekKey] = useState(() => {
    return localStorage.getItem("autoshorts_deepseek_key") || "";
  });
  const [deepseekModel, setDeepseekModel] = useState(() => {
    return localStorage.getItem("autoshorts_deepseek_model") || "";
  });
  const [geminiKey, setGeminiKey] = useState(() => {
    return localStorage.getItem("autoshorts_gemini_key") || "";
  });
  const [openaiKey, setOpenaiKey] = useState(() => {
    return localStorage.getItem("autoshorts_openai_key") || "";
  });
  const [openrouterKey, setOpenrouterKey] = useState(() => {
    return localStorage.getItem("autoshorts_openrouter_key") || "";
  });
  const [groqKey, setGroqKey] = useState(() => {
    return localStorage.getItem("autoshorts_groq_key") || "";
  });
  const [nvidiaKey, setNvidiaKey] = useState(() => {
    return localStorage.getItem("autoshorts_nvidia_key") || "";
  });
  const [nvidiaModel, setNvidiaModel] = useState(() => {
    const stored = localStorage.getItem("autoshorts_nvidia_model");
    return (stored && stored !== "stepfun-ai/step-5-preview") ? stored : "google/gemma-4-31b-it";
  });
  const [openrouterModel, setOpenrouterModel] = useState(() => {
    return localStorage.getItem("autoshorts_openrouter_model") || "google/gemini-2.5-flash";
  });
  const [youtubeBrowser, setYoutubeBrowser] = useState(() => {
    return localStorage.getItem("autoshorts_youtube_browser") || "auto";
  });
  const [youtubeCookiesPath, setYoutubeCookiesPath] = useState(() => {
    return localStorage.getItem("autoshorts_youtube_cookies_path") || "";
  });

  const [downloadingModelName, setDownloadingModelName] = useState<string | null>(null);
  const [modelDownloadStatus, setModelDownloadStatus] = useState("");
  const [modelDownloadProgress, setModelDownloadProgress] = useState(0);
  const [candidateFilter, setCandidateFilter] = useState<"all" | "selected" | "pool">("all");

  // Range-based candidate selection (1-based, inclusive candidate ranks).
  // `toValue === null` means "Last" (the final candidate).
  const [rangeFromValue, setRangeFromValue] = useState("1");
  const [rangeToValue, setRangeToValue] = useState<string | null>(null);

  // Echo-guard: the last range written through updateSelectionRange, in derived
  // form (to === null means "Last"). Lets the sync effect distinguish our own
  // writes echoing back through detail.candidates from external reloads.
  const lastAppliedRangeRef = useRef<{ from: number; to: number | null } | null>(null);

  // Serialized selection updates: a monotonic token taken per selection
  // request (rank range or Top-N). A response carrying an older token than
  // the latest issued request is stale and its result is discarded, so an
  // out-of-order invoke resolution can never overwrite a newer selection with
  // older data. The backend remains authoritative for all selection state.
  const selectionSeqRef = useRef(0);

  // Transient Top-N input state (the backend response is authoritative).
  const [topNValue, setTopNValue] = useState("");

  function getProjectNormalizedStatus(status: string): "ready" | "processing" | "failed" {
    const s = (status || "").toLowerCase();
    if (s === "ready" || s === "done" || s === "completed") return "ready";
    if (s === "failed" || s === "error") return "failed";
    return "processing";
  }

  function getProjectStatusLabel(status: string): string {
    const norm = getProjectNormalizedStatus(status);
    if (norm === "ready") return "Ready";
    if (norm === "failed") return "Failed";
    const s = (status || "").toLowerCase();
    if (s === "transcribing") return "Transcribing";
    if (s === "analyzing") return "Analyzing";
    if (s === "ingest") return "Ingesting";
    return "Processing";
  }

  const projectCounts = useMemo(() => {
    let ready = 0;
    let processing = 0;
    let failed = 0;
    for (const p of projects) {
      const cat = getProjectNormalizedStatus(p.status);
      if (cat === "ready") ready++;
      else if (cat === "processing") processing++;
      else if (cat === "failed") failed++;
    }
    return {
      all: projects.length,
      ready,
      processing,
      failed,
    };
  }, [projects]);

  const recentProjects = useMemo(() => {
    return [...projects]
      .sort((a, b) => {
        const timeB = new Date(b.updatedAt || b.createdAt).getTime() || 0;
        const timeA = new Date(a.updatedAt || a.createdAt).getTime() || 0;
        return timeB - timeA;
      })
      .slice(0, 5);
  }, [projects]);

  const filteredProjects = useMemo(() => {
    let list = [...projects];

    if (projectSearch.trim()) {
      const query = projectSearch.toLowerCase();
      list = list.filter((p) => {
        const title = (p.name || fileName(p.sourcePath)).toLowerCase();
        return title.includes(query);
      });
    }

    if (statusFilter !== "all") {
      list = list.filter((p) => getProjectNormalizedStatus(p.status) === statusFilter);
    }

    list.sort((a, b) => {
      if (sortOption === "recent") {
        const timeB = new Date(b.updatedAt || b.createdAt).getTime() || 0;
        const timeA = new Date(a.updatedAt || a.createdAt).getTime() || 0;
        return timeB - timeA;
      }
      if (sortOption === "date") {
        const timeB = new Date(b.createdAt).getTime() || 0;
        const timeA = new Date(a.createdAt).getTime() || 0;
        return timeB - timeA;
      }
      if (sortOption === "duration") {
        return (b.sourceDuration || 0) - (a.sourceDuration || 0);
      }
      return 0;
    });

    return list;
  }, [projects, projectSearch, statusFilter, sortOption]);

  const transcript = useMemo(() => {
    if (!detail?.transcript) return null;
    try {
      return JSON.parse(detail.transcript.rawJson) as NormalizedTranscript;
    } catch {
      return null;
    }
  }, [detail?.transcript]);

  const totalCandidateCount = detail?.candidates.length ?? 0;
  const selectedCount = detail?.candidates.filter((candidate) => candidate.selected).length ?? 0;
  const poolCount = totalCandidateCount - selectedCount;

  // Parsed range inputs. `to === null` means "Last" (the final candidate rank).
  const rangeFrom = Number.parseInt(rangeFromValue, 10);
  const rangeTo = rangeToValue === null ? totalCandidateCount : Number.parseInt(rangeToValue, 10);
  const rangeInputsValid =
    Number.isInteger(rangeFrom) &&
    Number.isInteger(rangeTo) &&
    rangeFrom >= 1 &&
    rangeTo <= totalCandidateCount &&
    rangeFrom <= rangeTo;
  const rangeSelectedCount = rangeInputsValid ? rangeTo - rangeFrom + 1 : 0;

  const displayedCandidates = useMemo(() => {
    if (!detail?.candidates) return [];
    if (candidateFilter === "selected") {
      return detail.candidates.filter((c) => c.selected);
    }
    if (candidateFilter === "pool") {
      return detail.candidates.filter((c) => !c.selected);
    }
    return detail.candidates;
  }, [detail?.candidates, candidateFilter]);

  const clipByCandidate = useMemo(() => {
    return new Map(detail?.clips.map((clip) => [clip.candidateId, clip]) ?? []);
  }, [detail?.clips]);
  const selectedCandidates = detail?.candidates.filter((candidate) => candidate.selected) ?? [];
  const selectedCutCount = selectedCandidates.filter((candidate) => {
    const clip = clipByCandidate.get(candidate.id);
    return clip?.status === "done" && Boolean(clip.outputPath);
  }).length;
  const selectedCaptionsCount = selectedCandidates.filter((candidate) => {
    const clip = clipByCandidate.get(candidate.id);
    return clip?.status === "done" && Boolean(clip.captionAssPath);
  }).length;

  const targetCandidate = useMemo(() => {
    if (!detail || !styleStudioTargetCandidateId) return null;
    return detail.candidates.find((c) => c.id === styleStudioTargetCandidateId) || null;
  }, [detail, styleStudioTargetCandidateId]);

  const targetCandidateIds = useMemo(() => {
    if (!detail) return [];
    if (styleStudioScope === "target" && styleStudioTargetCandidateId) {
      return [styleStudioTargetCandidateId];
    }
    if (styleStudioScope === "selected") {
      const ids = selectedCandidates.map((c) => c.id);
      if (ids.length > 0) return ids;
      return styleStudioTargetCandidateId ? [styleStudioTargetCandidateId] : [];
    }
    if (styleStudioScope === "all") {
      return detail.candidates.map((c) => c.id);
    }
    return styleStudioTargetCandidateId ? [styleStudioTargetCandidateId] : [];
  }, [detail, styleStudioScope, styleStudioTargetCandidateId, selectedCandidates]);

  const hasPreviousVersion = useMemo(() => {
    return targetCandidateIds.some((id) => previousRenderHistory.has(id));
  }, [targetCandidateIds, previousRenderHistory]);

  const filteredTranscriptSegments = useMemo(() => {
    if (!transcript) return [];
    if (!transcriptSearch.trim()) return transcript.segments;
    const q = transcriptSearch.toLowerCase().trim();
    return transcript.segments.filter(
      (seg) =>
        seg.text.toLowerCase().includes(q) ||
        (seg.speaker && seg.speaker.toLowerCase().includes(q))
    );
  }, [transcript, transcriptSearch]);

  const activePipelineStep = useMemo(() => {
    if (!detail) return "Transcript";
    if (!detail.transcript) return "Transcript";
    if (detail.candidates.length === 0) return "Moments";
    if (selectedCount === 0 || selectedCutCount < selectedCount) return "Cut";
    if (selectedCaptionsCount < selectedCount) return "Captions";
    return "Export";
  }, [detail, selectedCount, selectedCutCount, selectedCaptionsCount]);
  const canUseCloudKey = environment?.hasDeepgramKey || deepgramKey.trim().length > 0;
  const canUseClaude = environment?.hasAnthropicKey || anthropicKey.trim().length > 0;
  const canUseDeepseek = environment?.hasDeepseekKey || deepseekKey.trim().length > 0;
  const canUseGemini = environment?.hasGeminiKey || geminiKey.trim().length > 0;
  const canUseOpenai = environment?.hasOpenaiKey || openaiKey.trim().length > 0;
  const canUseOpenrouter = environment?.hasOpenrouterKey || openrouterKey.trim().length > 0;
  const canUseGroq = environment?.hasGroqKey || groqKey.trim().length > 0;
  const canUseNvidia = environment?.hasNvidiaKey || nvidiaKey.trim().length > 0;

  const canTranscribe = transcriptionEngine === "local"
    ? Boolean(environment?.hasLocalWhisperModel)
    : canUseCloudKey;

  const canUseActiveLlm = llmEngine === "local"
    ? Boolean(environment?.hasOllama)
    : llmEngine === "claude"
      ? canUseClaude
      : llmEngine === "deepseek"
        ? canUseDeepseek
        : llmEngine === "gemini"
          ? canUseGemini
          : llmEngine === "openai"
            ? canUseOpenai
            : llmEngine === "openrouter"
              ? canUseOpenrouter
              : llmEngine === "groq"
                ? canUseGroq
                : llmEngine === "nvidia"
                  ? canUseNvidia
                  : false;

  const isRezeMode = discoveryMode === "window_scoring";
  const effectiveTimestampEngine = llmEngine === "openrouter" ? "deepseek" : llmEngine;
  const canUseDiscoveryLlm = isRezeMode
    ? true
    : (effectiveTimestampEngine === "deepseek"
        ? canUseDeepseek
        : canUseActiveLlm);

  // Keep the From/To inputs in sync with the persisted selection when the
  // candidate list is (re)loaded from an external source (project switch,
  // moments regeneration, post-cut refresh). Skips our own writes echoing
  // back, so typing in the inputs is never clobbered mid-edit.
  useEffect(() => {
    const candidates = detail?.candidates;
    if (!candidates || candidates.length === 0) return;
    const selectedRanks = candidates.filter((c) => c.selected).map((c) => c.rank);
    if (selectedRanks.length === 0) {
      setTopNValue("0");
      setRangeFromValue("");
      setRangeToValue(null);
      lastAppliedRangeRef.current = null;
      return;
    }
    const from = Math.min(...selectedRanks);
    const to = Math.max(...selectedRanks);
    const isContiguous = selectedRanks.length === (to - from + 1);
    if (!isContiguous) {
      // Non-contiguous individual selection (e.g. #1, #36, #47)
      setRangeFromValue("");
      setRangeToValue(null);
      setTopNValue("");
      lastAppliedRangeRef.current = null;
      return;
    }
    const lastApplied = lastAppliedRangeRef.current;
    const isEcho = lastApplied !== null && lastApplied.from === from &&
      (lastApplied.to === null ? to === candidates.length : lastApplied.to === to);
    if (isEcho) return;
    setRangeFromValue(String(from));
    setRangeToValue(to === candidates.length ? null : String(to));
    if (from === 1) {
      setTopNValue(String(to));
    } else {
      setTopNValue("");
    }
  }, [detail?.candidates]);

  useEffect(() => {
    setTranscriptSearch("");
  }, [detail?.project.id]);

  useEffect(() => {
    if (!detail?.project?.id) {
      setSceneIntelStatus(null);
      return;
    }
    const pid = detail.project.id;
    let isCancelled = false;
    let unlistenFn: (() => void) | null = null;

    // Immediately provide a default idle status so the pill is ALWAYS visible without IPC delay
    setSceneIntelStatus((prev) => (prev?.projectId === pid ? prev : {
      projectId: pid,
      status: "idle",
      progress: 0,
      sceneCount: 0,
    }));

    // Hydrate initial status
    invoke<SceneIntelStatus>("get_scene_intelligence_status", { projectId: pid })
      .then((status) => {
        if (!isCancelled && status) setSceneIntelStatus(status);
      })
      .catch((err) => console.warn("[SceneIntel] Failed to query status:", err));

    // Listen to real-time events
    listen<SceneIntelStatus>("scene-intel-status", (event) => {
      if (!isCancelled && event.payload.projectId === pid) {
        setSceneIntelStatus(event.payload);
      }
    }).then((unlisten) => {
      if (isCancelled) {
        unlisten();
      } else {
        unlistenFn = unlisten;
      }
    });

    return () => {
      isCancelled = true;
      if (unlistenFn) unlistenFn();
    };
  }, [detail?.project?.id]);

  useEffect(() => {
    void refresh();
    const value = localStorage.getItem("autoshorts_onboarded");
    if (value === "true") {
      setIsOnboarded(true);
    } else {
      setIsOnboarded(false);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("autoshorts_transcription_engine", transcriptionEngine);
  }, [transcriptionEngine]);

  useEffect(() => {
    localStorage.setItem("autoshorts_scoring_mode", scoringMode);
  }, [scoringMode]);

  useEffect(() => {
    localStorage.setItem("autoshorts_llm_engine", llmEngine);
  }, [llmEngine]);

  useEffect(() => {
    localStorage.setItem("autoshorts_local_llm_model", localLlmModel);
  }, [localLlmModel]);

  useEffect(() => {
    localStorage.setItem("autoshorts_deepgram_key", deepgramKey);
  }, [deepgramKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_anthropic_key", anthropicKey);
  }, [anthropicKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_deepseek_key", deepseekKey);
  }, [deepseekKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_deepseek_model", deepseekModel);
  }, [deepseekModel]);

  useEffect(() => {
    localStorage.setItem("autoshorts_gemini_key", geminiKey);
  }, [geminiKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_openai_key", openaiKey);
  }, [openaiKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_openrouter_key", openrouterKey);
  }, [openrouterKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_groq_key", groqKey);
  }, [groqKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_nvidia_key", nvidiaKey);
  }, [nvidiaKey]);

  useEffect(() => {
    localStorage.setItem("autoshorts_nvidia_model", nvidiaModel);
  }, [nvidiaModel]);

  useEffect(() => {
    localStorage.setItem("autoshorts_openrouter_model", openrouterModel);
  }, [openrouterModel]);

  useEffect(() => {
    localStorage.setItem("autoshorts_youtube_browser", youtubeBrowser);
  }, [youtubeBrowser]);

  useEffect(() => {
    localStorage.setItem("autoshorts_youtube_cookies_path", youtubeCookiesPath);
  }, [youtubeCookiesPath]);

  const pullModelDirectly = async (modelName: string) => {
    setDownloadingModelName(modelName);
    setModelDownloadProgress(0);
    setModelDownloadStatus("Connecting to Ollama...");
    try {
      const unlisten = await listen<{
        status: string;
        completed?: number;
        total?: number;
        percentage?: number;
      }>("ollama-pull-progress", (event) => {
        const payload = event.payload;
        setModelDownloadStatus(payload.status);
        if (payload.percentage !== undefined && payload.percentage !== null) {
          setModelDownloadProgress(Math.round(payload.percentage));
        }
      });

      await invoke("pull_ollama_model", { modelName });
      unlisten();
      setModelDownloadStatus("Download complete!");
      setModelDownloadProgress(100);
      setTimeout(() => setDownloadingModelName(null), 500);
    } catch (err) {
      alert("Failed to download model: " + String(err));
      setDownloadingModelName(null);
    }
  };

  async function refresh(nextProjectId?: string) {
    setError(null);
    const [env, projectList] = await Promise.all([
      invoke<EnvironmentStatus>("environment_status"),
      invoke<Project[]>("list_projects"),
    ]);
    setEnvironment(env);
    setProjects(projectList);

    if (env?.llmProvider) {
      const p = env.llmProvider.toLowerCase();
      const validEngines = ["claude", "deepseek", "local", "gemini", "openai", "openrouter", "groq", "nvidia"];
      if (validEngines.includes(p)) {
        setLlmEngine(p as any);
      }
    }

    if (nextProjectId) {
      const nextDetail = await invoke<ProjectDetail>("get_project_detail", { projectId: nextProjectId });
      setDetail(nextDetail);
    } else {
      setDetail(null);
    }
  }

  async function run(action: BusyState, task: () => Promise<void>) {
    setBusy(action);
    setError(null);
    try {
      await task();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("idle");
    }
  }

  async function importMedia() {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: "Media",
          extensions: ["mp4", "mov", "mp3", "wav", "m4a"],
        },
      ],
    });
    if (typeof selected !== "string") return;
    setSelectedStyle("preset_minimal_capsule");
    setSelectedFramingMode("adaptive");
    setSelectedPresentationMode("autoshorts");
    setPlatformStrategy("douyin");
    setSubtitleLook("accent");
    setTitleStyle("comic");
    setTitleMotion(true);
    setTemplate("interview_zh");
    setMediaPathToImport(selected);
    setShowStyleModal(true);
  }

  async function handleYoutubeImport() {
    if (!youtubeUrl) return;
    setYoutubeStatus("checking");
    setYoutubeError(null);
    setError(null);
    setYoutubeAudioTracks([]);
    setSelectedAudioFormatId(null);
    setYoutubeVideoTitle(null);
    try {
      console.log("[YouTube Frontend] Probing media for:", youtubeUrl, "browser:", youtubeBrowser);
      const result = await invoke<YoutubeProbeResult>("probe_youtube_media", {
        url: youtubeUrl,
        browser: youtubeBrowser !== "auto" ? youtubeBrowser : null,
        cookiesPath: youtubeCookiesPath.trim() || null,
      });
      console.log("[YouTube Frontend] Probe media result:", result);
      const tracks = result.audioTracks || [];
      setYoutubeAudioTracks(tracks);
      setYoutubeVideoTitle(result.title || null);
      const chosenDefaultId = result.defaultAudioFormatId || (tracks.length > 0 ? tracks[0].formatId : null);
      setSelectedAudioFormatId(chosenDefaultId);

      if (!result.isSafe) {
        setYoutubeWarningLicense(result.license || "Unknown / Not specified");
        setYoutubeStatus("warning");
        return;
      }

      if (tracks.length > 1) {
        // Multi-track audio detected! Prompt user with interactive selector
        setYoutubeStatus("ready");
        return;
      }

      await executeYoutubeDownload(chosenDefaultId);
    } catch (err: any) {
      const errMsg = err?.toString() || "Failed to check YouTube video details.";
      console.error("[YouTube Frontend] Probe error:", errMsg);
      setYoutubeError(errMsg);
      setError(errMsg);
      setYoutubeStatus("idle");
    }
  }

  async function executeYoutubeDownload(audioFormatIdOverride?: string | null) {
    setYoutubeStatus("downloading");
    setYoutubeError(null);
    setError(null);
    const chosenFormatId = audioFormatIdOverride !== undefined ? audioFormatIdOverride : selectedAudioFormatId;
    try {
      console.log("[YouTube Frontend] Starting download for:", youtubeUrl, "audio format:", chosenFormatId, "browser:", youtubeBrowser);
      const downloadedPath = await invoke<string>("download_youtube_video", {
        url: youtubeUrl,
        audioFormatId: chosenFormatId || null,
        browser: youtubeBrowser !== "auto" ? youtubeBrowser : null,
        cookiesPath: youtubeCookiesPath.trim() || null,
      });
      console.log("[YouTube Frontend] Download completed successfully. Path:", downloadedPath);
      setYoutubeModalOpen(false);
      setYoutubeUrl("");
      setYoutubeStatus("idle");
      setYoutubeWarningLicense(null);
      setYoutubeError(null);
      setYoutubeAudioTracks([]);
      setSelectedAudioFormatId(null);
      setYoutubeVideoTitle(null);
      setSelectedStyle("preset_minimal_capsule");
      setSelectedFramingMode("adaptive");
      setSelectedPresentationMode("autoshorts");
      setPlatformStrategy("douyin");
      setSubtitleLook("accent");
      setTitleStyle("comic");
      setTitleMotion(true);
      setTemplate("interview_zh");
      setMediaPathToImport(downloadedPath);
      setShowStyleModal(true);
    } catch (err: any) {
      const errMsg = err?.toString() || "Failed to download YouTube video.";
      console.error("[YouTube Frontend] Download error:", errMsg);
      setYoutubeError(errMsg);
      setError(errMsg);
      setYoutubeStatus(youtubeAudioTracks.length > 1 ? "ready" : "idle");
    }
  }

  async function confirmImport(style: string | null, framingMode: string) {
    if (!mediaPathToImport) return;
    const selected = mediaPathToImport;
    const chosenStyle = style || "preset_minimal_capsule";
    const chosenFramingMode = framingMode;
    const chosenScoringMode = scoringMode;
    const chosenPresentationMode = selectedPresentationMode;
    const chosenPresentationConfig = JSON.stringify({
      platform_strategy: platformStrategy,
      platformStrategy: platformStrategy,
      subtitle_style: subtitleLook,
      subtitleLook: subtitleLook,
      title_style: "plain",
      titleStyle: "plain",
      title_motion: false,
      titleMotion: false,
      template: "none",
    });
    setMediaPathToImport(null);
    setShowStyleModal(false);

    let newProjectId: string | null = null;
    await run("import", async () => {
      const project = await invoke<Project>("create_project_from_path", {
        path: selected,
        transcriptionMode: transcriptionEngine === "local" ? "local" : "cloud",
        captionStyle: chosenStyle,
        framingMode: chosenFramingMode,
        presentationMode: chosenPresentationMode,
        presentationConfig: chosenPresentationConfig,
      });
      newProjectId = project.id;
      await refresh(project.id);
    });

    if (newProjectId) {
      await runAutoPipeline(newProjectId, chosenScoringMode);
    }
  }

  async function runAutoPipeline(projectId: string, overrideScoringMode?: "legacy" | "autoclip") {
    setError(null);
    const activeScoringMode = overrideScoringMode || scoringMode;
    const env = await invoke<EnvironmentStatus>("environment_status");

    // AutoClip Scoring requires the NVIDIA key BEFORE any billable work
    // (transcription included). The backend would hard-error at the start of
    // generate_candidates anyway; failing fast here avoids spending the
    // transcription call. Never silently falls back to Legacy Scoring.
    if (activeScoringMode === "autoclip" && !env.hasNvidiaKey) {
      setError("Import successful. AutoClip Scoring (NVIDIA) is selected but NVIDIA_API_KEY is missing. Configure the key in your .env or environment, or switch the scoring mode to Legacy Scoring.");
      return;
    }

    if (transcriptionEngine === "local") {
      if (!env.hasLocalWhisperModel) {
        setError("Import successful. Local Whisper GGML model (ggml-base.bin) is missing in your models directory. Please add it to start transcription.");
        return;
      }
    } else {
      const hasDG = env.hasDeepgramKey || deepgramKey.trim().length > 0;
      if (!hasDG) {
        setError("Import successful. Deepgram key is missing. Please add it to start transcription.");
        return;
      }
    }

    const isReze = discoveryMode === "window_scoring";
    const targetProvider = isReze
      ? (rezeProvider || "nvidia_diffusiongemma")
      : (llmEngine === "openrouter" ? "deepseek" : llmEngine);

    if (isReze) {
      // REZE window scoring uses NVIDIA NIM via backend with safe DeepSeek fallback
    } else if (targetProvider === "local") {
      if (!env.hasOllama) {
        setError("Import successful. Local Ollama server is not running at http://localhost:11434. Please start it to find viral moments.");
        return;
      }
    } else {
      const activeKey =
        targetProvider === "claude" ? anthropicKey :
          targetProvider === "deepseek" ? deepseekKey :
            targetProvider === "gemini" ? geminiKey :
              targetProvider === "openai" ? openaiKey :
                targetProvider === "groq" ? groqKey :
                  targetProvider === "nvidia" ? nvidiaKey : "";
      const hasActiveKey =
        targetProvider === "claude" ? (env.hasAnthropicKey || activeKey.trim().length > 0) :
          targetProvider === "deepseek" ? (env.hasDeepseekKey || activeKey.trim().length > 0) :
            targetProvider === "gemini" ? (env.hasGeminiKey || activeKey.trim().length > 0) :
              targetProvider === "openai" ? (env.hasOpenaiKey || activeKey.trim().length > 0) :
                targetProvider === "groq" ? (env.hasGroqKey || activeKey.trim().length > 0) :
                  targetProvider === "nvidia" ? (env.hasNvidiaKey || activeKey.trim().length > 0) : false;
      if (!hasActiveKey) {
        const engineName =
          targetProvider === "claude" ? "Claude" :
            targetProvider === "deepseek" ? "DeepSeek" :
              targetProvider === "gemini" ? "Gemini" :
                targetProvider === "openai" ? "OpenAI" :
                  targetProvider === "groq" ? "Groq" :
                    targetProvider === "nvidia" ? "NVIDIA NIM" : "LLM";
        setError(`Transcription complete. ${engineName} API Key is missing. Please add it in settings to analyze viral moments.`);
        return;
      }
    }

    // 1. Transcription
    try {
      setBusy("transcribe");
      await invoke<Transcript>("transcribe_project", {
        projectId,
        provider: transcriptionEngine,
        apiKey: transcriptionEngine === "deepgram" ? (deepgramKey.trim() || null) : null,
      });
      await refresh(projectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy("idle");
      return;
    }

    // 2. LLM Moments
    try {
      setBusy("moments");
      const targetKey = isReze
        ? ""
        : (targetProvider === "claude" ? anthropicKey.trim() :
            targetProvider === "deepseek" ? deepseekKey.trim() :
              targetProvider === "gemini" ? geminiKey.trim() :
                targetProvider === "openai" ? openaiKey.trim() :
                  targetProvider === "groq" ? groqKey.trim() :
                    targetProvider === "nvidia" ? nvidiaKey.trim() : "");
      const targetModel = isReze
        ? null
        : (targetProvider === "local" ? localLlmModel.trim() :
            targetProvider === "deepseek" ? (deepseekModel.trim() || "deepseek-chat") :
              targetProvider === "nvidia" ? ((nvidiaModel.trim() && nvidiaModel.trim() !== "stepfun-ai/step-5-preview") ? nvidiaModel.trim() : "google/gemma-4-31b-it") : null);

      await invoke<Candidate[]>("generate_candidates", {
        projectId,
        apiKey: targetKey || null,
        provider: targetProvider,
        modelName: targetModel,
        allowDemo: false,
        discoveryMode,
        discovery_mode: discoveryMode,
        rezeProvider: isReze ? (rezeProvider || "nvidia_diffusiongemma") : undefined,
        reze_provider: isReze ? (rezeProvider || "nvidia_diffusiongemma") : undefined,
        scoringMode: activeScoringMode,
        scoring_mode: activeScoringMode,
      });
      await refresh(projectId);
    } catch (err) {
      await refresh(projectId);
      const errMsg = String(err);
      if (targetProvider === "local" && (errMsg.includes("not found") || errMsg.includes("404"))) {
        if (window.confirm(`Ollama model "${localLlmModel}" is not downloaded. Would you like to download it now?`)) {
          setTimeout(() => {
            void pullModelDirectly(localLlmModel).then(() => {
              void refresh(projectId);
            });
          }, 100);
          return;
        }
      }
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("idle");
    }
  }

  async function renameProject(projectId: string) {
    const project = projects.find((p) => p.id === projectId);
    if (!project) return;
    const currentName = project.name || fileName(project.sourcePath);
    const newName = window.prompt("Rename Project:", currentName);
    if (newName === null) return;
    const trimmed = newName.trim();
    if (!trimmed) return;

    try {
      await invoke("rename_project", { projectId, name: trimmed });
      await refresh(detail?.project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function deleteProject(projectId: string) {
    const project = projects.find((p) => p.id === projectId);
    if (!project) return;
    const name = project.name || fileName(project.sourcePath);
    if (!window.confirm(`Are you sure you want to delete the project "${name}"?`)) return;

    try {
      await invoke("delete_project", { projectId });
      const nextActiveId = detail?.project.id === projectId ? null : detail?.project.id;
      await refresh(nextActiveId ?? undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function selectProject(projectId: string) {
    await run("idle", async () => {
      const nextDetail = await invoke<ProjectDetail>("get_project_detail", { projectId });
      setDetail(nextDetail);
    });
  }

  async function transcribe() {
    if (!detail) return;
    await run("transcribe", async () => {
      await invoke<Transcript>("transcribe_project", {
        projectId: detail.project.id,
        provider: transcriptionEngine,
        apiKey: transcriptionEngine === "deepgram" ? (deepgramKey.trim() || null) : null,
      });
      await refresh(detail.project.id);
    });
  }

  async function moments(allowDemo: boolean) {
    if (!detail) return;
    const currentProjectId = detail.project.id;
    await run("moments", async () => {
      const isReze = discoveryMode === "window_scoring";
      const targetProvider = isReze
        ? (rezeProvider || "nvidia_diffusiongemma")
        : (llmEngine === "openrouter" ? "deepseek" : llmEngine);

      const targetKey = isReze
        ? ""
        : (targetProvider === "claude"
            ? anthropicKey.trim()
            : targetProvider === "gemini"
              ? geminiKey.trim()
              : targetProvider === "openai"
                ? openaiKey.trim()
                : targetProvider === "groq"
                  ? groqKey.trim()
                  : targetProvider === "deepseek"
                    ? deepseekKey.trim()
                    : targetProvider === "nvidia"
                      ? nvidiaKey.trim()
                      : "");

      const targetModel = isReze
        ? null
        : (targetProvider === "local"
            ? localLlmModel.trim()
            : targetProvider === "deepseek"
              ? (deepseekModel.trim() || "deepseek-chat")
              : targetProvider === "nvidia"
                ? ((nvidiaModel.trim() && nvidiaModel.trim() !== "stepfun-ai/step-5-preview") ? nvidiaModel.trim() : "google/gemma-4-31b-it")
                : null);

      try {
        await invoke<Candidate[]>("generate_candidates", {
          projectId: currentProjectId,
          apiKey: targetKey || null,
          provider: targetProvider,
          modelName: targetModel,
          allowDemo,
          discoveryMode,
          discovery_mode: discoveryMode,
          rezeProvider: isReze ? (rezeProvider || "nvidia_diffusiongemma") : undefined,
          reze_provider: isReze ? (rezeProvider || "nvidia_diffusiongemma") : undefined,
          scoringMode,
          scoring_mode: scoringMode,
        });
        await refresh(currentProjectId);
      } catch (err) {
        await refresh(currentProjectId);
        const errMsg = String(err);
        if (targetProvider === "local" && (errMsg.includes("not found") || errMsg.includes("404"))) {
          if (window.confirm(`Ollama model "${localLlmModel}" is not downloaded. Would you like to download it now?`)) {
            setTimeout(() => {
              void pullModelDirectly(localLlmModel).then(() => {
                void refresh(currentProjectId);
              });
            }, 100);
            return;
          }
        }
        throw err;
      }
    });
  }

  // Range-based selection: marks candidates with ranks in the inclusive 1-based
  // range [from, to] as selected; all others become unselected. `to === null`
  // means "Last" (the final candidate). Invalid/empty input never triggers a cut.
  // Stale asynchronous responses are discarded via the monotonic
  // `selectionSeqRef` token, so an older invoke can never overwrite a newer
  // selection change.
  async function updateSelectionRange(from: number, to: number | null) {
    if (!detail) return;
    const total = detail.candidates.length;
    const endRank = to === null ? total : to;
    if (!Number.isInteger(from) || !Number.isInteger(endRank)) return;
    if (from < 1 || endRank < from || endRank > total) return;
    lastAppliedRangeRef.current = { from, to };
    const requestSeq = ++selectionSeqRef.current;
    const projectId = detail.project.id;
    await run("clipCount", async () => {
      const candidates = await invoke<Candidate[]>("set_selected_rank_range", {
        projectId,
        startRank: from,
        endRank,
      });
      if (requestSeq !== selectionSeqRef.current) return; // stale response — discard
      setDetail((prev) =>
        prev && prev.project.id === projectId ? { ...prev, candidates } : prev
      );
    });
  }

  // Top-N selection: marks ranks 1..N as selected through the backend's
  // authoritative set_selected_clip_count command (0 = no selection). The UI
  // never recomputes Top-N itself — it only displays backend state.
  async function updateTopN(count: number) {
    if (!detail) return;
    if (!Number.isInteger(count) || count < 0 || count > detail.candidates.length) return;
    const requestSeq = ++selectionSeqRef.current;
    const projectId = detail.project.id;
    await run("clipCount", async () => {
      const candidates = await invoke<Candidate[]>("set_selected_clip_count", {
        projectId,
        count,
      });
      if (requestSeq !== selectionSeqRef.current) return; // stale response — discard
      setDetail((prev) =>
        prev && prev.project.id === projectId ? { ...prev, candidates } : prev
      );
    });
  }

  // Individual candidate checkbox toggle: safely leverages updateSelectionRange or updateTopN(0)
  // ensuring the backend database remains 100% authoritative.
  // Individual candidate checkbox toggle: flips only that candidate's selection
  // via backend toggle_candidate_selection without forcing contiguous ranges.
  async function toggleCandidateSelection(candidate: Candidate) {
    if (!detail) return;
    const requestSeq = ++selectionSeqRef.current;
    const projectId = detail.project.id;
    await run("clipCount", async () => {
      const candidates = await invoke<Candidate[]>("toggle_candidate_selection", {
        candidateId: candidate.id,
      });
      if (requestSeq !== selectionSeqRef.current) return;
      setDetail((prev) =>
        prev && prev.project.id === projectId ? { ...prev, candidates } : prev
      );
    });
  }

  async function cutCandidate(candidateId: string) {
    if (!detail) return;
    // Immediate synchronous re-entrancy check: cannot start if batch cut is running or candidate is already in-flight
    if (isCuttingRef.current || activeCuttingCandidatesRef.current.has(candidateId)) {
      console.warn(`[Cut Guard] Candidate ${candidateId} cut already in progress. Ignoring duplicate invocation.`);
      return;
    }
    activeCuttingCandidatesRef.current.add(candidateId);
    setRenderingCandidateId(candidateId);
    setBusy("cut");
    setError(null);
    try {
      await invoke<string>("render_flat_clip_for_candidate", { candidateId });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      activeCuttingCandidatesRef.current.delete(candidateId);
      setRenderingCandidateId(null);
      setBusy("idle");
      await refresh(detail.project.id);
    }
  }

  async function cutSelected() {
    if (!detail) return;
    // Immediate synchronous re-entrancy check: only one batch loop can exist at a time
    if (isCuttingRef.current) {
      console.warn("[Cut Guard] Batch cut operation already in progress. Ignoring duplicate invocation.");
      return;
    }
    isCuttingRef.current = true;
    setBusy("cut");
    setError(null);
    try {
      for (const candidate of selectedCandidates) {
        setRenderingCandidateId(candidate.id);
        activeCuttingCandidatesRef.current.add(candidate.id);
        try {
          await invoke<string>("render_flat_clip_for_candidate", { candidateId: candidate.id });
        } finally {
          activeCuttingCandidatesRef.current.delete(candidate.id);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      isCuttingRef.current = false;
      activeCuttingCandidatesRef.current.clear();
      setRenderingCandidateId(null);
      setBusy("idle");
      await refresh(detail.project.id);
    }
  }

  function openStyleStudio(targetCandidateId: string | null, scope: "target" | "selected") {
    setStyleStudioTargetCandidateId(targetCandidateId);
    setStyleStudioScope(scope);

    const project = detail?.project;
    const currentFraming = (project?.framingMode as "adaptive" | "original") || "adaptive";
    setStyleStudioFraming(currentFraming === "original" ? "original" : "adaptive");
    setStyleStudioCaption(project?.captionStyle || "preset_minimal_capsule");

    const currentMode = project?.presentationMode === "autoclip" ? "autoclip" : "autoshorts";
    setStyleStudioPresentationMode(currentMode);

    if (project?.presentationConfig) {
      try {
        const cfg = JSON.parse(project.presentationConfig);
        setStyleStudioPlatformStrategy(
          cfg.platform_strategy || cfg.platformStrategy || "douyin"
        );
        setStyleStudioSubtitleLook(
          cfg.subtitle_style || cfg.subtitleLook || "accent"
        );
        setStyleStudioTitleStyle(
          cfg.title_style || cfg.titleStyle || "comic"
        );
        setStyleStudioTitleMotion(
          cfg.title_motion !== undefined
            ? Boolean(cfg.title_motion)
            : cfg.titleMotion !== undefined
            ? Boolean(cfg.titleMotion)
            : true
        );
        setStyleStudioTemplate(
          cfg.template || "interview_zh"
        );
      } catch (err) {
        console.warn("[Style Studio] Failed to parse project.presentationConfig:", err);
        setStyleStudioPlatformStrategy("douyin");
        setStyleStudioSubtitleLook("accent");
        setStyleStudioTitleStyle("comic");
        setStyleStudioTitleMotion(true);
        setStyleStudioTemplate("interview_zh");
      }
    } else {
      setStyleStudioPlatformStrategy("douyin");
      setStyleStudioSubtitleLook("accent");
      setStyleStudioTitleStyle("comic");
      setStyleStudioTitleMotion(true);
      setStyleStudioTemplate("interview_zh");
    }

    setStyleStudioOpen(true);
  }

  function openSingleClipStyleStudio(candidateId: string) {
    openStyleStudio(candidateId, "target");
  }

  function openBulkStyleStudio() {
    openStyleStudio(null, "selected");
  }

  async function applyStyleStudioAndRerender() {
    if (!detail) return;
    if (targetCandidateIds.length === 0) return;

    // 1. Snapshot current output paths into previousRenderHistory for reversion
    const nextHistory = new Map(previousRenderHistory);
    for (const cid of targetCandidateIds) {
      const clip = clipByCandidate.get(cid);
      if (clip?.outputPath) {
        nextHistory.set(cid, clip.outputPath);
      }
    }
    setPreviousRenderHistory(nextHistory);

    // 2. Persist updated framing, caption style, presentation mode, and presentation config to backend DB
    const presentationConfig = JSON.stringify({
      platform_strategy: styleStudioPlatformStrategy,
      platformStrategy: styleStudioPlatformStrategy,
      subtitle_style: styleStudioSubtitleLook,
      subtitleLook: styleStudioSubtitleLook,
      title_style: "plain",
      titleStyle: "plain",
      title_motion: false,
      titleMotion: false,
      template: "none",
    });

    try {
      await invoke("update_project_style", {
        projectId: detail.project.id,
        captionStyle: styleStudioCaption,
        framingMode: styleStudioFraming,
        presentationMode: styleStudioPresentationMode,
        presentationConfig,
      });
    } catch (err) {
      console.warn("[Style Studio] Failed to update project style in DB:", err);
    }

    // 3. Close Style Studio drawer immediately as required
    setStyleStudioOpen(false);

    // 4. Trigger rendering for target candidates
    if (targetCandidateIds.length === 1) {
      await cutCandidate(targetCandidateIds[0]);
    } else {
      if (isCuttingRef.current) {
        console.warn("[Style Studio Guard] Batch cut operation already in progress. Ignoring duplicate invocation.");
        return;
      }
      isCuttingRef.current = true;
      setBusy("cut");
      setError(null);
      try {
        for (const cid of targetCandidateIds) {
          setRenderingCandidateId(cid);
          activeCuttingCandidatesRef.current.add(cid);
          try {
            await invoke<string>("render_flat_clip_for_candidate", { candidateId: cid });
          } finally {
            activeCuttingCandidatesRef.current.delete(cid);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        isCuttingRef.current = false;
        activeCuttingCandidatesRef.current.clear();
        setRenderingCandidateId(null);
        setBusy("idle");
        await refresh(detail.project.id);
      }
    }
  }

  async function restorePreviousVersion() {
    if (!detail) return;
    const candidatesToRestore = targetCandidateIds.filter((id) => previousRenderHistory.has(id));
    if (candidatesToRestore.length === 0) return;

    setBusy("cut");
    setError(null);
    try {
      for (const cid of candidatesToRestore) {
        const prevPath = previousRenderHistory.get(cid);
        if (prevPath) {
          try {
            await invoke("restore_clip_output_path", {
              candidateId: cid,
              outputPath: prevPath,
            });
          } catch (err) {
            console.warn("[Style Studio] Failed to restore clip output path:", err);
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("idle");
      await refresh(detail.project.id);
      setStyleStudioOpen(false);
    }
  }

  if (isOnboarded === null) {
    return (
      <div className="onboarding-loading" style={{ display: 'grid', placeItems: 'center', height: '100vh', background: 'var(--bg-base)' }}>
        <Loader2 className="spin" size={32} color="var(--accent-primary)" />
      </div>
    );
  }

  if (isOnboarded === false) {
    return (
      <Onboarding
        environment={environment}
        onComplete={() => setIsOnboarded(true)}
        setTranscriptionEngine={setTranscriptionEngine}
        setLlmEngine={setLlmEngine}
        setLocalLlmModel={setLocalLlmModel}
        setDeepgramKey={setDeepgramKey}
        setAnthropicKey={setAnthropicKey}
        setDeepseekKey={setDeepseekKey}
        setGroqKey={setGroqKey}
        deepgramKey={deepgramKey}
        anthropicKey={anthropicKey}
        deepseekKey={deepseekKey}
        groqKey={groqKey}
        refreshEnv={() => refresh()}
      />
    );
  }

  return (
    <div className="app-shell-container">
      <main className="app-shell">
        <aside className="sidebar">
          <div
            className="brand-row"
            onClick={() => setDetail(null)}
            style={{ cursor: "pointer" }}
            title="Go to Home Dashboard"
          >
            <div className="brand-mark">
              <Scissors size={18} className="brand-icon" />
            </div>
            <div className="brand-text">
              <h1 className="brand-title">AutoShorts</h1>
              <p className="brand-subtitle">AI Video Clipper</p>
            </div>
          </div>

          <nav className="sidebar-nav">
            <button
              type="button"
              className={`nav-item ${detail === null ? "active" : ""}`}
              onClick={() => setDetail(null)}
              title="All Projects Dashboard"
            >
              <FolderOpen size={16} className="nav-icon" />
              <span className="nav-label">All Projects</span>
              <span className="nav-count">{projects.length}</span>
            </button>
          </nav>

          <div className="sidebar-section-header">
            <span className="sidebar-section-title">Recent</span>
            {recentProjects.length > 0 && (
              <span className="sidebar-section-count">{recentProjects.length}</span>
            )}
          </div>

          <div className="recent-projects-list" aria-label="Recent Projects">
            {recentProjects.map((project) => {
              const isActive = detail?.project.id === project.id;
              const name = project.name || fileName(project.sourcePath);
              const dateObj = new Date(project.updatedAt || project.createdAt);
              const dateStr = !isNaN(dateObj.getTime())
                ? dateObj.toLocaleDateString(undefined, { month: "short", day: "numeric" })
                : "";

              return (
                <button
                  key={project.id}
                  type="button"
                  className={`recent-project-item ${isActive ? "active" : ""}`}
                  onClick={() => void selectProject(project.id)}
                  title={name}
                >
                  <div className="recent-project-main">
                    {isActive ? (
                      <span className="active-dot" />
                    ) : (
                      <FileVideo size={13} className="recent-project-icon" />
                    )}
                    <span className="recent-project-name">{name}</span>
                  </div>
                  {dateStr && <span className="recent-project-time">{dateStr}</span>}
                </button>
              );
            })}
            {recentProjects.length === 0 && (
              <div className="recent-projects-empty">No projects yet</div>
            )}
          </div>

          <div className="sidebar-footer">
            <div className="user-profile">
              <div className="avatar-placeholder">AS</div>
              <div className="user-details">
                <span className="user-name">Local Studio</span>
                <span className="user-status">v0.1.3 Desktop</span>
              </div>
            </div>
            <button
              type="button"
              className={`sidebar-settings-btn ${showSettings ? "active" : ""}`}
              onClick={() => setShowSettings(!showSettings)}
              title="API & Model Settings"
            >
              <SlidersHorizontal size={15} />
            </button>
          </div>
        </aside>

        <section className="workspace">
          {detail ? (
            <>
              <header className="workspace-topbar">
                <div className="workspace-topbar-left">
                  <button
                    type="button"
                    className="workspace-back-btn"
                    onClick={() => setDetail(null)}
                    title="Back to All Projects"
                  >
                    <ChevronLeft size={16} />
                    <span>All Projects</span>
                  </button>
                  <div className="workspace-topbar-divider" />
                  <div className="workspace-title-group">
                    <h2 className="workspace-project-title">
                      {detail.project.name || fileName(detail.project.sourcePath)}
                    </h2>
                    <span className={`status-chip status-${getProjectNormalizedStatus(detail.project.status)}`}>
                      {getProjectStatusLabel(detail.project.status)}
                    </span>
                  </div>
                </div>
                <div className="workspace-topbar-actions">
                  <button
                    type="button"
                    className={`icon-button settings-toggle ${showSettings ? "active" : ""}`}
                    onClick={() => setShowSettings(!showSettings)}
                    title="API Settings"
                  >
                    <SlidersHorizontal size={15} />
                    <span>API Settings</span>
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => void refresh(detail.project.id)}
                    title="Refresh"
                    disabled={busy !== "idle"}
                  >
                    <RefreshCw size={15} className={busy !== "idle" ? "spin" : ""} />
                    <span>Refresh</span>
                  </button>
                </div>
              </header>

              {showSettings && (
                <SettingsPanel
                  transcriptionEngine={transcriptionEngine}
                  setTranscriptionEngine={setTranscriptionEngine}
                  llmEngine={llmEngine}
                  setLlmEngine={setLlmEngine}
                  deepgramKey={deepgramKey}
                  setDeepgramKey={setDeepgramKey}
                  anthropicKey={anthropicKey}
                  setAnthropicKey={setAnthropicKey}
                  deepseekKey={deepseekKey}
                  setDeepseekKey={setDeepseekKey}
                  deepseekModel={deepseekModel}
                  setDeepseekModel={setDeepseekModel}
                  geminiKey={geminiKey}
                  setGeminiKey={setGeminiKey}
                  openaiKey={openaiKey}
                  setOpenaiKey={setOpenaiKey}
                  openrouterKey={openrouterKey}
                  setOpenrouterKey={setOpenrouterKey}
                  openrouterModel={openrouterModel}
                  setOpenrouterModel={setOpenrouterModel}
                  groqKey={groqKey}
                  setGroqKey={setGroqKey}
                  nvidiaKey={nvidiaKey}
                  setNvidiaKey={setNvidiaKey}
                  nvidiaModel={nvidiaModel}
                  setNvidiaModel={setNvidiaModel}
                  localLlmModel={localLlmModel}
                  setLocalLlmModel={setLocalLlmModel}
                  pullModelDirectly={pullModelDirectly}
                  youtubeBrowser={youtubeBrowser}
                  setYoutubeBrowser={setYoutubeBrowser}
                  youtubeCookiesPath={youtubeCookiesPath}
                  setYoutubeCookiesPath={setYoutubeCookiesPath}
                  environment={environment}
                />
              )}

              {error && <div className="error-banner">{error}</div>}

              <nav className="workflow-nav" aria-label="Workflow Navigation">
                <div className="workflow-nav-track">
                  <div className={`workflow-pill ${detail.transcript ? "done" : ""} ${activePipelineStep === "Transcript" ? "active" : ""}`}>
                    <span className="workflow-step-icon">
                      {detail.transcript ? <Check size={13} strokeWidth={2.5} /> : <AudioLines size={14} />}
                    </span>
                    <span className="workflow-step-label">Transcript</span>
                  </div>
                  <div className="workflow-nav-arrow">→</div>
                  <div className={`workflow-pill ${detail.candidates.length > 0 ? "done" : ""} ${activePipelineStep === "Moments" ? "active" : ""}`}>
                    <span className="workflow-step-icon">
                      {detail.candidates.length > 0 ? <Check size={13} strokeWidth={2.5} /> : <Sparkles size={14} />}
                    </span>
                    <span className="workflow-step-label">Moments</span>
                  </div>
                  <div className="workflow-nav-arrow">→</div>
                  <div className={`workflow-pill ${selectedCount > 0 && selectedCutCount === selectedCount ? "done" : ""} ${activePipelineStep === "Cut" ? "active" : ""}`}>
                    <span className="workflow-step-icon">
                      {selectedCount > 0 && selectedCutCount === selectedCount ? <Check size={13} strokeWidth={2.5} /> : <Scissors size={14} />}
                    </span>
                    <span className="workflow-step-label">Cut</span>
                  </div>
                  <div className="workflow-nav-arrow">→</div>
                  <div className={`workflow-pill ${selectedCount > 0 && selectedCaptionsCount === selectedCount ? "done" : ""} ${activePipelineStep === "Captions" ? "active" : ""}`}>
                    <span className="workflow-step-icon">
                      {selectedCount > 0 && selectedCaptionsCount === selectedCount ? <Check size={13} strokeWidth={2.5} /> : <Captions size={14} />}
                    </span>
                    <span className="workflow-step-label">Captions</span>
                  </div>
                  <div className="workflow-nav-arrow">→</div>
                  <div className={`workflow-pill ${selectedCount > 0 && selectedCutCount === selectedCount && selectedCaptionsCount === selectedCount ? "done" : ""} ${activePipelineStep === "Export" ? "active" : ""}`}>
                    <span className="workflow-step-icon">
                      {selectedCount > 0 && selectedCutCount === selectedCount && selectedCaptionsCount === selectedCount ? <Check size={13} strokeWidth={2.5} /> : <Download size={14} />}
                    </span>
                    <span className="workflow-step-label">Export</span>
                  </div>
                </div>
              </nav>

              <div className="work-grid">
                <section className="panel transcript-panel">
                  <div className="panel-heading transcript-panel-heading">
                    <div className="transcript-heading-info">
                      <div className="transcript-title-row">
                        <h3>Transcript</h3>
                        <span className="transcript-count-chip">
                          {transcript
                            ? `${filteredTranscriptSegments.length}${transcriptSearch ? ` of ${transcript.segments.length}` : ""} segments`
                            : "No transcript"}
                        </span>
                      </div>
                      {transcript && (
                        <div className="transcript-search-wrap">
                          <Search size={13} className="transcript-search-icon" />
                          <input
                            type="text"
                            placeholder="Search transcript..."
                            value={transcriptSearch}
                            onChange={(e) => setTranscriptSearch(e.target.value)}
                            className="transcript-search-input"
                          />
                          {transcriptSearch && (
                            <button
                              type="button"
                              className="transcript-search-clear"
                              onClick={() => setTranscriptSearch("")}
                              title="Clear search"
                            >
                              ×
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                    <button
                      type="button"
                      className="transcribe-btn"
                      onClick={transcribe}
                      disabled={busy !== "idle" || !canTranscribe}
                      title="Transcribe Audio"
                    >
                      {busy === "transcribe" ? <Loader2 className="spin" size={14} /> : <AudioLines size={14} />}
                      <span>Transcribe</span>
                    </button>
                  </div>

                  {!canTranscribe && (
                    <div className="api-warning">
                      {transcriptionEngine === "local"
                        ? `⚠️ Local Whisper (Python package 'openai-whisper') is not installed. Run 'pip3 install openai-whisper' in your terminal.`
                        : "⚠️ Deepgram API Key is missing. Transcribing will not work. Please add your key in API Settings."}
                    </div>
                  )}

                  <div className="transcript-list">
                    {transcript ? (
                      filteredTranscriptSegments.length > 0 ? (
                        filteredTranscriptSegments.map((segment, index) => {
                          const isSegmentSelected = selectedCandidates.some(
                            (c) => segment.start < c.endSec && segment.end > c.startSec
                          );
                          return (
                            <article
                              key={`${segment.start}-${index}`}
                              className={`segment-row ${isSegmentSelected ? "highlighted" : ""}`}
                            >
                              <div className="segment-meta">
                                <span className="segment-timestamp">{formatTimestamp(segment.start)}</span>
                                {segment.speaker && (
                                  <span className="segment-speaker">{segment.speaker}</span>
                                )}
                              </div>
                              <p className="segment-text">{segment.text}</p>
                            </article>
                          );
                        })
                      ) : (
                        <EmptyState
                          icon={<Search size={24} />}
                          label={`No segments matching "${transcriptSearch}"`}
                        />
                      )
                    ) : (
                      <EmptyState icon={<AudioLines size={28} />} label="Transcript pending" />
                    )}
                  </div>
                </section>

                <section className="panel candidate-panel">
                  <div className="panel-heading candidate-panel-heading">
                    <div className="candidate-heading-left">
                      <div className="candidate-title-row">
                        <h3>Clip Candidates</h3>
                        <div className="candidate-count-chips">
                          <span className="count-chip-selected">{selectedCount} Selected</span>
                          <span className="count-chip-pool">{poolCount} Pool</span>
                        </div>
                      </div>
                    </div>
                    <div className="candidate-heading-actions">
                      <div className="candidate-filter-tabs">
                        <button
                          type="button"
                          className={`filter-tab ${candidateFilter === "all" ? "active" : ""}`}
                          onClick={() => setCandidateFilter("all")}
                        >
                          All ({totalCandidateCount})
                        </button>
                        <button
                          type="button"
                          className={`filter-tab ${candidateFilter === "selected" ? "active" : ""}`}
                          onClick={() => setCandidateFilter("selected")}
                        >
                          Selected ({selectedCount})
                        </button>
                        <button
                          type="button"
                          className={`filter-tab ${candidateFilter === "pool" ? "active" : ""}`}
                          onClick={() => setCandidateFilter("pool")}
                        >
                          Pool ({poolCount})
                        </button>
                      </div>

                      {selectedCutCount > 0 && (
                        <button
                          type="button"
                          className="restyle-selected-btn"
                          onClick={openBulkStyleStudio}
                          disabled={busy !== "idle"}
                          title="Restyle selected clips in Style Studio"
                        >
                          <SlidersHorizontal size={14} />
                          <span>Restyle Selected ({selectedCutCount})</span>
                        </button>
                      )}

                      <button
                        type="button"
                        className="cut-selected-btn"
                        onClick={cutSelected}
                        disabled={busy !== "idle" || selectedCount === 0 || !environment?.hasFfmpeg}
                        title="Cut Selected"
                      >
                        {busy === "cut" ? <Loader2 className="spin" size={14} /> : <Scissors size={14} />}
                        <span>Cut Selected ({selectedCount})</span>
                      </button>
                    </div>
                  </div>

                  {/* Secondary Controls Row */}
                  <div className="candidate-controls-row">
                    <div className="candidate-controls-left">
                      <div className="range-control-group">
                        <label className="range-field">
                          <span>From</span>
                          <input
                            type="number"
                            min="1"
                            max={totalCandidateCount || 1}
                            value={rangeFromValue}
                            onChange={(event) => {
                              setRangeFromValue(event.target.value);
                              void updateSelectionRange(
                                Number.parseInt(event.target.value, 10),
                                rangeToValue === null ? null : Number.parseInt(rangeToValue, 10)
                              );
                            }}
                            onBlur={(event) => {
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (!Number.isInteger(parsed) || parsed < 1) {
                                setRangeFromValue("1");
                              } else if (parsed > totalCandidateCount) {
                                setRangeFromValue(String(totalCandidateCount));
                              }
                            }}
                            disabled={busy !== "idle" && busy !== "clipCount"}
                            title="First candidate rank to select (1-based, inclusive)"
                          />
                        </label>
                        <span className="range-arrow">→</span>
                        <label className="range-field">
                          <span>To</span>
                          <input
                            type="number"
                            min="1"
                            max={totalCandidateCount || 1}
                            value={rangeToValue ?? ""}
                            placeholder="Last"
                            onChange={(event) => {
                              const raw = event.target.value;
                              setRangeToValue(raw === "" ? null : raw);
                              void updateSelectionRange(
                                Number.parseInt(rangeFromValue, 10),
                                raw === "" ? null : Number.parseInt(raw, 10)
                              );
                            }}
                            onBlur={(event) => {
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (Number.isInteger(parsed) && parsed > totalCandidateCount) {
                                setRangeToValue(String(totalCandidateCount));
                              }
                            }}
                            disabled={busy !== "idle" && busy !== "clipCount"}
                            title="Last candidate rank to select (1-based, inclusive). Empty = Last candidate."
                          />
                        </label>
                        <button
                          type="button"
                          className={`range-last-btn ${rangeToValue === null ? "active" : ""}`}
                          onClick={() => {
                            setRangeToValue(null);
                            void updateSelectionRange(Number.parseInt(rangeFromValue, 10), null);
                          }}
                          disabled={busy !== "idle" && busy !== "clipCount"}
                          title="Select through the final candidate"
                        >
                          Last
                        </button>
                      </div>

                      <div className="topn-control-group">
                        <label className="range-field">
                          <span>Top N</span>
                          <input
                            type="number"
                            min="0"
                            max={totalCandidateCount || 1}
                            value={topNValue}
                            placeholder="—"
                            onChange={(event) => {
                              const raw = event.target.value;
                              setTopNValue(raw);
                              const parsed = Number.parseInt(raw, 10);
                              if (Number.isInteger(parsed)) {
                                void updateTopN(parsed);
                              }
                            }}
                            onBlur={(event) => {
                              const parsed = Number.parseInt(event.target.value, 10);
                              if (!Number.isInteger(parsed) || parsed < 0) {
                                setTopNValue("");
                              } else if (parsed > totalCandidateCount) {
                                setTopNValue(String(totalCandidateCount));
                              }
                            }}
                            disabled={busy !== "idle" && busy !== "clipCount"}
                            title="Select the top N candidates by rank (backend-authoritative; 0 clears the selection)"
                          />
                        </label>
                      </div>
                    </div>

                    <div className="candidate-controls-right">
                      <select
                        value={discoveryMode}
                        onChange={(e) => {
                          const val = e.target.value;
                          setDiscoveryMode(val);
                          if (val === "window_scoring") {
                            setRezeProvider("nvidia_diffusiongemma");
                          } else {
                            setRezeProvider("");
                          }
                        }}
                        disabled={busy !== "idle"}
                        title="Candidate discovery engine: Timestamp Generation is the production default; REZE Window Scoring is the experimental alternative."
                        className="discovery-mode-select"
                      >
                        <option value="timestamp_generation">Discovery: Timestamp</option>
                        <option value="window_scoring">Discovery: REZE Window</option>
                      </select>

                      {(() => {
                        const effectiveStatus: SceneIntelStatus = sceneIntelStatus ?? {
                          projectId: detail.project.id,
                          status: "idle",
                          progress: 0,
                          sceneCount: 0,
                        };
                        return (
                          <div
                            className={`scene-intel-pill status-${effectiveStatus.status}`}
                            onClick={() => {
                              if (effectiveStatus.status === "idle" || effectiveStatus.status === "error") {
                                invoke<SceneIntelStatus>("trigger_scene_intelligence", { projectId: detail.project.id })
                                  .then((status) => { if (status) setSceneIntelStatus(status); })
                                  .catch((err) => console.warn("[SceneIntel] Failed to trigger:", err));
                              }
                            }}
                            title={
                              effectiveStatus.status === "running"
                                ? "GPU Scene Intelligence is scanning video for camera cuts in the background."
                                : effectiveStatus.status === "completed"
                                ? `Full scene intelligence ready (${effectiveStatus.sceneCount} camera cuts cached).`
                                : effectiveStatus.status === "idle"
                                ? "Click to start GPU Scene Intelligence scan in background."
                                : "Scene intelligence encountered an issue; click to retry."
                            }
                            style={{
                              cursor: (effectiveStatus.status === "idle" || effectiveStatus.status === "error") ? "pointer" : "default"
                            }}
                          >
                            {effectiveStatus.status === "running" && (
                              <>
                                <Loader2 className="spin" size={13} />
                                <span>Scene Intel {effectiveStatus.progress}%</span>
                              </>
                            )}
                            {effectiveStatus.status === "completed" && (
                              <>
                                <Check size={13} className="pill-check-icon" />
                                <span>Scene Intel Ready ({effectiveStatus.sceneCount})</span>
                              </>
                            )}
                            {effectiveStatus.status === "idle" && (
                              <>
                                <Sparkles size={13} className="pill-idle-icon" />
                                <span>Scan Scenes (GPU)</span>
                              </>
                            )}
                            {effectiveStatus.status === "error" && (
                              <>
                                <AlertCircle size={13} className="pill-error-icon" />
                                <span>Retry Scene Intel</span>
                              </>
                            )}
                          </div>
                        );
                      })()}

                      <button
                        type="button"
                        className="find-moments-btn"
                        onClick={() => void moments(false)}
                        disabled={busy !== "idle" || !detail.transcript || !canUseDiscoveryLlm}
                        title="Find Viral Moments"
                      >
                        {busy === "moments" ? <Loader2 className="spin" size={14} /> : <Sparkles size={14} />}
                        <span>Find Viral Moments</span>
                      </button>
                    </div>
                  </div>

                  {scoringMode === "autoclip" && environment !== null && !environment.hasNvidiaKey && (
                    <div className="api-warning">
                      ⚠️ AutoClip Scoring (NVIDIA) is the active scoring mode but NVIDIA_API_KEY is missing. Find Viral Moments will report a configuration error — it will not silently fall back to Legacy Scoring. Configure the key or switch the scoring mode back to Legacy.
                    </div>
                  )}

                  {!canUseDiscoveryLlm && (
                    <div className="api-warning">
                      {isRezeMode
                        ? "⚠️ NVIDIA API Key is required for REZE Window Scoring (DiffusionGemma). Please ensure NVIDIA_API_KEY is configured in your .env or environment."
                        : (effectiveTimestampEngine === "local"
                            ? "⚠️ Ollama local server is not running at http://localhost:11434. Moment detection will not work."
                            : `⚠️ ${effectiveTimestampEngine === "claude" ? "Claude" :
                              effectiveTimestampEngine === "gemini" ? "Gemini" :
                              effectiveTimestampEngine === "openai" ? "OpenAI" :
                              effectiveTimestampEngine === "groq" ? "Groq" : "DeepSeek"
                            } API Key is missing. Viral moment identification will not work. Please add your key in API Settings.`)}
                    </div>
                  )}

                  <div className="candidate-list">
                    {displayedCandidates.map((candidate) => {
                      const clip = clipByCandidate.get(candidate.id);
                      const isCut = clip?.status === "done" && Boolean(clip.outputPath);
                      return (
                        <article
                          key={candidate.id}
                          className={`candidate-card ${candidate.selected ? "selected" : ""}`}
                        >
                          <div className="candidate-card-header">
                            <div className="candidate-card-header-left">
                              <label
                                className="candidate-checkbox-label"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <input
                                  type="checkbox"
                                  className="candidate-checkbox"
                                  checked={candidate.selected}
                                  onChange={() => toggleCandidateSelection(candidate)}
                                  disabled={busy !== "idle" && busy !== "clipCount"}
                                  title={candidate.selected ? "Deselect candidate" : "Select candidate"}
                                />
                                <span className="candidate-rank-badge">#{candidate.rank}</span>
                              </label>
                              <span className="candidate-timerange">
                                {formatTimestamp(candidate.startSec)} – {formatTimestamp(candidate.endSec)}
                              </span>
                            </div>

                            <div className="candidate-card-header-right">
                              {(() => {
                                const scoreSource = candidateScoreSource(candidate);
                                if (scoreSource === "llm") {
                                  return (
                                    <span className="provenance-chip autoclip">
                                      AutoClip Score
                                    </span>
                                  );
                                }
                                return (
                                  <span className="provenance-chip multimodal">
                                    Multimodal
                                  </span>
                                );
                              })()}
                              <span className="candidate-score-pct">
                                {Math.round(candidate.score * 100)}% Match
                              </span>
                            </div>
                          </div>

                          <div className="candidate-card-body">
                            <h4 className="candidate-hook">{candidate.hook}</h4>
                            <p className="candidate-rationale">{candidate.rationale}</p>
                          </div>

                          <div className="candidate-card-footer">
                            <div className="candidate-footer-status">
                              <span className={`clip-status-indicator ${isCut ? "ready" : clip?.status === "error" ? "error" : "pending"}`}>
                                <span className="status-dot" />
                                {isCut ? "Cut ready" : clip?.status === "error" ? "Cut failed" : clip?.status === "rendering" ? "Rendering" : "Pending"}
                              </span>
                              {clip?.outputPath && (
                                <span className="clip-output-path" title={clip.outputPath}>
                                  {fileName(clip.outputPath)}
                                </span>
                              )}
                            </div>

                            <div className="candidate-footer-actions">
                              {isCut && (
                                <button
                                  type="button"
                                  className="candidate-action-restyle-btn"
                                  onClick={() => openSingleClipStyleStudio(candidate.id)}
                                  disabled={busy !== "idle"}
                                  title="Restyle this clip in Style Studio"
                                >
                                  <SlidersHorizontal size={13} />
                                  <span>Restyle</span>
                                </button>
                              )}
                              <button
                                type="button"
                                className={`candidate-action-cut-btn ${isCut ? "recut" : ""}`}
                                onClick={() => void cutCandidate(candidate.id)}
                                disabled={busy !== "idle" || !environment?.hasFfmpeg}
                                title={isCut ? "Re-cut this clip" : "Cut clip"}
                              >
                                {renderingCandidateId === candidate.id ? (
                                  <Loader2 className="spin" size={13} />
                                ) : (
                                  <Scissors size={13} />
                                )}
                                <span>{renderingCandidateId === candidate.id ? "Cutting..." : isCut ? "Re-cut" : "Cut"}</span>
                              </button>
                            </div>
                          </div>

                          {clip?.captionAssPath && (
                            <div className="candidate-subtitles-info">
                              Subtitles: {fileName(clip.captionAssPath)}
                            </div>
                          )}
                          {clip?.renderLog && <div className="render-log">{clip.renderLog}</div>}
                        </article>
                      );
                    })}
                    {displayedCandidates.length === 0 && totalCandidateCount > 0 && (
                      <EmptyState icon={<Sparkles size={28} />} label={`No candidates in "${candidateFilter}" view`} />
                    )}
                    {totalCandidateCount === 0 && <EmptyState icon={<Sparkles size={28} />} label="Moments pending" />}
                  </div>
                </section>
              </div>
            </>
          ) : (
            <div className="home-dashboard">
              {showSettings && (
                <SettingsPanel
                  transcriptionEngine={transcriptionEngine}
                  setTranscriptionEngine={setTranscriptionEngine}
                  llmEngine={llmEngine}
                  setLlmEngine={setLlmEngine}
                  deepgramKey={deepgramKey}
                  setDeepgramKey={setDeepgramKey}
                  anthropicKey={anthropicKey}
                  setAnthropicKey={setAnthropicKey}
                  deepseekKey={deepseekKey}
                  setDeepseekKey={setDeepseekKey}
                  deepseekModel={deepseekModel}
                  setDeepseekModel={setDeepseekModel}
                  geminiKey={geminiKey}
                  setGeminiKey={setGeminiKey}
                  openaiKey={openaiKey}
                  setOpenaiKey={setOpenaiKey}
                  openrouterKey={openrouterKey}
                  setOpenrouterKey={setOpenrouterKey}
                  openrouterModel={openrouterModel}
                  setOpenrouterModel={setOpenrouterModel}
                  groqKey={groqKey}
                  setGroqKey={setGroqKey}
                  nvidiaKey={nvidiaKey}
                  setNvidiaKey={setNvidiaKey}
                  nvidiaModel={nvidiaModel}
                  setNvidiaModel={setNvidiaModel}
                  localLlmModel={localLlmModel}
                  setLocalLlmModel={setLocalLlmModel}
                  pullModelDirectly={pullModelDirectly}
                  youtubeBrowser={youtubeBrowser}
                  setYoutubeBrowser={setYoutubeBrowser}
                  youtubeCookiesPath={youtubeCookiesPath}
                  setYoutubeCookiesPath={setYoutubeCookiesPath}
                  environment={environment}
                />
              )}

              {error && <div className="error-banner">{error}</div>}

              <header className="home-header">
                <div className="home-header-title-block">
                  <h2 className="home-title">All Projects</h2>
                  <p className="home-subtitle">Manage and open your video projects.</p>
                </div>
                <div className="home-header-actions">
                  <button
                    type="button"
                    className="primary-action compact"
                    onClick={importMedia}
                    disabled={busy !== "idle"}
                  >
                    {busy === "import" ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}
                    <span>Import Recording</span>
                  </button>
                  <button
                    type="button"
                    className="secondary-action-outlined compact"
                    onClick={() => {
                      setYoutubeUrl("");
                      setYoutubeStatus("idle");
                      setYoutubeWarningLicense(null);
                      setYoutubeError(null);
                      setYoutubeAudioTracks([]);
                      setSelectedAudioFormatId(null);
                      setYoutubeVideoTitle(null);
                      setYoutubeModalOpen(true);
                    }}
                    disabled={busy !== "idle" || !environment?.hasYtdlp}
                    title={!environment?.hasYtdlp ? "Please install yt-dlp to use this feature" : "Download a video from YouTube"}
                  >
                    <Youtube size={16} />
                    <span>Import from YouTube</span>
                  </button>
                </div>
              </header>

              <div className="dashboard-filter-bar">
                <div className="search-input-wrapper">
                  <Search size={15} className="search-icon-inline" />
                  <input
                    type="text"
                    placeholder="Search projects..."
                    value={projectSearch}
                    onChange={(e) => setProjectSearch(e.target.value)}
                  />
                  {projectSearch && (
                    <button
                      type="button"
                      className="clear-search-btn"
                      onClick={() => setProjectSearch("")}
                      title="Clear search"
                    >
                      ×
                    </button>
                  )}
                </div>

                <div className="pill-group" role="tablist">
                  <button
                    type="button"
                    className={`pill-btn ${statusFilter === "all" ? "active" : ""}`}
                    onClick={() => setStatusFilter("all")}
                  >
                    All <span className="pill-count">{projectCounts.all}</span>
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${statusFilter === "ready" ? "active" : ""}`}
                    onClick={() => setStatusFilter("ready")}
                  >
                    Ready <span className="pill-count">{projectCounts.ready}</span>
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${statusFilter === "processing" ? "active" : ""}`}
                    onClick={() => setStatusFilter("processing")}
                  >
                    Processing <span className="pill-count">{projectCounts.processing}</span>
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${statusFilter === "failed" ? "active" : ""}`}
                    onClick={() => setStatusFilter("failed")}
                  >
                    Failed <span className="pill-count">{projectCounts.failed}</span>
                  </button>
                </div>

                <div className="sort-dropdown-wrapper">
                  <ArrowUpDown size={14} className="sort-icon" />
                  <select
                    value={sortOption}
                    onChange={(e) => setSortOption(e.target.value as "recent" | "date" | "duration")}
                    className="sort-select"
                  >
                    <option value="recent">Recently Opened</option>
                    <option value="date">Date Created</option>
                    <option value="duration">Duration</option>
                  </select>
                </div>
              </div>

              {projects.length > 0 ? (
                filteredProjects.length > 0 ? (
                  <div className="projects-grid">
                    {filteredProjects.map((project) => {
                      const name = project.name || fileName(project.sourcePath);
                      const statusCategory = getProjectNormalizedStatus(project.status);
                      const statusLabel = getProjectStatusLabel(project.status);
                      const createdDate = new Date(project.createdAt);
                      const formattedDate = !isNaN(createdDate.getTime())
                        ? createdDate.toLocaleDateString(undefined, {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                          })
                        : "Unknown";

                      return (
                        <article key={project.id} className="project-card">
                          <div
                            className="project-card-thumbnail"
                            onClick={() => void selectProject(project.id)}
                            title={`Open ${name}`}
                          >
                            <div className="thumbnail-placeholder">
                              <Film size={28} className="thumbnail-icon" />
                            </div>
                            <span className="duration-badge">
                              {project.sourceDuration ? formatTime(project.sourceDuration) : "Probing..."}
                            </span>
                          </div>

                          <div className="project-card-header">
                            <h3
                              className="project-card-title"
                              onClick={() => void selectProject(project.id)}
                              title={name}
                            >
                              {name}
                            </h3>
                            <span className={`status-chip ${statusCategory}`}>
                              <span className="status-dot" />
                              {statusLabel}
                            </span>
                          </div>

                          <div className="project-card-meta">
                            <span className="meta-item" title="Duration">
                              <Clock size={12} />
                              {project.sourceDuration ? formatTime(project.sourceDuration) : "Probing..."}
                            </span>
                            <span className="meta-item" title="Created date">
                              <Calendar size={12} />
                              {formattedDate}
                            </span>
                          </div>

                          <div className="project-card-actions">
                            <button
                              type="button"
                              className="action-btn open-btn"
                              onClick={() => void selectProject(project.id)}
                            >
                              Open
                            </button>
                            <button
                              type="button"
                              className="action-btn rename-btn"
                              onClick={() => void renameProject(project.id)}
                            >
                              Rename
                            </button>
                            <button
                              type="button"
                              className="action-btn delete-btn"
                              onClick={() => void deleteProject(project.id)}
                            >
                              Delete
                            </button>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                ) : (
                  <div className="empty-dashboard-state">
                    <Search size={44} className="empty-state-icon" />
                    <h3>No matching projects</h3>
                    <p>No projects match your current search query or filter.</p>
                    <button
                      type="button"
                      className="action-btn rename-btn"
                      style={{ marginTop: "8px", padding: "0 16px" }}
                      onClick={() => {
                        setProjectSearch("");
                        setStatusFilter("all");
                      }}
                    >
                      Clear filters
                    </button>
                  </div>
                )
              ) : (
                <div className="empty-dashboard-state">
                  <Clapperboard size={48} className="empty-state-icon" />
                  <h3>No projects found</h3>
                  <p>Import your first recording to begin creating shorts.</p>
                  <div style={{ marginTop: "12px", display: "flex", gap: "10px" }}>
                    <button
                      type="button"
                      className="primary-action compact"
                      onClick={importMedia}
                      disabled={busy !== "idle"}
                    >
                      <Upload size={16} /> Import Recording
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <footer className="status-bar">
        <div className="status-bar-left">
          <span className="app-status-indicator">System Ready</span>
        </div>
        <div className="status-bar-right">
          <div className="status-indicators">
            <span className={`indicator ${environment?.hasFfmpeg ? "active" : ""}`} title="FFmpeg status">ffmpeg</span>
            <span className={`indicator ${environment?.hasFfprobe ? "active" : ""}`} title="FFprobe status">ffprobe</span>
            <span className={`indicator ${environment?.hasYtdlp ? "active" : ""}`} title="yt-dlp status">yt-dlp</span>
            <span className={`indicator ${environment?.hasLocalWhisperModel ? "active" : ""}`} title="Whisper Model status">Whisper Model</span>
            <span className={`indicator ${environment?.hasOllama ? "active" : ""}`} title="Ollama status">Ollama</span>
            <span className={`indicator ${canUseCloudKey ? "active" : ""}`} title="Deepgram Key status">Deepgram</span>
            <span className={`indicator ${canUseClaude ? "active" : ""}`} title="Claude Key status">Claude</span>
            <span className={`indicator ${canUseDeepseek ? "active" : ""}`} title="DeepSeek Key status">DeepSeek</span>
          </div>
        </div>
      </footer>

      {styleStudioOpen && detail && (
        <>
          <div
            className="style-studio-backdrop"
            onClick={() => setStyleStudioOpen(false)}
            title="Click to close Style Studio"
          />
          <aside className="style-studio-drawer" aria-label="Style Studio">
            {/* Header */}
            <div className="style-studio-header">
              <div className="style-studio-header-left">
                <h3 className="style-studio-title">Style Studio</h3>
                <p className="style-studio-subtitle">
                  Customize framing and captions without re-scoring.
                </p>
              </div>
              <button
                type="button"
                className="style-studio-close-btn"
                onClick={() => setStyleStudioOpen(false)}
                title="Close Style Studio"
              >
                <X size={18} />
              </button>
            </div>

            {/* Context Banner */}
            <div className="style-studio-context-banner">
              {styleStudioTargetCandidateId && targetCandidate ? (
                <>
                  <div className="style-studio-clip-badge">
                    <span>Clip #{targetCandidate.rank} — {formatTimestamp(targetCandidate.startSec)}…{formatTimestamp(targetCandidate.endSec)}</span>
                  </div>
                  <div className="style-studio-hook-snippet" title={targetCandidate.hook}>
                    {targetCandidate.hook}
                  </div>
                </>
              ) : (
                <>
                  <div className="bulk-badge-row">
                    <div className="bulk-count-badge">
                      <Layers size={13} />
                      <span>{targetCandidateIds.length} clips selected</span>
                    </div>
                  </div>
                  <div className="bulk-ranks-list" title={targetCandidateIds.map(id => {
                    const c = detail.candidates.find(cand => cand.id === id);
                    return c ? `#${c.rank}` : null;
                  }).filter(Boolean).join(", ")}>
                    Clips {targetCandidateIds.map(id => {
                      const c = detail.candidates.find(cand => cand.id === id);
                      return c ? `#${c.rank}` : null;
                    }).filter(Boolean).sort((a,b) => parseInt(a!.slice(1)) - parseInt(b!.slice(1))).join(", ")}
                  </div>
                </>
              )}
            </div>

            {/* Scrollable Body */}
            <div className="style-studio-body">
              {/* Section 1: Presentation Engine */}
              <section className="style-studio-section">
                <h4 className="style-studio-section-title">Presentation Engine</h4>
                <div className="style-studio-framing-grid">
                  <div
                    className={`style-studio-framing-card ${styleStudioPresentationMode === "autoshorts" ? "selected" : ""}`}
                    onClick={() => setStyleStudioPresentationMode("autoshorts")}
                  >
                    <div className="style-studio-framing-top">
                      <div className="framing-icon-box mint">
                        <Crop size={16} />
                      </div>
                      <div className="card-indicator">
                        {styleStudioPresentationMode === "autoshorts" ? (
                          <div className="indicator-check"><Check size={11} strokeWidth={3} /></div>
                        ) : (
                          <div className="indicator-empty" />
                        )}
                      </div>
                    </div>
                    <span className="style-studio-card-title">AutoShorts Native (Mode 1)</span>
                    <p className="style-studio-card-desc">BoT-SORT tracking, 1:1 midpoint retention &amp; kinetic ASS</p>
                  </div>

                  <div
                    className={`style-studio-framing-card ${styleStudioPresentationMode === "autoclip" ? "selected" : ""}`}
                    onClick={() => setStyleStudioPresentationMode("autoclip")}
                  >
                    <div className="style-studio-framing-top">
                      <div className="framing-icon-box slate">
                        <Clapperboard size={16} />
                      </div>
                      <div className="card-indicator">
                        {styleStudioPresentationMode === "autoclip" ? (
                          <div className="indicator-check"><Check size={11} strokeWidth={3} /></div>
                        ) : (
                          <div className="indicator-empty" />
                        )}
                      </div>
                    </div>
                    <span className="style-studio-card-title">AutoClip Studio (Mode 2)</span>
                    <p className="style-studio-card-desc">YuNet detector, hard cuts, mouth motion &amp; PIL TitleArt</p>
                  </div>
                </div>
              </section>

              {styleStudioPresentationMode === "autoshorts" ? (
                <>
                  {/* Mode 1 - Section 2: Framing */}
                  <section className="style-studio-section">
                    <h4 className="style-studio-section-title">Framing</h4>
                    <div className="style-studio-framing-grid">
                      <div
                        className={`style-studio-framing-card ${styleStudioFraming === "adaptive" ? "selected" : ""}`}
                        onClick={() => setStyleStudioFraming("adaptive")}
                      >
                        <div className="style-studio-framing-top">
                          <div className="framing-icon-box mint">
                            <Crop size={16} />
                          </div>
                          <div className="card-indicator">
                            {styleStudioFraming === "adaptive" ? (
                              <div className="indicator-check"><Check size={11} strokeWidth={3} /></div>
                            ) : (
                              <div className="indicator-empty" />
                            )}
                          </div>
                        </div>
                        <span className="style-studio-card-title">Adaptive Framing</span>
                        <p className="style-studio-card-desc">Dynamic speaker tracking and isolation</p>
                      </div>

                      <div
                        className={`style-studio-framing-card ${styleStudioFraming === "original" ? "selected" : ""}`}
                        onClick={() => setStyleStudioFraming("original")}
                      >
                        <div className="style-studio-framing-top">
                          <div className="framing-icon-box">
                            <Smartphone size={16} />
                          </div>
                          <div className="card-indicator">
                            {styleStudioFraming === "original" ? (
                              <div className="indicator-check"><Check size={11} strokeWidth={3} /></div>
                            ) : (
                              <div className="indicator-empty" />
                            )}
                          </div>
                        </div>
                        <span className="style-studio-card-title">Original 9:16</span>
                        <p className="style-studio-card-desc">Fixed portrait framing, DualFrame where supported</p>
                      </div>
                    </div>
                  </section>

                  {/* Mode 1 - Section 3: Caption Style */}
                  <section className="style-studio-section">
                    <h4 className="style-studio-section-title">Caption Style</h4>
                    <div className="style-studio-caption-grid">
                      {/* Clean */}
                      <div
                        className={`style-studio-caption-card ${
                          styleStudioCaption === "preset_minimal_capsule" || styleStudioCaption === "preset_cinematic_vlog"
                            ? "selected"
                            : ""
                        }`}
                        onClick={() => setStyleStudioCaption("preset_minimal_capsule")}
                      >
                        <div className="style-studio-preview-box">
                          <div className="preview-pill-clean">How sleep affects cortisol</div>
                        </div>
                        <span className="style-studio-card-title">Clean</span>
                        <p className="style-studio-card-desc">Minimal centered</p>
                      </div>

                      {/* Bold */}
                      <div
                        className={`style-studio-caption-card ${
                          styleStudioCaption === "preset_viral_bold" || styleStudioCaption === "preset_bhaukal_caption"
                            ? "selected"
                            : ""
                        }`}
                        onClick={() => setStyleStudioCaption("preset_viral_bold")}
                      >
                        <div className="style-studio-preview-box">
                          <div className="preview-text-bold">HORMOZI VIRAL</div>
                        </div>
                        <span className="style-studio-card-title">Bold</span>
                        <p className="style-studio-card-desc">Punchy uppercase</p>
                      </div>

                      {/* Word Highlight */}
                      <div
                        className={`style-studio-caption-card ${
                          styleStudioCaption === "preset_mrbeast_pop" ||
                          styleStudioCaption === "preset_dynamic_editorial" ||
                          styleStudioCaption === "preset_t7"
                            ? "selected"
                            : ""
                        }`}
                        onClick={() => setStyleStudioCaption("preset_mrbeast_pop")}
                      >
                        <div className="style-studio-preview-box">
                          <div className="preview-text-highlight">
                            Morning <span className="word-accent">cortisol</span>
                          </div>
                        </div>
                        <span className="style-studio-card-title">Highlight</span>
                        <p className="style-studio-card-desc">Word accent</p>
                      </div>
                    </div>

                    {/* Preset Chips */}
                    <div className="style-studio-presets-row">
                      <span className="style-studio-presets-label">Preset Selector:</span>
                      <div className="style-studio-presets-list">
                        {CAPTION_PRESETS.map((preset) => (
                          <button
                            key={preset.id}
                            type="button"
                            className={`style-studio-preset-chip ${styleStudioCaption === preset.id ? "active" : ""}`}
                            onClick={() => setStyleStudioCaption(preset.id)}
                          >
                            {preset.name}
                          </button>
                        ))}
                      </div>
                    </div>
                  </section>
                </>
              ) : (
                /* Mode 2: AutoClip Studio Controls */
                <section className="style-studio-section" style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                  <h4 className="style-studio-section-title">AutoClip Studio Controls</h4>

                  {/* Platform Strategy */}
                  <div>
                    <span className="style-studio-presets-label" style={{ display: "block", marginBottom: "6px" }}>
                      Platform Strategy
                    </span>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "8px" }}>
                      {[
                        { id: "douyin", name: "抖音 9:16", desc: "9:16 Blur" },
                        { id: "tiktok", name: "TikTok 9:16", desc: "9:16 Crop" },
                        { id: "youtube_shorts", name: "YouTube Shorts", desc: "9:16 Crop" },
                        { id: "bilibili", name: "Bilibili 16:9", desc: "16:9 Fit" },
                        { id: "xiaohongshu", name: "小红书 9:16", desc: "9:16 Blur Box" },
                      ].map((plat) => (
                        <div
                          key={plat.id}
                          className={`style-studio-framing-card ${styleStudioPlatformStrategy === plat.id ? "selected" : ""}`}
                          style={{ padding: "8px 10px" }}
                          onClick={() => setStyleStudioPlatformStrategy(plat.id as AutoClipPlatformStrategy)}
                        >
                          <div className="style-studio-framing-top">
                            <span className="style-studio-card-title" style={{ fontSize: "12px" }}>{plat.name}</span>
                            <div className="card-indicator">
                              {styleStudioPlatformStrategy === plat.id ? (
                                <div className="indicator-check"><Check size={10} strokeWidth={3} /></div>
                              ) : (
                                <div className="indicator-empty" />
                              )}
                            </div>
                          </div>
                          <p className="style-studio-card-desc" style={{ fontSize: "10.5px" }}>{plat.desc}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Subtitle Look */}
                  <div>
                    <span className="style-studio-presets-label" style={{ display: "block", marginBottom: "6px" }}>
                      Subtitle Look (≤2 lines, CJK/Latin width-aware)
                    </span>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "8px" }}>
                      {[
                        { id: "clean", name: "Clean", desc: "White text, 2px outline" },
                        { id: "bold", name: "Bold", desc: "Large bold white, 3px outline" },
                        { id: "box", name: "Box", desc: "Translucent dark card box" },
                        { id: "accent", name: "Accent", desc: "Bold yellow, high contrast" },
                      ].map((sub) => (
                        <div
                          key={sub.id}
                          className={`style-studio-framing-card ${styleStudioSubtitleLook === sub.id ? "selected" : ""}`}
                          style={{ padding: "8px 10px" }}
                          onClick={() => setStyleStudioSubtitleLook(sub.id as AutoClipSubtitleLook)}
                        >
                          <div className="style-studio-framing-top">
                            <span className="style-studio-card-title" style={{ fontSize: "12px" }}>{sub.name}</span>
                            <div className="card-indicator">
                              {styleStudioSubtitleLook === sub.id ? (
                                <div className="indicator-check"><Check size={10} strokeWidth={3} /></div>
                              ) : (
                                <div className="indicator-empty" />
                              )}
                            </div>
                          </div>
                          <p className="style-studio-card-desc" style={{ fontSize: "10.5px" }}>{sub.desc}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              )}

              {/* Section 3: Scope Selection */}
              <section className="style-studio-section">
                <h4 className="style-studio-section-title">Apply To</h4>
                <div className="style-studio-scope-row">
                  {styleStudioTargetCandidateId && (
                    <button
                      type="button"
                      className={`style-studio-scope-pill ${styleStudioScope === "target" ? "active" : ""}`}
                      onClick={() => setStyleStudioScope("target")}
                    >
                      This clip
                    </button>
                  )}
                  <button
                    type="button"
                    className={`style-studio-scope-pill ${styleStudioScope === "selected" ? "active" : ""}`}
                    onClick={() => setStyleStudioScope("selected")}
                  >
                    Selected clips ({selectedCandidates.length})
                  </button>
                  <button
                    type="button"
                    className={`style-studio-scope-pill ${styleStudioScope === "all" ? "active" : ""}`}
                    onClick={() => setStyleStudioScope("all")}
                  >
                    All clips ({detail.candidates.length})
                  </button>
                </div>

                {/* Warning note */}
                <div className="style-studio-warning-box">
                  <AlertTriangle size={15} />
                  <span>
                    Changes will affect rendered output. The previous successful render is retained until the new render completes.
                  </span>
                </div>
              </section>

              {/* Section 4: Summary & Recommendation */}
              <section className="style-studio-section">
                <h4 className="style-studio-section-title">Summary</h4>
                <div className="style-studio-summary-card">
                  <div className="style-studio-summary-row">
                    <span className="style-studio-summary-label">Applying to</span>
                    <span className="style-studio-summary-value">
                      {styleStudioScope === "target" && targetCandidate
                        ? `Clip #${targetCandidate.rank}`
                        : styleStudioScope === "selected"
                        ? `${targetCandidateIds.length} selected clips`
                        : `All ${targetCandidateIds.length} clips in project`}
                    </span>
                  </div>
                  <div className="style-studio-summary-row">
                    <span className="style-studio-summary-label">Engine</span>
                    <span className="style-studio-summary-value">
                      {styleStudioPresentationMode === "autoshorts" ? "AutoShorts Native (Mode 1)" : "AutoClip Studio (Mode 2)"}
                    </span>
                  </div>
                  {styleStudioPresentationMode === "autoshorts" ? (
                    <>
                      <div className="style-studio-summary-row">
                        <span className="style-studio-summary-label">Framing</span>
                        <span className="style-studio-summary-value">
                          {styleStudioFraming === "adaptive" ? "Adaptive Framing" : "Original 9:16"}
                        </span>
                      </div>
                      <div className="style-studio-summary-row">
                        <span className="style-studio-summary-label">Captions</span>
                        <span className="style-studio-summary-value">
                          {CAPTION_PRESETS.find(p => p.id === styleStudioCaption)?.name || styleStudioCaption}
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="style-studio-summary-row">
                        <span className="style-studio-summary-label">Platform</span>
                        <span className="style-studio-summary-value" style={{ textTransform: "capitalize" }}>
                          {styleStudioPlatformStrategy.replace("_", " ")}
                        </span>
                      </div>
                      <div className="style-studio-summary-row">
                        <span className="style-studio-summary-label">Subtitle Look</span>
                        <span className="style-studio-summary-value" style={{ textTransform: "capitalize" }}>
                          {styleStudioSubtitleLook}
                        </span>
                      </div>
                    </>
                  )}
                  <div className="style-studio-summary-row">
                    <span className="style-studio-summary-label">Estimated renders</span>
                    <span className="style-studio-summary-value">
                      {targetCandidateIds.length} {targetCandidateIds.length === 1 ? "clip" : "clips"}
                    </span>
                  </div>
                </div>

                <div className="style-studio-recommendation-card">
                  <Sparkles size={15} />
                  <span>
                    {styleStudioPresentationMode === "autoshorts"
                      ? "AI Recommendation: Adaptive Framing with high-contrast captions delivers highest completion rate on short-form platforms."
                      : "AutoClip Studio: CJK 2-line subtitles, optical-flow mouth tracking, and platform packaging optimized for viral retention."}
                  </span>
                </div>
              </section>
            </div>

            {/* Footer */}
            <div className="style-studio-footer">
              <div className="style-studio-footer-actions">
                <button
                  type="button"
                  className="style-studio-btn-restore"
                  onClick={restorePreviousVersion}
                  disabled={!hasPreviousVersion || busy !== "idle"}
                  title={hasPreviousVersion ? "Restore previous render version" : "No previous render version stored"}
                >
                  <RotateCcw size={13} />
                  <span>Restore Previous</span>
                </button>
                <button
                  type="button"
                  className="style-studio-btn-apply"
                  onClick={applyStyleStudioAndRerender}
                  disabled={busy !== "idle" || targetCandidateIds.length === 0 || !environment?.hasFfmpeg}
                  title="Apply styles and re-render target clips"
                >
                  <span>Apply &amp; Re-render{targetCandidateIds.length > 1 ? ` (${targetCandidateIds.length})` : ""}</span>
                  <ArrowRight size={14} />
                </button>
              </div>
              <p className="style-studio-disclaimer">
                Restyling changes caption appearance and framing only. Scoring, candidate ranking, and transcript are unchanged.
              </p>
            </div>
          </aside>
        </>
      )}

      {showStyleModal && (
        <div className="style-modal-overlay">
          <div className="style-modal">
            {/* Modal Header */}
            <div className="style-modal-header">
              <div className="modal-header-left">
                <button
                  type="button"
                  className="modal-back-btn"
                  onClick={() => {
                    setShowStyleModal(false);
                    setMediaPathToImport(null);
                  }}
                  aria-label="Go back"
                  title="Cancel and close"
                >
                  <ArrowLeft size={18} />
                </button>
                <div className="modal-header-text">
                  <h1 className="style-modal-title">Configure &amp; Start Processing</h1>
                  <p className="style-modal-subtitle">
                    Choose your framing, captions, and scoring settings before AutoShorts processes this video.
                  </p>
                </div>
              </div>
              {mediaFilename && (
                <div className="modal-media-chip" title={mediaPathToImport || ""}>
                  <Film size={14} className="modal-media-chip-icon" />
                  <span className="modal-media-chip-text">{mediaFilename}</span>
                </div>
              )}
            </div>

            {/* Scrollable Modal Body */}
            <div className="style-modal-body">
              {/* Section 1: Presentation Engine */}
              <section className="modal-section">
                <div className="modal-section-header">
                  <h2 className="modal-section-title">Presentation Engine</h2>
                  <span className="modal-section-step">Step 1 of 3 · Engine Architecture</span>
                </div>
                <div className="framing-grid">
                  {/* Mode 1: AutoShorts Native */}
                  <div
                    className={`framing-card ${selectedPresentationMode === "autoshorts" ? "selected" : ""}`}
                    onClick={() => setSelectedPresentationMode("autoshorts")}
                  >
                    <div className="card-top-row">
                      <div className="card-icon-box mint">
                        <Crop size={18} />
                      </div>
                      <div className="card-indicator">
                        {selectedPresentationMode === "autoshorts" ? (
                          <div className="indicator-check"><Check size={13} strokeWidth={3} /></div>
                        ) : (
                          <div className="indicator-empty" />
                        )}
                      </div>
                    </div>
                    <div className="card-title-row">
                      <span className="card-title">AutoShorts Native (Mode 1)</span>
                      <span className="card-badge mint">DEFAULT</span>
                    </div>
                    <p className="card-desc">
                      YOLOv8/11 + BoT-SORT continuous tracking, 1:1 midpoint retention, kinetic ASS subtitles.
                    </p>
                  </div>

                  {/* Mode 2: AutoClip Studio */}
                  <div
                    className={`framing-card ${selectedPresentationMode === "autoclip" ? "selected" : ""}`}
                    onClick={() => setSelectedPresentationMode("autoclip")}
                  >
                    <div className="card-top-row">
                      <div className="card-icon-box slate">
                        <Clapperboard size={18} />
                      </div>
                      <div className="card-indicator">
                        {selectedPresentationMode === "autoclip" ? (
                          <div className="indicator-check"><Check size={13} strokeWidth={3} /></div>
                        ) : (
                          <div className="indicator-empty" />
                        )}
                      </div>
                    </div>
                    <div className="card-title-row">
                      <span className="card-title">AutoClip Studio (Mode 2)</span>
                      <span className="card-badge slate">STUDIO</span>
                    </div>
                    <p className="card-desc">
                      OpenCV YuNet, hard cuts, mouth optical flow, hold-segment framing, blurred fit fallback, CJK 2-line subtitles, PIL TitleArt.
                    </p>
                  </div>
                </div>
              </section>

              {/* Section 2: Presentation Styling */}
              <section className="modal-section">
                <div className="modal-section-header">
                  <h2 className="modal-section-title">
                    {selectedPresentationMode === "autoshorts" ? "Presentation Styling" : "AutoClip Studio Styling"}
                  </h2>
                  <span className="modal-section-step">Step 2 of 3 · Visual &amp; Layout Styling</span>
                </div>

                {selectedPresentationMode === "autoshorts" ? (
                  <>
                    {/* Mode 1 Framing options */}
                    <div style={{ marginBottom: "16px" }}>
                      <span className="presets-row-label" style={{ display: "block", marginBottom: "8px", fontWeight: 600 }}>
                        Framing Mode
                      </span>
                      <div className="framing-grid">
                        <div
                          className={`framing-card ${selectedFramingMode === "adaptive" ? "selected" : ""}`}
                          onClick={() => setSelectedFramingMode("adaptive")}
                        >
                          <div className="card-top-row">
                            <div className="card-icon-box mint">
                              <Crop size={18} />
                            </div>
                            <div className="card-indicator">
                              {selectedFramingMode === "adaptive" ? (
                                <div className="indicator-check"><Check size={13} strokeWidth={3} /></div>
                              ) : (
                                <div className="indicator-empty" />
                              )}
                            </div>
                          </div>
                          <div className="card-title-row">
                            <span className="card-title">Adaptive Framing</span>
                            <span className="card-badge mint">SMART</span>
                          </div>
                          <p className="card-desc">
                            AutoShorts dynamically tracks and reframes speakers, isolating the active speaker when two or more are present.
                          </p>
                        </div>

                        <div
                          className={`framing-card ${selectedFramingMode === "original" ? "selected" : ""}`}
                          onClick={() => setSelectedFramingMode("original")}
                        >
                          <div className="card-top-row">
                            <div className="card-icon-box">
                              <Smartphone size={18} />
                            </div>
                            <div className="card-indicator">
                              {selectedFramingMode === "original" ? (
                                <div className="indicator-check"><Check size={13} strokeWidth={3} /></div>
                              ) : (
                                <div className="indicator-empty" />
                              )}
                            </div>
                          </div>
                          <div className="card-title-row">
                            <span className="card-title">Original 9:16</span>
                          </div>
                          <p className="card-desc">
                            Preserves original recording framing at 9:16 aspect ratio. Includes DualFrame layout when multiple speakers are detected.
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Mode 1 Caption presets */}
                    <span className="presets-row-label" style={{ display: "block", marginBottom: "8px", fontWeight: 600 }}>
                      Caption Preset
                    </span>
                    <div className="style-grid-c">
                      {/* Clean */}
                      <div
                        className={`style-card-c ${selectedStyle === "preset_minimal_capsule" || selectedStyle === "preset_cinematic_vlog" ? "selected" : ""}`}
                        onClick={() => setSelectedStyle("preset_minimal_capsule")}
                      >
                        <div className="style-preview-box-c">
                          <div className="preview-dots">
                            <span /><span />
                          </div>
                          <div className="preview-badge-clean">
                            How sleep affects cortisol
                          </div>
                        </div>
                        <div className="card-info-row">
                          <div className="card-info">
                            <span className="card-title">Clean</span>
                            <p className="card-desc-compact">Minimal, centered captions.</p>
                          </div>
                          <div className="card-indicator">
                            {selectedStyle === "preset_minimal_capsule" || selectedStyle === "preset_cinematic_vlog" ? (
                              <div className="indicator-check"><Check size={12} strokeWidth={3} /></div>
                            ) : (
                              <div className="indicator-empty" />
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Bold */}
                      <div
                        className={`style-card-c ${selectedStyle === "preset_viral_bold" || selectedStyle === "preset_bhaukal_caption" ? "selected" : ""}`}
                        onClick={() => setSelectedStyle("preset_viral_bold")}
                      >
                        <div className="style-preview-box-c">
                          <div className="preview-text-bold">
                            HORMOZI VIRAL
                          </div>
                        </div>
                        <div className="card-info-row">
                          <div className="card-info">
                            <span className="card-title">Bold</span>
                            <p className="card-desc-compact">High-contrast, punchy captions.</p>
                          </div>
                          <div className="card-indicator">
                            {selectedStyle === "preset_viral_bold" || selectedStyle === "preset_bhaukal_caption" ? (
                              <div className="indicator-check"><Check size={12} strokeWidth={3} /></div>
                            ) : (
                              <div className="indicator-empty" />
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Word Highlight */}
                      <div
                        className={`style-card-c ${selectedStyle === "preset_mrbeast_pop" || selectedStyle === "preset_dynamic_editorial" || selectedStyle === "preset_t7" ? "selected" : ""}`}
                        onClick={() => setSelectedStyle("preset_mrbeast_pop")}
                      >
                        <div className="style-preview-box-c">
                          <div className="preview-text-highlight">
                            The morning <span className="word-accent">cortisol</span> surge
                          </div>
                        </div>
                        <div className="card-info-row">
                          <div className="card-info">
                            <span className="card-title">Word Highlight</span>
                            <p className="card-desc-compact">Word-by-word accented karaoke.</p>
                          </div>
                          <div className="card-indicator">
                            {selectedStyle === "preset_mrbeast_pop" || selectedStyle === "preset_dynamic_editorial" || selectedStyle === "preset_t7" ? (
                              <div className="indicator-check"><Check size={12} strokeWidth={3} /></div>
                            ) : (
                              <div className="indicator-empty" />
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Secondary row for specific presets */}
                    <div className="style-presets-row">
                      <span className="presets-row-label">Specific Preset:</span>
                      <div className="preset-pills-list">
                        {CAPTION_PRESETS.map((preset) => (
                          <button
                            key={preset.id}
                            type="button"
                            className={`preset-pill-chip ${selectedStyle === preset.id ? "active" : ""}`}
                            onClick={() => setSelectedStyle(preset.id)}
                          >
                            {preset.name}
                          </button>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  /* Mode 2: AutoClip Isolated Controls */
                  <div className="autoclip-isolated-controls" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                    {/* Platform Strategy */}
                    <div>
                      <span className="presets-row-label" style={{ display: "block", marginBottom: "8px", fontWeight: 600 }}>
                        Platform Strategy
                      </span>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "10px" }}>
                        {[
                          { id: "douyin", name: "抖音 9:16", desc: "9:16 Blur Backdrop" },
                          { id: "tiktok", name: "TikTok 9:16", desc: "9:16 Tight Face Crop" },
                          { id: "youtube_shorts", name: "YouTube Shorts", desc: "9:16 Portrait Crop" },
                          { id: "bilibili", name: "Bilibili 16:9", desc: "16:9 Landscape Fit" },
                          { id: "xiaohongshu", name: "小红书 9:16", desc: "9:16 Blur + Card" },
                        ].map((plat) => (
                          <div
                            key={plat.id}
                            className={`framing-card ${platformStrategy === plat.id ? "selected" : ""}`}
                            style={{ padding: "10px" }}
                            onClick={() => setPlatformStrategy(plat.id as AutoClipPlatformStrategy)}
                          >
                            <div className="card-top-row">
                              <span className="card-title" style={{ fontSize: "13px" }}>{plat.name}</span>
                              <div className="card-indicator">
                                {platformStrategy === plat.id ? (
                                  <div className="indicator-check"><Check size={11} strokeWidth={3} /></div>
                                ) : (
                                  <div className="indicator-empty" />
                                )}
                              </div>
                            </div>
                            <p className="card-desc" style={{ fontSize: "11px", margin: 0 }}>{plat.desc}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Subtitle Look */}
                    <div>
                      <span className="presets-row-label" style={{ display: "block", marginBottom: "8px", fontWeight: 600 }}>
                        Subtitle Look (≤2 lines per screen, CJK/Latin width-aware)
                      </span>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px" }}>
                        {[
                          { id: "clean", name: "Clean", desc: "White text, 2px outline" },
                          { id: "bold", name: "Bold", desc: "Large bold white, 3px outline" },
                          { id: "box", name: "Box", desc: "Translucent dark card box" },
                          { id: "accent", name: "Accent", desc: "Bold yellow, high contrast" },
                        ].map((sub) => (
                          <div
                            key={sub.id}
                            className={`framing-card ${subtitleLook === sub.id ? "selected" : ""}`}
                            style={{ padding: "10px" }}
                            onClick={() => setSubtitleLook(sub.id as AutoClipSubtitleLook)}
                          >
                            <div className="card-top-row">
                              <span className="card-title" style={{ fontSize: "13px" }}>{sub.name}</span>
                              <div className="card-indicator">
                                {subtitleLook === sub.id ? (
                                  <div className="indicator-check"><Check size={11} strokeWidth={3} /></div>
                                ) : (
                                  <div className="indicator-empty" />
                                )}
                              </div>
                            </div>
                            <p className="card-desc" style={{ fontSize: "11px", margin: 0 }}>{sub.desc}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </section>

              {/* Section 3: Scoring Backend */}
              <section className="modal-section">
                <div className="modal-section-header">
                  <div className="title-with-info">
                    <h2 className="modal-section-title">Scoring Backend</h2>
                    <span
                      className="info-tooltip-wrap"
                      title="Determines how viral moments, speech hooks, and retention peaks are identified."
                    >
                      <Info size={14} />
                    </span>
                  </div>
                  <span className="modal-section-step">Step 3 of 3 · Model Engine</span>
                </div>
                <div className="scoring-stack">
                  {/* Legacy Scoring */}
                  <div
                    className={`scoring-card ${scoringMode === "legacy" ? "selected" : ""}`}
                    onClick={() => setScoringMode("legacy")}
                  >
                    <div className="scoring-card-left">
                      <div className="card-icon-box mint">
                        <ShieldCheck size={20} />
                      </div>
                      <div className="scoring-card-content">
                        <div className="card-title-row">
                          <span className="card-title">Legacy Scoring — Recommended</span>
                          <span className="card-badge default">DEFAULT</span>
                        </div>
                        <p className="card-desc">
                          Fast, offline composite scoring. No external API required. Recommended for all users.
                        </p>
                        <div className="card-meta-chips">
                          <span className="meta-chip mint"><Zap size={12} /> ~45s Processing</span>
                          <span className="meta-dot">•</span>
                          <span className="meta-chip mint"><Lock size={12} /> 100% On-device Privacy</span>
                        </div>
                      </div>
                    </div>
                    <div className="card-indicator">
                      {scoringMode === "legacy" ? (
                        <div className="indicator-check"><Check size={13} strokeWidth={3} /></div>
                      ) : (
                        <div className="indicator-empty" />
                      )}
                    </div>
                  </div>

                  {/* AutoClip Holistic Scoring */}
                  <div
                    className={`scoring-card ${scoringMode === "autoclip" ? "selected" : ""}`}
                    onClick={() => setScoringMode("autoclip")}
                  >
                    <div className="scoring-card-left">
                      <div className="card-icon-box slate">
                        <Cpu size={20} />
                      </div>
                      <div className="scoring-card-content">
                        <div className="card-title-row">
                          <span className="card-title">AutoClip Holistic Scoring — NVIDIA</span>
                          <span className="card-badge amber">
                            <Key size={11} /> External API · API Key Required
                          </span>
                        </div>
                        <p className="card-desc">
                          Multi-dimensional editorial scoring via NVIDIA NIM external API. Requires a configured NVIDIA API key and may incur API usage costs.
                        </p>
                        <div className="card-meta-chips">
                          <span className="meta-chip"><Cloud size={12} /> Requires Internet</span>
                          <span className="meta-dot">•</span>
                          <span className="meta-chip"><SlidersHorizontal size={12} /> Deep Nuance Extraction</span>
                        </div>
                      </div>
                    </div>
                    <div className="card-indicator">
                      {scoringMode === "autoclip" ? (
                        <div className="indicator-check"><Check size={13} strokeWidth={3} /></div>
                      ) : (
                        <div className="indicator-empty" />
                      )}
                    </div>
                  </div>
                </div>

                {scoringMode === "autoclip" && environment !== null && !environment.hasNvidiaKey && (
                  <div className="api-warning">
                    ⚠️ AutoClip Scoring (NVIDIA) is selected but NVIDIA_API_KEY is missing. Configure it in your .env or environment, or switch back to Legacy Scoring — moment finding will report this error instead of falling back.
                  </div>
                )}
              </section>
            </div>

            {/* Modal Footer */}
            <div className="style-modal-footer">
              <button
                type="button"
                className="btn-modal-cancel"
                onClick={() => {
                  setShowStyleModal(false);
                  setMediaPathToImport(null);
                }}
              >
                Cancel
              </button>
              <div className="footer-right-group">
                <span className="footer-helper-note">
                  This will begin transcription and analysis.
                </span>
                <button
                  type="button"
                  className="btn-modal-confirm"
                  disabled={selectedPresentationMode === "autoshorts" ? !selectedStyle : false}
                  onClick={() => {
                    confirmImport(selectedStyle, selectedFramingMode);
                  }}
                >
                  <span>Start Processing</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      {downloadingModelName && (
        <div className="onboarding-overlay" style={{ zIndex: 20000 }}>
          <div className="onboarding-card" style={{ maxWidth: '480px', textAlign: 'center' }}>
            <div className="onboarding-header compact" style={{ textAlign: 'center' }}>
              <h2>Downloading Ollama Model</h2>
              <p>Downloading model weights for "{downloadingModelName}". Please do not close the app.</p>
            </div>

            <div className="download-progress-container">
              <div className="download-loader">
                <Loader2 className="spin" size={48} />
              </div>

              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: `${modelDownloadProgress}%` }}></div>
              </div>

              <div className="download-stats">
                <span className="download-status">{modelDownloadStatus}</span>
                <span className="download-percentage">{modelDownloadProgress}%</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {youtubeModalOpen && (
        <div className="style-modal-overlay">
          <div className="style-modal" style={{ maxWidth: "520px" }}>
            <div className="style-modal-header">
              <h3>Import from YouTube</h3>
              <p>Paste a YouTube URL below to download and import it directly.</p>
            </div>
            <div style={{ padding: "1rem" }}>
              <input
                type="text"
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={(e) => {
                  setYoutubeUrl(e.target.value);
                  if (youtubeAudioTracks.length > 0 || youtubeVideoTitle) {
                    setYoutubeAudioTracks([]);
                    setSelectedAudioFormatId(null);
                    setYoutubeVideoTitle(null);
                    setYoutubeWarningLicense(null);
                    setYoutubeStatus("idle");
                  }
                }}
                disabled={youtubeStatus === "checking" || youtubeStatus === "downloading"}
                style={{ width: "100%", padding: "0.75rem", borderRadius: "8px", border: "1px solid var(--border)", background: "var(--background)", color: "var(--foreground)", fontSize: "1rem", marginBottom: "1rem" }}
              />

              {youtubeVideoTitle && (
                <div style={{ marginBottom: "1rem", padding: "0.6rem 0.85rem", background: "rgba(255, 255, 255, 0.04)", borderRadius: "8px", border: "1px solid var(--border)", fontSize: "0.88rem" }}>
                  <div style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--muted-foreground)", marginBottom: "0.25rem", fontWeight: 600 }}>Video Found</div>
                  <div style={{ fontWeight: 500, color: "var(--foreground)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{youtubeVideoTitle}</div>
                </div>
              )}

              {youtubeAudioTracks.length > 1 && (
                <div style={{ marginBottom: "1.25rem", padding: "0.85rem", background: "rgba(99, 102, 241, 0.08)", borderRadius: "8px", border: "1px solid rgba(99, 102, 241, 0.3)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.4rem" }}>
                    <AudioLines size={16} style={{ color: "var(--primary, #6366f1)" }} />
                    <label style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--foreground)" }}>
                      Multiple Audio Tracks Detected ({youtubeAudioTracks.length})
                    </label>
                  </div>
                  <p style={{ margin: "0 0 0.5rem 0", fontSize: "0.8rem", color: "var(--muted-foreground)" }}>
                    Select which audio language to download with the video:
                  </p>
                  <select
                    value={selectedAudioFormatId || ""}
                    onChange={(e) => setSelectedAudioFormatId(e.target.value)}
                    disabled={youtubeStatus === "downloading"}
                    style={{
                      width: "100%",
                      padding: "0.65rem 0.75rem",
                      borderRadius: "6px",
                      border: "1px solid var(--border)",
                      background: "var(--background)",
                      color: "var(--foreground)",
                      fontSize: "0.9rem",
                      cursor: "pointer",
                      fontWeight: 500,
                    }}
                  >
                    {youtubeAudioTracks.map((track) => (
                      <option key={track.formatId} value={track.formatId}>
                        {track.isOriginal ? `⭐ ${track.languageName} (Original - Recommended)` : `${track.languageName}`}
                        {track.bitrateKbps ? ` [~${Math.round(track.bitrateKbps)} kbps]` : ""}
                      </option>
                    ))}
                  </select>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted-foreground)", marginTop: "0.4rem" }}>
                    ⭐ Original audio track is pre-selected by default to avoid translated dubs.
                  </div>
                </div>
              )}

              {youtubeError && (
                <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid #ef4444", padding: "0.75rem 1rem", borderRadius: "8px", marginBottom: "1rem", color: "#f87171", fontSize: "0.85rem", maxHeight: "150px", overflowY: "auto" }}>
                  <div style={{ fontWeight: "bold", marginBottom: "0.25rem" }}>Download Failed</div>
                  <div style={{ whiteSpace: "pre-wrap" }}>{youtubeError}</div>
                </div>
              )}

              {(youtubeStatus === "warning" || youtubeStatus === "downloading") && youtubeWarningLicense && (
                <div style={{ background: "rgba(255, 165, 0, 0.1)", border: "1px solid orange", padding: "1rem", borderRadius: "8px", marginBottom: "1rem", color: "orange" }}>
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem", fontWeight: "bold" }}>
                    <AlertTriangle size={20} />
                    Copyright Warning
                  </div>
                  <p style={{ margin: 0, fontSize: "0.9rem" }}>
                    This video is not explicitly marked for reuse (Creative Commons). Its license appears to be: <strong>{youtubeWarningLicense}</strong>.
                    {youtubeStatus === "warning" && (<><br/><br/>Clipping this video may lead to copyright strikes. Are you sure you want to proceed?</>)}
                  </p>
                </div>
              )}

              {youtubeStatus === "downloading" && (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.75rem", padding: "1.5rem 0", color: "var(--foreground)" }}>
                  <Loader2 className="spin" size={24} />
                  <span style={{ fontSize: "1rem" }}>Downloading video…</span>
                </div>
              )}

              <div className="style-modal-actions" style={{ marginTop: "1rem" }}>
                <button 
                  className="btn-cancel" 
                  onClick={() => {
                    setYoutubeModalOpen(false);
                    setYoutubeUrl("");
                    setYoutubeStatus("idle");
                    setYoutubeWarningLicense(null);
                    setYoutubeError(null);
                    setYoutubeAudioTracks([]);
                    setSelectedAudioFormatId(null);
                    setYoutubeVideoTitle(null);
                  }}
                  disabled={youtubeStatus === "checking" || youtubeStatus === "downloading"}
                >
                  Cancel
                </button>
                {youtubeStatus === "warning" ? (
                  <button className="btn-confirm" onClick={() => executeYoutubeDownload()}>
                    Yes, I understand the risks
                  </button>
                ) : youtubeStatus === "ready" ? (
                  <button className="btn-confirm" onClick={() => executeYoutubeDownload()} style={{ minWidth: "120px" }}>
                    Download Video
                  </button>
                ) : youtubeStatus === "downloading" ? null : (
                  <button 
                    className="btn-confirm" 
                    onClick={handleYoutubeImport}
                    disabled={!youtubeUrl || youtubeStatus !== "idle"}
                    style={{ minWidth: "120px" }}
                  >
                    {youtubeStatus === "checking" ? <><Loader2 className="spin" size={18} /> Checking...</> :
                     "Check & Download"}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

interface SettingsPanelProps {
  transcriptionEngine: "deepgram" | "local";
  setTranscriptionEngine: (engine: "deepgram" | "local") => void;
  llmEngine: "claude" | "deepseek" | "local" | "gemini" | "openai" | "openrouter" | "groq" | "nvidia";
  setLlmEngine: (engine: "claude" | "deepseek" | "local" | "gemini" | "openai" | "openrouter" | "groq" | "nvidia") => void;
  deepgramKey: string;
  setDeepgramKey: (key: string) => void;
  anthropicKey: string;
  setAnthropicKey: (key: string) => void;
  deepseekKey: string;
  setDeepseekKey: (key: string) => void;
  deepseekModel: string;
  setDeepseekModel: (m: string) => void;
  geminiKey: string;
  setGeminiKey: (key: string) => void;
  openaiKey: string;
  setOpenaiKey: (key: string) => void;
  openrouterKey: string;
  setOpenrouterKey: (key: string) => void;
  openrouterModel: string;
  setOpenrouterModel: (m: string) => void;
  groqKey: string;
  setGroqKey: (key: string) => void;
  nvidiaKey: string;
  setNvidiaKey: (key: string) => void;
  nvidiaModel: string;
  setNvidiaModel: (m: string) => void;
  localLlmModel: string;
  setLocalLlmModel: (m: string) => void;
  pullModelDirectly: (model: string) => Promise<void>;
  youtubeBrowser: string;
  setYoutubeBrowser: (b: string) => void;
  youtubeCookiesPath: string;
  setYoutubeCookiesPath: (p: string) => void;
  environment: EnvironmentStatus | null;
}

function SettingsPanel({
  transcriptionEngine,
  setTranscriptionEngine,
  llmEngine,
  setLlmEngine,
  deepgramKey,
  setDeepgramKey,
  anthropicKey,
  setAnthropicKey,
  deepseekKey,
  setDeepseekKey,
  deepseekModel,
  setDeepseekModel,
  geminiKey,
  setGeminiKey,
  openaiKey,
  setOpenaiKey,
  openrouterKey,
  setOpenrouterKey,
  openrouterModel,
  setOpenrouterModel,
  groqKey,
  setGroqKey,
  nvidiaKey,
  setNvidiaKey,
  nvidiaModel,
  setNvidiaModel,
  localLlmModel,
  setLocalLlmModel,
  pullModelDirectly,
  youtubeBrowser,
  setYoutubeBrowser,
  youtubeCookiesPath,
  setYoutubeCookiesPath,
  environment,
}: SettingsPanelProps) {
  return (
    <div className="settings-panel">
      <div className="key-stack-horizontal">
        <label>
          <span>Transcription Engine</span>
          <select
            value={transcriptionEngine}
            onChange={(event) => setTranscriptionEngine(event.target.value as "deepgram" | "local")}
          >
            <option value="local">Local Whisper (Offline)</option>
            <option value="deepgram">Deepgram (Cloud)</option>
          </select>
          <span>LLM Engine</span>
          <select
            value={llmEngine}
            onChange={(event) => setLlmEngine(event.target.value as any)}
          >
            <option value="nvidia">NVIDIA NIM (Gemma 4 31B IT)</option>
            <option value="local">Ollama (Offline Local)</option>
            <option value="claude">Claude (Cloud)</option>
            <option value="deepseek">DeepSeek (Cloud)</option>
            <option value="gemini">Google Gemini (Cloud)</option>
            <option value="openai">OpenAI (Cloud)</option>
            <option value="openrouter">OpenRouter (Cloud)</option>
            <option value="groq">Groq (Cloud)</option>
          </select>
        </label>
        {transcriptionEngine === "deepgram" && (
          <label>
            <span>Deepgram API Key</span>
            <input
              value={deepgramKey}
              onChange={(event) => setDeepgramKey(event.target.value)}
              placeholder={environment?.hasDeepgramKey ? "Loaded from env" : "Optional (Deepgram API Key)"}
              type="password"
            />
          </label>
        )}
        {llmEngine === "claude" && (
          <label>
            <span>Claude API Key</span>
            <input
              value={anthropicKey}
              onChange={(event) => setAnthropicKey(event.target.value)}
              placeholder={environment?.hasAnthropicKey ? "Loaded from env" : "Optional (Claude API Key)"}
              type="password"
            />
          </label>
        )}
        {llmEngine === "deepseek" && (
          <>
            <label>
              <span>DeepSeek API Key</span>
              <input
                value={deepseekKey}
                onChange={(event) => setDeepseekKey(event.target.value)}
                placeholder={environment?.hasDeepseekKey ? "Loaded from env" : "Optional (DeepSeek API Key)"}
                type="password"
              />
            </label>
            <label>
              <span>DeepSeek Model</span>
              <input
                value={deepseekModel}
                onChange={(event) => setDeepseekModel(event.target.value)}
                placeholder="Optional (e.g. deepseek-chat, deepseek-v4-pro)"
                type="text"
              />
            </label>
          </>
        )}
        {llmEngine === "gemini" && (
          <label>
            <span>Gemini API Key</span>
            <input
              value={geminiKey}
              onChange={(event) => setGeminiKey(event.target.value)}
              placeholder={environment?.hasGeminiKey ? "Loaded from env" : "Optional (Gemini API Key)"}
              type="password"
            />
          </label>
        )}
        {llmEngine === "openai" && (
          <label>
            <span>OpenAI API Key</span>
            <input
              value={openaiKey}
              onChange={(event) => setOpenaiKey(event.target.value)}
              placeholder={environment?.hasOpenaiKey ? "Loaded from env" : "Optional (OpenAI API Key)"}
              type="password"
            />
          </label>
        )}
        {llmEngine === "openrouter" && (
          <>
            <label>
              <span>OpenRouter API Key</span>
              <input
                value={openrouterKey}
                onChange={(event) => setOpenrouterKey(event.target.value)}
                placeholder={environment?.hasOpenrouterKey ? "Loaded from env" : "Optional (OpenRouter API Key)"}
                type="password"
              />
            </label>
            <label>
              <span>OpenRouter Model</span>
              <input
                value={openrouterModel}
                onChange={(event) => setOpenrouterModel(event.target.value)}
                placeholder="Optional (e.g. google/gemini-2.5-flash, deepseek/deepseek-r1)"
                type="text"
              />
            </label>
          </>
        )}
        {llmEngine === "groq" && (
          <label>
            <span>Groq API Key</span>
            <input
              value={groqKey}
              onChange={(event) => setGroqKey(event.target.value)}
              placeholder={environment?.hasGroqKey ? "Loaded from env" : "Optional (Groq API Key)"}
              type="password"
            />
          </label>
        )}
        {llmEngine === "nvidia" && (
          <>
            <label>
              <span>NVIDIA API Key</span>
              <input
                value={nvidiaKey}
                onChange={(event) => setNvidiaKey(event.target.value)}
                placeholder={environment?.hasNvidiaKey ? "Loaded from env" : "Optional (NVIDIA API Key)"}
                type="password"
              />
            </label>
            <label>
              <span>NVIDIA Model</span>
              <input
                value={nvidiaModel}
                onChange={(event) => setNvidiaModel(event.target.value)}
                placeholder="Optional (e.g. google/gemma-4-31b-it)"
                type="text"
              />
            </label>
          </>
        )}

        {llmEngine === "local" && (
          <label>
            <span>Ollama Model Name</span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                value={localLlmModel}
                onChange={(event) => setLocalLlmModel(event.target.value)}
                placeholder="e.g. llama3.2, qwen2.5:7b"
                type="text"
              />
              <button
                type="button"
                className="icon-button"
                style={{ minHeight: '36px', height: '36px' }}
                onClick={() => pullModelDirectly(localLlmModel)}
              >
                <Download size={14} /> Pull
              </button>
            </div>
          </label>
        )}

        <label>
          <span>YouTube Authentication Browser</span>
          <select
            value={youtubeBrowser}
            onChange={(event) => setYoutubeBrowser(event.target.value)}
          >
            <option value="auto">Auto (Try Unauthenticated, then Installed Browsers)</option>
            <option value="firefox">Firefox</option>
            <option value="chrome">Google Chrome</option>
            <option value="edge">Microsoft Edge</option>
            <option value="brave">Brave Browser</option>
            <option value="opera">Opera</option>
            <option value="safari">Safari (macOS)</option>
            <option value="none">None (Strictly Unauthenticated)</option>
            <option value="custom">Custom cookies.txt File</option>
          </select>
        </label>

        {youtubeBrowser === "custom" && (
          <label>
            <span>Custom cookies.txt Path</span>
            <input
              value={youtubeCookiesPath}
              onChange={(event) => setYoutubeCookiesPath(event.target.value)}
              placeholder="e.g. C:\Users\user\cookies.txt"
              type="text"
            />
          </label>
        )}
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '16px', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
        <button
          type="button"
          className="icon-button"
          style={{ background: 'rgba(239, 68, 68, 0.08)', borderColor: 'rgba(239, 68, 68, 0.2)', color: '#f87171' }}
          onClick={() => {
            if (window.confirm("Are you sure you want to reset your configuration and restart onboarding from scratch?")) {
              localStorage.clear();
              window.location.reload();
            }
          }}
        >
          Reset App Configuration & Onboarding
        </button>
      </div>
    </div>
  );
}

function StatusPill({ label, active }: { label: string; active?: boolean }) {
  return (
    <div className={`status-pill ${active ? "active" : ""}`}>
      <BadgeCheck size={14} />
      {label}
    </div>
  );
}

function PipelineStep({ icon, label, done }: { icon: React.ReactNode; label: string; done: boolean }) {
  return (
    <div className={`pipeline-step ${done ? "done" : ""}`}>
      {icon}
      <span>{label}</span>
    </div>
  );
}

function EmptyState({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="empty-state">
      {icon}
      <span>{label}</span>
    </div>
  );
}

function fileName(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.floor(seconds % 60);
  return `${minutes}:${remaining.toString().padStart(2, "0")}`;
}

interface OnboardingProps {
  environment: EnvironmentStatus | null;
  onComplete: () => void;
  setTranscriptionEngine: (engine: "deepgram" | "local") => void;
  setLlmEngine: (engine: "claude" | "deepseek" | "local" | "groq") => void;
  setLocalLlmModel: (model: string) => void;
  setDeepgramKey: (key: string) => void;
  setAnthropicKey: (key: string) => void;
  setDeepseekKey: (key: string) => void;
  setGroqKey: (key: string) => void;
  deepgramKey: string;
  anthropicKey: string;
  deepseekKey: string;
  groqKey: string;
  refreshEnv: () => Promise<void>;
}

function Onboarding({
  environment,
  onComplete,
  setTranscriptionEngine,
  setLlmEngine,
  setLocalLlmModel,
  setDeepgramKey,
  setAnthropicKey,
  setDeepseekKey,
  setGroqKey,
  deepgramKey: initialDeepgramKey,
  anthropicKey: initialAnthropicKey,
  deepseekKey: initialDeepseekKey,
  groqKey: initialGroqKey,
  refreshEnv,
}: OnboardingProps) {
  const [setupMode, setSetupMode] = useState<"choose" | "local" | "cloud" | "downloading">("choose");
  const [selectedModel, setSelectedModel] = useState<string>("llama3.2");

  const [dgKey, setDgKey] = useState(initialDeepgramKey);
  const [antKey, setAntKey] = useState(initialAnthropicKey);
  const [dsKey, setDsKey] = useState(initialDeepseekKey);
  const [grKey, setGrKey] = useState(initialGroqKey);

  const [downloadStatus, setDownloadStatus] = useState("Initializing download...");
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [checkingOllama, setCheckingOllama] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyWhisperCommand = () => {
    navigator.clipboard.writeText("pip3 install -U openai-whisper");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCloudSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!dgKey.trim()) {
      setError("Deepgram API Key is required for cloud mode.");
      return;
    }
    if (!antKey.trim() && !dsKey.trim() && !grKey.trim()) {
      setError("Please provide at least one LLM Key (Claude, DeepSeek, or Groq).");
      return;
    }

    setTranscriptionEngine("deepgram");
    setDeepgramKey(dgKey.trim());
    localStorage.setItem("autoshorts_deepgram_key", dgKey.trim());
    localStorage.setItem("autoshorts_transcription_engine", "deepgram");

    if (antKey.trim()) {
      setLlmEngine("claude");
      setAnthropicKey(antKey.trim());
      localStorage.setItem("autoshorts_anthropic_key", antKey.trim());
      localStorage.setItem("autoshorts_llm_engine", "claude");
    } else if (dsKey.trim()) {
      setLlmEngine("deepseek");
      setDeepseekKey(dsKey.trim());
      localStorage.setItem("autoshorts_deepseek_key", dsKey.trim());
      localStorage.setItem("autoshorts_llm_engine", "deepseek");
    } else if (grKey.trim()) {
      setLlmEngine("groq");
      setGroqKey(grKey.trim());
      localStorage.setItem("autoshorts_groq_key", grKey.trim());
      localStorage.setItem("autoshorts_llm_engine", "groq");
    }

    localStorage.setItem("autoshorts_onboarded", "true");
    onComplete();
  };

  const startLocalSetup = async () => {
    setError(null);
    setCheckingOllama(true);
    setDownloadProgress(0);

    await refreshEnv();

    let isOllamaRunning = false;
    try {
      const currentEnv = await invoke<EnvironmentStatus>("environment_status");
      isOllamaRunning = currentEnv.hasOllama;
    } catch (e) {
      // ignore
    }

    setCheckingOllama(false);

    if (!isOllamaRunning) {
      setSetupMode("downloading");
      setDownloadStatus("Ollama not found. Starting automatic installer...");

      try {
        const unlistenInstall = await listen<string>("ollama-install-status", (event) => {
          setDownloadStatus(event.payload);
        });

        await invoke("install_ollama");
        unlistenInstall();
      } catch (err) {
        setError("Automatic installation failed: " + String(err) + ". Please install it manually from ollama.com.");
        setSetupMode("local");
        return;
      }
    }

    setSetupMode("downloading");
    setDownloadStatus("Ollama connected. Initiating model download...");

    try {
      const unlisten = await listen<{
        status: string;
        completed?: number;
        total?: number;
        percentage?: number;
      }>("ollama-pull-progress", (event) => {
        const payload = event.payload;
        setDownloadStatus(payload.status);
        if (payload.percentage !== undefined && payload.percentage !== null) {
          setDownloadProgress(Math.round(payload.percentage));
        }
      });

      await invoke("pull_ollama_model", { modelName: selectedModel });

      unlisten();

      setTranscriptionEngine("local");
      setLlmEngine("local");
      setLocalLlmModel(selectedModel);

      localStorage.setItem("autoshorts_transcription_engine", "local");
      localStorage.setItem("autoshorts_llm_engine", "local");
      localStorage.setItem("autoshorts_local_llm_model", selectedModel);
      localStorage.setItem("autoshorts_onboarded", "true");

      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSetupMode("local");
    }
  };

  return (
    <div className="onboarding-overlay">
      <div className="onboarding-card">
        {setupMode === "choose" && (
          <>
            <div className="onboarding-header">
              <div className="brand-mark large">
                <Clapperboard size={36} />
              </div>
              <h2>Welcome to AutoShorts</h2>
              <p>Long recording in. Short clips out. Select how you would like to run the studio.</p>
            </div>

            <div className="onboarding-choices">
              <div className="choice-card clickable" onClick={() => setSetupMode("local")}>
                <div className="choice-icon">
                  <Database size={28} />
                </div>
                <h3>Fully Offline & Private</h3>
                <p>Process everything locally on your computer. Private, secure, and completely free.</p>
                <div className="choice-badge local">Offline (Ollama)</div>
              </div>

              <div className="choice-card clickable" onClick={() => setSetupMode("cloud")}>
                <div className="choice-icon">
                  <Cloud size={28} />
                </div>
                <h3>Cloud APIs</h3>
                <p>Use high-speed cloud services for transcription and analysis. No local GPU needed.</p>
                <div className="choice-badge cloud">API Keys Required</div>
              </div>
            </div>
          </>
        )}

        {setupMode === "local" && (
          <div className="local-setup-flow">
            <div className="onboarding-header compact">
              <h2>Configure Offline Mode</h2>
              <p>Follow these steps to set up your local studio.</p>
            </div>

            {error && <div className="error-banner" style={{ marginBottom: "16px" }}>{error}</div>}

            <div className="setup-steps">
              <div className="setup-step">
                <div className="step-num">1</div>
                <div className="step-body">
                  <h4>Install Python Whisper</h4>
                  <p>Open your terminal and run the following command to install the transcription engine:</p>
                  <div className="code-block-container">
                    <code>pip3 install -U openai-whisper</code>
                    <button type="button" className="copy-btn" onClick={copyWhisperCommand}>
                      {copied ? <Check size={14} /> : <Copy size={14} />}
                      {copied ? "Copied!" : "Copy"}
                    </button>
                  </div>
                  {environment?.hasLocalWhisperModel ? (
                    <span className="step-check success"><BadgeCheck size={14} /> Whisper installed in Python!</span>
                  ) : (
                    <span className="step-check warning">⚠️ Python package 'whisper' not detected yet. Run the command above.</span>
                  )}
                </div>
              </div>

              <div className="setup-step">
                <div className="step-num">2</div>
                <div className="step-body">
                  <h4>Set up local LLM (Ollama)</h4>
                  <p>
                    Ollama must be installed and running on your machine.
                    If you don't have it installed, you can download it from <a href="https://ollama.com" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-primary)', textDecoration: 'underline' }}>ollama.com</a>.
                  </p>
                  <p>Select a model to download:</p>

                  <div className="model-cards">
                    <div
                      className={`model-card ${selectedModel === "llama3.2" ? "active" : ""}`}
                      onClick={() => setSelectedModel("llama3.2")}
                    >
                      <div className="model-card-header">
                        <h5>LLaMA 3.2 3B</h5>
                        <span className="model-size">1.9 GB</span>
                      </div>
                      <p>Requires 8GB+ RAM. Recommended for standard setups. Fast and efficient.</p>
                    </div>

                    <div
                      className={`model-card ${selectedModel === "qwen2.5:3b" ? "active" : ""}`}
                      onClick={() => setSelectedModel("qwen2.5:3b")}
                    >
                      <div className="model-card-header">
                        <h5>Qwen 2.5 3B</h5>
                        <span className="model-size">2.0 GB</span>
                      </div>
                      <p>Requires 8GB+ RAM. Excellent coding and logical reasoning abilities.</p>
                    </div>

                    <div
                      className={`model-card ${selectedModel === "qwen2.5:7b" ? "active" : ""}`}
                      onClick={() => setSelectedModel("qwen2.5:7b")}
                    >
                      <div className="model-card-header">
                        <h5>Qwen 2.5 7B</h5>
                        <span className="model-size">4.7 GB</span>
                      </div>
                      <p>Requires 16GB+ RAM. High-quality moment detection and hook precision.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="onboarding-actions">
              <button type="button" className="icon-button" onClick={() => setSetupMode("choose")}>Back</button>
              <button
                type="button"
                className="primary-action compact"
                onClick={startLocalSetup}
                disabled={checkingOllama}
              >
                {checkingOllama ? <Loader2 className="spin" size={18} /> : null}
                {checkingOllama ? "Checking Ollama..." : "Download & Start Setup"}
              </button>
            </div>
          </div>
        )}

        {setupMode === "cloud" && (
          <form className="cloud-setup-flow" onSubmit={handleCloudSubmit}>
            <div className="onboarding-header compact">
              <h2>Configure Cloud APIs</h2>
              <p>Add your keys below. AutoShorts will route transcription and analysis to the cloud.</p>
            </div>

            {error && <div className="error-banner" style={{ marginBottom: "16px" }}>{error}</div>}

            <div className="form-stack">
              <div className="input-group">
                <label>Deepgram API Key *</label>
                <input
                  type="password"
                  value={dgKey}
                  onChange={(e) => setDgKey(e.target.value)}
                  placeholder="Insert your Deepgram API Key (for transcription)"
                />
              </div>

              <div className="input-group">
                <label>Claude API Key</label>
                <input
                  type="password"
                  value={antKey}
                  onChange={(e) => setAntKey(e.target.value)}
                  placeholder="Insert your Anthropic API Key (moment detection)"
                />
              </div>

              <div className="input-group">
                <label>DeepSeek API Key</label>
                <input
                  type="password"
                  value={dsKey}
                  onChange={(e) => setDsKey(e.target.value)}
                  placeholder="Insert your DeepSeek API Key (alternative moment detection)"
                />
              </div>

              <div className="input-group">
                <label>Groq API Key</label>
                <input
                  type="password"
                  value={grKey}
                  onChange={(e) => setGrKey(e.target.value)}
                  placeholder="Insert your Groq API Key (alternative moment detection)"
                />
              </div>
              <p className="form-help">* Deepgram Key + at least one LLM Key (Claude, DeepSeek, or Groq) is required.</p>
            </div>

            <div className="onboarding-actions">
              <button type="button" className="icon-button" onClick={() => setSetupMode("choose")}>Back</button>
              <button type="submit" className="primary-action compact">Save & Start</button>
            </div>
          </form>
        )}

        {setupMode === "downloading" && (
          <div className="downloading-flow">
            <div className="onboarding-header compact">
              <h2>Downloading Local Model</h2>
              <p>Please wait while your local environment is downloaded. Do not close the application.</p>
            </div>

            <div className="download-progress-container">
              <div className="download-loader">
                <Loader2 className="spin" size={48} />
              </div>

              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: `${downloadProgress}%` }}></div>
              </div>

              <div className="download-stats">
                <span className="download-status">{downloadStatus}</span>
                <span className="download-percentage">{downloadProgress}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
