import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import {
  AudioLines,
  BadgeCheck,
  Captions,
  Check,
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
  createdAt: string;
  updatedAt: string;
};

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

function App() {
  const [environment, setEnvironment] = useState<EnvironmentStatus | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [busy, setBusy] = useState<BusyState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [renderingCandidateId, setRenderingCandidateId] = useState<string | null>(null);
  const [showStyleModal, setShowStyleModal] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null);
  const [selectedFramingMode, setSelectedFramingMode] = useState<string>("original");
  // Candidate discovery engine for "Find Viral Moments": "timestamp_generation"
  // is the production default; "window_scoring" (REZE) is the experimental
  // opt-in. Per generation call — flip and re-run to compare on the same project.
  const [discoveryMode, setDiscoveryMode] = useState<string>("timestamp_generation");
  const [rezeProvider, setRezeProvider] = useState<string>("");
  const [mediaPathToImport, setMediaPathToImport] = useState<string | null>(null);

  const [youtubeModalOpen, setYoutubeModalOpen] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [youtubeStatus, setYoutubeStatus] = useState<"idle" | "checking" | "warning" | "downloading">("idle");
  const [youtubeWarningLicense, setYoutubeWarningLicense] = useState<string | null>(null);
  const [youtubeError, setYoutubeError] = useState<string | null>(null);

  // Synchronous in-flight guards against rapid duplicate clicks / concurrent triggers
  const isCuttingRef = useRef(false);
  const activeCuttingCandidatesRef = useRef(new Set<string>());

  // Persistence logic from localStorage
  const [isOnboarded, setIsOnboarded] = useState<boolean | null>(null);
  const [transcriptionEngine, setTranscriptionEngine] = useState<"deepgram" | "local">(() => {
    return (localStorage.getItem("autoshorts_transcription_engine") as "deepgram" | "local") || "local";
  });
  const [llmEngine, setLlmEngine] = useState<"claude" | "deepseek" | "local" | "gemini" | "openai" | "openrouter" | "groq">(() => {
    return (localStorage.getItem("autoshorts_llm_engine") as "claude" | "deepseek" | "local" | "gemini" | "openai" | "openrouter" | "groq") || "local";
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
  const canUseCloudKey = environment?.hasDeepgramKey || deepgramKey.trim().length > 0;
  const canUseClaude = environment?.hasAnthropicKey || anthropicKey.trim().length > 0;
  const canUseDeepseek = environment?.hasDeepseekKey || deepseekKey.trim().length > 0;
  const canUseGemini = environment?.hasGeminiKey || geminiKey.trim().length > 0;
  const canUseOpenai = environment?.hasOpenaiKey || openaiKey.trim().length > 0;
  const canUseOpenrouter = environment?.hasOpenrouterKey || openrouterKey.trim().length > 0;
  const canUseGroq = environment?.hasGroqKey || groqKey.trim().length > 0;

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
    if (selectedRanks.length === 0) return;
    const from = Math.min(...selectedRanks);
    const to = Math.max(...selectedRanks);
    const lastApplied = lastAppliedRangeRef.current;
    const isEcho = lastApplied !== null && lastApplied.from === from &&
      (lastApplied.to === null ? to === candidates.length : lastApplied.to === to);
    if (isEcho) return;
    setRangeFromValue(String(from));
    setRangeToValue(to === candidates.length ? null : String(to));
  }, [detail?.candidates]);

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
      const validEngines = ["claude", "deepseek", "local", "gemini", "openai", "openrouter", "groq"];
      if (validEngines.includes(p)) {
        const stored = localStorage.getItem("autoshorts_llm_engine");
        if (!stored || stored === "local" || stored === "deepseek" || stored === "openrouter") {
          setLlmEngine(p as any);
        }
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
    setSelectedStyle(null);
    setMediaPathToImport(selected);
    setShowStyleModal(true);
  }

  async function handleYoutubeImport() {
    if (!youtubeUrl) return;
    setYoutubeStatus("checking");
    setYoutubeError(null);
    setError(null);
    try {
      console.log("[YouTube Frontend] Checking copyright for:", youtubeUrl, "browser:", youtubeBrowser);
      const result = await invoke<{isSafe: boolean; license: string | null}>("check_youtube_copyright", {
        url: youtubeUrl,
        browser: youtubeBrowser !== "auto" ? youtubeBrowser : null,
        cookiesPath: youtubeCookiesPath.trim() || null,
      });
      console.log("[YouTube Frontend] Copyright check result:", result);
      if (!result.isSafe) {
        setYoutubeWarningLicense(result.license || "Unknown / Not specified");
        setYoutubeStatus("warning");
        return;
      }
      await executeYoutubeDownload();
    } catch (err: any) {
      const errMsg = err?.toString() || "Failed to check YouTube video copyright.";
      console.error("[YouTube Frontend] Copyright check error:", errMsg);
      setYoutubeError(errMsg);
      setError(errMsg);
      setYoutubeStatus("idle");
    }
  }

  async function executeYoutubeDownload() {
    setYoutubeStatus("downloading");
    setYoutubeError(null);
    setError(null);
    try {
      console.log("[YouTube Frontend] Starting download for:", youtubeUrl, "browser:", youtubeBrowser);
      const downloadedPath = await invoke<string>("download_youtube_video", {
        url: youtubeUrl,
        browser: youtubeBrowser !== "auto" ? youtubeBrowser : null,
        cookiesPath: youtubeCookiesPath.trim() || null,
      });
      console.log("[YouTube Frontend] Download completed successfully. Path:", downloadedPath);
      setYoutubeModalOpen(false);
      setYoutubeUrl("");
      setYoutubeStatus("idle");
      setYoutubeWarningLicense(null);
      setYoutubeError(null);
      setSelectedStyle(null);
      setMediaPathToImport(downloadedPath);
      setShowStyleModal(true);
    } catch (err: any) {
      const errMsg = err?.toString() || "Failed to download YouTube video.";
      console.error("[YouTube Frontend] Download error:", errMsg);
      setYoutubeError(errMsg);
      setError(errMsg);
      setYoutubeStatus("idle");
    }
  }

  async function confirmImport(style: string | null, framingMode: string) {
    if (!mediaPathToImport || !style) return;
    const selected = mediaPathToImport;
    const chosenStyle = style;
    const chosenFramingMode = framingMode;
    setMediaPathToImport(null);
    setSelectedStyle(null);
    setSelectedFramingMode("original");
    setShowStyleModal(false);

    let newProjectId: string | null = null;
    await run("import", async () => {
      const project = await invoke<Project>("create_project_from_path", {
        path: selected,
        transcriptionMode: transcriptionEngine === "local" ? "local" : "cloud",
        captionStyle: chosenStyle,
        framingMode: chosenFramingMode,
      });
      newProjectId = project.id;
      await refresh(project.id);
    });

    if (newProjectId) {
      await runAutoPipeline(newProjectId);
    }
  }

  async function runAutoPipeline(projectId: string) {
    setError(null);
    const env = await invoke<EnvironmentStatus>("environment_status");

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
                targetProvider === "groq" ? groqKey : "";
      const hasActiveKey =
        targetProvider === "claude" ? (env.hasAnthropicKey || activeKey.trim().length > 0) :
          targetProvider === "deepseek" ? (env.hasDeepseekKey || activeKey.trim().length > 0) :
            targetProvider === "gemini" ? (env.hasGeminiKey || activeKey.trim().length > 0) :
              targetProvider === "openai" ? (env.hasOpenaiKey || activeKey.trim().length > 0) :
                targetProvider === "groq" ? (env.hasGroqKey || activeKey.trim().length > 0) : false;
      if (!hasActiveKey) {
        const engineName =
          targetProvider === "claude" ? "Claude" :
            targetProvider === "deepseek" ? "DeepSeek" :
              targetProvider === "gemini" ? "Gemini" :
                targetProvider === "openai" ? "OpenAI" :
                  targetProvider === "groq" ? "Groq" : "LLM";
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
                  targetProvider === "groq" ? groqKey.trim() : "");
      const targetModel = isReze
        ? null
        : (targetProvider === "local" ? localLlmModel.trim() :
            targetProvider === "deepseek" ? (deepseekModel.trim() || "deepseek-chat") : null);

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
                    : "");

      const targetModel = isReze
        ? null
        : (targetProvider === "local"
            ? localLlmModel.trim()
            : targetProvider === "deepseek"
              ? (deepseekModel.trim() || "deepseek-chat")
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
  async function updateSelectionRange(from: number, to: number | null) {
    if (!detail) return;
    const total = detail.candidates.length;
    const endRank = to === null ? total : to;
    if (!Number.isInteger(from) || !Number.isInteger(endRank)) return;
    if (from < 1 || endRank < from || endRank > total) return;
    lastAppliedRangeRef.current = { from, to };
    await run("clipCount", async () => {
      const candidates = await invoke<Candidate[]>("set_selected_rank_range", {
        projectId: detail.project.id,
        startRank: from,
        endRank: endRank,
      });
      setDetail({ ...detail, candidates });
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
              <Clapperboard size={20} />
            </div>
            <div>
              <h1>AutoShorts</h1>
              <p>Long recording in. Short clips out.</p>
            </div>
          </div>

          <button className="primary-action" onClick={importMedia} disabled={busy !== "idle"}>
            {busy === "import" ? <Loader2 className="spin" size={18} /> : <FileVideo size={18} />}
            Import recording
          </button>
          <button 
            className="secondary-action" 
            onClick={() => setYoutubeModalOpen(true)} 
            disabled={busy !== "idle" || !environment?.hasYtdlp}
            title={!environment?.hasYtdlp ? "Please install yt-dlp to use this feature" : "Download a video from YouTube"}
            style={{ width: "100%", padding: "0.75rem", borderRadius: "10px", marginTop: "0.5rem", display: "flex", gap: "0.5rem", alignItems: "center", justifyContent: "center", border: "1px solid var(--border)", background: "transparent", color: "var(--foreground)", cursor: "pointer", fontSize: "0.95rem" }}
          >
            <Youtube size={18} />
            Import from YouTube
          </button>

          <section className="project-list" aria-label="Projects">
            <button
              className={`project-row ${!detail ? "active" : ""}`}
              onClick={() => setDetail(null)}
            >
              <Clapperboard size={15} />
              <span>All Projects</span>
              <ChevronRight size={14} />
            </button>

            {projects.map((project) => (
              <button
                key={project.id}
                className={`project-row ${detail?.project.id === project.id ? "active" : ""}`}
                onClick={() => void selectProject(project.id)}
              >
                <FileVideo size={15} />
                <span>{project.name || fileName(project.sourcePath)}</span>
                <ChevronRight size={14} />
              </button>
            ))}
          </section>
        </aside>

        <section className="workspace">
          {detail ? (
            <>
              <header className="topbar">
                <div className="project-info">
                  <div className="eyebrow">{detail.project.status}</div>
                  <h2>{detail.project.name || fileName(detail.project.sourcePath)}</h2>
                </div>
                <div className="topbar-actions">
                  <button
                    className={`icon-button settings-toggle ${showSettings ? "active" : ""}`}
                    onClick={() => setShowSettings(!showSettings)}
                    title="API Settings"
                  >
                    <SlidersHorizontal size={16} />
                    <span>API Settings</span>
                  </button>
                  <button className="icon-button" onClick={() => void refresh(detail.project.id)} title="Refresh">
                    <RefreshCw size={18} />
                  </button>
                </div>
              </header>

              {showSettings && (
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
              )}

              {error && <div className="error-banner">{error}</div>}

              <div className="pipeline-strip">
                <PipelineStep icon={<AudioLines size={16} />} label="Transcript" done={Boolean(detail.transcript)} />
                <PipelineStep icon={<Sparkles size={16} />} label="Moments" done={detail.candidates.length > 0} />
                <PipelineStep icon={<Scissors size={16} />} label="Cut" done={selectedCount > 0 && selectedCutCount === selectedCount} />
                <PipelineStep icon={<Captions size={16} />} label="Captions" done={selectedCount > 0 && selectedCaptionsCount === selectedCount} />
                <PipelineStep icon={<Download size={16} />} label="Export" done={selectedCount > 0 && selectedCutCount === selectedCount} />
              </div>

              <div className="work-grid">
                <section className="panel transcript-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Transcript</h3>
                      <p>{transcript ? `${transcript.segments.length} segments` : "No transcript"}</p>
                    </div>
                    <div className="button-pair">
                      <button onClick={transcribe} disabled={busy !== "idle" || !canTranscribe}>
                        {busy === "transcribe" ? <Loader2 className="spin" size={16} /> : <AudioLines size={16} />}
                        Transcribe
                      </button>
                    </div>
                  </div>

                  {!canTranscribe && (
                    <div className="api-warning">
                      {transcriptionEngine === "local"
                        ? `⚠️ Local Whisper (Python package 'openai-whisper') is not installed. Run 'pip3 install openai-whisper' in your terminal.`
                        : "⚠️ Deepgram API Key is missing. Transcribing will not work. Please add your key in API Settings."}
                    </div>
                  )}

                  <div className="transcript-list">
                    {transcript?.segments.map((segment, index) => (
                      <article key={`${segment.start}-${index}`} className="segment-row">
                        <span>{formatTime(segment.start)}</span>
                        <p>{segment.text}</p>
                      </article>
                    )) ?? <EmptyState icon={<AudioLines size={28} />} label="Transcript pending" />}
                  </div>
                </section>

                <section className="panel candidate-panel">
                  <div className="panel-heading">
                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <h3>Clip Candidates</h3>
                        <span style={{ fontSize: "0.75rem", padding: "2px 8px", borderRadius: "12px", background: "rgba(142, 230, 199, 0.15)", color: "var(--accent-primary)", border: "1px solid rgba(142, 230, 199, 0.3)", fontWeight: 600 }}>
                          v9.3 Multimodal (Visual ✅ | Audio ✅ | Temporal ✅)
                        </span>
                      </div>
                      <p>
                        {totalCandidateCount > 0
                          ? `ALL CANDIDATES: ${totalCandidateCount} | SELECTED: ${selectedCount} | POOL: ${poolCount}`
                          : "No candidates"}
                      </p>
                    </div>
                    <div className="button-pair">
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
                        title="Candidate discovery engine: Timestamp Generation is the production default; REZE Window Scoring is the experimental alternative. Re-run Find Viral Moments to compare."
                        style={{
                          fontSize: "0.8rem",
                          padding: "6px 8px",
                          borderRadius: "8px",
                          border: "1px solid var(--border-color, #333)",
                          background: "var(--bg-secondary, #1a1a1a)",
                          color: "inherit",
                        }}
                      >
                        <option value="timestamp_generation">Discovery: Timestamp Generation</option>
                        <option value="window_scoring">Discovery: REZE Window Scoring</option>
                      </select>
                      <button onClick={cutSelected} disabled={busy !== "idle" || selectedCount === 0 || !environment?.hasFfmpeg}>
                        {busy === "cut" ? <Loader2 className="spin" size={16} /> : <Scissors size={16} />}
                        Cut Selected ({selectedCount})
                      </button>
                      <button onClick={() => void moments(false)} disabled={busy !== "idle" || !detail.transcript || !canUseDiscoveryLlm}>
                        {busy === "moments" ? <Loader2 className="spin" size={16} /> : <Sparkles size={16} />}
                        Find Viral Moments
                      </button>
                    </div>
                  </div>

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

                  {totalCandidateCount > 0 && (
                    <div className="candidate-toolbar">
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

                      <div className="range-control">
                        <label className="range-field">
                          <span>From</span>
                          <input
                            type="number"
                            min="1"
                            max={totalCandidateCount}
                            value={rangeFromValue}
                            onChange={(event) => {
                              setRangeFromValue(event.target.value);
                              void updateSelectionRange(Number.parseInt(event.target.value, 10), rangeToValue === null ? null : Number.parseInt(rangeToValue, 10));
                            }}
                            onBlur={(event) => {
                              // Clamp out-of-range / empty input on blur
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
                            max={totalCandidateCount}
                            value={rangeToValue ?? ""}
                            placeholder="Last"
                            onChange={(event) => {
                              const raw = event.target.value;
                              setRangeToValue(raw === "" ? null : raw);
                              void updateSelectionRange(Number.parseInt(rangeFromValue, 10), raw === "" ? null : Number.parseInt(raw, 10));
                            }}
                            onBlur={(event) => {
                              // Clamp out-of-range input on blur; empty stays "Last"
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
                        <span
                          className={`range-count ${rangeInputsValid ? "" : "invalid"}`}
                          title="Derived selection count: end - start + 1"
                        >
                          {rangeInputsValid ? rangeSelectedCount : "—"}
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="candidate-list">
                    {displayedCandidates.map((candidate) => {
                      const clip = clipByCandidate.get(candidate.id);
                      const isCut = clip?.status === "done" && Boolean(clip.outputPath);
                      return (
                        <article key={candidate.id} className={`candidate-card ${candidate.selected ? "selected" : ""}`}>
                          {/* 9:16 portrait mockup preview placeholder representing vertical formats */}
                          <div className="portrait-preview-container">
                            <div className="portrait-preview-mock">
                              {isCut ? (
                                <div className="mock-video-active">
                                  <Play size={20} className="play-icon-mock" />
                                </div>
                              ) : (
                                <div className="mock-video-inactive">
                                  <span>9:16</span>
                                </div>
                              )}
                            </div>
                            <div className="candidate-rank">
                              <span>#{candidate.rank}</span>
                              {candidate.selected && <Check size={14} />}
                            </div>
                          </div>

                          <div className="candidate-body">
                            <div className="candidate-meta">
                              <span>{formatTime(candidate.startSec)} - {formatTime(candidate.endSec)}</span>
                              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                {candidate.selected ? (
                                  <span style={{ fontSize: "0.68rem", padding: "1px 6px", borderRadius: "8px", background: "rgba(142, 230, 199, 0.2)", color: "var(--accent-primary)", border: "1px solid rgba(142, 230, 199, 0.4)", fontWeight: 700 }}>
                                    ★ Selected
                                  </span>
                                ) : (
                                  <span style={{ fontSize: "0.68rem", padding: "1px 6px", borderRadius: "8px", background: "rgba(255, 255, 255, 0.05)", color: "var(--text-muted)", border: "1px solid rgba(255, 255, 255, 0.1)" }}>
                                    Pool
                                  </span>
                                )}
                                {candidate.rationale.includes("[v9.2 Fallback]") ? (
                                  <span style={{ fontSize: "0.68rem", padding: "1px 6px", borderRadius: "8px", background: "rgba(255, 179, 71, 0.15)", color: "#ffb347", border: "1px solid rgba(255, 179, 71, 0.3)" }}>
                                    v9.2 Fallback
                                  </span>
                                ) : (
                                  <span style={{ fontSize: "0.68rem", padding: "1px 6px", borderRadius: "8px", background: "rgba(142, 230, 199, 0.15)", color: "var(--accent-primary)", border: "1px solid rgba(142, 230, 199, 0.3)" }}>
                                    ✨ Multimodal
                                  </span>
                                )}
                                <span className="candidate-score">{Math.round(candidate.score * 100)}% Match</span>
                              </div>
                            </div>
                            <h4>{candidate.hook}</h4>
                            <p className="candidate-rationale">{candidate.rationale}</p>

                            <div className="candidate-actions">
                              <span className={`clip-status ${isCut ? "ready" : clip?.status === "error" ? "error" : ""}`}>
                                {isCut ? "Cut ready" : clip?.status === "error" ? "Cut failed" : clip?.status ?? "Pending"}
                              </span>
                              <button
                                className="cut-button"
                                onClick={() => void cutCandidate(candidate.id)}
                                disabled={busy !== "idle" || !environment?.hasFfmpeg}
                              >
                                {renderingCandidateId === candidate.id ? (
                                  <Loader2 className="spin" size={14} />
                                ) : (
                                  <Scissors size={14} />
                                )}
                                {renderingCandidateId === candidate.id ? "Cutting..." : isCut ? "Re-cut" : "Cut"}
                              </button>
                            </div>
                            {clip?.outputPath && <div className="output-path">{clip.outputPath}</div>}
                            {clip?.captionAssPath && (
                              <div className="output-path" style={{ background: "rgba(142, 230, 199, 0.05)", borderColor: "var(--accent-primary)", color: "var(--accent-primary)", marginTop: "4px" }}>
                                Subtitles: {clip.captionAssPath}
                              </div>
                            )}
                            {clip?.appliedFeatures && (() => {
                              try {
                                const f = JSON.parse(clip.appliedFeatures) as {
                                  smartPacing?: boolean;
                                  hookEndingOptimization?: boolean;
                                  audioIntelligence?: boolean;
                                  captionIntelligence?: boolean;
                                };
                                const badges: Array<{ label: string; color: string }> = [
                                  f.hookEndingOptimization ? { label: "✂ Hook/End", color: "#7eb8f7" } : null,
                                  f.smartPacing            ? { label: "⚡ Pacing",  color: "#f7c77e" } : null,
                                  f.audioIntelligence      ? { label: "🎚 Audio",   color: "#8ee6c7" } : null,
                                  f.captionIntelligence    ? { label: "💬 Caption Intelligence", color: "#c084fc" } : null,
                                ].filter((b): b is { label: string; color: string } => b !== null);
                                if (badges.length === 0) return null;
                                return (
                                  <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", marginTop: "6px" }}>
                                    {badges.map((b) => (
                                      <span
                                        key={b.label}
                                        title={`Feature applied to this clip: ${b.label}`}
                                        style={{
                                          fontSize: "0.68rem",
                                          padding: "1px 7px",
                                          borderRadius: "8px",
                                          background: `${b.color}22`,
                                          color: b.color,
                                          border: `1px solid ${b.color}55`,
                                          fontWeight: 600,
                                        }}
                                      >
                                        {b.label}
                                      </span>
                                    ))}
                                  </div>
                                );
                              } catch {
                                return null;
                              }
                            })()}
                            {clip?.renderLog && <div className="render-log">{clip.renderLog}</div>}
                          </div>
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
              <header className="home-header">
                <div>
                  <h2>All Projects</h2>
                  <p>Select a project below or import a new media file to get started.</p>
                </div>
                <button className="primary-action compact" onClick={importMedia} disabled={busy !== "idle"}>
                  {busy === "import" ? <Loader2 className="spin" size={18} /> : <FileVideo size={18} />}
                  Import recording
                </button>
                <button 
                  className="secondary-action compact" 
                  onClick={() => setYoutubeModalOpen(true)} 
                  disabled={busy !== "idle" || !environment?.hasYtdlp}
                  title={!environment?.hasYtdlp ? "Please install yt-dlp to use this feature" : "Download a video from YouTube"}
                  style={{ display: "flex", gap: "0.5rem", alignItems: "center", padding: "0.5rem 1rem", borderRadius: "8px", border: "1px solid var(--border)", background: "transparent", color: "var(--foreground)", cursor: "pointer", fontSize: "0.95rem", marginLeft: "1rem" }}
                >
                  <Youtube size={18} />
                  Import from YouTube
                </button>
              </header>

              {projects.length > 0 ? (
                <div className="projects-grid">
                  {projects.map((project) => {
                    const name = project.name || fileName(project.sourcePath);
                    return (
                      <article key={project.id} className="project-card">
                        <div className="project-card-header">
                          <FileVideo size={24} className="project-card-icon" />
                          <span className="project-card-status">{project.status}</span>
                        </div>
                        <h3 className="project-card-title">{name}</h3>
                        <div className="project-card-meta">
                          <span>Duration: {project.sourceDuration ? formatTime(project.sourceDuration) : "Probing..."}</span>
                          <span>Created: {new Date(project.createdAt).toLocaleDateString()}</span>
                        </div>
                        <div className="project-card-actions">
                          <button className="action-btn open-btn" onClick={() => void selectProject(project.id)}>
                            Open
                          </button>
                          <button className="action-btn rename-btn" onClick={() => void renameProject(project.id)}>
                            Rename
                          </button>
                          <button className="action-btn delete-btn" onClick={() => void deleteProject(project.id)}>
                            Delete
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="empty-dashboard-state">
                  <Clapperboard size={48} className="empty-state-icon" />
                  <h3>No projects found</h3>
                  <p>Import your first recording to begin creating shorts.</p>
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

      {showStyleModal && (
        <div className="style-modal-overlay">
          <div className="style-modal">
            <div className="style-modal-header">
              <h3>Choose Caption Style</h3>
              <p>Select how your automated captions should look on the portrait short-form video clips.</p>
            </div>

            <div className="framing-section">
              <div className="framing-section-title">Framing Mode</div>
              <div className="framing-grid">
                <div
                  className={`framing-card ${selectedFramingMode === "original" ? "selected" : ""}`}
                  onClick={() => setSelectedFramingMode("original")}
                >
                  <div className="framing-card-title">Original 9:16</div>
                  <div className="framing-card-desc">The unchanged pipeline — center-cropped 9:16 with DualFrame split-screen when two speakers appear close together.</div>
                </div>
                <div
                  className={`framing-card ${selectedFramingMode === "adaptive" ? "selected" : ""}`}
                  onClick={() => setSelectedFramingMode("adaptive")}
                >
                  <div className="framing-card-title">Adaptive Framing</div>
                  <div className="framing-card-desc">Composition derived from the real footage — never split-screen. Two people who fit comfortably stay in one natural shot; subjects far apart get an active-speaker focus. Final canvas stays 9:16.</div>
                </div>
              </div>
            </div>

            <div className="style-grid">
              <div
                className={`style-card ${selectedStyle === "preset_viral_bold" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("preset_viral_bold")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-viral-bold">
                    <span className="preview-word-highlight">HORMOZI</span> VIRAL
                  </span>
                </div>
                <div className="style-card-title">Hormozi Viral</div>
                <div className="style-card-desc">Bold uppercase font with high-impact neon green & yellow scale pop (Hormozi style).</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "preset_mrbeast_pop" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("preset_mrbeast_pop")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-narrative-pop">
                    <span className="preview-word-pop">Narrative</span> pop stories
                  </span>
                </div>
                <div className="style-card-title">Narrative Pop</div>
                <div className="style-card-desc">Dynamic two-line sentence phrasing with punchy yellow progressive karaoke bounce.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "preset_minimal_capsule" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("preset_minimal_capsule")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-minimal-capsule">
                    <span className="preview-word-active">Minimal</span> <span className="preview-word-inactive">capsule reveal</span>
                  </span>
                </div>
                <div className="style-card-title">Minimal Capsule</div>
                <div className="style-card-desc">Clean aesthetic text with smooth word-by-word opacity reveal inside a semi-transparent capsule box.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "preset_cinematic_vlog" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("preset_cinematic_vlog")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-cinematic-vlog">Cinematic vlog subtitles</span>
                </div>
                <div className="style-card-title">Cinematic Vlog</div>
                <div className="style-card-desc">Understated lower-third typography with soft drop shadow and smooth fade-in for documentary & vlog storytelling.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "preset_dynamic_editorial" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("preset_dynamic_editorial")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-dynamic-editorial">
                    <span className="preview-word-emphasis">DYNAMIC</span> <span className="preview-word-editorial">editorial</span>
                  </span>
                </div>
                <div className="style-card-title">Dynamic Editorial Kinetic</div>
                <div className="style-card-desc">Multi-object kinetic typography with multi-directional entrance motion, mixed typography lockups, and synchronized group exit fades.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "preset_bhaukal_caption" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("preset_bhaukal_caption")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-bhaukal">
                    <span className="preview-word-context">you will</span> <span className="preview-word-hero">Believe</span>
                  </span>
                </div>
                <div className="style-card-title">Bhaukal caption</div>
                <div className="style-card-desc">Editorial dual-typography with progressive phrase builds, white monochrome scale hierarchy, hero metrics, and a persistent hook strip.</div>
              </div>

              <div
                className={`style-card ${selectedStyle === "preset_t7" ? "selected" : ""}`}
                onClick={() => setSelectedStyle("preset_t7")}
              >
                <div className="style-preview-box">
                  <span className="preview-text-t7">
                    <span className="preview-word-t7-normal">believe in</span> <span className="preview-word-t7-emph">yourself</span>
                  </span>
                </div>
                <div className="style-card-title">T7 Reference Style</div>
                <div className="style-card-desc">Phrase-chunked rolling captions driven by speech rhythm: short 1-3 word chunks appear as stable units, previous chunk in plain white above the newly arrived solid-yellow chunk, Matt Bold, no stroke or box, fixed lower position.</div>
              </div>
            </div>

            <div className="style-modal-actions">
              <button className="btn-cancel" onClick={() => { setShowStyleModal(false); setMediaPathToImport(null); setSelectedStyle(null); }}>
                Cancel
              </button>
              <button
                className="btn-confirm"
                disabled={!selectedStyle}
                onClick={() => { if (selectedStyle) confirmImport(selectedStyle, selectedFramingMode); }}
              >
                Confirm & Import
              </button>
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
          <div className="style-modal" style={{ maxWidth: "500px" }}>
            <div className="style-modal-header">
              <h3>Import from YouTube</h3>
              <p>Paste a YouTube URL below to download and import it directly.</p>
            </div>
            <div style={{ padding: "1rem" }}>
              <input
                type="text"
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                disabled={youtubeStatus !== "idle"}
                style={{ width: "100%", padding: "0.75rem", borderRadius: "8px", border: "1px solid var(--border)", background: "var(--background)", color: "var(--foreground)", fontSize: "1rem", marginBottom: "1rem" }}
              />

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
                  onClick={() => { setYoutubeModalOpen(false); setYoutubeUrl(""); setYoutubeStatus("idle"); setYoutubeWarningLicense(null); setYoutubeError(null); }}
                  disabled={youtubeStatus === "checking" || youtubeStatus === "downloading"}
                >
                  Cancel
                </button>
                {youtubeStatus === "warning" ? (
                  <button className="btn-confirm" onClick={executeYoutubeDownload}>
                    Yes, I understand the risks
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
