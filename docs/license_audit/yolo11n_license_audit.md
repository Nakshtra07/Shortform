# Software & Model License Audit: `yolo11n.pt` and Associated Weights

**Date:** 2026-09-28  
**Audit Target:** `autoshorts/src-tauri/models/yolo11n.pt`, `yolo11n-seg.pt`, and `face_detection_yunet_2023mar.onnx`  
**Workspace:** `D:\College\Autoshorts 11.0`  
**Auditor:** AutoShorts Engineering Team  
**Disclaimer:** This document is an engineering technical audit documenting upstream software licenses, technical dependencies, and architecture options. It does not constitute formal legal advice.

---

## 1. Inventory of Shipped Weights & Local Notices

Direct inspection of `autoshorts/src-tauri/models/`:

| Artifact | File Size | Upstream Origin | Local Notice / License File Present? |
|---|:---:|---|:---:|
| `yolo11n.pt` | 5,613,764 bytes (~5.35 MB) | Ultralytics YOLO11 (nano detection) | **NO** (`models/` contains no LICENSE or NOTICE file) |
| `yolo11n-seg.pt` | 6,182,636 bytes (~5.90 MB) | Ultralytics YOLO11 (nano segmentation) | **NO** (`models/` contains no LICENSE or NOTICE file) |
| `face_detection_yunet_2023mar.onnx` | 232,589 bytes (~0.22 MB) | OpenCV Zoo (YuNet Face Detector) | **NO** (`models/` contains no LICENSE or NOTICE file) |

---

## 2. Authoritative Upstream Licensing

### 2.1 Ultralytics YOLO11 (`yolo11n.pt` & `yolo11n-seg.pt`)
- **Author / Upstream Repository:** Ultralytics LLC (`https://github.com/ultralytics/ultralytics`)
- **Upstream License:** **GNU Affero General Public License v3.0 (AGPL-3.0)**
- **Commercial Policy:** Ultralytics operates a dual-licensing business model:
  1. **AGPL-3.0 (Open-Source Path):** Free to use, subject to strict reciprocal copyleft obligations:
     - Any modified version or work combining the library/models must also be released under AGPL-3.0.
     - Complete source code must be made accessible to all distributed recipients.
     - **Network Interaction Clause (Section 13):** If the application provides functionality over a network (e.g., web interface, cloud backend, API), the complete corresponding source code must be made available to all remote network users.
  2. **Ultralytics Enterprise License (Commercial/Closed-Source Path):** Paid proprietary license required if organizations want to distribute closed-source binaries, offer commercial SaaS without disclosing source code, or bypass AGPL copyleft obligations.

### 2.2 OpenCV YuNet (`face_detection_yunet_2023mar.onnx`)
- **Author / Upstream Repository:** OpenCV Model Zoo (`https://github.com/opencv/opencv_zoo`)
- **Upstream License:** **Apache License 2.0**
- **Commercial Policy:** Permissive open-source license. Permits commercial use, modification, distribution, and closed-source proprietary distribution, requiring only copyright notices and disclaimers.

---

## 3. Engineering Implications for AutoShorts

### 3.1 Local Development Environment (Status Quo)
- In a private, local development workspace, invoking `yolo11n.pt` through `ultralytics` in Python does not trigger distribution or network obligations.
- Local tests and internal benchmarks run lawfully under fair open development terms.

### 3.2 Desktop Binary Distribution (Tauri Installer)
- If AutoShorts desktop installers (e.g. MSI, NSIS, dmg) bundle `yolo11n.pt` or `yolo11n-seg.pt` directly into the distributed application package:
  - This constitutes binary distribution under copyright law.
  - If AutoShorts is intended to be a proprietary, commercial application, bundling AGPL-3.0 weights creates copyleft exposure. Ultralytics asserts that incorporating their weights or libraries requires either full AGPL-3.0 compliance (open-sourcing the entire AutoShorts application) or an Enterprise License.

### 3.3 Cloud / SaaS Deployment
- If AutoShorts' video processing engine is hosted in the cloud where remote users upload videos and receive shorts:
  - AGPL-3.0 Section 13 mandates providing all remote users with complete source code of the backend processing system.

---

## 4. Technical Resolution Options

As instructed by project directives, **no new model bundles will be added at this stage**. The following technical options are available for project leadership:

### Option A: Open-Source AutoShorts Under AGPL-3.0
- Publish AutoShorts as an open-source tool under AGPL-3.0.
- Fully satisfies Ultralytics copyleft requirements with zero licensing fees.

### Option B: Procure Ultralytics Enterprise License
- Obtain a commercial enterprise agreement from Ultralytics LLC for commercial closed-source redistribution.
- Allows proprietary binaries and SaaS hosting without source disclosure.

### Option C: User-Supplied Weights (On-Demand Model Download)
- Remove `yolo11n.pt` and `yolo11n-seg.pt` from the default application installer.
- Adopt the standard pattern used by Ollama, Whisper, and Stable Diffusion tools:
  - The application binary contains no AGPL code or weights.
  - On first run, the user is prompted to download model weights directly from upstream repositories into their local `%LOCALAPPDATA%` or application cache.
  - The software functions as a generic runtime; the user exercises their individual right to download and use the weights.

### Option D: Transition to Permissive Models (Apache-2.0 / MIT)
- Replace YOLO11 with models trained under permissive licenses:
  - For face and active speaker tracking: OpenCV YuNet (`face_detection_yunet_2023mar.onnx`, **Apache-2.0**) is already shipped and integrated.
  - For body/person bounding boxes: YOLO-NAS (Deci AI, **Apache-2.0**) or RT-DETR (Baidu/Paddle, **Apache-2.0**).

---

## 5. Open Questions for Stakeholders

1. **Commercial Intent:** Is AutoShorts intended to be released as a proprietary commercial SaaS/desktop product, or as an open-source creator tool?
2. **Distribution Topology:** Will the production desktop build be distributed as a self-contained bundle, or will heavy ML weights be downloaded on-demand by the client during initial setup?
3. **Legal Evaluation of Subprocess Boundary:** Does executing `speaker_tracker.py` as an external subprocess via `std::process::Command` insulate the host Tauri Rust application under AGPL interpretation? (Note: Ultralytics publicly maintains that any tool relying on YOLO is covered, but legal precedents vary by jurisdiction).
