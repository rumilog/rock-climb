# Hand Tracking SDK / Package Landscape (single RGB webcam) — 2026-04-14

## Goal (project-specific)
Identify **fast, accessible** hand-tracking options that can run from **one cheap RGB webcam** (no depth, no gloves) and plausibly support:

- **Real-time teleoperation** (low latency, stable temporal output)
- **Finger pose / joint-angle control** (not just gestures)
- **Occlusion robustness** improvement when fused with Quest 2 finger tracking
- **Low barrier to adoption** for labs (license, install friction, hardware cost)

This document focuses on *off-the-shelf* SDKs / OSS that could power the webcam side of a Quest+webcam fusion system.

---

## Executive takeaways

- **Best default baseline (open-source, easy, fast):** **MediaPipe Hands / Hand Landmarker** (21 landmarks per hand; widely used; strong “tracking vs detection” architecture for speed).
- **Best “research-grade” monocular 3D mesh (fast but not as simple / licensing caveats):** **WiLoR (CVPR 2025)** for MANO-based hand reconstruction. Great for “3D hand mesh” papers, but heavier + MANO licensing complicates “fully open” pipelines.
- **Best web-only option:** `@handtracking.io/yoha` exists, but it is **unmaintained** (still useful to mention as a web baseline, not ideal for a serious system).
- **Best commercial comparator (not low-cost target):** **Ultraleap** (excellent tracking quality / occlusion handling with IR sensors). Useful as an “upper bound” comparison in a paper, but it violates the “only a cheap webcam” constraint.

---

## 1) MediaPipe Hands / Hand Landmarker (Google)

### What it gives you
- Per hand: **21 landmarks** (2D + an estimated depth-like coordinate) and a “world landmarks” representation depending on API/solution.
- Runs in a **detector + tracker** style: palm detection runs intermittently; per-frame landmark tracking is lighter. This is directly relevant to latency.

### Why it fits your niche
- Most accessible path to “good enough” webcam tracking at interactive rates.
- Works well as the webcam “assist” signal even if it’s not perfectly accurate in 3D.
- Has built-in confidences (detection/tracking confidence knobs; per-landmark visibility-like signals depending on API) that are useful for **fusion gating**.

### Performance/latency notes (practical)
- Python users commonly report ~**25–35 ms**/frame for 2-hand tracking at moderate resolutions on CPU depending on settings; spikes can occur on re-detection events.
- The newer **Tasks** API (Hand Landmarker) is the actively supported path and is the one to prefer when you eventually implement.

### License
- MediaPipe is widely distributed under **Apache 2.0** (confirm per component when you lock exact package versions).

### Useful references
- MediaPipe Hands original paper: `https://arxiv.org/pdf/2006.10214`
- MediaPipe Hands overview (legacy solutions docs): `https://chuoling.github.io/mediapipe/solutions/hands.html`
- Latency discussion (Python): `https://github.com/google-ai-edge/mediapipe/issues/5789`

### Key caveat for teleop
MediaPipe’s “3D” is not a calibrated metric 3D pose from a single RGB camera; it is often sufficient for **relative finger configuration**, but not for absolute metric reconstruction. For your project, that’s okay if you fuse in **joint-angle space**.

---

## 2) WiLoR (CVPR 2025) — monocular 3D hand reconstruction (MANO)

### What it gives you
- Multi-hand detection + high-fidelity 3D reconstruction (MANO pose/shape + camera) from monocular RGB.
- Demonstrated smooth tracking without explicit temporal modeling (still typically benefits from smoothing downstream for teleop).

### Why it might be useful
If you want the paper to claim **3D hand reconstruction** rather than “landmarks → angles”, WiLoR is a strong candidate for the webcam side.

### Why it may be overkill / not the best “low-income labs” story
- **MANO model** itself is not permissively licensed; it requires separate download and has restrictions.
- The WiLoR repo indicates **CC-BY-NC-ND** constraints for parts/models; this complicates “industry-friendly open-source” positioning.

### References
- Project page: `https://rolpotamias.github.io/WiLoR/`
- Repo: `https://github.com/rolpotamias/WiLoR`
- arXiv: `https://arxiv.org/abs/2409.12259`

---

## 3) YOHA (handtracking-io) — web landmark tracker

### What it gives you
- Web-focused hand tracking engine; includes a 21-landmark output for a single hand.

### Why it’s probably not the main path
- The repository indicates it is **currently unmaintained**.

### Reference
- Repo: `https://github.com/handtracking-io/yoha`

---

## 4) “Triangulation via multiple webcams” (not your current constraint, but relevant context)

Even though you want **one** webcam, it’s worth noting the existence of low-cost multi-webcam approaches to motivate the “single extra view is enough” idea:

- **THETA (2026 arXiv)** uses **three** webcams and a learned classifier to estimate discretized finger joint angles for robotic-hand control.
  - Reference: `https://arxiv.org/html/2601.07768v1`

This is useful related work to cite (as “multi-webcam is possible but increases setup complexity; we target the 1-webcam add-on”).

---

## 5) Commercial / “upper bound” comparators (not the target solution)

### Ultraleap (Leap Motion / IR stereo)
- Generally regarded as significantly more robust than inside-out RGB hand tracking under many occlusion scenarios.
- Not “cheap webcam”, but it’s a strong **upper bound** system to compare against in a paper discussion.

References (general):
- Ultraleap press/info: `https://www.ultraleap.com/company/news/press-release/gemini-v5-hand-tracking/`

---

## 6) Meta Quest hand tracking “confidence” signals (useful for fusion gating)

Even if your webcam module is the main focus, your fusion needs a way to detect “Quest is hallucinating”.

Meta’s Unity Interaction SDK exposes a **high/low confidence** hand tracking signal:
- `HandTrackingConfidenceProvider.TryGetTrackingConfidence(key, out bool isTrackingHighConfidence)`
  - Reference (Interaction SDK v68): `https://developers.meta.com/horizon/reference/interaction/v68/class_oculus_interaction_hand_tracking_confidence_provider`

Also relevant (Quest hand tracking features):
- Fast Motion Mode (60 Hz) / Wide Motion Mode (out-of-FOV plausibility)
  - Reference: `https://developers.meta.com/horizon/documentation/unity/unity-handtracking-overview/`

Important nuance: this is Unity-side / SDK-side. Whether you can access an analogous confidence value in your current Unity→UDP pipeline depends on what you currently serialize in `HandController.cs`. For the eventual system, exposing **per-hand confidence** (and ideally per-finger/landmark quality if available) is high leverage.

---

## Recommendation for your “accessible single-webcam” niche

If the priority is a system that other labs can reproduce easily:

- **Baseline A (open, practical):** MediaPipe Hand Landmarker → compute LEAP-relevant angles → temporal smoothing → fusion gating with Quest confidence/kinematic checks.
- **Baseline B (stronger but less open):** WiLoR (MANO) → angles → fusion (note licensing).
- **Paper hook:** “single extra view + confidence-aware fusion” reduces catastrophic finger hallucinations without requiring multi-camera rigs or point clouds.

