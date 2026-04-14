# Webcam Assist (Quest+1 RGB webcam) — design notes and package choices (2026-04-14)

## Problem restatement (your setup)
Quest 2 finger tracking occasionally enters a **prediction / hallucination** regime under self-occlusion (hand blocks fingers). Those errors are *worse than noise* because they can send the LEAP motors to a wrong pose, corrupt demos, and hurt safety.

You want a **single cheap RGB webcam** viewing the hand from a different angle to reduce these failures, ideally without requiring calibration-heavy multi-cam rigs or point clouds.

---

## What “good” looks like for teleop data collection
For teleoperation, you usually care more about:

- **Continuity** (no sudden jumps)
- **Plausibility** (joint limits, realistic coupling)
- **Low latency** (operator trust and control feel)
- **Graceful degradation** (when vision fails, output should “freeze” or drift safely, not snap wildly)

The webcam module should be judged on these criteria, not just MPJPE.

---

## System concept: confidence-aware “assist”, not a hard switch
The safest pattern is **Quest-first** with webcam assistance:

- If Quest seems confident and consistent → pass-through.
- If Quest seems wrong → blend toward webcam estimate and increase smoothing.
- If both are uncertain → hold last safe pose + mild relaxation to neutral.

This avoids introducing latency in the common case.

---

## Package shortlist (webcam hand tracking)

### Tier 1: “Ship it” baseline
**MediaPipe Hand Landmarker / Hands**
- Pros: fast, widely supported, Apache-ish ecosystem, many examples, cross-platform.
- Cons: monocular 3D is approximate; occlusion still exists; may swap fingers in extreme poses.

### Tier 2: “3D reconstruction” baseline (researchy)
**WiLoR (CVPR 2025)**
- Pros: higher-fidelity 3D hand model (MANO).
- Cons: MANO licensing + repo/model licensing constraints; likely heavier than MediaPipe; may complicate “accessible” story.

### Tier 3: web-only / optional mention
**YOHA**
- Pros: easy demo in browser.
- Cons: unmaintained.

Commercial (not target):
- Ultraleap as an “upper bound” comparator if you want a reference for what excellent tracking feels like.

See the package landscape doc for links: `tasks/hand_tracking_sdk_landscape_2026-04-14.md`.

---

## How to map webcam output to your 16-DoF LEAP control (conceptually)
You ultimately need a 16-value joint target vector consistent with what you already send to the LEAP hand.

Two reasonable approaches (both single-webcam compatible):

### A) Geometry-first (no training required)
- Use MediaPipe’s 21 landmarks.
- Compute per-finger flexion proxies from landmark angles in the 2D image plane (and/or the provided z-like coordinate).
- Fit a small set of scaling/offset parameters to match your LEAP joint conventions.

Pros: very accessible, minimal data collection.
Cons: less accurate under foreshortening; thumb is hard.

### B) Lightweight learned regressor (still accessible)
- Input: webcam landmarks (21×(x,y,z,confidence)) over a short window.
- Output: 16 joint targets.
- Training data: collect paired webcam landmarks + “teacher” targets (Quest when un-occluded; plus your own prompted poses).

Pros: can learn your specific LEAP joint mapping and thumb quirks.
Cons: requires a small dataset + training pipeline (but can be modest).

---

## Confidence signals and gating ideas

### Quest-side signals (high leverage)
Meta’s SDK exposes a high/low confidence flag in Unity (`HandTrackingConfidenceProvider.TryGetTrackingConfidence`).

Even if you don’t use the full Unity Interaction SDK, the paper/system can incorporate:
- **Quest-provided confidence** (if you can export it)
- **Kinematic sanity checks** (limit violations, velocity spikes, discontinuities)

### Webcam-side signals
From MediaPipe-like trackers you can use:
- detection/tracking confidences
- landmark visibility / presence heuristics
- re-detection events as a “possible glitch” marker

### Simple, publishable fusion rule (concept)
Per joint \(j\), define a reliability weight \(w_j \in [0,1]\) for Quest:

- \(w_j \approx 1\) when Quest is confident + consistent
- \(w_j \approx 0\) when Quest is low confidence or fails sanity checks

Then:
\n- fused_j = w_j * quest_j + (1-w_j) * webcam_j
\nwith temporal smoothing + projection into joint limits.

---

## Evaluation ideas (no code yet; just what you’d measure)

### Tracking-level (desk test)
- Glitch rate: frames violating limits or showing implausible jumps
- Recovery time after occlusion ends
- Spectral energy in high-frequency band (jitter proxy)

### Teleop/data collection-level (robot)
- Number of “bad episodes” due to hand pose glitches
- Frequency of safety clipping / saturations
- Operator workload (optional)

---

## What this suggests for a paper contribution
The novelty is not “we track hands with a webcam” (lots of work exists), but:

- **single extra view** specifically to mitigate HMD self-occlusion failures
- **confidence-aware fusion** tuned for *teleoperation safety* and *data quality*
- evaluation on a real dexterous-hand teleop/data-collection pipeline (your LEAP+Franka stack)

