# Webcam Assist evaluation protocol (paper-shaped) — 2026-04-14

## Purpose
Define a **reproducible evaluation** for “Quest + single RGB webcam” finger teleoperation robustness that can support a publishable writeup.

The protocol is designed to measure the specific failure mode you described:
- **self-occlusion → Quest hallucinates finger pose → wrong robot-hand command**

It includes both (A) tracking-level metrics and (B) downstream teleoperation/data-collection impact.

---

## Systems to compare (baselines)

### B0 — Quest-only
Your current pipeline: Quest finger estimates used directly for LEAP control (or forwarded as-is).

### B1 — Webcam-only (if feasible)
Webcam tracker drives LEAP joints directly (useful to show webcam alone is not sufficient everywhere; also clarifies the value of fusion).

### B2 — Quest + Webcam Assist (proposed)
Confidence-aware fusion: Quest-first, blend toward webcam when Quest likely wrong.

### Optional reference (not required, but good to mention)
- **Commercial upper bound** (e.g., Ultraleap) if you ever want to contextualize “what great tracking looks like”. Not necessary for the core claim.

---

## A) Tracking-level evaluation (desk test, fast iteration)

### A.1 Data collection setup
Record synchronized streams for each trial:
- Quest finger output (the same values you use for teleop)
- Webcam video (RGB)
- Webcam tracker output (landmarks + any confidence)
- (If available) Quest confidence (high/low) and/or any native tracking quality indicator

Store logs so you can replay offline and compute metrics consistently.

### A.2 Occlusion scenario suite (minimum)
Each scenario should be recorded as a short clip (10–20 s) and repeated \(N\) times.

**S1: Finger curl behind palm**
- Slowly close hand to fist with fingers fully occluded by palm for 1–2 s, then reopen.

**S2: Thumb crosses index**
- Thumb moves across index / pinch-like poses where thumb occludes other digits.

**S3: Hand yaw rotation**
- Rotate hand left/right relative to Quest cameras while flexing fingers (induces foreshortening + partial occlusion).

**S4: Two-hand occlusion (if you use both hands)**
- One hand occludes the other briefly; useful if your teleop uses two hands or bimanual gestures.

**S5: Object occlusion**
- Hold a small object (cup/phone) and perform grasp; fingers partially hidden.

**S6: Fast motion burst**
- Quick open-close-open sequence to test temporal stability.

Practical note: the webcam should be placed to be a *complementary view* (e.g., monitor-top, slightly downward) to maximize “assist” benefit.

### A.3 Metrics (tracking-level)
These metrics should be computed on the **16 teleop joint targets** (or whatever you consider the control vector), since that’s what matters for LEAP actuation.

#### M1: Glitch rate
Count frames where any of the following occurs:
- joint limit violation (pre-clip)
- large instantaneous jump: \(|\Delta \theta_j| > T_j\) for any joint \(j\)
- implausible configuration detector triggers (see M4)

Report:
- glitches per minute
- % frames glitched

#### M2: Recovery time
After an occlusion interval ends, time until the signal returns to a “plausible” regime:
- no limit violations
- velocities below threshold
- stable tracking confidence (if available)

Report median and 95th percentile recovery times.

#### M3: Jitter / smoothness
Compute on each joint:
- total variation: \(\sum_t |\theta_{t+1}-\theta_t|\)
- optionally jerk proxy: \(\sum_t |\Delta^2 \theta_t|\)

Report aggregated statistics across joints and trials.

#### M4: Plausibility score (teleop-specific)
Define a simple plausibility score using constraints you already care about:
- joint limits
- expected coupling (e.g., DIP vs PIP monotonicity / typical ranges)
- saturation frequency (how often you would hit safety clips)

This metric is not “biomechanically perfect”; it just needs to correlate with “this would have ruined a demo”.

#### M5: Latency impact (operator-visible)
Measure added delay from fusion path:
- If fusion is Quest-first, it should be near-zero in the “good tracking” regime.
Report: median and worst-case added latency during occlusion transitions.

### A.4 Reporting format
Per scenario \(S_k\):
- bar plot: glitch rate (B0/B1/B2)
- box plot: recovery time (B0/B2)
- line plot: one representative trajectory showing a Quest hallucination corrected by assist

---

## B) Teleoperation/data-collection evaluation (robot impact)

### B.1 Standardized teleop task
Pick a minimal, repeatable teleop task that is sensitive to finger pose errors, such as:
- “close to pinch” then “close to fist” then “open” while keeping arm steady
- or a simple grasp-and-release routine on a standardized object/hold

Keep arm teleop unchanged; only change finger control condition.

### B.2 Trial structure
For each condition (B0/B2):
- run \(N\) trials (e.g., 20–30) per session
- randomize order across sessions (to reduce learning effects)

### B.3 Metrics (robot/data)

#### R1: Episode quality yield
If you have a “good/bad episode” marking flow:
- % episodes rated “good” without hand-related issues
- reasons for bad episodes (categorize: finger glitch, arm issue, camera issue, etc.)

#### R2: Safety clip / saturation events
Count how often motor commands hit safety constraints / clips.

#### R3: Task success / stability
Define a binary success criterion appropriate for your setup:
- stable grasp achieved within X seconds
- object/hold remains stable under gentle tug for Y seconds

#### R4: Operator workload (optional, fast)
3–6 questions (mini NASA-TLX or a short custom scale):
- perceived control
- trust in finger pose
- frustration during occlusions

### B.4 Analysis
Given small N, use:
- success rates with Wilson 95% CI
- paired tests if within-subject (same operator) across conditions

---

## Minimal “paper claims” supported by this protocol
If B2 improves:
- M1 glitch rate ↓
- M2 recovery time ↓
- R1 episode yield ↑
without significantly harming latency (M5),
then you have a strong story: **single RGB webcam assistance materially improves real teleop data collection quality under occlusion**.

---

## Notes on feasibility and keeping it accessible
- This protocol deliberately avoids requiring motion-capture GT.
- You can start with tracking-level metrics (A) in a day, then graduate to robot trials (B).
- The evaluation is compatible with either a pure-geometry webcam baseline or a learned landmark→angle mapping.

