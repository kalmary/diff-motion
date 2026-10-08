# Project Goal & Problem Specification

## 1. Executive Summary
The objective of this project is to reliably detect, segregate, and track a localized region of interest (an observed object) from an airborne camera feed aboard an unmanned aerial vehicle (UAV/drone). 

The core challenge is separating target-induced pixel dynamics from background-induced pixel dynamics across continuous frames, where the relative motion profiles of both entities diverge significantly and unpredictably.

---

## 2. Problem Statement

### 2.1 The Visual Discrepancy Problem
A camera mounted on an aerial platform captures frames composed of two fundamentally distinct dynamic layers:
1. **Background Layer (Global Scene):** The global environment (terrain, structures, vegetation, horizon) observed from a moving perspective.
2. **Target Layer (Observed Object):** A localized cluster of pixels representing a distinct physical entity within the scene.

The system must isolate the target cluster purely or primarily through motion discrepancy and localized signal variation, without prior semantic knowledge of what the object looks like.

### 2.2 Kinematic & Dynamic Divergence
The separation problem manifests in two primary, opposing operational regimes:

* **Regime A (High Target Volatility, Quasi-Static Background):**
  * The background exhibits slow, smooth drift or subtle parallax.
  * The target undergoes sharp, high-magnitude pixel transitions (sudden acceleration, erratic maneuvers, or internal dynamic changes).
  * *Challenge:* Rapid spatial displacement between consecutive frames where target association cannot rely on small-displacement assumptions.

* **Regime B (Global Platform Drift, Localized Micro-Displacement):**
  * The background moves rapidly and uniformly across the entire frame due to drone translation, panning, or attitude adjustments.
  * The target exhibits subtle, low-amplitude, or occasional displacements relative to the scene, or moves counter to the dominant background vector.
  * *Challenge:* Dominant global motion masks local target motion. Standard temporal differentiation treats the entire frame as changed, overwhelming the target signal with global noise.

---

## 3. Operational Constraints & Environmental Factors

### 3.1 Platform Characteristics
* **Unstable Perspective:** The sensor is subject to high-frequency vibrations, jitter, wind buffeting, and sudden gimbal adjustments.
* **Varying Elevation & Scale:** The distance between the drone and the ground/target may vary dynamically, inducing apparent scale changes and continuous parallax gradients.

### 3.2 Target Characteristics
* **No Class-Specific Geometry:** The target cannot be assumed to fit canonical computer vision classes (e.g., standard road vehicles or pedestrians).
* **Variable Pixel Coverage:** The target footprint can vary from a small cluster (sub-1% of screen area) to a prominent localized area.
* **Non-Rigid / Variable Appearance:** Lighting changes, specular reflections, attitude changes, and partial occlusions can drastically alter the target's intensity profile over short time windows.

### 3.3 Scene Constraints
* **Parallax Complexity:** Three-dimensional scene structures (buildings, trees, terrain relief) move at differing apparent velocities depending on depth, complicating single-vector background assumptions.
* **Low Signal-to-Noise Ratio (SNR):** Camera sensor noise, compression artifacts, and environmental factors (glare, shadows, low-contrast terrain) introduce spurious pixel shifts.

---

## 4. System Requirements & Invariants

| ID | Criterion | Requirement Specification |
| :--- | :--- | :--- |
| **REQ-01** | **Discrepancy Detection** | Reliably discern whether anomalous pixel changes stem from a discrete localized body or platform motion. |
| **REQ-02** | **Regime Agnosticism** | Maintain spatial coherence across both operational regimes (fast background/slow target and slow background/fast target) without manual mode switching. |
| **REQ-03** | **Continuous Localization** | Provide a continuous spatial reference (e.g., bounding coordinates, centroid) for the target as long as dynamic divergence persists. |
| **REQ-04** | **Occlusion / Stagnation Tolerance** | Define and handle edge cases where the target temporarily matches the background velocity profile exactly (zero relative delta). |
| **REQ-05** | **Semantic Mask Generation** | Determine a semantic mask based on observations. Output must be a binary mask where the UAV is 1 and the background is 0. |

---

## 5. Explicit Non-Goals (Scope Boundaries)

* **No Neural Networks / Deep Learning:** The system must be a purely deterministic algorithm based entirely on optical flow separations. No AI models, CNNs, or learning-based segmentation techniques are allowed.
* **No Prescribed Algorithm (Beyond Optical Flow):** While optical flow and deterministic methods are mandated, the specific pipeline (e.g. dense vs. sparse, RANSAC, homography) remains open for evaluation.
* **No Flight Control / Gimbal Actuation:** The project scope is bounded by video stream ingestion, processing, and output of target coordinates; drone piloting and flight dynamics control loops are decoupled from this specification.

---

## 6. Success Metrics & Evaluation

1. **Precision of Spatial Segregation:** Accuracy of distinguishing target pixel boundaries or centroids without bleeding into the background motion field.
2. **False Discovery Rate (FDR):** Minimization of false-positive clusters triggered by parallax edges, camera jitter, or sensor noise.
3. **Temporal Tracking Continuity:** Number of track losses or identity swaps per minute of flight footage across varied terrain textures.
4. **Latency Budget:** Frame-to-output latency must remain low and deterministic enough to enable downstream real-time consumers.