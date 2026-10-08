# Computer Vision Algorithm Design Prompt: Dynamic Motion-Based Segmentation

**You are an expert computer vision engineer and researcher.** I need you to design a robust, highly efficient algorithm for a real-time tracking application running on an NVIDIA Jetson Nano mounted on an unmanned aerial vehicle (UAV).

## System Context & Inputs
1. **Hardware**: NVIDIA Jetson Nano (compute and memory constrained).
2. **Input Feed**: A continuous live video stream from a camera mounted on the UAV.
3. **Current Pipeline**: We are NOT using any prior object detection models like YOLO. The entire detection and tracking system must rely purely on visual data and motion characteristics.

## The Core Problem
The challenge is extracting the moving target UAV from the scene without prior semantic knowledge of what it looks like. 

**Our goal is to produce a dense semantic mask (a binary mask where the target UAV is 1 and the background is 0), and derive a bounding box from this mask.**

However, because the camera is mounted on a flying drone, we face severe dynamic discrepancy challenges:
1. **Global Motion**: The background moves rapidly across the frame due to our drone's translation, panning, or attitude adjustments.
2. **Local Motion**: The target UAV moves independently within the scene.
3. **Parallax Complexity**: The background has 3D depth (buildings, trees, terrain), meaning the global background motion isn't a simple uniform 2D translation.
4. **Low Signal-to-Noise**: Camera jitter, wind buffeting, and scale changes are constant.

## Your Task
Propose a highly efficient computer vision algorithm (or a combination of lightweight techniques) to isolate the target entirely based on spatial and temporal motion discrepancy between consecutive frames.

**Requirements for your solution:**
- **MUST BE PURELY DETERMINISTIC:** Do not suggest *any* neural networks, deep learning segmentation models, or CNNs (no YOLO, no SAM, no AI models). The solution must be a strictly deterministic computer vision pipeline based on **optical flow separations**.
- Address how to perform global ego-motion compensation to isolate the local optical flow of the target.
- Address how to handle edge cases where the target temporarily matches the background velocity profile exactly (zero relative delta/stagnation).
- Detail the exact mathematical or algorithmic steps (e.g., specific sparse/dense optical flow variants like Lucas-Kanade or Farneback, affine/homography transforms, RANSAC, flow magnitude/angle thresholding, morphological operations, etc.).
- Explain why your proposed approach minimizes false-positive clusters triggered by parallax edges or camera jitter.

Please outline the exact pipeline step-by-step so my engineering agent can implement it in Rust using OpenCV.
