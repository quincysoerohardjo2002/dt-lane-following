# dt-lane-following

A **Duckietown lane-following system with object detection** for the Duckiebot. This project enables a Duckiebot to autonomously follow lane markings on a Duckietown road and optionally detect and stop for duckie obstacles using an ONNX-based neural network.

Built on ROS (Robot Operating System) and the Duckietown software stack.

---

## How it works

The system is a pipeline of ROS nodes. Each node handles one step of the processing chain:

```
Camera image
  │
  ├──► Anti-Instagram (color correction)
  │         │
  │         ▼
  │    Line Detector (detects white/yellow/red line segments)
  │         │
  │         ▼
  │    Ground Projection (projects segments to ground plane)
  │         │
  │         ▼
  │    Lane Filter (estimates lateral offset and heading)
  │         │
  │         ▼
  │    Lane Controller (PID control → wheel commands)
  │
  └──► Object Detection (ONNX model → detect duckies)
            │
            ├── Standalone mode: sends wheel commands directly (drive / stop)
            └── Combined mode: publishes obstacle distance to Lane Controller
```

In **lane-following mode**, the camera image flows through color correction, line detection, ground projection, and filtering to estimate where the robot is in the lane. The lane controller then computes wheel velocities to keep the robot centered.

In **object-detection mode**, the camera image is fed into a YOLOv8-based ONNX model that detects duckie obstacles. When an obstacle is detected within 0.5 m, the robot stops.

In **combined mode**, both pipelines run simultaneously. The object detection node publishes obstacle distance information to the lane controller, which handles both lane keeping and obstacle stopping.

---

## Packages

| Package | Description |
|---|---|
| `lane_control` | PID lane controller. Converts the estimated lane pose (lateral offset `d` and heading angle `phi`) into wheel commands. Supports stop-line slowdown and obstacle distance readings. |
| `lane_filter` | Bayesian histogram filter that estimates the robot's position within the lane (`d` and `phi`) from ground-projected line segments. |
| `line_detector` | Detects white, yellow, and red line segments in camera images using color thresholds. |
| `ground_projection` | Projects detected line segments from image (pixel) coordinates to ground-plane (metric) coordinates using the camera's extrinsic calibration (homography). |
| `object_detection` | ONNX-based obstacle detection node. Runs a trained neural network (`best (1).onnx`) on camera images to detect duckies. Uses ground projection to estimate distance to detected objects. Operates in standalone or combined mode. |
| `anti_instagram` | Applies color correction to camera images to compensate for varying lighting conditions. |
| `fsm` | Finite State Machine that manages the robot's operational mode (e.g., lane following). |
| `stop_line_filter` | Detects stop lines (red lines) from ground-projected segments and signals the lane controller. |
| `led_emitter` | Controls the Duckiebot's LED patterns. |
| `image_processing` | Utilities for image decoding and rectification. |
| `visualization_tools` | RViz visualization helpers for lane segments and lane pose. |
| `duckietown_demos` | Contains the ROS launch files (`master.launch` and the demo-specific launch files) that wire all nodes together. |

---

## Launchers

The `launchers/` directory contains shell scripts that each start a different operating mode:

### `lane-following.sh`

Starts the **full lane-following stack** without object detection.

Nodes launched: anti-instagram, line detector, ground projection, lane filter, lane controller, FSM, LED emitter, and visualization.

### `object-detection.sh`

Starts **only the object detection node** in standalone mode.

The robot drives straight at a fixed speed and stops when a duckie obstacle is detected within range. No lane following is active — the detection node sends wheel commands directly.

### `lane-following-with-detection.sh`

Starts the **full lane-following stack combined with object detection**.

Both the lane-following pipeline and the object detection node run simultaneously. In this mode, the object detection node does *not* send wheel commands directly. Instead, it publishes obstacle distance information to the lane controller, which handles both staying in the lane and stopping for obstacles.

### `default.sh`

Delegates to the platform-specific default launcher (`dt-launcher-default-<ROBOT_TYPE>`).

---

## How to run

**Prerequisites**: Make sure you have an up-to-date `dts` shell and that your Duckiebot is reachable on the network.

### 1. Build the project on your Duckiebot

```bash
dts devel build -f -H YOURBOTNAME.local
```

### 2. Run a launcher

Choose the mode you want:

**Lane following only:**
```bash
dts devel run -L lane-following -H YOURBOTNAME.local
```

**Object detection only (standalone):**
```bash
dts devel run -L object-detection -H YOURBOTNAME.local
```

**Lane following with object detection (combined):**
```bash
dts devel run -L lane-following-with-detection -H YOURBOTNAME.local
```

Replace `YOURBOTNAME` with the hostname of your Duckiebot.

---

## Project structure

```
dt-lane-following/
├── assets/                  # ONNX model file, entrypoint & environment scripts
├── Dockerfile               # Builds the Docker image for the Duckiebot
├── dtproject/               # Duckietown project metadata
├── launchers/               # Shell scripts for each run mode
│   ├── default.sh
│   ├── lane-following.sh
│   ├── lane-following-with-detection.sh
│   └── object-detection.sh
├── packages/                # ROS packages (see table above)
│   ├── anti_instagram/
│   ├── duckietown_demos/    # Launch files (master.launch, etc.)
│   ├── fsm/
│   ├── ground_projection/
│   ├── image_processing/
│   ├── lane_control/
│   ├── lane_filter/
│   ├── led_emitter/
│   ├── line_detector/
│   ├── object_detection/
│   ├── stop_line_filter/
│   └── visualization_tools/
├── dependencies-apt.txt     # System (apt) dependencies
├── dependencies-py3.txt     # Python dependencies
└── README.md
```

---

## Configuration

Tunable parameters are stored in YAML config files inside each package's `config/` directory. Key examples:

- **Lane controller PID gains**: `packages/lane_control/config/lane_controller_node/default.yaml`
- **Object detection thresholds**: `packages/object_detection/include/object_detection_model/config.py` (confidence threshold, stop distance, forward PWM)
- **Object detection mode**: `packages/object_detection/config/object_detection_node/default.yaml` (standalone: `publish_wheels: true`) vs `combined.yaml` (`publish_wheels: false`)
- **Lane filter settings**: `packages/lane_filter/config/`
- **Line detector settings**: `packages/line_detector/config/`