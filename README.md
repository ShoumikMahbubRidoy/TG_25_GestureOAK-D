# TG_25_GestureOAK-D - Comprehensive Documentation

**English** | [日本語](#日本語版)

---

## 🌟 Project Overview

**TG_25_GestureOAK-D** is a real-time hand detection and swipe gesture recognition system optimized for the **Luxonis OAK-D-PRO** camera. The system leverages infrared (IR) stereo cameras with depth sensing to provide robust hand tracking in challenging lighting conditions, specifically designed for an operating distance of **80–160 cm**.

### Key Features
- **IR-Based Hand Detection**: Utilizes stereo IR cameras for dark environment operation
- **MediaPipe Integration**: Employs palm detection and hand landmark neural networks
- **Swipe Gesture Recognition**: Detects left-to-right swipe gestures with velocity and distance validation
- **UDP Communication**: Sends swipe notifications to external systems (`192.168.10.10:6001`)
- **Depth Filtering**: Uses stereo depth maps to filter false positives
- **Standalone Executables**: PyInstaller-based `.exe` files for deployment

---

## 📋 Table of Contents

1. [What Has Been Implemented](#what-has-been-implemented)
2. [Technical Architecture](#technical-architecture)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Installation Guide](#installation-guide)
6. [Application Architecture](#application-architecture)
7. [Running the Application](#running-the-application)
8. [Executable (.exe) Handling](#executable-exe-handling)
9. [Implementation Details](#implementation-details)
10. [Troubleshooting](#troubleshooting)
11. [Known Issues](#known-issues)
12. [Future Roadmap](#future-roadmap)

---

## 🎯 What Has Been Implemented

### Core Components

#### 1. **OAK-D Camera Interface** (`oak_camera.py`)
- **Purpose**: Hardware abstraction layer for OAK-D device communication
- **Implementation**:
  - Dual-mode support (RGB and IR cameras)
  - Dynamic resolution configuration (640×480 default)
  - Frame rate control (30 FPS target)
  - DepthAI pipeline initialization
  - Non-blocking frame acquisition
- **Technical Details**:
  - Uses `depthai` SDK for device communication
  - Implements `ColorCamera` and `MonoCamera` nodes
  - `ImageManip` node for real-time resizing
  - XLinkOut queues for host-device data transfer

#### 2. **Hand Detector** (`hand_detector.py`)
- **Purpose**: Real-time hand detection and landmark extraction
- **Implementation**:
  - **Palm Detection Network**: SSD-based palm localization (128×128 input)
  - **Hand Landmark Network**: 21-point hand skeleton extraction (224×224 input)
  - **Postprocessing Network**: NMS and score filtering
  - **Script Node**: On-device orchestration of detection pipeline
  - **IR Enhancement**: CLAHE + bilateral filtering for low-light conditions
  - **Depth-Based Filtering**: Distance-aware variance tolerance (300–2000 mm range)
- **Technical Pipeline**:
  ```
  IR Camera → ImageManip (resize) → Palm NN → Postproc NN → 
  Landmark NN → Script (manager) → Host Queue
  ```
- **Why This Approach**:
  - **IR cameras** provide consistent performance in dark environments
  - **On-device processing** reduces latency (no host-side inference)
  - **Depth filtering** eliminates false positives from background objects
  - **MediaPipe models** offer pre-trained accuracy for hand detection

#### 3. **Swipe Detector** (`swipe_detector.py`)
- **Purpose**: Robust left-to-right swipe gesture recognition
- **Implementation**:
  - **State Machine**: `IDLE → DETECTING → VALIDATING → CONFIRMED`
  - **Trajectory Buffering**: Deque-based position history (18 frames)
  - **Timestamp-Based Velocity**: FPS-independent speed calculation
  - **Multi-Criteria Validation**:
    - Minimum distance: 90 pixels
    - Duration: 0.2–2.0 seconds
    - Velocity: 35–900 px/s
    - Y-axis deviation: ≤35% of X-axis travel
  - **Cooldown Mechanism**: 0.8s debounce to prevent repeated triggers
  - **UDP Notification**: Non-blocking socket communication
- **Why This Approach**:
  - **State machine** provides clear gesture progression tracking
  - **Velocity-based** detection is FPS-independent (robust across hardware)
  - **Multi-criteria** validation reduces false positives
  - **Cooldown** prevents rapid re-triggering during continuous motion

#### 4. **Applications**

##### Hand Tracking App (`hand_tracking_app.py`)
- **Purpose**: Integrated hand detection with swipe recognition
- **Features**:
  - Real-time hand landmark visualization
  - Bounding box rendering
  - Depth information overlay
  - Gesture classification (if enabled)
  - Swipe progress indicator
  - Statistics display (FPS, swipe count, filtered false positives)
- **Use Case**: Development, debugging, and demonstration

##### Motion Swipe App (`motion_swipe_app.py`)
- **Purpose**: Specialized swipe detection with trajectory visualization
- **Features**:
  - Swipe trail rendering (last 18 positions)
  - Detection zone overlay
  - Detailed progress metrics
  - Configurable sensitivity presets
- **Use Case**: Swipe gesture tuning and validation

##### Swipe Detection App (`swipe_detection_app.py`)
- **Purpose**: Minimal swipe-only interface
- **Features**:
  - Lightweight UI focused on swipe events
  - Real-time state display
  - Performance optimization for production
- **Use Case**: Production deployment

#### 5. **Launcher GUI** (`TG25_Launcher.py`)
- **Purpose**: User-friendly executable management
- **Implementation**:
  - Tkinter-based graphical interface
  - Start/Stop worker process control
  - Graceful shutdown via stop flag file
  - Process monitoring and status display
- **Why This Approach**:
  - **Separate launcher** prevents main app from blocking UI
  - **Stop flag file** enables clean shutdown without forced termination
  - **Process group management** ensures proper cleanup

#### 6. **Utility Modules**

##### FPS Counter (`FPS.py`)
- Rolling window FPS calculation
- Global average FPS tracking
- Elapsed time measurement

##### MediaPipe Utils (`mediapipe_utils.py`)
- Hand region data structures
- Landmark coordinate transformations
- Gesture recognition logic (finger counting)

##### Template Manager Script (`template_manager_script_solo.py`)
- On-device script template for DepthAI Script node
- Coordinates palm detection → landmark extraction pipeline
- Implements NMS, region rotation, and score filtering

---

## 🏗️ Technical Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Host Application                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  main.py (Menu Controller)                             │ │
│  │    ├─ Camera Test (oak_camera.py)                      │ │
│  │    ├─ Hand Tracking App                                │ │
│  │    ├─ Swipe Detection App                              │ │
│  │    └─ Motion Swipe App                                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Detection Layer                                       │ │
│  │    ├─ HandDetector (hand_detector.py)                  │ │
│  │    └─ SwipeDetector (swipe_detector.py) ─────UDP────┐  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                   DepthAI Pipeline
                              │
┌─────────────────────────────────────────────────────────────┐
│                      OAK-D Device                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  IR Mono Cameras (LEFT + RIGHT)                        │ │
│  │    ↓                ↓                                  │ │
│  │  StereoDepth   MonoCamera → ImageManip → RGB888p       │ │
│  │    ↓                ↓                                  │ │
│  │  depth_out      cam_out (XLinkOut)                     │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Neural Networks (on VPU)                              │ │
│  │    ├─ Palm Detection NN (palm_detection_sh4.blob)      │ │
│  │    ├─ Postproc NN (PDPostProcessing_top2_sh1.blob)     │ │
│  │    └─ Landmark NN (hand_landmark_lite_sh4.blob)        │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Script Node (Manager)                                 │ │
│  │    ├─ Coordinates NN execution                         │ │
│  │    ├─ Implements NMS and filtering                     │ │
│  │    └─ Outputs marshalled results → manager_out         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Sequence

```
1. IR Camera Capture (400p @ 30fps)
   ↓
2. ImageManip → Resize to 640×480, Convert to RGB888p
   ↓
3. IR Enhancement (CLAHE + Bilateral Filter)
   ↓
4. Palm Detection NN → Bounding boxes
   ↓
5. Postprocessing NN → NMS, Top-2 hands
   ↓
6. Landmark NN → 21 keypoints per hand
   ↓
7. Script Manager → Serialize results (marshal)
   ↓
8. Host Queue → Python HandDetector.get_frame_and_hands()
   ↓
9. Depth Filtering → Validate hand regions (300–2000mm)
   ↓
10. SwipeDetector.update() → Trajectory analysis
    ↓
11. Swipe Confirmation → UDP packet to 192.168.10.10:6001
```

---

## 📦 Prerequisites

### Hardware Requirements
- **OAK-D-PRO** or **OAK-D** camera with IR stereo capability
- USB 3.0 port (minimum; USB 3.1+ recommended for higher throughput)
- Windows 10/11 (64-bit) or Linux (Ubuntu 20.04+)

### Software Requirements
- **Python 3.10–3.12** (3.12 recommended for latest features)
- **pip** package manager (latest version)
- **Git** (for cloning repository)
- **Virtual environment support** (venv module)

### System Libraries (Linux)
```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libusb-1.0-0-dev \
    libudev-dev \
    python3-dev \
    python3-pip \
    libopencv-dev
```

### System Libraries (Windows)
- **Visual C++ Redistributable** (automatically installed with Python)
- **USB 3.0 drivers** (usually built-in; check Device Manager)

---

## 🔧 Environment Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D.git
cd TG_25_GestureOAK-D
git checkout Hand-Gesture  # Ensure you're on the correct branch
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

**Why Virtual Environment?**
- Isolates project dependencies from system Python
- Prevents version conflicts with other projects
- Enables reproducible builds

### Step 3: Upgrade pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 📥 Installation Guide

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependency Breakdown**:
- **depthai** (≥2.24.0): OAK-D SDK for device communication and pipeline management
- **opencv-python** (≥4.8.0): Computer vision library for frame processing and display
- **numpy** (≥1.24.0): Numerical operations for landmark transformations
- **mediapipe** (≥0.10.0): Pre-trained hand detection models (optional, for reference)
- **imutils** (≥0.5.4): Convenience functions for OpenCV operations
- **pyyaml** (≥6.0): Configuration file parsing (if using YAML configs)

### Verify Installation
```bash
# Check Python version
python --version  # Should show 3.10.x or 3.11.x or 3.12.x

# Check installed packages
pip list | grep -E "depthai|opencv|numpy"

# Test OAK-D connection
python -c "import depthai as dai; print(dai.__version__); print(dai.Device.getAllAvailableDevices())"
```

**Expected Output**:
```
2.24.0.0  # or later
[<depthai.DeviceInfo ...>]  # Should list your OAK-D device
```

### Environment Verification Script
```bash
python probe_dai.py
```

**What It Checks**:
- DepthAI library import
- OAK-D device enumeration
- USB connection speed
- Camera sensor availability

---

## 🏛️ Application Architecture

### Module Structure
```
src/gesture_oak/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── oak_camera.py          # Camera abstraction layer
├── detection/
│   ├── __init__.py
│   ├── hand_detector.py       # MediaPipe-based hand detection
│   ├── motion_detector.py     # Motion-based detection (alternative)
│   ├── motion_swipe_detector.py
│   └── swipe_detector.py      # Gesture recognition logic
├── logic/
│   ├── __init__.py
│   └── gesture_classifier.py # Finger counting, gestures
├── apps/
│   ├── __init__.py
│   ├── hand_tracking_app.py   # Main hand tracking demo
│   ├── swipe_detection_app.py # Swipe-focused demo
│   └── motion_swipe_app.py    # Motion-based swipe demo
└── utils/
    ├── FPS.py                 # FPS measurement
    ├── mediapipe_utils.py     # Data structures, helpers
    └── template_manager_script_solo.py  # On-device script
```

### Entry Points
1. **main.py**: Interactive menu for running different applications
2. **run_hand_tracking.py**: Direct execution of hand tracking app
3. **TG25_Launcher.py**: GUI launcher for executable management
4. **probe_dai.py**: Device diagnostic tool

### Design Patterns

#### 1. **Factory Pattern** (OAKCamera)
```python
# oak_camera.py
class OAKCamera:
    def setup_pipeline(self) -> dai.Pipeline:
        # Dynamically creates RGB or IR pipeline based on use_rgb flag
```

#### 2. **State Machine** (SwipeDetector)
```python
# swipe_detector.py
class SwipeState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    VALIDATING = "validating"
    CONFIRMED = "confirmed"
```

#### 3. **Template Method** (HandDetector)
```python
# hand_detector.py
def build_manager_script(self) -> str:
    # Uses string template for on-device script generation
    template = Template(raw_code)
    return template.substitute(params...)
```

---

## 🚀 Running the Application

### Method 1: Interactive Menu (Recommended)
```bash
python main.py
```

**Menu Options**:
```
1. Test camera connection        # Verify OAK-D setup
2. Run hand tracking app         # Full-featured hand detection + swipe
3. Run swipe detection app       # Swipe-focused interface
4. Run motion-based swipe        # Alternative motion detection
5. Exit
```

**When to Use Each**:
- **Option 1**: First-time setup, troubleshooting connection issues
- **Option 2**: Development, debugging, demonstration
- **Option 3**: Production swipe detection
- **Option 4**: Experimental motion-based approach

### Method 2: Direct Execution
```bash
# Hand tracking with swipe detection
python run_hand_tracking.py

# Or via module
python -m gesture_oak.apps.hand_tracking_app
```

### Method 3: Using UV (Fast Python Package Manager)
```bash
# Install uv if not already installed
pip install uv

# Run with uv (faster startup)
uv run python main.py
```

### Runtime Controls

#### Keyboard Shortcuts
- **`q`**: Quit application
- **`s`**: Save current frame to disk (JPEG format)
- **`r`**: Reset swipe statistics

#### Console Output
```
OAK-D Hand Detection Demo with Swipe Detection
=============================================
Press 'q' to quit
Press 's' to save current frame
Press 'r' to reset swipe statistics

Connected to device: OAK-D-PRO
USB Speed: SUPER_SPEED
Hand detection started. Showing live preview...

Hand 1: right (confidence: 0.957)
  Gesture: FIVE
 LEFT-TO-RIGHT SWIPE DETECTED! (Total: 1)
```

### Expected Performance
- **FPS**: 25–30 FPS (IR mode on OAK-D-PRO)
- **Latency**: ~50–80ms (camera to display)
- **Detection Range**: 80–160 cm optimal, 40–200 cm extended
- **Swipe Success Rate**: ~95% under good conditions

---

## 🗂️ Executable (.exe) Handling

### Building Executables

The project uses **PyInstaller** to create standalone executables for Windows deployment.

#### Prerequisites for Building
```bash
pip install pyinstaller
```

#### Build Scripts

##### 1. Build Hand Tracking Worker
```bash
# Windows
build.bat

# Or manually
pyinstaller TG25_HandTracking.spec
```

**Spec File Highlights** (`TG25_HandTracking.spec`):
```python
a = Analysis(
    ['run_hand_tracking.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),  # Bundle neural network blobs
        ('src/gesture_oak/utils/template_manager_script_solo.py',
         'src/gesture_oak/utils'),
    ],
    hiddenimports=[
        'depthai',
        'cv2',
        'numpy',
        # ... all dependencies
    ],
    ...
)
```

##### 2. Build Launcher GUI
```bash
pyinstaller TG25_Launcher.spec
```

##### 3. Build Diagnostic Tool
```bash
pyinstaller probe_dai.spec
```

### Executable Distribution

After building, the `dist/` folder contains:
```
dist/
├── TG25_HandTracking/
│   ├── TG25_HandTracking.exe  # Main worker application
│   ├── models/                # Neural network blobs
│   └── [dependencies]         # DLLs, libraries
├── TG25_Launcher.exe          # GUI launcher
└── probe_dai.exe              # Diagnostic tool
```

### Deployment Instructions

1. **Copy entire folder** to target machine:
   ```
   TG25_GestureOAK-D_Deployment/
   ├── TG25_Launcher.exe
   ├── TG25_HandTracking.exe
   ├── models/
   └── (all DLLs from dist/)
   ```

2. **Run `TG25_Launcher.exe`** to start the GUI

3. **Click "Start Hand Tracking"** to launch worker

### Troubleshooting Executables

#### Issue: "Cannot find models folder"
**Solution**: Ensure `models/` is in the same directory as `.exe`

#### Issue: "Import Error: No module named 'depthai'"
**Solution**: PyInstaller may have missed dependencies. Add to `hiddenimports` in `.spec` file:
```python
hiddenimports=[
    'depthai',
    'depthai._version',  # Add submodules
    ...
]
```

#### Issue: Worker doesn't stop gracefully
**Solution**: Check stop flag file path:
```python
# In TG25_Launcher.py
stop_file = Path(_exe_dir()) / "TG25_STOP.flag"

# In run_hand_tracking.py
stop_file_path = os.environ.get("TG25_STOP_FILE", "")
```

---

## 🛠️ Implementation Details

### Hand Detection Pipeline

#### 1. Camera Initialization
```python
# hand_detector.py
cam_left = pipeline.createMonoCamera()
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setFps(30)

cam_right = pipeline.createMonoCamera()
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setFps(30)
```

**Why IR Cameras**:
- Consistent performance in low light
- Less affected by color variations (skin tone, clothing)
- Native 400p resolution matches MediaPipe input requirements

#### 2. Stereo Depth Configuration
```python
depth = pipeline.createStereoDepth()
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(True)    # Eliminates invalid disparities
depth.setSubpixel(True)          # Improves depth precision
```

**Technical Explanation**:
- **HIGH_DENSITY**: Optimized for indoor scenes with small objects (hands)
- **7×7 Median Filter**: Removes speckle noise from depth map
- **Left-Right Check**: Validates disparity consistency between stereo pairs
- **Subpixel**: Enables fractional disparity values for smoother depth

#### 3. IR Frame Enhancement
```python
def enhance_ir_frame(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter (edge-preserving smoothing)
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
```

**Why This Approach**:
- **CLAHE**: Enhances local contrast without over-amplifying noise
- **clipLimit=2.0**: Prevents excessive amplification in homogeneous regions
- **tileGridSize=(8,8)**: Balances local and global contrast
- **Bilateral Filter**: Smooths noise while preserving hand edges

#### 4. Depth-Based Filtering
```python
def filter_hands_by_depth(self, hands, depth_frame):
    for hand in hands:
        cx, cy = hand.rect_x_center_a, hand.rect_y_center_a
        d_center = depth_frame[cy, cx]
        
        # Valid range: 300–2000mm (30cm – 2m)
        if not (300 <= d_center <= 2000):
            continue
        
        # Region of interest around hand center
        half = 18 if d_center < 1000 else 26  # Larger ROI for distant hands
        roi = depth_frame[cy-half:cy+half, cx-half:cx+half]
        
        avg = np.mean(roi[roi > 0])
        std = np.std(roi[roi > 0])
        
        # Distance-aware tolerance
        std_limit = 80.0 + 0.08 * max(0.0, (avg - 800.0))
        if std <= std_limit:
            hand.depth = avg
            hand.depth_confidence = 1.0 - (std / std_limit)
```

**Technical Rationale**:
- **Variable std_limit**: Farther hands have less depth precision
- **ROI size scaling**: Adapts to hand apparent size at different distances
- **Confidence score**: Quantifies reliability of depth measurement

### Swipe Detection Algorithm

#### State Transition Logic
```python
IDLE:
    - Buffer hand positions with timestamps
    - Check for consistent rightward motion (3 consecutive frames)
    - Transition to DETECTING if motion detected

DETECTING:
    - Continue buffering trajectory
    - Calculate accumulated distance
    - Timeout if duration > max_duration (2.0s)
    - Abort if movement reverses left
    - Transition to VALIDATING when distance >= min_distance (90px)

VALIDATING:
    - Verify duration within [0.2s, 2.0s]
    - Calculate average velocity: distance / duration
    - Check velocity bounds [35 px/s, 900 px/s]
    - Verify Y-axis deviation ≤ 35% of X-travel
    - Confirm no significant backward motion (< -12px jumps)
    - Transition to CONFIRMED if all checks pass

CONFIRMED:
    - Increment swipe counter
    - Send UDP packet "Swipe" to 192.168.10.10:6001
    - Apply cooldown (0.8s) to prevent rapid re-triggers
    - Transition back to IDLE
```

#### Velocity Calculation
```python
# FPS-independent velocity (pixels per second)
times = np.array(self.time_buffer)
poses = np.array(self.position_buffer)

total_dx = poses[-1, 0] - poses[0, 0]  # Horizontal displacement
duration = times[-1] - times[0]        # Elapsed time
velocity = total_dx / duration         # px/s
```

**Why Timestamp-Based**:
- Independent of frame rate variations
- Handles dropped frames gracefully
- Provides accurate speed measurement

---

## 🐛 Troubleshooting

### Issue 1: Camera Not Detected

**Symptoms**:
```
Failed to connect to OAK-D: [X_LINK_DEVICE_NOT_FOUND]
```

**Solutions**:

1. **Check USB Connection**:
   ```bash
   # Linux
   lsusb | grep "03e7"  # Luxonis VID
   
   # Windows (Device Manager)
   # Look for "Movidius MyriadX" or "OAK-D" under USB devices
   ```

2. **Update USB Drivers** (Windows):
   - Download Zadig: https://zadig.akeo.ie/
   - Replace driver with WinUSB

3. **Grant USB Permissions** (Linux):
   ```bash
   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

4. **Test with Diagnostic Tool**:
   ```bash
   python probe_dai.py
   ```

### Issue 2: Low FPS / Stuttering

**Symptoms**:
- Frame rate drops below 20 FPS
- Jerky camera preview

**Solutions**:

1. **Check USB Speed**:
   ```python
   # Should show USB3 SUPER_SPEED or SUPER_PLUS
   device.getUsbSpeed()
   ```

2. **Reduce Queue Sizes** (if buffering):
   ```python
   # In hand_detector.py
   self.q_video = self.device.getOutputQueue(
       name="cam_out", 
       maxSize=2,  # Reduce from 4 to 2
       blocking=False
   )
   ```

3. **Disable IR Enhancement** (for testing):
   ```python
   # Comment out in hand_detector.py
   # frame = self.enhance_ir_frame(raw_frame)
   frame = raw_frame
   ```

4. **Optimize OpenCV**:
   ```python
   import cv2
   cv2.setUseOptimized(True)
   cv2.setNumThreads(0)  # Auto-detect optimal thread count
   ```

### Issue 3: False Positive Swipe Detections

**Symptoms**:
- Swipes trigger without hand movement
- Random objects cause detections

**Solutions**:

1. **Increase Minimum Distance**:
   ```python
   swipe_detector = SwipeDetector(
       min_distance=120,  # Increase from 90
   )
   ```

2. **Stricter Y-Deviation**:
   ```python
   swipe_detector = SwipeDetector(
       max_y_deviation=0.25,  # Reduce from 0.35
   )
   ```

3. **Tighten Velocity Bounds**:
   ```python
   swipe_detector = SwipeDetector(
       min_velocity=50,   # Increase from 35
       max_velocity=700,  # Reduce from 900
   )
   ```

4. **Enable Stricter Depth Filtering**:
   ```python
   # In hand_detector.py
   std_limit = 60.0 + 0.06 * max(0.0, (avg - 800.0))  # Reduce tolerance
   ```

### Issue 4: Hand Not Detected at Distance

**Symptoms**:
- Hand visible on screen but no detection at 100+ cm
- Works only at close range (<80 cm)

**Solutions**:

1. **Lower Palm Detection Threshold**:
   ```python
   detector = HandDetector(
       pd_score_thresh=0.08,  # Reduce from 0.10
   )
   ```

2. **Increase Buffer Size** (more history):
   ```python
   swipe_detector = SwipeDetector(
       buffer_size=24,  # Increase from 18
   )
   ```

3. **Adjust Depth Range**:
   ```python
   # In hand_detector.py - filter_hands_by_depth()
   if not (200 <= d_center <= 2500):  # Extend range
   ```

4. **Verify IR Illumination**:
   - OAK-D-PRO has active IR emitters
   - Check if IR LEDs are visible (use phone camera)

### Issue 5: "ImportError" on Executable

**Symptoms**:
```
ImportError: No module named 'depthai'
ModuleNotFoundError: No module named 'cv2'
```

**Solutions**:

1. **Rebuild with All Hidden Imports**:
   ```python
   # In .spec file
   hiddenimports=[
       'depthai',
       'cv2',
       'numpy',
       'numpy.core',
       'numpy.core._multiarray_umath',
       'mediapipe',
       'marshal',
       'collections',
       'collections.abc',
       'socket',
       'time',
       'pathlib',
   ]
   ```

2. **Bundle Binary Dependencies**:
   ```python
   # In .spec file
   binaries=[
       (r'C:\path\to\.venv\Lib\site-packages\depthai\*.dll', 'depthai'),
   ]
   ```

3. **Use PyInstaller Hooks**:
   ```bash
   pip install pyinstaller-hooks-contrib
   pyinstaller --additional-hooks-dir=. TG25_HandTracking.spec
   ```

### Issue 6: UDP Packets Not Received

**Symptoms**:
- Swipe detected but no network traffic
- Target system doesn't receive "Swipe" message

**Solutions**:

1. **Verify Network Configuration**:
   ```bash
   # Test UDP connectivity
   # On target (192.168.10.10):
   nc -u -l 6001
   
   # On source machine:
   echo "test" | nc -u 192.168.10.10 6001
   ```

2. **Check Firewall Rules**:
   ```bash
   # Windows
   netsh advfirewall firewall add rule name="Allow UDP 6001" dir=out action=allow protocol=UDP localport=6001
   
   # Linux
   sudo ufw allow out 6001/udp
   ```

3. **Test Socket Directly**:
   ```python
   import socket
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   sock.sendto(b"Test", ("192.168.10.10", 6001))
   print("Sent test packet")
   ```

4. **Enable Socket Debug**:
   ```python
   # In swipe_detector.py
   try:
       self._udp_sock.sendto(b"Swipe", self._udp_target)
       print(f"UDP sent to {self._udp_target}")
   except Exception as e:
       print(f"UDP error: {e}")
   ```

### Issue 7: Launcher Can't Stop Worker

**Symptoms**:
- "Stop" button doesn't terminate worker
- Worker process remains after launcher exit

**Solutions**:

1. **Verify Stop Flag Path**:
   ```python
   # In TG25_Launcher.py
   print(f"Stop flag: {self.stop_file}")
   
   # In run_hand_tracking.py
   stop_file_path = os.environ.get("TG25_STOP_FILE", "")
   print(f"Monitoring stop flag: {stop_file_path}")
   ```

2. **Check File System Permissions**:
   ```bash
   # Ensure launcher can write to working directory
   touch TG25_STOP.flag && rm TG25_STOP.flag
   ```

3. **Force Termination** (as last resort):
   ```python
   # In TG25_Launcher.py - stop_worker()
   import psutil
   try:
       proc = psutil.Process(self.proc.pid)
       for child in proc.children(recursive=True):
           child.kill()
       proc.kill()
   except Exception as e:
       print(f"Force kill error: {e}")
   ```

---

## ⚠️ Known Issues

### 1. FPS Counter Anomaly
**Description**: Occasionally reports unrealistic FPS (>100,000)  
**Root Cause**: Rolling window calculation edge case when `elapsed < 1e-6`  
**Impact**: Visual only; doesn't affect performance  
**Workaround**: Use global average FPS instead  
**Status**: Low priority fix

### 2. Left Hand Detection Instability
**Description**: Left hand less reliable beyond 100 cm  
**Root Cause**: MediaPipe model bias toward right hand  
**Impact**: Reduced detection range for left hand  
**Workaround**: Use right hand for distant operations  
**Status**: Investigating alternative models

### 3. Background False Positives
**Description**: Clothing, hair, or ears occasionally trigger hand detection  
**Root Cause**: IR reflectance similarities  
**Impact**: Occasional false detections  
**Workaround**: Use depth filtering, stricter thresholds  
**Status**: Ongoing tuning

### 4. Depth Map Holes
**Description**: Depth unavailable in certain frame regions  
**Root Cause**: Insufficient texture in stereo images  
**Impact**: Some valid hands fail depth filtering  
**Workaround**: Lower `std_limit` tolerance  
**Status**: Hardware limitation

---

## 🗺️ Future Roadmap

### Phase 1: Gesture Expansion (Q2 2024)
- [ ] Finger count recognition (1-5)
- [ ] Static gestures (peace, thumbs-up, OK)
- [ ] Fist/palm open detection
- [ ] Dynamic gesture vocabulary

### Phase 2: Multi-Hand Support (Q3 2024)
- [ ] Simultaneous tracking of 2 hands
- [ ] Hand-hand interaction gestures
- [ ] Coordinated swipe detection

### Phase 3: Robustness Improvements (Q4 2024)
- [ ] Custom training dataset collection
- [ ] Fine-tuned MediaPipe models
- [ ] Adaptive thresholding based on environment
- [ ] Machine learning for false positive reduction

### Phase 4: Extended Features (2025)
- [ ] 3D gesture recognition using world landmarks
- [ ] Pose estimation integration
- [ ] Gesture macros and sequences
- [ ] Remote configuration via web UI

### Phase 5: Optimization (Ongoing)
- [ ] Fix FPS counter edge cases
- [ ] Reduce latency to <30ms
- [ ] GPU acceleration for preprocessing
- [ ] Custom neural network quantization

---

# 日本語版

## 🌟 プロジェクト概要

**TG_25_GestureOAK-D** は、**Luxonis OAK-D-PRO** カメラ向けに最適化されたリアルタイム手検出およびスワイプジェスチャー認識システムです。本システムは、赤外線（IR）ステレオカメラと深度センシングを活用し、厳しい照明条件下でも堅牢な手追跡を実現します。**80〜160 cm** の動作距離に特化して設計されています。

### 主な機能
- **IRベース手検出**: ステレオIRカメラによる暗所環境での動作
- **MediaPipe統合**: パーム検出およびハンドランドマークニューラルネットワークの採用
- **スワイプジェスチャー認識**: 速度と距離検証を伴う左から右へのスワイプ検出
- **UDP通信**: 外部システム（`192.168.10.10:6001`）へのスワイプ通知送信
- **深度フィルタリング**: ステレオ深度マップによる誤検出除去
- **スタンドアローン実行ファイル**: PyInstallerベースの`.exe`展開

---

## 📋 目次

1. [実装済み機能](#実装済み機能)
2. [技術アーキテクチャ](#技術アーキテクチャ-1)
3. [前提条件](#前提条件-1)
4. [環境セットアップ](#環境セットアップ-1)
5. [インストールガイド](#インストールガイド-1)
6. [アプリケーションアーキテクチャ](#アプリケーションアーキテクチャ-1)
7. [アプリケーション実行](#アプリケーション実行)
8. [実行ファイル（.exe）の取り扱い](#実行ファイルexeの取り扱い)
9. [実装の詳細](#実装の詳細)
10. [トラブルシューティング](#トラブルシューティング-1)
11. [既知の問題](#既知の問題-1)
12. [今後のロードマップ](#今後のロードマップ-1)

---

## 🎯 実装済み機能

### コアコンポーネント

#### 1. **OAK-Dカメラインターフェース** (`oak_camera.py`)
- **目的**: OAK-Dデバイス通信のためのハードウェア抽象化レイヤー
- **実装内容**:
  - デュアルモード対応（RGBおよびIRカメラ）
  - 動的解像度設定（デフォルト640×480）
  - フレームレート制御（目標30 FPS）
  - DepthAIパイプライン初期化
  - ノンブロッキングフレーム取得
- **技術詳細**:
  - `depthai` SDKによるデバイス通信
  - `ColorCamera` および `MonoCamera` ノードの実装
  - リアルタイムリサイズのための `ImageManip` ノード
  - ホスト・デバイス間データ転送用XLinkOutキュー

#### 2. **ハンド検出器** (`hand_detector.py`)
- **目的**: リアルタイム手検出およびランドマーク抽出
- **実装内容**:
  - **パーム検出ネットワーク**: SSDベースの手のひら位置特定（128×128入力）
  - **ハンドランドマークネットワーク**: 21点の手骨格抽出（224×224入力）
  - **後処理ネットワーク**: NMSおよびスコアフィルタリング
  - **スクリプトノード**: オンデバイス検出パイプラインのオーケストレーション
  - **IR強調処理**: 低照度条件向けCLAHE + バイラテラルフィルタリング
  - **深度ベースフィルタリング**: 距離対応分散許容（300〜2000 mm範囲）
- **技術パイプライン**:
  ```
  IRカメラ → ImageManip（リサイズ） → Palm NN → Postproc NN → 
  Landmark NN → Script（マネージャー） → ホストキュー
  ```
- **このアプローチを採用した理由**:
  - **IRカメラ** は暗所環境で一貫したパフォーマンスを提供
  - **オンデバイス処理** によりレイテンシ削減（ホスト側推論なし）
  - **深度フィルタリング** により背景物体からの誤検出を排除
  - **MediaPipeモデル** は手検出用の事前学習済み精度を提供

#### 3. **スワイプ検出器** (`swipe_detector.py`)
- **目的**: 堅牢な左から右へのスワイプジェスチャー認識
- **実装内容**:
  - **ステートマシン**: `IDLE → DETECTING → VALIDATING → CONFIRMED`
  - **軌跡バッファリング**: Dequeベース位置履歴（18フレーム）
  - **タイムスタンプベース速度**: FPS非依存速度計算
  - **多基準検証**:
    - 最小距離: 90ピクセル
    - 持続時間: 0.2〜2.0秒
    - 速度: 35〜900 px/s
    - Y軸偏差: X軸移動の≤35%
  - **クールダウンメカニズム**: 0.8秒デバウンスによる連続トリガー防止
  - **UDP通知**: ノンブロッキングソケット通信
- **このアプローチを採用した理由**:
  - **ステートマシン** により明確なジェスチャー進行追跡が可能
  - **速度ベース** 検出はFPS非依存（ハードウェア間で堅牢）
  - **多基準** 検証により誤検出を削減
  - **クールダウン** により連続動作中の急速な再トリガーを防止

#### 4. **アプリケーション**

##### ハンドトラッキングアプリ (`hand_tracking_app.py`)
- **目的**: スワイプ認識統合手検出
- **機能**:
  - リアルタイム手ランドマーク可視化
  - バウンディングボックス描画
  - 深度情報オーバーレイ
  - ジェスチャー分類（有効時）
  - スワイプ進行状況インジケーター
  - 統計表示（FPS、スワイプ回数、フィルタリングされた誤検出）
- **ユースケース**: 開発、デバッグ、デモンストレーション

##### モーションスワイプアプリ (`motion_swipe_app.py`)
- **目的**: 軌跡可視化を伴う特化型スワイプ検出
- **機能**:
  - スワイプトレイル描画（直近18位置）
  - 検出ゾーンオーバーレイ
  - 詳細進行状況メトリクス
  - 設定可能な感度プリセット
- **ユースケース**: スワイプジェスチャーチューニングおよび検証

##### スワイプ検出アプリ (`swipe_detection_app.py`)
- **目的**: 最小限のスワイプ専用インターフェース
- **機能**:
  - スワイプイベントに焦点を当てた軽量UI
  - リアルタイム状態表示
  - 本番環境向けパフォーマンス最適化
- **ユースケース**: 本番デプロイメント

#### 5. **ランチャーGUI** (`TG25_Launcher.py`)
- **目的**: ユーザーフレンドリーな実行ファイル管理
- **実装内容**:
  - Tkinterベースグラフィカルインターフェース
  - ワーカープロセス開始/停止制御
  - 停止フラグファイル経由のグレースフルシャットダウン
  - プロセス監視およびステータス表示
- **このアプローチを採用した理由**:
  - **別個のランチャー** によりメインアプリがUIをブロックするのを防止
  - **停止フラグファイル** により強制終了なしのクリーンシャットダウンが可能
  - **プロセスグループ管理** により適切なクリーンアップを保証

#### 6. **ユーティリティモジュール**

##### FPSカウンター (`FPS.py`)
- ローリングウィンドウFPS計算
- グローバル平均FPS追跡
- 経過時間測定

##### MediaPipe Utils (`mediapipe_utils.py`)
- 手領域データ構造
- ランドマーク座標変換
- ジェスチャー認識ロジック（指カウント）

##### テンプレートマネージャースクリプト (`template_manager_script_solo.py`)
- DepthAI Scriptノード用オンデバイススクリプトテンプレート
- パーム検出 → ランドマーク抽出パイプラインの調整
- NMS、領域回転、スコアフィルタリングの実装

---

## 🏗️ 技術アーキテクチャ

### システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                      ホストアプリケーション                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  main.py (メニューコントローラー)                        │ │
│  │    ├─ カメラテスト (oak_camera.py)                      │ │
│  │    ├─ ハンドトラッキングアプリ                        　 │ │
│  │    ├─ スワイプ検出アプリ                                │ │
│  │    └─ モーションスワイプアプリ                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  検出レイヤー                                           │ │
│  │    ├─ HandDetector (hand_detector.py)                  │ │
│  │    └─ SwipeDetector (swipe_detector.py) ─────UDP────┐  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                   DepthAIパイプライン
                              │
┌─────────────────────────────────────────────────────────────┐
│                      OAK-Dデバイス                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  IRモノカメラ (LEFT + RIGHT)                            │ │
│  │    ↓                ↓                                  │ │
│  │  StereoDepth   MonoCamera → ImageManip → RGB888p       │ │
│  │    ↓                ↓                                  │ │
│  │  depth_out      cam_out (XLinkOut)                     │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ニューラルネットワーク (VPU上)                          │ │
│  │    ├─ パーム検出NN (palm_detection_sh4.blob)            │ │
│  │    ├─ 後処理NN (PDPostProcessing_top2_sh1.blob)         │ │
│  │    └─ ランドマークNN (hand_landmark_lite_sh4.blob)      │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  スクリプトノード (マネージャー)                          │ │
│  │    ├─ NN実行の調整                                      │ │
│  │    ├─ NMSおよびフィルタリングの実装                       │ │
│  │    └─ マーシャルされた結果を出力 → manager_out            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### データフローシーケンス

```
1. IRカメラキャプチャ (400p @ 30fps)
   ↓
2. ImageManip → 640×480にリサイズ、RGB888pに変換
   ↓
3. IR強調処理 (CLAHE + バイラテラルフィルター)
   ↓
4. パーム検出NN → バウンディングボックス
   ↓
5. 後処理NN → NMS、上位2つの手
   ↓
6. ランドマークNN → 手ごとに21キーポイント
   ↓
7. スクリプトマネージャー → 結果をシリアライズ (marshal)
   ↓
8. ホストキュー → Python HandDetector.get_frame_and_hands()
   ↓
9. 深度フィルタリング → 手領域を検証 (300〜2000mm)
   ↓
10. SwipeDetector.update() → 軌跡分析
    ↓
11. スワイプ確認 → 192.168.10.10:6001へUDPパケット送信
```

---

## 📦 前提条件

### ハードウェア要件
- IRステレオ機能付き **OAK-D-PRO** または **OAK-D** カメラ
- USB 3.0ポート（最小; より高いスループットのためUSB 3.1+推奨）
- Windows 10/11（64ビット）またはLinux（Ubuntu 20.04+）

### ソフトウェア要件
- **Python 3.10〜3.12**（最新機能のため3.12推奨）
- **pip** パッケージマネージャー（最新バージョン）
- **Git**（リポジトリクローン用）
- **仮想環境サポート**（venvモジュール）

### システムライブラリ (Linux)
```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libusb-1.0-0-dev \
    libudev-dev \
    python3-dev \
    python3-pip \
    libopencv-dev
```

### システムライブラリ (Windows)
- **Visual C++ 再頒布可能パッケージ**（Pythonと共に自動インストール）
- **USB 3.0ドライバー**（通常組み込み; デバイスマネージャーで確認）

---

## 🔧 環境セットアップ

### ステップ1: リポジトリのクローン
```bash
git clone https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D.git
cd TG_25_GestureOAK-D
git checkout Hand-Gesture  # 正しいブランチにいることを確認
```

### ステップ2: 仮想環境の作成
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

**仮想環境を使う理由**:
- プロジェクトの依存関係をシステムPythonから分離
- 他のプロジェクトとのバージョン競合を防止
- 再現可能なビルドを実現

### ステップ3: pipのアップグレード
```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 📥 インストールガイド

### 依存関係のインストール
```bash
pip install -r requirements.txt
```

**依存関係の内訳**:
- **depthai** (≥2.24.0): デバイス通信およびパイプライン管理のためのOAK-D SDK
- **opencv-python** (≥4.8.0): フレーム処理および表示用コンピュータビジョンライブラリ
- **numpy** (≥1.24.0): ランドマーク変換用数値演算
- **mediapipe** (≥0.10.0): 事前学習済み手検出モデル（参照用オプション）
- **imutils** (≥0.5.4): OpenCV操作用便利関数
- **pyyaml** (≥6.0): 設定ファイルパース（YAML設定使用時）

### インストール検証
```bash
# Pythonバージョン確認
python --version  # 3.10.x または 3.11.x または 3.12.x と表示されるべき

# インストール済みパッケージ確認
pip list | grep -E "depthai|opencv|numpy"

# OAK-D接続テスト
python -c "import depthai as dai; print(dai.__version__); print(dai.Device.getAllAvailableDevices())"
```

**期待される出力**:
```
2.24.0.0  # またはそれ以降
[<depthai.DeviceInfo ...>]  # OAK-Dデバイスがリストされるべき
```

### 環境検証スクリプト
```bash
python probe_dai.py
```

**チェック内容**:
- DepthAIライブラリのインポート
- OAK-Dデバイスの列挙
- USB接続速度
- カメラセンサーの可用性

---

## 🏛️ アプリケーションアーキテクチャ

### モジュール構造
```
src/gesture_oak/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── oak_camera.py          # カメラ抽象化レイヤー
├── detection/
│   ├── __init__.py
│   ├── hand_detector.py       # MediaPipeベース手検出
│   ├── motion_detector.py     # モーションベース検出（代替）
│   ├── motion_swipe_detector.py
│   └── swipe_detector.py      # ジェスチャー認識ロジック
├── logic/
│   ├── __init__.py
│   └── gesture_classifier.py # 指カウント、ジェスチャー
├── apps/
│   ├── __init__.py
│   ├── hand_tracking_app.py   # メインハンドトラッキングデモ
│   ├── swipe_detection_app.py # スワイプ専用デモ
│   └── motion_swipe_app.py    # モーションベーススワイプデモ
└── utils/
    ├── FPS.py                 # FPS測定
    ├── mediapipe_utils.py     # データ構造、ヘルパー
    └── template_manager_script_solo.py  # オンデバイススクリプト
```

### エントリーポイント
1. **main.py**: 異なるアプリケーション実行用インタラクティブメニュー
2. **run_hand_tracking.py**: ハンドトラッキングアプリの直接実行
3. **TG25_Launcher.py**: 実行ファイル管理用GUIランチャー
4. **probe_dai.py**: デバイス診断ツール

---

## 🚀 アプリケーション実行

### 方法1: インタラクティブメニュー（推奨）
```bash
python main.py
```

**メニューオプション**:
```
1. Test camera connection        # OAK-Dセットアップ検証
2. Run hand tracking app         # フル機能手検出 + スワイプ
3. Run swipe detection app       # スワイプ専用インターフェース
4. Run motion-based swipe        # 代替モーション検出
5. Exit
```

**使用場面**:
- **オプション1**: 初回セットアップ、接続問題のトラブルシューティング
- **オプション2**: 開発、デバッグ、デモンストレーション
- **オプション3**: 本番スワイプ検出
- **オプション4**: 実験的モーションベースアプローチ

### 方法2: 直接実行
```bash
# スワイプ検出付きハンドトラッキング
python run_hand_tracking.py

# またはモジュール経由
python -m gesture_oak.apps.hand_tracking_app
```

### 方法3: UV使用（高速Pythonパッケージマネージャー）
```bash
# uvがまだインストールされていない場合
pip install uv

# uvで実行（高速起動）
uv run python main.py
```

### ランタイム制御

#### キーボードショートカット
- **`q`**: アプリケーション終了
- **`s`**: 現在のフレームをディスクに保存（JPEG形式）
- **`r`**: スワイプ統計をリセット

#### コンソール出力
```
OAK-D Hand Detection Demo with Swipe Detection
=============================================
Press 'q' to quit
Press 's' to save current frame
Press 'r' to reset swipe statistics

Connected to device: OAK-D-PRO
USB Speed: SUPER_SPEED
Hand detection started. Showing live preview...

Hand 1: right (confidence: 0.957)
  Gesture: FIVE
 LEFT-TO-RIGHT SWIPE DETECTED! (Total: 1)
```

### 期待されるパフォーマンス
- **FPS**: 25〜30 FPS（OAK-D-PRO IRモード）
- **レイテンシ**: 〜50〜80ms（カメラから表示まで）
- **検出範囲**: 80〜160 cm 最適、40〜200 cm 拡張
- **スワイプ成功率**: 良好な条件下で〜95%

---

## 🗂️ 実行ファイル（.exe）の取り扱い

### 実行ファイルのビルド

プロジェクトは **PyInstaller** を使用してWindows展開用スタンドアローン実行ファイルを作成します。

#### ビルド前提条件
```bash
pip install pyinstaller
```

#### ビルドスクリプト

##### 1. ハンドトラッキングワーカーのビルド
```bash
# Windows
build.bat

# または手動で
pyinstaller run_hand_tracking.spec
```

**Specファイルハイライト** (`run_hand_tracking.spec`):
```python
a = Analysis(
    ['run_hand_tracking.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),  # ニューラルネットワークblobをバンドル
        ('src/gesture_oak/utils/template_manager_script_solo.py',
         'src/gesture_oak/utils'),
    ],
    hiddenimports=[
        'depthai',
        'cv2',
        'numpy',
        # ... すべての依存関係
    ],
    ...
)
```

##### 2. ランチャーGUIのビルド
```bash
pyinstaller TG25_Launcher.spec
```

##### 3. 診断ツールのビルド
```bash
pyinstaller probe_dai.spec
```

### 実行ファイルの配布

ビルド後、`dist/` フォルダには以下が含まれます:
```
dist/
├── TG25_HandTracking/
│   ├── TG25_HandTracking.exe  # メインワーカーアプリケーション
│   ├── models/                # ニューラルネットワークblob
│   └── [依存関係]             # DLL、ライブラリ
├── TG25_Launcher.exe          # GUIランチャー
└── probe_dai.exe              # 診断ツール
```

### 展開手順

1. **フォルダ全体をコピー** してターゲットマシンへ:
   ```
   TG25_GestureOAK-D_Deployment/
   ├── TG25_Launcher.exe
   ├── TG25_HandTracking.exe
   ├── models/
   └── (dist/からのすべてのDLL)
   ```

2. **`TG25_Launcher.exe` を実行** してGUIを起動

3. **「Start Hand Tracking」をクリック** してワーカーを起動

### 実行ファイルのトラブルシューティング

#### 問題: 「modelsフォルダが見つかりません」
**解決策**: `models/` が `.exe` と同じディレクトリにあることを確認

#### 問題: 「Import Error: No module named 'depthai'」
**解決策**: PyInstallerが依存関係を見逃した可能性。`.spec` ファイルの `hiddenimports` に追加:
```python
hiddenimports=[
    'depthai',
    'depthai._version',  # サブモジュールを追加
    ...
]
```

#### 問題: ワーカーがグレースフルに停止しない
**解決策**: 停止フラグファイルパスを確認:
```python
# TG25_Launcher.py内
stop_file = Path(_exe_dir()) / "TG25_STOP.flag"

# run_hand_tracking.py内
stop_file_path = os.environ.get("TG25_STOP_FILE", "")
```

---

## 🛠️ 実装の詳細

### 手検出パイプライン

#### 1. カメラ初期化
```python
# hand_detector.py
cam_left = pipeline.createMonoCamera()
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setFps(30)

cam_right = pipeline.createMonoCamera()
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setFps(30)
```

**IRカメラを使う理由**:
- 低照度で一貫したパフォーマンス
- 色変化（肌色、衣服）の影響を受けにくい
- ネイティブ400p解像度がMediaPipe入力要件と一致

#### 2. ステレオ深度設定
```python
depth = pipeline.createStereoDepth()
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(True)    # 無効な視差を排除
depth.setSubpixel(True)          # 深度精度を向上
```

**技術説明**:
- **HIGH_DENSITY**: 小物体（手）のある屋内シーン用に最適化
- **7×7メディアンフィルター**: 深度マップからスペックルノイズを除去
- **Left-Right Check**: ステレオペア間の視差一貫性を検証
- **Subpixel**: より滑らかな深度のために分数視差値を有効化

#### 3. IRフレーム強調処理
```python
def enhance_ir_frame(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # CLAHE（コントラスト制限適応ヒストグラム均等化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # バイラテラルフィルター（エッジ保存平滑化）
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
```

**このアプローチを採用した理由**:
- **CLAHE**: ノイズを過度に増幅せずに局所コントラストを強調
- **clipLimit=2.0**: 均質領域での過度の増幅を防止
- **tileGridSize=(8,8)**: 局所と全体のコントラストのバランス
- **バイラテラルフィルター**: 手のエッジを保持しながらノイズを平滑化

#### 4. 深度ベースフィルタリング
```python
def filter_hands_by_depth(self, hands, depth_frame):
    for hand in hands:
        cx, cy = hand.rect_x_center_a, hand.rect_y_center_a
        d_center = depth_frame[cy, cx]
        
        # 有効範囲: 300〜2000mm（30cm - 2m）
        if not (300 <= d_center <= 2000):
            continue
        
        # 手中心周辺の関心領域
        half = 18 if d_center < 1000 else 26  # 遠い手には大きいROI
        roi = depth_frame[cy-half:cy+half, cx-half:cx+half]
        
        avg = np.mean(roi[roi > 0])
        std = np.std(roi[roi > 0])
        
        # 距離対応許容値
        std_limit = 80.0 + 0.08 * max(0.0, (avg - 800.0))
        if std <= std_limit:
            hand.depth = avg
            hand.depth_confidence = 1.0 - (std / std_limit)
```

**技術的根拠**:
- **可変std_limit**: 遠い手は深度精度が低い
- **ROIサイズスケーリング**: 異なる距離での手の見かけサイズに適応
- **信頼度スコア**: 深度測定の信頼性を定量化

### スワイプ検出アルゴリズム

#### 状態遷移ロジック
```python
IDLE:
    - タイムスタンプ付き手位置をバッファリング
    - 一貫した右方向動作をチェック（連続3フレーム）
    - 動作検出時にDETECTINGに遷移

DETECTING:
    - 軌跡のバッファリングを継続
    - 累積距離を計算
    - 持続時間 > max_duration（2.0秒）でタイムアウト
    - 左への動きで中止
    - 距離 >= min_distance（90px）時にVALIDATINGに遷移

VALIDATING:
    - 持続時間が[0.2秒、2.0秒]内であることを検証
    - 平均速度を計算: 距離 / 持続時間
    - 速度範囲[35 px/s、900 px/s]をチェック
    - Y軸偏差 ≤ X移動の35%を検証
    - 有意な後退動作がないことを確認（< -12pxジャンプ）
    - すべてのチェックが通過した場合CONFIRMEDに遷移

CONFIRMED:
    - スワイプカウンターを増分
    - 192.168.10.10:6001へUDPパケット「Swipe」を送信
    - クールダウン（0.8秒）を適用して急速な再トリガーを防止
    - IDLEに戻る
```

#### 速度計算
```python
# FPS非依存速度（ピクセル毎秒）
times = np.array(self.time_buffer)
poses = np.array(self.position_buffer)

total_dx = poses[-1, 0] - poses[0, 0]  # 水平変位
duration = times[-1] - times[0]        # 経過時間
velocity = total_dx / duration         # px/s
```

**タイムスタンプベースを使う理由**:
- フレームレート変動に依存しない
- ドロップフレームを適切に処理
- 正確な速度測定を提供

---

## 🐛 トラブルシューティング

### 問題1: カメラが検出されない

**症状**:
```
Failed to connect to OAK-D: [X_LINK_DEVICE_NOT_FOUND]
```

**解決策**:

1. **USB接続を確認**:
   ```bash
   # Linux
   lsusb | grep "03e7"  # Luxonis VID
   
   # Windows（デバイスマネージャー）
   # USBデバイス下で「Movidius MyriadX」または「OAK-D」を探す
   ```

2. **USBドライバーを更新**（Windows）:
   - Zadigをダウンロード: https://zadig.akeo.ie/
   - ドライバーをWinUSBに置き換え

3. **USB権限を付与**（Linux）:
   ```bash
   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

4. **診断ツールでテスト**:
   ```bash
   python probe_dai.py
   ```

### 問題2: 低FPS / カクつき

**症状**:
- フレームレートが20 FPS未満に低下
- カメラプレビューがカクカク

**解決策**:

1. **USB速度を確認**:
   ```python
   # USB3 SUPER_SPEEDまたはSUPER_PLUSと表示されるべき
   device.getUsbSpeed()
   ```

2. **キューサイズを削減**（バッファリングしている場合）:
   ```python
   # hand_detector.py内
   self.q_video = self.device.getOutputQueue(
       name="cam_out", 
       maxSize=2,  # 4から2に削減
       blocking=False
   )
   ```

3. **IR強調処理を無効化**（テスト用）:
   ```python
   # hand_detector.py内でコメントアウト
   # frame = self.enhance_ir_frame(raw_frame)
   frame = raw_frame
   ```

4. **OpenCVを最適化**:
   ```python
   import cv2
   cv2.setUseOptimized(True)
   cv2.setNumThreads(0)  # 最適なスレッド数を自動検出
   ```

### 問題3: 誤検出スワイプ

**症状**:
- 手の動きなしにスワイプがトリガーされる
- ランダムな物体が検出を引き起こす

**解決策**:

1. **最小距離を増やす**:
   ```python
   swipe_detector = SwipeDetector(
       min_distance=120,  # 90から増加
   )
   ```

2. **より厳しいY偏差**:
   ```python
   swipe_detector = SwipeDetector(
       max_y_deviation=0.25,  # 0.35から削減
   )
   ```

3. **速度範囲を狭める**:
   ```python
   swipe_detector = SwipeDetector(
       min_velocity=50,   # 35から増加
       max_velocity=700,  # 900から削減
   )
   ```

4. **より厳しい深度フィルタリングを有効化**:
   ```python
   # hand_detector.py内
   std_limit = 60.0 + 0.06 * max(0.0, (avg - 800.0))  # 許容値を削減
   ```

### 問題4: 距離で手が検出されない

**症状**:
- 画面上に手が見えるが100+ cmで検出されない
- 近距離（<80 cm）でのみ動作

**解決策**:

1. **パーム検出閾値を下げる**:
   ```python
   detector = HandDetector(
       pd_score_thresh=0.08,  # 0.10から削減
   )
   ```

2. **バッファサイズを増やす**（履歴を増やす）:
   ```python
   swipe_detector = SwipeDetector(
       buffer_size=24,  # 18から増加
   )
   ```

3. **深度範囲を調整**:
   ```python
   # hand_detector.py - filter_hands_by_depth()内
   if not (200 <= d_center <= 2500):  # 範囲を拡張
   ```

4. **IR照明を確認**:
   - OAK-D-PROはアクティブIRエミッターを持つ
   - IR LEDが見えるか確認（スマホカメラを使用）

### 問題5: 実行ファイルで「ImportError」

**症状**:
```
ImportError: No module named 'depthai'
ModuleNotFoundError: No module named 'cv2'
```

**解決策**:

1. **すべての隠しインポートで再ビルド**:
   ```python
   # .specファイル内
   hiddenimports=[
       'depthai',
       'cv2',
       'numpy',
       'numpy.core',
       'numpy.core._multiarray_umath',
       'mediapipe',
       'marshal',
       'collections',
       'collections.abc',
       'socket',
       'time',
       'pathlib',
   ]
   ```

2. **バイナリ依存関係をバンドル**:
   ```python
   # .specファイル内
   binaries=[
       (r'C:\path\to\.venv\Lib\site-packages\depthai\*.dll', 'depthai'),
   ]
   ```

3. **PyInstallerフックを使用**:
   ```bash
   pip install pyinstaller-hooks-contrib
   pyinstaller --additional-hooks-dir=. run_hand_tracking.spec
   ```

### 問題6: UDPパケットが受信されない

**症状**:
- スワイプ検出されたがネットワークトラフィックなし
- ターゲットシステムが「Swipe」メッセージを受信しない

**解決策**:

1. **ネットワーク構成を確認**:
   ```bash
   # UDP接続をテスト
   # ターゲット（192.168.10.10）で:
   nc -u -l 6001
   
   # ソースマシンで:
   echo "test" | nc -u 192.168.10.10 6001
   ```

2. **ファイアウォールルールを確認**:
   ```bash
   # Windows
   netsh advfirewall firewall add rule name="Allow UDP 6001" dir=out action=allow protocol=UDP localport=6001
   
   # Linux
   sudo ufw allow out 6001/udp
   ```

3. **ソケットを直接テスト**:
   ```python
   import socket
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   sock.sendto(b"Test", ("192.168.10.10", 6001))
   print("テストパケット送信")
   ```

4. **ソケットデバッグを有効化**:
   ```python
   # swipe_detector.py内
   try:
       self._udp_sock.sendto(b"Swipe", self._udp_target)
       print(f"UDP送信: {self._udp_target}")
   except Exception as e:
       print(f"UDPエラー: {e}")
   ```

### 問題7: ランチャーがワーカーを停止できない

**症状**:
- 「停止」ボタンがワーカーを終了しない
- ランチャー終了後もワーカープロセスが残る

**解決策**:

1. **停止フラグパスを検証**:
   ```python
   # TG25_Launcher.py内
   print(f"停止フラグ: {self.stop_file}")
   
   # run_hand_tracking.py内
   stop_file_path = os.environ.get("TG25_STOP_FILE", "")
   print(f"停止フラグ監視中: {stop_file_path}")
   ```

2. **ファイルシステム権限を確認**:
   ```bash
   # ランチャーが作業ディレクトリに書き込めることを確認
   touch TG25_STOP.flag && rm TG25_STOP.flag
   ```

3. **強制終了**（最後の手段として）:
   ```python
   # TG25_Launcher.py - stop_worker()内
   import psutil
   try:
       proc = psutil.Process(self.proc.pid)
       for child in proc.children(recursive=True):
           child.kill()
       proc.kill()
   except Exception as e:
       print(f"強制終了エラー: {e}")
   ```

---

## ⚠️ 既知の問題

### 1. FPSカウンター異常
**説明**: 時々非現実的なFPS（>100,000）を報告  
**根本原因**: `elapsed < 1e-6` 時のローリングウィンドウ計算エッジケース  
**影響**: 視覚的のみ; パフォーマンスには影響なし  
**回避策**: 代わりにグローバル平均FPSを使用  
**ステータス**: 低優先度修正

### 2. 左手検出の不安定性
**説明**: 100 cm以上では左手の信頼性が低い  
**根本原因**: MediaPipeモデルの右手へのバイアス  
**影響**: 左手の検出範囲が縮小  
**回避策**: 遠距離操作には右手を使用  
**ステータス**: 代替モデルを調査中

### 3. 背景誤検出
**説明**: 衣服、髪、または耳が時々手検出をトリガー  
**根本原因**: IR反射率の類似性  
**影響**: 時折の誤検出  
**回避策**: 深度フィルタリング、より厳しい閾値を使用  
**ステータス**: 継続的なチューニング中

### 4. 深度マップの穴
**説明**: フレームの特定領域で深度が利用不可  
**根本原因**: ステレオ画像のテクスチャ不足  
**影響**: 一部の有効な手が深度フィルタリングで失敗  
**回避策**: `std_limit` 許容値を下げる  
**ステータス**: ハードウェア制限

---

## 🗺️ 今後のロードマップ

### フェーズ1: ジェスチャー拡張（2024年Q2）
- [ ] 指カウント認識（1-5）
- [ ] 静的ジェスチャー（ピース、サムズアップ、OK）
- [ ] 拳/手のひら開閉検出
- [ ] 動的ジェスチャー語彙

### フェーズ2: 両手サポート（2024年Q3）
- [ ] 2つの手の同時追跡
- [ ] 手-手インタラクションジェスチャー
- [ ] 協調スワイプ検出

### フェーズ3: 堅牢性向上（2024年Q4）
- [ ] カスタムトレーニングデータセット収集
- [ ] 微調整されたMediaPipeモデル
- [ ] 環境ベースの適応閾値設定
- [ ] 誤検出削減のための機械学習

### フェーズ4: 拡張機能（2025年）
- [ ] ワールドランドマークを使用した3Dジェスチャー認識
- [ ] ポーズ推定統合
- [ ] ジェスチャーマクロとシーケンス
- [ ] Web UI経由のリモート設定

### フェーズ5: 最適化（継続中）
- [ ] FPSカウンターエッジケースの修正
- [ ] レイテンシを<30msに削減
- [ ] 前処理のためのGPUアクセラレーション
- [ ] カスタムニューラルネットワーク量子化

---

## 📜 ライセンス

MIT License

---

## 👥 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

---

## 📞 サポート

問題が発生した場合:
1. [トラブルシューティングセクション](#トラブルシューティング-1)を確認
2. [既知の問題](#既知の問題-1)を確認
3. GitHubでissueを作成
4. 詳細ログとシステム情報を含める

---

**プロジェクト維持**: ShoumikMahbubRidoy  
**リポジトリ**: https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D  
**ブランチ**: Hand-Gesture