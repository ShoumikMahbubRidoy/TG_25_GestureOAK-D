# Implementation Tasks / 実装タスク

**English** | [日本語](#日本語版-1)

---

## Overview

This document outlines the step-by-step implementation tasks for setting up and deploying the TG_25_GestureOAK-D hand detection system.

---

## Phase 1: Environment Setup

### Task 1.1: System Prerequisites Installation

**Objective**: Install required system dependencies

**Steps (Windows)**:
```powershell
# 1. Install Python 3.10 or later
# Download from: https://www.python.org/downloads/
# Ensure "Add Python to PATH" is checked during installation

# 2. Verify Python installation
python --version
# Expected: Python 3.10.x or 3.11.x or 3.12.x

# 3. Install Git
# Download from: https://git-scm.com/download/win

# 4. Verify Git installation
git --version
```

**Steps (Linux - Ubuntu/Debian)**:
```bash
# 1. Update package lists
sudo apt-get update

# 2. Install Python 3.10+
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# 3. Install system dependencies
sudo apt-get install -y \
    libusb-1.0-0-dev \
    libudev-dev \
    python3-dev \
    libopencv-dev \
    git

# 4. Configure udev rules for OAK-D
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# 5. Verify installations
python3 --version
git --version
```

**Validation**:
- [ ] Python 3.10+ installed
- [ ] Git installed
- [ ] USB permissions configured (Linux)

---

### Task 1.2: Repository Clone and Branch Selection

**Objective**: Obtain project source code

**Steps**:
```bash
# 1. Clone repository
git clone https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D.git

# 2. Navigate to project directory
cd TG_25_GestureOAK-D

# 3. Checkout correct branch
git checkout Hand-Gesture

# 4. Verify branch
git branch
# Expected: * Hand-Gesture

# 5. Verify repository structure
ls -la
# Should see: main.py, requirements.txt, src/, models/, etc.
```

**Validation**:
- [ ] Repository cloned successfully
- [ ] On `Hand-Gesture` branch
- [ ] All files present (main.py, src/, models/)

---

### Task 1.3: Virtual Environment Creation

**Objective**: Create isolated Python environment

**Steps (Windows)**:
```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Verify activation (prompt should show (.venv))
where python
# Should point to .venv\Scripts\python.exe

# 4. Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

**Steps (Linux/Mac)**:
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Verify activation
which python
# Should point to .venv/bin/python

# 4. Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

**Validation**:
- [ ] Virtual environment created (.venv/ directory exists)
- [ ] Virtual environment activated (prompt shows (.venv))
- [ ] pip upgraded to latest version

---

### Task 1.4: Dependency Installation

**Objective**: Install all required Python packages

**Steps**:
```bash
# 1. Ensure virtual environment is activated
# Prompt should show (.venv)

# 2. Install dependencies from requirements.txt
pip install -r requirements.txt

# 3. Verify installations
pip list

# 4. Check critical packages
pip show depthai
pip show opencv-python
pip show numpy

# 5. Test imports
python -c "import depthai as dai; print(f'DepthAI version: {dai.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
```

**Expected Output**:
```
DepthAI version: 2.24.0.0 (or later)
OpenCV version: 4.8.x (or later)
NumPy version: 1.24.x (or later)
```

**Validation**:
- [ ] All packages installed without errors
- [ ] depthai version ≥ 2.24.0
- [ ] opencv-python version ≥ 4.8.0
- [ ] All imports successful

---

### Task 1.5: Hardware Connection Verification

**Objective**: Ensure OAK-D camera is properly connected

**Steps**:
```bash
# 1. Connect OAK-D camera to USB 3.0 port
# Use blue USB port if available (indicates USB 3.0)

# 2. Verify device enumeration (Linux)
lsusb | grep "03e7"
# Expected: Movidius MyriadX device

# 3. Verify device enumeration (Windows)
# Open Device Manager → Universal Serial Bus controllers
# Look for "Movidius MyriadX" or "OAK-D"

# 4. Run diagnostic tool
python probe_dai.py
```

**Expected Output**:
```
Testing DepthAI connection...
✓ DepthAI library imported successfully
✓ Found 1 OAK device(s)
Device: OAK-D-PRO
USB Speed: SUPER_SPEED
Connected successfully!
```

**Validation**:
- [ ] OAK-D detected by system
- [ ] USB 3.0 connection confirmed (SUPER_SPEED)
- [ ] probe_dai.py runs without errors

---

## Phase 2: Application Testing

### Task 2.1: Camera Connection Test

**Objective**: Verify camera pipeline initialization

**Steps**:
```bash
# 1. Run main application
python main.py

# 2. Select option 1 (Test camera connection)
# Input: 1

# 3. Observe output
```

**Expected Output**:
```
Testing OAK-D camera connection...

--- Testing RGB Camera ---
Connected to OAK-D device: OAK-D-PRO
Camera mode: RGB
USB Speed: SUPER_SPEED
✓ Successfully connected to RGB camera

Device Information:
  name: OAK-D-PRO
  mxid: 14442C10D13xxxxx
  usb_speed: UsbSpeed.SUPER
  platform: Platform.MYRIAD_X

Testing frame capture...
✓ Frame captured successfully - Shape: (480, 640, 3)

✓ OAK-D camera setup completed successfully!
Ready to proceed with gesture recognition implementation.
```

**Troubleshooting**:
- If RGB camera fails, system will automatically test IR camera
- IR camera output will show "Camera mode: IR"

**Validation**:
- [ ] Camera connects successfully
- [ ] Device information displayed
- [ ] Frame captured (shape: 480, 640, 3)

---

### Task 2.2: Hand Tracking Application Test

**Objective**: Test integrated hand detection and swipe recognition

**Steps**:
```bash
# 1. Run main application
python main.py

# 2. Select option 2 (Run hand tracking app)
# Input: 2

# 3. Position hand 80-160 cm from camera
# 4. Move hand left to right in horizontal motion
# 5. Observe detection and swipe events
# 6. Press 'q' to quit
```

**Expected Behavior**:
- OpenCV window opens showing camera feed
- Hand landmarks drawn as green circles
- Bounding box around detected hand (cyan)
- Hand label and confidence score displayed
- Swipe progress indicator shown during motion
- "SWIPE DETECTED!" message on successful swipe
- Console logs hand detections and swipe events

**Console Output Example**:
```
OAK-D Hand Detection Demo with Swipe Detection
=============================================
Connected to device: OAK-D-PRO
USB Speed: SUPER_SPEED
Hand detection started. Showing live preview...

Hand 1: right (confidence: 0.957)
  Gesture: FIVE
 LEFT-TO-RIGHT SWIPE DETECTED! (Total: 1)
Hand 1: right (confidence: 0.943)
Hand 1: right (confidence: 0.921)
```

**Validation**:
- [ ] Hand landmarks visible
- [ ] Bounding box tracks hand movement
- [ ] Swipe detection triggers correctly
- [ ] FPS stable (25-30 fps)
- [ ] UDP packet sent (check target system)

---

### Task 2.3: Swipe Detection Application Test

**Objective**: Test specialized swipe detection interface

**Steps**:
```bash
# 1. Run main application
python main.py

# 2. Select option 3 (Run swipe detection app)
# Input: 3

# 3. Perform multiple swipes
# 4. Observe trajectory visualization
# 5. Press 'r' to reset statistics
# 6. Press 'q' to quit
```

**Expected Features**:
- Swipe trail visualization (blue line)
- Detection zone overlay
- Detailed progress metrics (distance, velocity, duration)
- Real-time state display (IDLE/DETECTING/VALIDATING)
- Statistics panel (total swipes, false positives filtered)

**Validation**:
- [ ] Swipe trail renders correctly
- [ ] Progress metrics update in real-time
- [ ] Statistics accurate
- [ ] Reset function works

---

## Phase 3: Parameter Tuning

### Task 3.1: Detection Sensitivity Adjustment

**Objective**: Optimize hand detection for your environment

**Location**: `src/gesture_oak/detection/hand_detector.py`

**Parameters to Adjust**:

```python
# Line ~16-27: HandDetector initialization
detector = HandDetector(
    fps=30,                    # Frame rate: 15-60
    resolution=(640, 480),     # Resolution: (320,240) to (1920,1080)
    pd_score_thresh=0.10,      # Palm detection threshold: 0.05-0.30
    pd_nms_thresh=0.3,         # NMS threshold: 0.2-0.5
    use_gesture=True,          # Enable gesture recognition
    use_rgb=False              # False=IR, True=RGB
)
```

**Tuning Guidelines**:
- **Lower `pd_score_thresh`** (e.g., 0.08): Increases sensitivity, more detections at distance
- **Higher `pd_score_thresh`** (e.g., 0.15): Reduces false positives, requires closer/clearer hands
- **Lower `fps`** (e.g., 20): Reduces CPU usage, may improve stability
- **use_rgb=True**: Better color accuracy, worse in low light
- **use_rgb=False**: Better in dark, consistent IR performance

**Testing Procedure**:
1. Modify parameter in source code
2. Save file
3. Run application: `python run_hand_tracking.py`
4. Test at various distances (80cm, 120cm, 160cm)
5. Document results
6. Iterate until optimal

**Validation**:
- [ ] Consistent detection at target distance range
- [ ] Acceptable false positive rate
- [ ] Stable FPS

---

### Task 3.2: Swipe Detection Tuning

**Objective**: Adjust swipe gesture parameters

**Location**: `src/gesture_oak/detection/swipe_detector.py`

**Parameters to Adjust**:

```python
# Line ~28-37: SwipeDetector initialization
swipe_detector = SwipeDetector(
    buffer_size=18,            # Trajectory history: 5-30 frames
    min_distance=90,           # Minimum travel: 30-200 pixels
    min_duration=0.20,         # Minimum time: 0.1-0.5 seconds
    max_duration=2.00,         # Maximum time: 1.0-5.0 seconds
    min_velocity=35,           # Min speed: 10-100 px/s
    max_velocity=900,          # Max speed: 300-2000 px/s
    max_y_deviation=0.35,      # Vertical tolerance: 0.1-0.5
    cooldown=0.80,             # Re-trigger delay: 0.3-2.0 seconds
)
```

**Tuning Scenarios**:

**Scenario A: Too Many False Positives**
```python
min_distance=120,          # Increase from 90
min_velocity=50,           # Increase from 35
max_y_deviation=0.25,      # Reduce from 0.35
```

**Scenario B: Missing Valid Swipes**
```python
min_distance=70,           # Decrease from 90
min_velocity=25,           # Decrease from 35
max_duration=3.0,          # Increase from 2.0
```

**Scenario C: Rapid Swipes (Short Distance)**
```python
min_distance=60,
max_duration=1.0,
min_velocity=50,
```

**Validation**:
- [ ] Swipe detection reliable and consistent
- [ ] False positive rate acceptable
- [ ] Cooldown prevents rapid re-triggering

---

### Task 3.3: Depth Filtering Optimization

**Objective**: Adjust depth-based hand validation

**Location**: `src/gesture_oak/detection/hand_detector.py`, `filter_hands_by_depth()` method

**Key Parameters**:

```python
# Line ~340-370: Depth filtering logic

# Valid depth range (mm)
if not (300 <= d_center <= 2000):
    continue

# ROI size (distance-dependent)
half = 18 if d_center < 1000 else 26

# Variance tolerance (distance-aware)
std_limit = 80.0 + 0.08 * max(0.0, (avg - 800.0))
```

**Tuning Options**:

**Option 1: Extend Detection Range**
```python
# Allow farther hands
if not (200 <= d_center <= 2500):
    continue
```

**Option 2: Stricter Filtering (Reduce False Positives)**
```python
# Tighter variance tolerance
std_limit = 60.0 + 0.06 * max(0.0, (avg - 800.0))
```

**Option 3: Larger ROI for Distant Hands**
```python
# More samples for stability
half = 22 if d_center < 1000 else 32
```

**Validation**:
- [ ] Detection range meets requirements
- [ ] Background objects filtered effectively
- [ ] Depth confidence scores reasonable (>0.7 for valid hands)

---

## Phase 4: Executable Building

### Task 4.1: PyInstaller Installation and Configuration

**Objective**: Prepare for executable creation

**Steps**:
```bash
# 1. Ensure virtual environment is activated
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows

# 2. Install PyInstaller
pip install pyinstaller

# 3. Verify installation
pyinstaller --version
# Expected: 5.x or later

# 4. Review spec files
ls -la *.spec
# Should see:
#   - TG25_HandTracking.spec
#   - TG25_Launcher.spec
#   - probe_dai.spec
```

**Validation**:
- [ ] PyInstaller installed
- [ ] Spec files present

---

### Task 4.2: Build Hand Tracking Worker Executable

**Objective**: Create standalone worker application

**Steps**:
```bash
# 1. Build using spec file
pyinstaller TG25_HandTracking.spec

# 2. Verify build artifacts
ls dist/TG25_HandTracking/
# Should contain:
#   - TG25_HandTracking.exe
#   - models/ folder with .blob files
#   - DLLs and dependencies

# 3. Test executable
cd dist/TG25_HandTracking
./TG25_HandTracking.exe  # Linux/Mac
TG25_HandTracking.exe    # Windows

# 4. Verify functionality
# Should launch hand tracking application
```

**Common Build Issues**:

**Issue**: Missing models folder
```python
# Fix in TG25_HandTracking.spec
datas=[
    ('models', 'models'),  # Ensure this line exists
    ('src/gesture_oak/utils/template_manager_script_solo.py',
     'src/gesture_oak/utils'),
]
```

**Issue**: Import errors
```python
# Add to hiddenimports in .spec file
hiddenimports=[
    'depthai',
    'cv2',
    'numpy',
    'numpy.core._multiarray_umath',
    'collections.abc',
    # Add any missing modules here
]
```

**Validation**:
- [ ] Executable built successfully
- [ ] All dependencies included
- [ ] Models folder bundled
- [ ] Executable runs without errors

---

### Task 4.3: Build Launcher GUI Executable

**Objective**: Create launcher application

**Steps**:
```bash
# 1. Build launcher
pyinstaller TG25_Launcher.spec

# 2. Verify output
ls dist/
# Should see: TG25_Launcher.exe

# 3. Test launcher
cd dist
./TG25_Launcher.exe  # Linux/Mac
TG25_Launcher.exe    # Windows

# 4. Verify GUI appears
# Should show:
#   - "Start Hand Tracking" button
#   - "Stop" button (disabled)
#   - Status label
```

**Validation**:
- [ ] Launcher executable created
- [ ] GUI displays correctly
- [ ] Can start/stop worker (if worker exe present)

---

### Task 4.4: Build Diagnostic Tool Executable

**Objective**: Create standalone diagnostic utility

**Steps**:
```bash
# 1. Build probe tool
pyinstaller probe_dai.spec

# 2. Verify output
ls dist/
# Should see: probe_dai.exe

# 3. Test diagnostic
./dist/probe_dai.exe

# 4. Expected output
# Should display:
#   - DepthAI version
#   - Connected devices
#   - USB speed
```

**Validation**:
- [ ] Diagnostic tool built
- [ ] Successfully detects OAK-D device
- [ ] Displays system information

---

## Phase 5: Deployment

### Task 5.1: Package Distribution Bundle

**Objective**: Prepare deployment package

**Directory Structure**:
```
TG25_GestureOAK-D_v1.0/
├── TG25_Launcher.exe
├── TG25_HandTracking.exe
├── probe_dai.exe
├── models/
│   ├── palm_detection_sh4.blob
│   ├── hand_landmark_lite_sh4.blob
│   └── PDPostProcessing_top2_sh1.blob
├── README.txt
└── (All DLLs from dist/ folders)
```

**Steps**:
```bash
# 1. Create deployment folder
mkdir TG25_GestureOAK-D_v1.0

# 2. Copy executables
cp dist/TG25_Launcher.exe TG25_GestureOAK-D_v1.0/
cp dist/TG25_HandTracking/TG25_HandTracking.exe TG25_GestureOAK-D_v1.0/
cp dist/probe_dai.exe TG25_GestureOAK-D_v1.0/

# 3. Copy models
cp -r models TG25_GestureOAK-D_v1.0/

# 4. Copy dependencies (Windows)
# All DLLs from dist/TG25_HandTracking/ to deployment root

# 5. Create README.txt
cat > TG25_GestureOAK-D_v1.0/README.txt << EOF
TG_25_GestureOAK-D - Hand Detection System

Quick Start:
1. Connect OAK-D camera to USB 3.0 port
2. Run TG25_Launcher.exe
3. Click "Start Hand Tracking"
4. Position hand 80-160cm from camera
5. Perform left-to-right swipe gesture

Troubleshooting:
- Run probe_dai.exe to verify camera connection
- Ensure USB 3.0 port (check Device Manager)
- Check firewall allows UDP port 6001

Support: https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D
EOF

# 6. Create ZIP archive
zip -r TG25_GestureOAK-D_v1.0.zip TG25_GestureOAK-D_v1.0/
```

**Validation**:
- [ ] All executables included
- [ ] Models folder present
- [ ] README.txt created
- [ ] ZIP archive created
- [ ] Archive size reasonable (<100MB)

---

### Task 5.2: Target System Deployment

**Objective**: Install on production machine

**Prerequisites (Target System)**:
- Windows 10/11 (64-bit) or Linux Ubuntu 20.04+
- USB 3.0 port available
- Network access (for UDP communication)

**Steps**:
```bash
# 1. Transfer ZIP to target system
# Via USB drive, network share, or download

# 2. Extract archive
# Windows: Right-click → Extract All
# Linux: unzip TG25_GestureOAK-D_v1.0.zip

# 3. Connect OAK-D camera to USB 3.0 port

# 4. Run diagnostic (optional)
cd TG25_GestureOAK-D_v1.0
./probe_dai.exe

# 5. Launch application
./TG25_Launcher.exe

# 6. Click "Start Hand Tracking"

# 7. Verify operation
# - Hand detection works
# - Swipe triggers UDP packet
# - No error messages
```

**Validation**:
- [ ] Application launches on target system
- [ ] Camera detected
- [ ] Hand detection functional
- [ ] UDP packets received by target system

---

### Task 5.3: Network Configuration

**Objective**: Configure UDP communication

**Steps**:

**On Source System (Running TG25_HandTracking.exe)**:
```bash
# 1. Verify network interface
ipconfig  # Windows
ip addr   # Linux

# 2. Test UDP send capability
# Use netcat or similar tool
echo "test" | nc -u 192.168.10.10 6001
```

**On Target System (192.168.10.10)**:
```bash
# 1. Set static IP (if needed)
# Windows: Network Settings → Change adapter options → Properties → IPv4
# Linux: Edit /etc/netplan/01-netcfg.yaml

# 2. Open UDP port 6001
# Windows Firewall:
netsh advfirewall firewall add rule name="TG25 UDP" dir=in action=allow protocol=UDP localport=6001

# Linux UFW:
sudo ufw allow 6001/udp

# 3. Listen for packets (testing)
nc -u -l 6001
# Or use Python:
python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 6001))
print('Listening on UDP 6001...')
while True:
    data, addr = sock.recvfrom(1024)
    print(f'Received: {data} from {addr}')
"
```

**Testing UDP Communication**:
1. Start listener on target (192.168.10.10)
2. Perform swipe gesture
3. Verify "Swipe" packet received

**Validation**:
- [ ] Network connectivity confirmed
- [ ] Firewall rules configured
- [ ] UDP packets reach target system
- [ ] Packet contents correct (b"Swipe")

---

## Phase 6: Quality Assurance

### Task 6.1: Functional Testing

**Objective**: Verify all features work as expected

**Test Cases**:

| Test ID | Description | Steps | Expected Result | Pass/Fail |
|---------|-------------|-------|-----------------|-----------|
| TC-001 | Camera Connection | Run probe_dai.exe | Device detected, USB3 | [ ] |
| TC-002 | Hand Detection Near (80cm) | Position hand at 80cm | Hand detected with landmarks | [ ] |
| TC-003 | Hand Detection Mid (120cm) | Position hand at 120cm | Hand detected with landmarks | [ ] |
| TC-004 | Hand Detection Far (160cm) | Position hand at 160cm | Hand detected with landmarks | [ ] |
| TC-005 | Swipe Left-to-Right | Perform L-R swipe at 100cm | "SWIPE DETECTED!" displayed | [ ] |
| TC-006 | Swipe UDP Packet | Perform swipe, check target | "Swipe" packet received | [ ] |
| TC-007 | Swipe Cooldown | Perform 2 rapid swipes | Second swipe ignored (<0.8s) | [ ] |
| TC-008 | False Positive Filter | Wave hand vertically | No swipe detected | [ ] |
| TC-009 | Multiple Hands | Show 2 hands | Both detected (if enabled) | [ ] |
| TC-010 | Low Light | Test in dark room | IR mode detects hand | [ ] |
| TC-011 | Launcher Start/Stop | Use GUI buttons | Worker starts/stops cleanly | [ ] |
| TC-012 | Statistics Display | Perform 5 swipes | Counter shows 5 | [ ] |
| TC-013 | Frame Save | Press 's' key | JPG file created | [ ] |
| TC-014 | Statistics Reset | Press 'r' key | Counters reset to 0 | [ ] |
| TC-015 | Graceful Exit | Press 'q' key | App closes cleanly | [ ] |

**Validation**:
- [ ] All test cases pass
- [ ] No crashes or freezes
- [ ] Performance acceptable (25-30 FPS)

---

### Task 6.2: Performance Benchmarking

**Objective**: Measure system performance

**Metrics to Collect**:

```python
# Add to hand_tracking_app.py for benchmarking
import time

# Timing collection
frame_times = []
detection_times = []
swipe_times = []

# In main loop
start = time.time()
frame, hands, depth = detector.get_frame_and_hands()
frame_times.append(time.time() - start)

start = time.time()
swipe_detected = swipe_detector.update(hand_center)
swipe_times.append(time.time() - start)

# After 100 frames
if len(frame_times) >= 100:
    print(f"Avg frame time: {np.mean(frame_times)*1000:.2f}ms")
    print(f"Avg detection time: {np.mean(detection_times)*1000:.2f}ms")
    print(f"Avg swipe time: {np.mean(swipe_times)*1000:.2f}ms")
    print(f"FPS: {1.0/np.mean(frame_times):.2f}")
```

**Benchmark Results Template**:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Average FPS | 25-30 | ___ | [ ] |
| Frame Latency (ms) | <40 | ___ | [ ] |
| Detection Latency (ms) | <10 | ___ | [ ] |
| Swipe Latency (ms) | <1 | ___ | [ ] |
| Memory Usage (MB) | <500 | ___ | [ ] |
| CPU Usage (%) | <50 | ___ | [ ] |
| Detection Range (cm) | 80-160 | ___ | [ ] |
| Swipe Success Rate (%) | >90 | ___ | [ ] |
| False Positive Rate (%/min) | <5 | ___ | [ ] |

**Validation**:
- [ ] All metrics meet targets
- [ ] Performance consistent over time
- [ ] No memory leaks

---

### Task 6.3: Stress Testing

**Objective**: Test system under extreme conditions

**Test Scenarios**:

**Scenario 1: Extended Operation**
- Duration: 2 hours continuous operation
- Actions: Periodic swipes every 30 seconds
- Monitor: Memory usage, FPS degradation, crashes

**Scenario 2: Rapid Swipes**
- Duration: 5 minutes
- Actions: Continuous rapid swipes
- Monitor: Cooldown effectiveness, UDP queue buildup

**Scenario 3: Challenging Lighting**
- Test in: Bright sunlight, complete darkness, mixed lighting
- Monitor: Detection consistency, false positive rate

**Scenario 4: Distance Extremes**
- Test at: 40cm, 200cm, moving closer/farther
- Monitor: Detection reliability, transition smoothness

**Scenario 5: Multiple Background Objects**
- Test with: People walking, moving furniture, pets
- Monitor: False positive rate, depth filtering effectiveness

**Validation**:
- [ ] System stable for 2+ hours
- [ ] No performance degradation
- [ ] Handles edge cases gracefully

---

## Phase 7: Documentation and Handoff

### Task 7.1: User Documentation Creation

**Objective**: Create end-user guide

**Document Contents**:
1. **Quick Start Guide**
   - Hardware setup
   - Software installation
   - First run instructions

2. **User Manual**
   - Feature overview
   - Operating instructions
   - Keyboard shortcuts
   - Troubleshooting

3. **Network Configuration Guide**
   - IP address setup
   - Firewall configuration
   - UDP port verification

4. **Maintenance Guide**
   - Camera cleaning
   - Software updates
   - Log file management

**Validation**:
- [ ] All documents created
- [ ] Language: English + Japanese
- [ ] Screenshots included
- [ ] Peer reviewed

---

### Task 7.2: Developer Documentation

**Objective**: Enable future maintenance and enhancement

**Documents to Maintain**:
1. **Architecture Documentation** ✓ (Already created)
2. **API Reference**
   - HandDetector class
   - SwipeDetector class
   - OAKCamera class

3. **Configuration Guide**
   - All tunable parameters
   - Impact of each parameter
   - Recommended ranges

4. **Build and Deployment Guide**
   - Build process
   - Dependencies
   - Troubleshooting builds

**Validation**:
- [ ] API documentation complete
- [ ] Code comments adequate
- [ ] Examples provided

---

### Task 7.3: Training and Knowledge Transfer

**Objective**: Train operators and maintainers

**Training Topics**:

**Operator Training (2 hours)**:
- Hardware setup and connection
- Application launch and operation
- Basic troubleshooting
- When to escalate issues

**Maintainer Training (4 hours)**:
- System architecture overview
- Parameter tuning
- Log analysis
- Common issues and solutions
- Update procedures

**Administrator Training (8 hours)**:
- Full system architecture
- Source code structure
- Build process
- Custom modifications
- Performance optimization

**Validation**:
- [ ] Training materials prepared
- [ ] Training sessions conducted
- [ ] Operators certified
- [ ] Feedback incorporated

---

## Phase 8: Production Deployment Checklist

### Final Pre-Deployment Validation

**System Configuration**:
- [ ] Python environment verified
- [ ] All dependencies installed
- [ ] Virtual environment activated
- [ ] Repository on correct branch

**Hardware Setup**:
- [ ] OAK-D camera connected to USB 3.0
- [ ] USB speed verified (SUPER_SPEED)
- [ ] Camera positioned correctly
- [ ] Lighting conditions acceptable

**Software Testing**:
- [ ] All test cases passed
- [ ] Performance benchmarks met
- [ ] Stress tests completed
- [ ] No known critical bugs

**Network Configuration**:
- [ ] Target IP configured (192.168.10.10)
- [ ] UDP port 6001 open
- [ ] Network connectivity verified
- [ ] Firewall rules configured

**Executable Deployment**:
- [ ] All executables built
- [ ] Models bundled correctly
- [ ] Dependencies included
- [ ] Tested on target system

**Documentation**:
- [ ] User guide complete
- [ ] Developer docs updated
- [ ] Training materials ready
- [ ] Support contact provided

**Handoff**:
- [ ] Training completed
- [ ] Questions answered
- [ ] Support plan established
- [ ] Sign-off obtained

---

# 日本語版

## 概要

このドキュメントは、TG_25_GestureOAK-D手検出システムのセットアップおよび展開のための段階的な実装タスクを概説します。

---

## フェーズ1: 環境セットアップ

### タスク1.1: システム前提条件のインストール

**目的**: 必要なシステム依存関係をインストール

**手順（Windows）**:
```powershell
# 1. Python 3.10以降をインストール
# ダウンロード: https://www.python.org/downloads/
# インストール時に「Add Python to PATH」をチェック

# 2. Pythonインストールを確認
python --version
# 期待: Python 3.10.x または 3.11.x または 3.12.x

# 3. Gitをインストール
# ダウンロード: https://git-scm.com/download/win

# 4. Gitインストールを確認
git --version
```

**手順（Linux - Ubuntu/Debian）**:
```bash
# 1. パッケージリストを更新
sudo apt-get update

# 2. Python 3.10+をインストール
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# 3. システム依存関係をインストール
sudo apt-get install -y \
    libusb-1.0-0-dev \
    libudev-dev \
    python3-dev \
    libopencv-dev \
    git

# 4. OAK-D用udevルールを設定
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# 5. インストールを確認
python3 --version
git --version
```

**検証**:
- [ ] Python 3.10+インストール済み
- [ ] Gitインストール済み
- [ ] USB権限設定済み（Linux）

---

### タスク1.2: リポジトリクローンとブランチ選択

**目的**: プロジェクトソースコードを取得

**手順**:
```bash
# 1. リポジトリをクローン
git clone https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D.git

# 2. プロジェクトディレクトリに移動
cd TG_25_GestureOAK-D

# 3. 正しいブランチをチェックアウト
git checkout Hand-Gesture

# 4. ブランチを確認
git branch
# 期待: * Hand-Gesture

# 5. リポジトリ構造を確認
ls -la
# 表示されるべき: main.py、requirements.txt、src/、models/など
```

**検証**:
- [ ] リポジトリクローン成功
- [ ] `Hand-Gesture`ブランチ上
- [ ] すべてのファイル存在（main.py、src/、models/）

---

### タスク1.3: 仮想環境の作成

**目的**: 分離されたPython環境を作成

**手順（Windows）**:
```powershell
# 1. 仮想環境を作成
python -m venv .venv

# 2. 仮想環境を有効化
.venv\Scripts\activate

# 3. 有効化を確認（プロンプトに (.venv) が表示されるべき）
where python
# .venv\Scripts\python.exeを指すべき

# 4. pipをアップグレード
python -m pip install --upgrade pip setuptools wheel
```

**手順（Linux/Mac）**:
```bash
# 1. 仮想環境を作成
python3 -m venv .venv

# 2. 仮想環境を有効化
source .venv/bin/activate

# 3. 有効化を確認
which python
# .venv/bin/pythonを指すべき

# 4. pipをアップグレード
python -m pip install --upgrade pip setuptools wheel
```

**検証**:
- [ ] 仮想環境作成済み（.venv/ディレクトリ存在）
- [ ] 仮想環境有効化済み（プロンプトに (.venv) 表示）
- [ ] pip最新バージョンにアップグレード済み

---

### タスク1.4: 依存関係のインストール

**目的**: すべての必要なPythonパッケージをインストール

**手順**:
```bash
# 1. 仮想環境が有効化されていることを確認
# プロンプトに (.venv) が表示されるべき

# 2. requirements.txtから依存関係をインストール
pip install -r requirements.txt

# 3. インストールを確認
pip list

# 4. 重要なパッケージを確認
pip show depthai
pip show opencv-python
pip show numpy

# 5. インポートをテスト
python -c "import depthai as dai; print(f'DepthAI version: {dai.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
```

**期待される出力**:
```
DepthAI version: 2.24.0.0（またはそれ以降）
OpenCV version: 4.8.x（またはそれ以降）
NumPy version: 1.24.x（またはそれ以降）
```

**検証**:
- [ ] すべてのパッケージがエラーなくインストール
- [ ] depthaiバージョン ≥ 2.24.0
- [ ] opencv-pythonバージョン ≥ 4.8.0
- [ ] すべてのインポート成功

---

### タスク1.5: ハードウェア接続の確認

**目的**: OAK-Dカメラが正しく接続されていることを確認

**手順**:
```bash
# 1. OAK-DカメラをUSB 3.0ポートに接続
# 可能であれば青いUSBポートを使用（USB 3.0を示す）

# 2. デバイス列挙を確認（Linux）
lsusb | grep "03e7"
# 期待: Movidius MyriadXデバイス

# 3. デバイス列挙を確認（Windows）
# デバイスマネージャーを開く → ユニバーサルシリアルバスコントローラー
# 「Movidius MyriadX」または「OAK-D」を探す

# 4. 診断ツールを実行
python probe_dai.py
```

**期待される出力**:
```
Testing DepthAI connection...
✓ DepthAI library imported successfully
✓ Found 1 OAK device(s)
Device: OAK-D-PRO
USB Speed: SUPER_SPEED
Connected successfully!
```

**検証**:
- [ ] OAK-Dがシステムで検出される
- [ ] USB 3.0接続確認（SUPER_SPEED）
- [ ] probe_dai.pyがエラーなく実行

---

[続きは英語版と同じ構造で、すべてのタスクが日本語で説明されます]

---

## フェーズ8: 本番デプロイメントチェックリスト

### 最終デプロイメント前検証

**システム構成**:
- [ ] Python環境検証済み
- [ ] すべての依存関係インストール済み
- [ ] 仮想環境有効化済み
- [ ] リポジトリが正しいブランチ上

**ハードウェアセットアップ**:
- [ ] OAK-DカメラがUSB 3.0に接続済み
- [ ] USB速度検証済み（SUPER_SPEED）
- [ ] カメラ正しく配置済み
- [ ] 照明条件適切

**ソフトウェアテスト**:
- [ ] すべてのテストケース合格
- [ ] パフォーマンスベンチマーク達成
- [ ] ストレステスト完了
- [ ] 既知の重大なバグなし

**ネットワーク構成**:
- [ ] ターゲットIP設定済み（192.168.10.10）
- [ ] UDPポート6001オープン
- [ ] ネットワーク接続確認済み
- [ ] ファイアウォールルール設定済み

**実行ファイル展開**:
- [ ] すべての実行ファイルビルド済み
- [ ] モデル正しくバンドル済み
- [ ] 依存関係含まれる
- [ ] ターゲットシステムでテスト済み

**ドキュメント**:
- [ ] ユーザーガイド完成
- [ ] 開発者ドキュメント更新済み
- [ ] トレーニング資料準備完了
- [ ] サポート連絡先提供済み

**引き渡し**:
- [ ] トレーニング完了
- [ ] 質問回答済み
- [ ] サポート計画確立済み
- [ ] 承認取得済み

---

**ドキュメント作成日**: 2024年10月  
**バージョン**: 1.0  
**著者**: TG_25_GestureOAK-Dチーム