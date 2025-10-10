# Troubleshooting Guide / トラブルシューティングガイド

**English** | [日本語](#日本語版-1)

---

## Quick Diagnosis

Use this flowchart to identify your issue category:

```
Problem?
├─ Camera not detected → [Hardware Issues](#hardware-issues)
├─ Application won't start → [Software Issues](#software-issues)
├─ Hand not detected → [Detection Issues](#detection-issues)
├─ Swipe not triggering → [Swipe Detection Issues](#swipe-detection-issues)
├─ Low FPS / Lag → [Performance Issues](#performance-issues)
├─ Executable errors → [Executable Issues](#executable-issues)
└─ Network/UDP problems → [Network Issues](#network-issues)
```

---

## Hardware Issues

### Issue H-001: Camera Not Detected

**Symptoms**:
```
Failed to connect to OAK-D: [X_LINK_DEVICE_NOT_FOUND]
RuntimeError: No devices found!
```

**Diagnosis Steps**:

**Step 1**: Check Physical Connection
```bash
# Unplug and replug USB cable
# Try different USB port (prefer USB 3.0 - blue port)
# Ensure cable is fully inserted
```

**Step 2**: Verify Device Enumeration

*Windows*:
```powershell
# Open Device Manager (devmgmt.msc)
# Look under "Universal Serial Bus controllers"
# Should see "Movidius MyriadX" or similar

# If device shows with yellow exclamation:
# Right-click → Update Driver → Browse → Let me pick
# Select "libusbK" or "WinUSB" driver
```

*Linux*:
```bash
# Check USB devices
lsusb | grep "03e7"
# Expected: Bus XXX Device XXX: ID 03e7:XXXX Movidius Ltd.

# If not found, check dmesg
dmesg | tail -20
# Look for USB connection messages
```

**Step 3**: Install/Update Drivers

*Windows - Using Zadig*:
```
1. Download Zadig: https://zadig.akeo.ie/
2. Run as Administrator
3. Options → List All Devices
4. Select "Movidius MyriadX"
5. Select "WinUSB" driver
6. Click "Replace Driver"
7. Reboot system
```

*Linux - udev Rules*:
```bash
# Create udev rule
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/80-movidius.rules

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Reconnect camera
```

**Step 4**: Test with Diagnostic Tool
```bash
python probe_dai.py
```

**Resolution**:
- [ ] Device appears in system (lsusb / Device Manager)
- [ ] Driver installed correctly
- [ ] probe_dai.py detects camera
- [ ] USB speed shows SUPER_SPEED (USB 3.0)

---

### Issue H-002: USB Speed Too Slow

**Symptoms**:
```
USB Speed: HIGH_SPEED (expected: SUPER_SPEED)
Warning: USB 2.0 connection detected
Low FPS (<15)
```

**Diagnosis**:
```python
# Check current USB speed
python -c "
import depthai as dai
with dai.Device() as device:
    speed = device.getUsbSpeed()
    print(f'USB Speed: {speed}')
    if speed != dai.UsbSpeed.SUPER and speed != dai.UsbSpeed.SUPER_PLUS:
        print('WARNING: Not using USB 3.0!')
"
```

**Solutions**:

1. **Use USB 3.0 Port** (Blue port on most systems)
2. **Replace USB Cable** (Ensure it's USB 3.0 rated)
3. **Update USB 3.0 Controller Driver**:
   - Windows: Device Manager → USB 3.0 Controller → Update Driver
   - Linux: `sudo apt-get update && sudo apt-get upgrade`

4. **Disable USB Power Management** (Windows):
```
Device Manager → USB Root Hub → Properties → Power Management
Uncheck "Allow computer to turn off this device"
```

**Validation**:
- USB speed: `SUPER` or `SUPER_PLUS`
- FPS: 25-30
- No bandwidth warnings

---

### Issue H-003: Camera Disconnects Randomly

**Symptoms**:
- Application runs for a few minutes then crashes
- Error: `X_LINK_COMMUNICATION_NOT_OPEN`
- Camera LED turns off unexpectedly

**Causes & Solutions**:

**Cause 1: Power Supply Insufficient**
```
Solution: Use powered USB hub with dedicated power adapter
Avoid: USB hubs without external power
Avoid: Laptop USB ports when on battery
```

**Cause 2: USB Cable Quality**
```
Solution: Use high-quality shielded USB 3.0 cable
Max Length: 3 meters (10 feet)
Avoid: Cheap/thin cables, extension cables
```

**Cause 3: Thermal Throttling**
```
Solution: Ensure adequate ventilation around camera
Check: Camera housing should not be hot to touch
Add: Small fan if operating in enclosed space
```

**Cause 4: Power Management**
```bash
# Windows - Disable USB Selective Suspend
Control Panel → Power Options → Change plan settings
→ Change advanced power settings
→ USB settings → USB selective suspend → Disabled

# Linux - Disable autosuspend
echo -1 | sudo tee /sys/bus/usb/devices/*/power/autosuspend_delay_ms
```

**Validation**:
- Camera runs continuously for 2+ hours
- No disconnection errors
- Temperature stable

---

## Software Issues

### Issue S-001: Python Import Errors

**Symptoms**:
```
ImportError: No module named 'depthai'
ModuleNotFoundError: No module named 'cv2'
```

**Diagnosis**:
```bash
# Check if virtual environment is activated
which python  # Linux/Mac
where python  # Windows
# Should point to .venv/bin/python or .venv\Scripts\python.exe

# Check installed packages
pip list | grep -E "depthai|opencv|numpy"
```

**Solutions**:

**Solution 1: Activate Virtual Environment**
```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# PowerShell
.venv\Scripts\Activate.ps1
```

**Solution 2: Reinstall Dependencies**
```bash
# Ensure venv is activated
pip install --force-reinstall -r requirements.txt
```

**Solution 3: Check Python Version**
```bash
python --version
# Must be 3.10, 3.11, or 3.12

# If wrong version, recreate venv with correct Python
python3.10 -m venv .venv
```

**Validation**:
- [ ] Virtual environment activated
- [ ] `import depthai` succeeds
- [ ] `import cv2` succeeds
- [ ] `import numpy` succeeds

---

### Issue S-002: Application Won't Start

**Symptoms**:
```
python main.py
# No output, hangs, or immediate exit
```

**Diagnosis Steps**:

**Step 1: Check for Syntax Errors**
```bash
python -m py_compile main.py
# Should complete without errors
```

**Step 2: Run with Verbose Output**
```bash
python -u main.py 2>&1 | tee output.log
# -u: unbuffered output
# 2>&1: capture stderr
# tee: save to file
```

**Step 3: Test Individual Components**
```bash
# Test camera module
python -c "from src.gesture_oak.core.oak_camera import OAKCamera; print('OK')"

# Test detector module
python -c "from src.gesture_oak.detection.hand_detector import HandDetector; print('OK')"

# Test swipe detector
python -c "from src.gesture_oak.detection.swipe_detector import SwipeDetector; print('OK')"
```

**Common Causes**:

1. **Missing models/ folder**
```bash
ls -la models/
# Should contain:
# palm_detection_sh4.blob
# hand_landmark_lite_sh4.blob
# PDPostProcessing_top2_sh1.blob
```

2. **Permission issues**
```bash
# Check file permissions
ls -l main.py
# Should be readable: -rw-r--r-- or similar

# Fix if needed
chmod +x main.py
```

3. **Python path issues**
```bash
# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%           # Windows CMD
```

**Validation**:
- [ ] main.py runs and shows menu
- [ ] All imports successful
- [ ] models/ folder present and accessible

---

### Issue S-003: OpenCV Display Window Not Appearing

**Symptoms**:
- Application runs without errors
- No video window appears
- Console shows frame processing

**Diagnosis**:
```python
# Test OpenCV display capability
python -c "
import cv2
import numpy as np
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imshow('Test', test_img)
print('Window created. Press any key...')
cv2.waitKey(0)
cv2.destroyAllWindows()
"
```

**Solutions**:

**Solution 1: Install GUI Backend (Linux)**
```bash
# For GTK backend
sudo apt-get install -y python3-tk libgtk-3-dev

# For Qt backend
sudo apt-get install -y python3-pyqt5

# Reinstall opencv with GUI support
pip uninstall opencv-python
pip install opencv-python
```

**Solution 2: Set Display Variable (Linux SSH/Remote)**
```bash
# If running over SSH, enable X11 forwarding
ssh -X user@host

# Set DISPLAY variable
export DISPLAY=:0

# Or use VNC/Remote Desktop instead
```

**Solution 3: Use Headless Mode**
```python
# Modify hand_tracking_app.py to save frames instead of display
# Replace cv2.imshow() with:
cv2.imwrite(f'frame_{frame_count:04d}.jpg', frame)
```

**Validation**:
- [ ] Test window appears
- [ ] Application displays video feed
- [ ] Window is interactive (can click, move)

---

## Detection Issues

### Issue D-001: No Hand Detected

**Symptoms**:
- Camera feed visible
- No hand landmarks drawn
- Console shows "Hands: 0"

**Diagnosis Checklist**:

1. **Check Distance**
```
Optimal: 80-160 cm (31-63 inches)
Minimum: 40 cm (16 inches)
Maximum: 200 cm (79 inches)
```

2. **Check Lighting**
```python
# Check frame brightness
python -c "
import cv2
import numpy as np
# Capture a frame
frame = ... # your frame
avg_brightness = np.mean(frame)
print(f'Average brightness: {avg_brightness}')
# Should be: 50-200 for good detection
"
```

3. **Check Hand Visibility**
```
✓ Hand fully in frame (not cut off)
✓ Palm facing camera
✓ Fingers visible (not hidden)
✓ No gloves or coverings
```

4. **Check Detection Threshold**
```python
# In hand_detector.py
detector = HandDetector(
    pd_score_thresh=0.10,  # Try lowering to 0.05 or 0.08
)
```

**Solutions**:

**Solution 1: Adjust Distance**
- Move hand to 100cm (40 inches) from camera
- Gradually move closer or farther to find sweet spot

**Solution 2: Improve Lighting**
```
If using RGB mode:
  - Add lighting to scene
  - Avoid backlighting
  - Use diffuse light (not direct)

If using IR mode:
  - Check IR LEDs are working (use phone camera to see)
  - Ensure no IR blocking materials nearby
```

**Solution 3: Lower Detection Threshold**
```python
# Edit src/gesture_oak/detection/hand_detector.py
# Line ~18
detector = HandDetector(
    pd_score_thresh=0.05,  # More sensitive (was 0.10)
    pd_nms_thresh=0.3,
)
```

**Solution 4: Switch Camera Mode**
```python
# Try RGB mode instead of IR
detector = HandDetector(
    use_rgb=True,  # Was: use_rgb=False
)
```

**Validation**:
- [ ] Hand landmarks appear
- [ ] Bounding box tracks hand
- [ ] Confidence score > 0.5
- [ ] Detection consistent for 5+ seconds

---

### Issue D-002: Hand Detected But Unstable

**Symptoms**:
- Hand appears and disappears rapidly
- Landmarks jitter/flicker
- Confidence score fluctuates

**Diagnosis**:
```python
# Monitor detection stability
# Add to hand_tracking_app.py:
detection_count = 0
total_frames = 0
for ... in main_loop:
    total_frames += 1
    if len(hands) > 0:
        detection_count += 1
    if total_frames % 100 == 0:
        stability = detection_count / total_frames
        print(f'Stability: {stability:.2%}')
# Target: >85% stability
```

**Causes & Solutions**:

**Cause 1: Distance Too Far**
```
Solution: Move hand closer (100-120cm optimal)
Check: depth value should be 800-1200mm
```

**Cause 2: Motion Blur**
```
Solution: Move hand slower
Check: Ensure 30 FPS is maintained
Adjust: Reduce FPS to 20 if needed for better exposure
```

**Cause 3: Background Interference**
```
Solution: Remove background objects with similar IR reflectance
Check: Ensure plain background
Test: Cover background with dark cloth
```

**Cause 4: Depth Filtering Too Strict**
```python
# In hand_detector.py - filter_hands_by_depth()
# Increase tolerance
std_limit = 100.0 + 0.10 * max(0.0, (avg - 800.0))  # Was: 80.0 + 0.08
```

**Validation**:
- [ ] Detection stability > 85%
- [ ] Landmarks don't jitter
- [ ] Consistent depth readings

---

### Issue D-003: Wrong Hand Detected (Left vs Right)

**Symptoms**:
- Right hand labeled as "left"
- Hand label flips randomly
- Handedness score near 0.5

**Explanation**:
MediaPipe's handedness classifier is trained on specific hand poses. Mirror-image confusion can occur.

**Solutions**:

**Solution 1: Improve Hand Pose**
```
✓ Show clear palm to camera
✓ Spread fingers slightly
✓ Avoid side views
✓ Keep hand upright (fingers pointing up)
```

**Solution 2: Use Handedness Averaging**
```python
# In hand_detector.py, enable handedness averaging
# (Already implemented in Script node template)
# Averages handedness over multiple frames for stability
```

**Solution 3: Ignore Handedness**
```python
# If left/right distinction not needed
# Simply use hand.label = "hand" for all detections
```

**Workaround**:
- Use right hand for all operations (more reliably detected)
- If left hand required, position it carefully with clear palm view

---

## Swipe Detection Issues

### Issue SW-001: Swipe Not Triggering

**Symptoms**:
- Hand detected correctly
- Horizontal motion performed
- "SWIPE DETECTED!" never appears
- Console shows no swipe events

**Diagnosis**:
```python
# Enable debug output in swipe_detector.py
# Add to update() method:
progress = self.get_current_swipe_progress()
if progress and progress['state'] != 'idle':
    print(f"State: {progress['state']}, Distance: {progress['distance']:.0f}, Velocity: {progress['velocity']:.0f}")
```

**Common Causes**:

**Cause 1: Insufficient Distance**
```
Minimum required: 90 pixels
Your motion: Check debug output
Solution: Make larger sweeping motion (20-30cm travel)
```

**Cause 2: Too Slow Motion**
```
Minimum velocity: 35 px/s
Your velocity: Check debug output
Solution: Move hand faster (complete swipe in 0.5-1 second)
```

**Cause 3: Too Much Vertical Deviation**
```
Maximum Y deviation: 35% of X travel
Your deviation: Check debug output
Solution: Keep hand at same height while swiping
```

**Cause 4: Cooldown Period**
```
Default cooldown: 0.8 seconds
Solution: Wait 1 second between swipes
```

**Solutions**:

**Solution 1: Adjust Swipe Parameters**
```python
# Edit src/gesture_oak/detection/swipe_detector.py
swipe_detector = SwipeDetector(
    min_distance=60,       # Reduce from 90
    min_velocity=25,       # Reduce from 35
    max_y_deviation=0.45,  # Increase from 0.35
    cooldown=0.5,          # Reduce from 0.8
)
```

**Solution 2: Practice Swipe Motion**
```
Technique:
1. Position hand at left side of camera view
2. Keep palm facing camera
3. Move smoothly to right side (about 30cm)
4. Maintain constant height
5. Complete motion in about 0.8 seconds
```

**Solution 3: Check Buffer Size**
```python
# If swipes are very fast, increase buffer
swipe_detector = SwipeDetector(
    buffer_size=24,  # Increase from 18 for more history
)
```

**Validation**:
- [ ] Debug output shows progression through states
- [ ] Distance reaches >90 pixels
- [ ] Velocity in range 35-900 px/s
- [ ] Y deviation <35%
- [ ] Swipe confirmation appears

---

### Issue SW-002: Too Many False Positive Swipes

**Symptoms**:
- Swipes trigger without intentional motion
- Random movements cause detections
- Statistics show high swipe count

**Diagnosis**:
```python
# Monitor false positive rate
# Let application run for 5 minutes without intentional swipes
# Count: Auto-triggered swipes
# Target: <3 false positives per 5 minutes
```

**Solutions**:

**Solution 1: Increase Minimum Distance**
```python
swipe_detector = SwipeDetector(
    min_distance=120,  # Increase from 90
)
```

**Solution 2: Stricter Velocity Bounds**
```python
swipe_detector = SwipeDetector(
    min_velocity=50,   # Increase from 35
    max_velocity=700,  # Reduce from 900
)
```

**Solution 3: Reduce Y Deviation Tolerance**
```python
swipe_detector = SwipeDetector(
    max_y_deviation=0.25,  # Reduce from 0.35
)
```

**Solution 4: Enable Stricter Depth Filtering**
```python
# In hand_detector.py
std_limit = 60.0 + 0.06 * max(0.0, (avg - 800.0))  # Tighter tolerance
```

**Solution 5: Increase Cooldown**
```python
swipe_detector = SwipeDetector(
    cooldown=1.2,  # Increase from 0.8
)
```

**Validation**:
- [ ] False positive rate <3 per 5 minutes
- [ ] Intentional swipes still detected
- [ ] No swipes during idle time

---

### Issue SW-003: UDP Packets Not Received

**Symptoms**:
- Swipe detected in application
- Console shows "SWIPE DETECTED!"
- Target system (192.168.10.10) doesn't receive packet

**Diagnosis**:

**Step 1: Verify Network Connectivity**
```bash
# Ping target
ping 192.168.10.10

# Should show replies with low latency (<50ms)
```

**Step 2: Test UDP Directly**
```python
# On source machine (running gesture app)
python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b'TEST', ('192.168.10.10', 6001))
print('Test packet sent')
"

# On target machine (192.168.10.10)
python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 6001))
print('Listening on UDP 6001...')
data, addr = sock.recvfrom(1024)
print(f'Received: {data} from {addr}')
"
```

**Step 3: Check Firewall**
```bash
# Windows - Check firewall rules
netsh advfirewall firewall show rule name=all | findstr "6001"

# Linux - Check UFW
sudo ufw status | grep 6001

# Linux - Check iptables
sudo iptables -L -n | grep 6001
```

**Solutions**:

**Solution 1: Add Firewall Rule**
```bash
# Windows (run as Administrator)
netsh advfirewall firewall add rule name="TG25 UDP Out" dir=out action=allow protocol=UDP localport=any remoteport=6001
netsh advfirewall firewall add rule name="TG25 UDP In" dir=in action=allow protocol=UDP localport=6001

# Linux
sudo ufw allow out 6001/udp
sudo ufw allow in 6001/udp
```

**Solution 2: Verify IP Address**
```python
# In swipe_detector.py __init__()
print(f"UDP target: {self._udp_target}")
# Should show: ('192.168.10.10', 6001)

# Verify target IP is correct for your network
# Check with: ip addr (Linux) or ipconfig (Windows)
```

**Solution 3: Test with Wireshark**
```
1. Install Wireshark on target system
2. Start capture on network interface
3. Filter: udp.port == 6001
4. Perform swipe gesture
5. Verify UDP packet appears in capture
```

**Solution 4: Check Socket Creation**
```python
# Add debug to swipe_detector.py __init__()
try:
    self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._udp_target = ("192.168.10.10", 6001)
    print(f"UDP socket created: {self._udp_sock}")
except Exception as e:
    print(f"UDP socket error: {e}")
    self._udp_sock = None
```

**Validation**:
- [ ] Ping successful to 192.168.10.10
- [ ] Test UDP packet received
- [ ] Firewall allows UDP 6001
- [ ] Swipe triggers packet in Wireshark
- [ ] Target application receives packet

---

## Performance Issues

### Issue P-001: Low FPS (Frame Rate)

**Symptoms**:
- FPS counter shows <20
- Video appears choppy/laggy
- High CPU usage

**Diagnosis**:
```python
# Profile performance bottlenecks
import time

# Add to hand_tracking_app.py main loop:
t1 = time.time()
frame, hands, depth = detector.get_frame_and_hands()
t2 = time.time()
swipe_detected = swipe_detector.update(hand_center)
t3 = time.time()
# Draw and display
t4 = time.time()

print(f"Detection: {(t2-t1)*1000:.1f}ms, Swipe: {(t3-t2)*1000:.1f}ms, Draw: {(t4-t3)*1000:.1f}ms")
```

**Solutions**:

**Solution 1: Reduce Resolution**
```python
detector = HandDetector(
    resolution=(480, 360),  # Reduce from (640, 480)
)
```

**Solution 2: Lower Target FPS**
```python
detector = HandDetector(
    fps=20,  # Reduce from 30
)
```

**Solution 3: Disable IR Enhancement**
```python
# In hand_detector.py - get_frame_and_hands()
# Comment out:
# frame = self.enhance_ir_frame(raw_frame)
frame = raw_frame
```

**Solution 4: Reduce Queue Sizes**
```python
# In hand_detector.py - connect()
self.q_video = self.device.getOutputQueue(
    name="cam_out", 
    maxSize=1,  # Reduce from 2
    blocking=False
)
```

**Solution 5: Optimize OpenCV**
```python
# Add to top of hand_tracking_app.py
import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Adjust based on CPU cores
```

**Solution 6: Close Other Applications**
```bash
# Free up system resources
# Close browser, video players, etc.
# Check CPU usage: Task Manager (Windows) or top (Linux)
```

**Validation**:
- [ ] FPS ≥ 25
- [ ] CPU usage <60%
- [ ] Smooth video playback
- [ ] No frame drops

---

### Issue P-002: High Memory Usage / Memory Leak

**Symptoms**:
- Memory usage grows over time
- Application slows down after running for hours
- System becomes unresponsive

**Diagnosis**:
```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())

# Add to main loop (every 100 frames):
if frame_count % 100 == 0:
    mem = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {mem:.1f} MB")
```

**Solutions**:

**Solution 1: Clear Display Buffer**
```python
# In main loop, after cv2.imshow()
cv2.waitKey(1)
# Don't accumulate window events
```

**Solution 2: Limit Buffer Sizes**
```python
# In swipe_detector.py
swipe_detector = SwipeDetector(
    buffer_size=12,  # Reduce from 18
)
```

**Solution 3: Release Frames Explicitly**
```python
# After processing frame
frame = None
hands = None
depth_frame = None
```

**Solution 4: Periodic Cleanup**
```python
# Add to main loop
import gc
if frame_count % 1000 == 0:
    gc.collect()
```

**Validation**:
- [ ] Memory usage stable (<500 MB)
- [ ] No growth over 2-hour test
- [ ] Application responsive

---

## Executable Issues

### Issue E-001: Executable Won't Run

**Symptoms**:
```
TG25_HandTracking.exe
# Double-click does nothing OR
# Flash of console window then closes
```

**Diagnosis**:
```bash
# Run from command line to see errors
cd dist/TG25_HandTracking
TG25_HandTracking.exe

# Or (Windows):
cmd /k TG25_HandTracking.exe
# /k keeps window open after exit
```

**Common Errors & Solutions**:

**Error 1: "Missing DLL"**
```
Solution: Copy all DLLs from dist/TG25_HandTracking/ to same folder as .exe
Check: msvcp140.dll, vcruntime140.dll, etc.
Install: Visual C++ Redistributable if needed
```

**Error 2: "models folder not found"**
```
Solution: Ensure models/ folder is in same directory as .exe
Check: models/palm_detection_sh4.blob exists
```

**Error 3: "ImportError: No module named X"**
```
Solution: Rebuild with missing module in hiddenimports
Edit: TG25_HandTracking.spec
Add to hiddenimports: ['missing_module_name']
Rebuild: pyinstaller TG25_HandTracking.spec
```

**Validation**:
- [ ] Executable runs from command line
- [ ] No DLL errors
- [ ] models/ folder accessible
- [ ] Camera detected

---

### Issue E-002: Launcher Can't Control Worker

**Symptoms**:
- Click "Start" - nothing happens
- Worker doesn't stop when clicking "Stop"
- Status stays "idle"

**Diagnosis**:
```python
# Add debug to TG25_Launcher.py
def start_worker(self):
    print(f"Attempting to start worker")
    print(f"Base dir: {self.base}")
    print(f"Stop file: {self.stop_file}")
    cmd, env = self._worker_cmd()
    print(f"Command: {cmd}")
    print(f"Env TG25_STOP_FILE: {env.get('TG25_STOP_FILE')}")
    # ... rest of method
```

**Solutions**:

**Solution 1: Verify Worker Executable Path**
```python
# In TG25_Launcher.py - _worker_cmd()
exe = self.base / "TG25_HandTracking.exe"
print(f"Looking for: {exe}")
print(f"Exists: {exe.exists()}")
```

**Solution 2: Check Stop Flag Mechanism**
```python
# In run_hand_tracking.py, add at start of main():
stop_file_path = os.environ.get("TG25_STOP_FILE", "")
print(f"Stop flag path: {stop_file_path}")

# In main loop, add:
if stop_file and stop_file.exists():
    print("Stop flag detected!")
```

**Solution 3: Test Manual Start**
```bash
# Set environment variable manually
set TG25_STOP_FILE=C:\path\to\TG25_STOP.flag  # Windows
export TG25_STOP_FILE=/path/to/TG25_STOP.flag  # Linux

# Run worker
TG25_HandTracking.exe

# In another terminal, create stop flag
echo "stop" > C:\path\to\TG25_STOP.flag

# Worker should exit gracefully
```

**Validation**:
- [ ] Launcher successfully starts worker
- [ ] Worker process appears in Task Manager
- [ ] Stop button terminates worker
- [ ] Status label updates correctly

---

## Network Issues

### Issue N-001: Cannot Ping Target IP

**Symptoms**:
```bash
ping 192.168.10.10
# Request timeout or Destination host unreachable
```

**Diagnosis & Solutions**:

**Step 1: Verify IP Configuration**
```bash
# Check your IP address
ipconfig  # Windows
ip addr   # Linux

# Ensure you're on same subnet as target
# Example: Your IP: 192.168.10.5, Target: 192.168.10.10 ✓
# Example: Your IP: 192.168.1.5, Target: 192.168.10.10 ✗
```

**Step 2: Check Physical Connection**
```
✓ Ethernet cable connected
✓ Link lights on network adapter blinking
✓ Same network switch/router
```

**Step 3: Configure Static IP** (if needed)
```bash
# Windows
Control Panel → Network Connections → Adapter Properties → IPv4
IP: 192.168.10.5
Subnet: 255.255.255.0
Gateway: 192.168.10.1

# Linux
sudo nano /etc/netplan/01-netcfg.yaml
# Add:
network:
  version: 2
  ethernets:
    eth0:
      addresses: [192.168.10.5/24]
      gateway4: 192.168.10.1

sudo netplan apply
```

**Validation**:
- [ ] Ping successful
- [ ] Round-trip time <50ms
- [ ] 0% packet loss

---

# 日本語版

## クイック診断

このフローチャートを使用して問題カテゴリを特定:

```
問題？
├─ カメラが検出されない → [ハードウェア問題](#ハードウェア問題)
├─ アプリケーションが起動しない → [ソフトウェア問題](#ソフトウェア問題)
├─ 手が検出されない → [検出問題](#検出問題)
├─ スワイプがトリガーされない → [スワイプ検出問題](#スワイプ検出問題)
├─ 低FPS / ラグ → [パフォーマンス問題](#パフォーマンス問題)
├─ 実行ファイルエラー → [実行ファイル問題](#実行ファイル問題)
└─ ネットワーク/UDP問題 → [ネットワーク問題](#ネットワーク問題)
```

---

## ハードウェア問題

### 問題 H-001: カメラが検出されない

**症状**:
```
Failed to connect to OAK-D: [X_LINK_DEVICE_NOT_FOUND]
RuntimeError: No devices found!
```

**診断手順**:

**手順1**: 物理接続を確認
```bash
# USBケーブルを抜き差し
# 別のUSBポートを試す（USB 3.0 - 青いポートを優先）
# ケーブルが完全に挿入されていることを確認
```

**手順2**: デバイス列挙を確認

*Windows*:
```powershell
# デバイスマネージャーを開く（devmgmt.msc）
# 「ユニバーサルシリアルバスコントローラー」を確認
# 「Movidius MyriadX」または類似のデバイスが表示されるべき

# デバイスに黄色い感嘆符が表示される場合:
# 右クリック → ドライバーの更新 → コンピューターを参照 → 一覧から選択
# 「libusbK」または「WinUSB」ドライバーを選択
```

*Linux*:
```bash
# USBデバイスを確認
lsusb | grep "03e7"
# 期待: Bus XXX Device XXX: ID 03e7:XXXX Movidius Ltd.

# 見つからない場合、dmesgを確認
dmesg | tail -20
# USB接続メッセージを探す
```

**手順3**: ドライバーをインストール/更新

*Windows - Zadigを使用*:
```
1. Zadigをダウンロード: https://zadig.akeo.ie/
2. 管理者として実行
3. オプション → すべてのデバイスをリスト
4. 「Movidius MyriadX」を選択
5. 「WinUSB」ドライバーを選択
6. 「ドライバーを置き換える」をクリック
7. システムを再起動
```

*Linux - udevルール*:
```bash
# udevルールを作成
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/80-movidius.rules

# ルールをリロード
sudo udevadm control --reload-rules
sudo udevadm trigger

# カメラを再接続
```

**手順4**: 診断ツールでテスト
```bash
python probe_dai.py
```

**解決**:
- [ ] デバイスがシステムに表示される（lsusb / デバイスマネージャー）
- [ ] ドライバーが正しくインストールされている
- [ ] probe_dai.pyがカメラを検出
- [ ] USB速度がSUPER_SPEED（USB 3.0）を表示

[続きは英語版と同じ構造で、すべての問題が日本語で説明されます]

---

**ドキュメント作成日**: 2024年10月  
**バージョン**: 1.0  
**著者**: TG_25_GestureOAK-Dチーム  
**サポート**: https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D