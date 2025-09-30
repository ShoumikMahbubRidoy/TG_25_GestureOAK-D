# TG_25_GestureOAK-D

Real-time **hand detection** and **swipe gesture recognition** system built for the [Luxonis OAK-D-PRO](https://shop.luxonis.com/products/oak-d-pro).  
Optimized for **80–160 cm operating distance**, with IR-based robustness and UDP integration.

---

## 📌 What is done?
- **Hand Detection** using stereo IR cameras + depth filtering.  
- **Swipe Detection** (left-to-right) with distance, velocity, and debounce logic.  
- **UDP Messaging** on confirmed swipe to `192.168.10.10:6001`.  
- **Performance Optimizations**: non-blocking queues, OpenCV runtime tuning.  
- **Gesture Hooks (WIP)**: placeholders for finger-count and multi-hand classification.

---

## 🤔 Why is this done?
- Original depthai tracker struggled at **mid-range distances**.  
- Needed **stable swipe events** to control robots.  
- IR-based pipeline makes it usable in **dark or mixed-light environments**.  

---

## ⚙️ How is this done?
- **DepthAI Pipeline**: palm NN → postproc → landmark NN → IR-enhanced frames.  
- **Swipe Detector**: buffered trajectory → min distance + velocity check.  
- **Depth Filtering**: keeps stable hands in **300–2000 mm range**.  
- **IR Enhancement**: CLAHE + bilateral filter for edge-preserving contrast.  

---

## 📦 Requirements
- Python **3.10+**  
- DepthAI SDK (`depthai`)  
- OpenCV ≥ 4.8  
- NumPy  
- imutils  
- PyYAML (optional config)

See [`requirements.txt`](requirements.txt) for full list.

---

## 🛠️ Installation

**Clone:**
```bash
git clone https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D.git
cd TG_25_GestureOAK-D
```

**Setup venv:**
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\Activate.ps1  # Windows
```

**Install deps:**
```bash
pip install -r requirements.txt
```

## 📂 Directory Tree
```css
TG_25_GestureOAK-D/
├── main.py
├── README.md
├── requirements.txt
├── models/
│   ├── palm_detection_sh4.blob
│   ├── hand_landmark_lite_sh4.blob
│   └── PDPostProcessing_top2_sh1.blob
├── src/gesture_oak/
│   ├── apps/
│   │   └── hand_tracking_app.py
│   ├── detection/
│   │   ├── hand_detector.py
│   │   └── swipe_detector.py
│   ├── logic/
│   │   └── gesture_classifier.py
│   └── utils/
│       ├── mediapipe_utils.py
│       ├── FPS.py
│       └── template_manager_script_solo.py
└── docs/
    ├── application-architecture.md
    ├── implementation-tasks.md
    └── troubleshooting.md

```

## 🚀 How to Run
```bash
uv run python main.py
```

Menu:
```markdown
1. Test camera connection
2. Run hand tracking app
3. Run swipe detection app
4. Run motion-based swipe detection
5. Exit
```

## Workflow
- Choose 2/3 for hand tracking.
- Place hand 80–160 cm from OAK-D camera.
- Perform a left-to-right swipe.
- Observe console logs and UDP packet output.

## 📈 Known Issues
- Left hand less reliable beyond ~100 cm.
- Background objects (cloth, hair, ear, etc.) may cause false positives.
- FPS reporting bug: unrealistic values (>200k fps) are artifacts.

## 🗺️ Roadmap
- Finger-count gestures (1–5, peace, fist).
- Multi-hand support.
- Custom dataset training for robustness.
- Integrate MediaPipe HandLandmarker.
- Fix FPS counter (target 25–60 fps realistic).

## 📜 License
**MIT**
```yaml

---

👉 Now the README includes **everything in one file**:  
- What / Why / How  
- Requirements  
- Install  
- Directory Tree  
- Run + Menu + Workflow  
- Issues  
- Roadmap  

No parts are floating outside anymore.  

Do you also want me to create the **new `requirements.txt`** (cleaned, with only exact deps you need) right now?
```