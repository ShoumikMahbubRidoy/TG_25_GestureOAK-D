# TG_25_GestureOAK-D

Real-time **hand detection** and **swipe gesture recognition** system built for the [Luxonis OAK-D-PRO](https://shop.luxonis.com/products/oak-d-pro).  リアルタイム **ハンド検出** と **スワイプジェスチャー認識** システム。  
[Luxonis OAK-D-PRO](https://shop.luxonis.com/products/oak-d-pro) 用に構築。  
Optimized for **80–160 cm operating distance**, with IR-based robustness and UDP integration. **80〜160 cm 動作距離** に最適化され、IR（赤外線）ベースで高い堅牢性とUDP連携を実現。

---

## 📌 What is done? 実装済み機能
- **Hand Detection ハンド検出** using stereo IR cameras + depth filtering.  ステレオIRカメラ + 深度フィルタリング。  
- **Swipe Detection スワイプ検出** (left-to-right) with distance, velocity, and debounce logic.  左→右の動きを距離・速度・デバウンスで判定。  
- **UDP Messaging 送信** on confirmed swipe to `192.168.10.10:6001`.  スワイプ確定時に `192.168.10.10:6001` へ送信。 
- **Performance Optimizations 性能最適化**: non-blocking queues, OpenCV runtime tuning.  ノンブロッキングキュー、OpenCV最適化。 
- **Gesture Hooks (WIP) ジェスチャーフック (開発中)**: placeholders for finger-count and multi-hand classification. 指の本数カウント、両手対応の拡張。

---

## 🤔 Why is this done? なぜ開発したか
- Original depthai tracker struggled at **mid-range distances**.  既存の DepthAI トラッカーは **中距離検出に弱点** があった。
- IR-based pipeline makes it usable in **dark or mixed-light environments**.  **暗所や混合光環境** でも動作可能にするため、IRベースで設計。 

---

## ⚙️ How is this done? 実現方法
- **DepthAI Pipeline DepthAI パイプライン**: palm NN → postproc → landmark NN → IR-enhanced frames.  Palm NN → PostProc → Landmark NN → IR強調フレーム。  
- **Swipe Detector スワイプ検出器**: buffered trajectory → min distance + velocity check.  バッファリングされた軌跡を距離・速度閾値で判定。  
- **Depth Filtering 深度フィルタリング**: keeps stable hands in **300–2000 mm range**.  **300〜2000 mm** の範囲で安定した手だけを保持。
- **IR Enhancement 強調処理**: CLAHE + bilateral filter for edge-preserving contrast.  CLAHE + バイラテラルフィルタで輪郭を保持しつつノイズ低減。  

---

## 📦 Requirements 必要環境
- Python **3.10+**  
- DepthAI SDK (`depthai`)  
- OpenCV ≥ 4.8  
- NumPy  
- imutils  
- PyYAML (optional config)

See [`requirements.txt`](requirements.txt) for full list.
詳細は [`requirements.txt`](requirements.txt) を参照。


---

## 🛠️ Installation インストール手順

**Clone リポジトリ取得:**
```bash
git clone https://github.com/ShoumikMahbubRidoy/TG_25_GestureOAK-D.git
cd TG_25_GestureOAK-D
```

**Setup venv 仮想環境作成:**
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\Activate.ps1  # Windows
```

**Install deps 依存関係インストール:**
```bash
pip install -r requirements.txt
```
---

## 📂 Directory Tree ディレクトリ構造
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

---

## 🚀 How to Run 実行方法
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

---

## Workflow ワークフロー
- Choose 2/3 for hand tracking. 2 または 3 を選択して実行。
- Place hand 80–160 cm from OAK-D camera. OAK-D カメラから 80〜160 cm の距離に手を置く
- Perform a left-to-right swipe. 左から右へスワイプ を行う。
- Observe console logs and UDP packet output. コンソールログとUDP送信を確認。

---

## 📈 Known Issues 既知の課題
- Left hand less reliable beyond ~100 cm. 左手は 100 cm 以上での検出が不安定。
- Background objects (cloth, hair, ear, etc.) may cause false positives. 背景の布・髪・耳などが誤検出される可能性あり。
- FPS reporting bug: unrealistic values (>200k fps) are artifacts. FPS表示にバグがあり、20万fps以上の非現実的値が出ることがある。

---

## 🗺️ Roadmap 今後のロードマップ
- Finger-count gestures (1–5, peace, fist). 指本数のジェスチャー認識（1〜5本、ピース、グー）。
- Multi-hand support. 両手同時サポート。
- Custom dataset training for robustness. 独自データセットでの学習による精度向上。
- Integrate MediaPipe HandLandmarker. MediaPipe HandLandmarker との統合。
- Fix FPS counter (target 25–60 fps realistic). FPSカウンタ修正（目標: 実測25〜60fps）。

---

## 📜 License ライセンス
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

```