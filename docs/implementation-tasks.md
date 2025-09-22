# OAK-D ジェスチャー認識 実装タスクリスト

## プロジェクト概要
OAK-D AIカメラを使用したMediaPipeベースのリアルタイム手検出・ジェスチャー認識システムの実装

### アーキテクチャ
- **Hand Detection**: MediaPipe Palm Detection (128x128)
- **Hand Landmarks**: MediaPipe Hand Landmarks (224x224)
- **Gesture Recognition**: MediaPipe Gesture Classification
- **推論場所**: OAK-D VPU (エッジ推論)

---

## Phase 1: 基盤構築 ✅ **完了**

### ✅ Task 1: OAK-Dカメラの基本セットアップと接続確認
- [x] depthai-coreライブラリのインストール
- [x] OAK-Dデバイスの検出・接続確認
- [x] 基本的なカメラ制御クラスの実装 (`OAKCamera`)
- [x] デバイス情報の取得機能

### ✅ Task 2: RGB映像の取得とプレビュー表示機能の実装
- [x] RGB映像ストリームの設定
- [x] フレーム取得ループの実装
- [x] OpenCVでのリアルタイム表示
- [x] FPS計測・表示機能 (`FPS`クラス)

---

## Phase 2: MediaPipe Hand Detection実装 ✅ **完了**

### ✅ Task 3: MediaPipe Palm Detectionモデルの統合
- [x] palm_detection_sh4.blobモデルの準備
- [x] 128x128入力サイズでの前処理実装
- [x] OAK-D Neural Networkノードの設定
- [x] 検出結果の座標変換・正規化

### ✅ Task 4: MediaPipe Hand Landmarksモデルの統合
- [x] hand_landmark_lite_sh4.blobモデルの準備
- [x] 224x224入力サイズでの前処理実装
- [x] 21点手ランドマークの検出・描画
- [x] 手の向き（左右）判定機能

### ✅ Task 5: HandDetectorクラスの実装
- [x] DepthAIパイプラインの構築
- [x] Manager Scriptによる推論制御
- [x] 手検出結果の抽出・処理
- [x] エラーハンドリング実装

---

## Phase 3: Gesture Recognition実装 ✅ **完了**

### ✅ Task 6: MediaPipeベースジェスチャー認識
- [x] 基本的なジェスチャーパターンの実装
- [x] 手ランドマークからの特徴抽出
- [x] ジェスチャー分類結果の表示
- [x] 信頼度スコアの表示

### ✅ Task 7: 手検出デモアプリケーションの作成
- [x] リアルタイム手検出・追跡機能
- [x] 手ランドマーク・バウンディングボックス描画
- [x] ジェスチャー認識結果の表示
- [x] フレーム保存機能

---

## Phase 4: 統合・最適化 ✅ **完了**

### ✅ Task 8: メインアプリケーションの統合
- [x] カメラテスト機能
- [x] 手検出デモ選択機能
- [x] インタラクティブメニューシステム
- [x] エラーハンドリング・例外処理

### ✅ Task 9: パフォーマンス監視とUI改善
- [x] FPSカウンター表示
- [x] 検出手数の表示
- [x] セッション統計情報
- [x] リアルタイム情報表示

---

## Phase 5: スワイプ検出機能 ✅ **完了**

### ✅ Task 10: 左から右へのスワイプ検出機能の実装
- [x] SwipeDetectorクラスの設計・実装
- [x] 多層検証システム（方向・速度・直線性・継続性）
- [x] 状態機械による段階的検出
- [x] 誤検知フィルタリング機能

### ✅ Task 11: スワイプ検出デモアプリケーションの作成
- [x] 手検出デモへのスワイプ検出統合
- [x] スワイプ専用デモアプリケーション
- [x] リアルタイム軌跡表示・進行度表示
- [x] 3段階精度設定（Strict/Normal/Loose）

### ✅ Task 12: 視覚的フィードバックと統計機能
- [x] スワイプ検出時のアニメーション表示
- [x] リアルタイム統計情報（検出数・除外数）
- [x] 詳細なデバッグ情報表示
- [x] セッション統計とパフォーマンス監視

### ✅ Task 13: パラメータ調整機能とドキュメント
- [x] 用途別プリセット設定
- [x] リアルタイムモード切替機能
- [x] 設定・調整ガイドの作成
- [x] クラス構成図とAPI仕様書

---

## Phase 6: 今後の拡張予定 🔄

### 🔄 Task 14: 他方向スワイプ検出の拡張
- [ ] 右から左へのスワイプ検出
- [ ] 上下方向のスワイプ検出
- [ ] 斜め方向のスワイプ検出
- [ ] 複合ジェスチャー（L字、円形等）

### 🔄 Task 15: 深度情報を活用した3D手追跡
- [ ] OAK-Dの深度センサーの活用
- [ ] 3D座標での手位置追跡
- [ ] 奥行き方向のジェスチャー認識
- [ ] 3D空間でのインタラクション

### 🔄 Task 16: 複数手同時追跡とID管理
- [ ] 複数の手の同時検出・追跡
- [ ] 手のID割り当て・管理機能
- [ ] 個別手の状態管理
- [ ] 両手協調ジェスチャー認識

### 🔄 Task 17: 高度な動作パターン認識
- [ ] 手の動作軌跡の記録機能
- [ ] 時系列データの解析
- [ ] 複雑な動作パターンの認識
- [ ] 機械学習による動作分類

### 🔄 Task 18: 設定管理とUI改善
- [ ] 設定ファイル（YAML/JSON）の実装
- [ ] リアルタイムパラメータ調整UI
- [ ] プロファイル別設定管理
- [ ] Webベース設定インターフェース

---

## 技術スペック

### 使用技術
- **Camera**: OAK-D (DepthAI)
- **Deep Learning Framework**: OpenVINO / MediaPipe
- **Hand Detection**: MediaPipe Palm Detection
- **Hand Landmarks**: MediaPipe Hand Landmarks  
- **Gesture Recognition**: MediaPipe Gesture Classification
- **Language**: Python 3.8+
- **Package Manager**: uv

### 現在の性能
- **FPS**: ~30 FPS (640x480)
- **レイテンシ**: リアルタイム処理
- **検出精度**: MediaPipeベース高精度
- **メモリ使用量**: 効率的なVPU利用

### 実装済みファイル構造
```
src/gesture_oak/
├── core/
│   └── oak_camera.py          # OAK-Dカメラ制御
├── detection/
│   ├── hand_detector.py       # MediaPipe手検出エンジン
│   └── swipe_detector.py      # スワイプ検出エンジン
├── apps/
│   ├── hand_tracking_app.py   # 手検出アプリケーション
│   └── swipe_detection_app.py # スワイプ検出アプリケーション
└── utils/
    ├── FPS.py                 # FPS計測
    └── mediapipe_utils.py     # MediaPipe utilities

docs/
├── implementation-tasks.md    # 実装タスクリスト
├── troubleshooting.md        # トラブルシューティング
├── swipe-detection-guide.md  # スワイプ検出設定ガイド
└── application-architecture.md # アプリケーション全体構成図
```

---

## 注意点・制約
- OAK-D Myriad X VPUのメモリ制限 (512MB)
- MediaPipeモデルの入力サイズ制約
- リアルタイム性の要求
- OpenVINO 2021.4対応