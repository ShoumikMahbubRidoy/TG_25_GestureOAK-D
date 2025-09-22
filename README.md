# OAK-D Gesture Recognition

OAK-D AIカメラを使用してリアルタイムで手のジェスチャーを認識するPythonアプリケーションです。

## プロジェクト構造

```
TG_25_GestureOAK-D/
├── src/gesture_oak/           # メインパッケージ
│   ├── core/                  # コアカメラ機能
│   │   └── oak_camera.py     # OAK-Dカメララッパー
│   ├── detection/             # 手検出モジュール
│   │   └── hand_detector.py  # MediaPipe手検出器
│   ├── utils/                 # ユーティリティモジュール
│   │   ├── FPS.py            # FPSカウンター
│   │   ├── mediapipe_utils.py # MediaPipeユーティリティ
│   │   └── template_manager_script_solo.py
│   └── demos/                 # デモアプリケーション
│       └── hand_detection_demo.py
├── models/                    # AIモデル (.blobファイル)
├── scripts/                   # 開発用スクリプト
├── tests/                     # ユニットテスト
├── docs/                      # ドキュメント
├── main.py                    # メインアプリケーションエントリー
└── pyproject.toml            # プロジェクト設定
```

## 機能

- ✅ OAK-Dカメラ接続とセットアップ
- ✅ MediaPipeベース手検出
- ✅ リアルタイム手ランドマーク追跡
- ✅ ジェスチャー認識
- ✅ FPSモニタリング
- 🔄 ジェスチャー分類（開発中）

## インストール

### 前提条件

- Python 3.9以上
- OAK-Dカメラ
- USB-Cケーブル
- uv (Pythonパッケージマネージャー)

### セットアップ

```bash
# 依存関係のインストール
uv sync

# または開発モードでインストール
uv pip install -e .
```

## 使用方法

### メインアプリケーション
```bash
uv run python main.py
```

### 直接デモ実行
```bash
uv run python scripts/run_demo.py
```

### 開発モードでの実行
```bash
# パッケージを編集可能モードでインストール
uv sync

# アプリケーション実行
uv run python main.py
```

## 使用方法

1. アプリケーションを起動
2. カメラの前に手を置く
3. ジェスチャーを実行
4. 認識結果を画面で確認

## 対応ジェスチャー

- 指の本数（1-5本）
- 手の形状（握り拳、開いた手など）
- カスタムジェスチャー

## ファイル構造

```
TG_25_GestureOAK-D/
├── main.py                 # メインアプリケーション
├── gesture_detector.py     # ジェスチャー認識モジュール
├── oak_camera.py          # OAK-Dカメラ制御
├── utils.py               # ユーティリティ関数
├── pyproject.toml         # uvプロジェクト設定
├── requirements.txt       # 依存関係（uv用）
└── README.md             # このファイル
```

## 技術仕様

- **カメラ**: OAK-D (DepthAI)
- **手の検出**: MediaPipe Hands
- **ジェスチャー認識**: カスタムアルゴリズム
- **表示**: OpenCV
- **言語**: Python 3.8+
- **パッケージマネージャー**: uv
