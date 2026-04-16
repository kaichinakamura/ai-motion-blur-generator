# Project: AI Motion Blur Generator (MVP)

## 1. プロジェクト概要
高フレームレートで撮影された動画に対し、AI（RIFE）によるフレーム補間とブレンディング技術を用いて、物理的に正しい「モーションブラー」を後付けするMac用デスクトップアプリ。

## 2. 技術スタック
### Frontend (GUI)
- **Framework:** Tauri (v2推奨)
- **UI Library:** React + Tailwind CSS
- **Features:** - ファイルドラッグ＆ドロップ (D&D)
    - ネイティブファイルダイアログ（Finder連携）
    - ダークモード基調のモダンなビデオツールUI

### Backend (AI Core / Sidecar)
- **Language:** Python 3.10+
- **Package Manager:** `uv` (高速かつクリーンな環境管理)
- **AI Model:** RIFE (Real-Time Intermediate Flow Estimation)
- **ML Framework:** PyTorch (Apple Siliconの **MPS: Metal Performance Shaders** を使用)
- **Video Engine:** FFmpeg (フレーム分解、再合成、音声保持)

### Distribution
- **Output:** macOS Native Application (.app)

## 3. 画面仕様 (UI/UX)
1. **入力エリア:**
   - 画面中央に大きなD&Dエリア。「またはファイルを選択」ボタンでFinder起動。
2. **ブラー設定:**
   - **Shutter Angle (Slider/Input):** 0° 〜 360° (デフォルト 180°)。
3. **出力設定 (Export Settings):**
   - **Format:** Dropdown (.mp4, .mov)
   - **Resolution:** Input/Dropdown (Original, 1080p, 4K等)
   - **Bitrate:** Input (Mbps)
4. **進行状況:**
   - プログレスバーと現在のステータス表示（例: "AI補間中...", "レンダリング中..."）

## 4. 処理ロジック (コア仕様)
### Step 1: フレーム分解
FFmpegを使用して、入力動画を連番画像に分解する。

### Step 2: フレーム補間 (AI)
RIFEを用いて、オリジナルフレーム間に中間フレームを生成する。
- 補間倍率: 固定（例: 8倍）またはシャッターアングルに応じて動的に決定。

### Step 3: モーションブラー合成 (Weighted Blending)
指定されたシャッターアングルに基づき、補間されたサブフレームを積算・平均化して1枚の「ブラー付きフレーム」を生成する。
- **計算式:** シャッターアングルが $180^\circ$ の場合、1フレーム分の時間のうち前後の $50\%$ に相当するサブフレームのみを合成に使用する。

### Step 4: 動画結合
合成された連番画像をFFmpegで結合。元の動画から音声を抽出し、同期させて最終出力ファイルを作成する。

## 5. 開発環境の構築方針 (Antigravityへの指示)
1. **Tauriプロジェクトの初期化:** `pnpm create tauri-app`
2. **Python Sidecarのセットアップ:** - `src-tauri/bin/python` ディレクトリを作成。
   - `uv` を使用して、`torch` (MPS対応), `opencv-python`, `ffmpeg-python` を含む実行環境を構築。
3. **通信プロトコル:**
   - Tauri (Rust) から Python Sidecar を `Command` API経由で呼び出し、JSON形式で進捗をフロントエンドに通知する。

## 6. 注意事項
- **Apple Silicon最適化:** `torch.device("mps")` を明示的に使用すること。
- **メモリ管理:** 4K動画処理時にVRAM（共有メモリ）が不足しないよう、フレーム単位またはタイル単位での処理を検討すること。