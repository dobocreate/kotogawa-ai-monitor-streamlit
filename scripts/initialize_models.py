#!/usr/bin/env python3
"""
基本モデルと適応モデルの初期化スクリプト
デモデータを使用して両モデルを学習
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from river_streaming_prediction_v2 import RiverStreamingPredictor
import json
import shutil
from datetime import datetime


def train_model_with_demo_data(model, demo_file_path="sample/demo_training_data.json"):
    """デモデータでモデルを学習"""
    # ファイルパスを絶対パスに変換
    demo_path = Path(__file__).parent.parent / demo_file_path
    
    if not demo_path.exists():
        raise FileNotFoundError(f"デモデータが見つかりません: {demo_path}")
    
    with open(demo_path, 'r', encoding='utf-8') as f:
        demo_data = json.load(f)
    
    print(f"デモデータで学習中... ({len(demo_data['records'])}レコード)")
    
    # プログレスバー用
    total_records = len(demo_data['records'])
    
    for i, record in enumerate(demo_data['records']):
        if i % 50 == 0:
            progress = (i / total_records) * 100
            print(f"  進捗: {i}/{total_records} ({progress:.1f}%)")
        
        # 現在のデータから特徴量を抽出
        current_data = record['current']
        
        # 予測を実行（学習のため）
        predictions = model.predict_one(current_data)
        
        # 学習実行（futureデータを直接渡す）
        model.learn_one(current_data, record['future'])
    
    # 最終進捗
    print(f"  進捗: {total_records}/{total_records} (100.0%)")
    print("学習完了")
    
    # モデルの統計情報を表示
    print(f"\n学習後のモデル統計:")
    print(f"  総学習サンプル数: {model.n_samples}")
    if hasattr(model.mae_metric, 'get'):
        mae = model.mae_metric.get()
        if mae > 0:
            print(f"  平均絶対誤差 (MAE): {mae:.3f}")
    
    return model


def initialize_all_models():
    """すべてのモデルを初期化"""
    print("=== モデル初期化開始 ===")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # モデルディレクトリの作成
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # バックアップディレクトリの作成
    backup_dir = models_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    
    # 1. 基本モデルの作成
    print("\n1. 基本モデルの作成")
    print("-" * 40)
    
    base_model = RiverStreamingPredictor()
    base_model = train_model_with_demo_data(base_model)
    
    # 基本モデルを保存
    base_model_path = models_dir / "river_base_model_v2.pkl"
    base_model.model_path = str(base_model_path)
    base_model.save_model()
    print(f"基本モデルを保存しました: {base_model_path}")
    
    # 2. 適応モデルの初期状態を作成
    print("\n2. 適応モデルの初期状態を作成")
    print("-" * 40)
    
    adaptive_model = RiverStreamingPredictor()
    adaptive_model = train_model_with_demo_data(adaptive_model)
    
    # 適応モデルの初期状態を保存
    adaptive_initial_path = models_dir / "river_adaptive_model_initial.pkl"
    adaptive_model.model_path = str(adaptive_initial_path)
    adaptive_model.save_model()
    print(f"適応モデルの初期状態を保存しました: {adaptive_initial_path}")
    
    # 3. 適応モデルとして配置
    print("\n3. 適応モデルを配置")
    print("-" * 40)
    
    adaptive_model_path = models_dir / "river_adaptive_model_v2.pkl"
    
    # 既存の適応モデルがある場合はバックアップ
    if adaptive_model_path.exists():
        backup_path = backup_dir / f"adaptive_model_backup_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        shutil.copy(adaptive_model_path, backup_path)
        print(f"既存の適応モデルをバックアップしました: {backup_path}")
    
    # 初期状態を適応モデルとしてコピー
    shutil.copy(adaptive_initial_path, adaptive_model_path)
    print(f"適応モデルを配置しました: {adaptive_model_path}")
    
    # 4. モデル情報を保存
    print("\n4. モデル情報を保存")
    print("-" * 40)
    
    model_info = {
        "initialized_at": datetime.now().isoformat(),
        "base_model": {
            "path": str(base_model_path),
            "samples": base_model.n_samples,
            "description": "デモデータで学習した基本モデル（固定）"
        },
        "adaptive_model": {
            "path": str(adaptive_model_path),
            "initial_path": str(adaptive_initial_path),
            "samples": adaptive_model.n_samples,
            "description": "デモデータで初期化した適応モデル（継続学習）"
        },
        "demo_data": {
            "path": "sample/demo_training_data.json",
            "records": 500,
            "patterns": [
                "安定した晴天時",
                "小雨時の緩やかな上昇",
                "大雨時の急激な上昇",
                "ダム放流による変化",
                "複合パターン"
            ]
        }
    }
    
    info_path = models_dir / "model_metadata.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    print(f"モデル情報を保存しました: {info_path}")
    
    print("\n=== モデル初期化完了 ===")
    print("\n初期化されたモデル:")
    print(f"  - 基本モデル: {base_model_path}")
    print(f"  - 適応モデル: {adaptive_model_path}")
    print(f"  - 適応モデル初期状態: {adaptive_initial_path}")
    print("\n次のステップ:")
    print("  1. river_dual_model_predictor.py を実装")
    print("  2. collect_data.py と streaming_train_with_diagnostics.py を更新")
    print("  3. GitHub Actions で適応モデルの週次リセットを設定")


if __name__ == "__main__":
    # まずデモデータを作成
    print("デモデータの作成を開始します...")
    from create_demo_training_data import create_demo_training_data
    create_demo_training_data()
    
    print("\n" + "=" * 50 + "\n")
    
    # モデルを初期化
    initialize_all_models()