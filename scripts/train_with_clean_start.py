#!/usr/bin/env python3
"""
クリーンな状態から学習を開始するスクリプト
（前のモデルの影響を受けないように）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os

def clean_start_training():
    """既存モデルを削除してから学習"""
    
    print("=== クリーンスタート学習 ===\n")
    
    # 既存モデルを一時的に移動
    model_files = [
        "models/river_base_model_v2.pkl",
        "models/river_adaptive_model_v2.pkl",
        "models/river_streaming_model_v2.pkl"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            backup_name = model_file.replace('.pkl', '_temp_backup.pkl')
            os.rename(model_file, backup_name)
            print(f"一時バックアップ: {model_file} → {backup_name}")
    
    print("\nクリーンな状態で学習を開始します...")
    print("=" * 50)
    
    # 学習を実行
    from create_training_data_from_demo_csv import train_models_with_demo_csv
    train_models_with_demo_csv()
    
    # バックアップを削除
    for model_file in model_files:
        backup_name = model_file.replace('.pkl', '_temp_backup.pkl')
        if os.path.exists(backup_name):
            os.remove(backup_name)
            print(f"\nバックアップ削除: {backup_name}")
    
    print("\n=== クリーンスタート学習完了 ===")

if __name__ == "__main__":
    clean_start_training()