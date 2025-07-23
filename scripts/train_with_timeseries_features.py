#!/usr/bin/env python3
"""
時系列特徴量とデータ前処理の最適化を含めた学習スクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os

def train_with_improvements():
    """改善されたモデルで学習"""
    
    print("=== 時系列特徴量とデータ前処理を改善した学習 ===\n")
    
    print("主な改善内容:")
    print("\n1. データ前処理の最適化")
    print("   - RobustScaler（外れ値に強い正規化）を採用")
    print("   - 外れ値検出器（HalfSpaceTrees）の導入")
    print("   - データ品質スコアの算出")
    print()
    
    print("2. 時系列特徴量の大幅追加")
    print("   - 指数移動平均（EMA）：短期・中期・長期")
    print("   - 単純移動平均（SMA）：3点、6点")
    print("   - ボリンジャーバンド的特徴量")
    print("   - 自己相関（ラグ1、3、6）")
    print("   - モメンタム（累積変化量）")
    print("   - トレンド転換検出（EMAクロスオーバー）")
    print()
    
    print("3. 物理的制約の導入")
    print("   - 変化率制限（10分あたり最大30cm）")
    print("   - トレンドベースの制約")
    print("   - データ品質による予測調整")
    print()
    
    # 既存モデルをバックアップ
    model_file = "models/river_streaming_model_v2.pkl"
    if os.path.exists(model_file):
        backup_name = model_file.replace('.pkl', '_before_timeseries.pkl')
        os.rename(model_file, backup_name)
        print(f"既存モデルをバックアップ: {backup_name}")
    
    print("\n改善されたモデルで学習を開始します...")
    print("=" * 50)
    
    # 学習を実行
    from create_training_data_from_demo_csv import train_models_with_demo_csv
    train_models_with_demo_csv()
    
    # バックアップを削除
    if os.path.exists(backup_name):
        os.remove(backup_name)
        print(f"\nバックアップ削除: {backup_name}")
    
    print("\n=== 学習完了 ===")
    print("\n期待される効果:")
    print("- より滑らかで現実的な予測")
    print("- 外れ値に対する頑健性の向上")
    print("- トレンドパターンの認識精度向上")
    print("- 目標の10cm以内により近い予測精度")

if __name__ == "__main__":
    train_with_improvements()