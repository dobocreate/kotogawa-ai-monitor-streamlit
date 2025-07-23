#!/usr/bin/env python3
"""
精度向上のために最適化されたモデルを学習するスクリプト
目標：30分～180分の予測誤差を10cm以内に
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from create_training_data_from_demo_csv import train_models_with_demo_csv
import subprocess

def main():
    print("=== 高精度予測モデルの学習 ===\n")
    
    print("改善内容:")
    print("1. モデルパラメータの最適化")
    print("   - モデル数: 15→50（メイン）、予測時間に応じて25-40")
    print("   - 木の深さ: 15→30（メイン）、予測時間に応じて20-25")
    print("   - より敏感なドリフト検出")
    print()
    
    print("2. 特徴量エンジニアリングの改善")
    print("   - 水位変化率（時間あたり）を追加")
    print("   - 加速度（変化率の変化）を追加")
    print("   - 急激な変化の検出フラグ")
    print("   - ダム放流変化の影響係数")
    print()
    
    print("3. 学習アルゴリズムの改善")
    print("   - 急激な変化パターンを重点的に学習")
    print("   - 10cm以上の変化は最大3回繰り返し学習")
    print("   - 長期予測でも5cm以上の変化を重視")
    print()
    
    print("学習を開始します...")
    print("-" * 50)
    
    # デモCSVデータで学習
    train_models_with_demo_csv()
    
    print("\n=== 学習完了 ===")
    print("\n期待される効果:")
    print("- 水位の急激な変化をより正確に予測")
    print("- 30分～180分先の予測誤差を大幅削減")
    print("- ベースラインへの引っ張りを解消")
    print()
    print("次のステップ:")
    print("1. AI学習結果ページで新しい精度を確認")
    print("2. 特に30分、60分、120分の予測誤差をチェック")
    print("3. 目標の10cm以内に近づいているか評価")

if __name__ == "__main__":
    main()