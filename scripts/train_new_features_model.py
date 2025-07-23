#!/usr/bin/env python3
"""
新しい特徴量で基本モデルと適応モデルを初期学習するスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from create_training_data_from_demo_csv import train_models_with_demo_csv
import subprocess

def main():
    print("=== 新しい特徴量でモデルを初期学習 ===\n")
    
    print("1. 既存モデルのバックアップ完了")
    print("   - models/backup_20250723_204109/ に保存済み\n")
    
    print("2. 新しい特徴量で基本モデルと適応モデルを学習中...")
    print("   - 過去90分の履歴データを使用")
    print("   - 50以上の新しい特徴量を生成")
    print("   - 遅延パターンを自動学習\n")
    
    # デモCSVデータで学習
    train_models_with_demo_csv()
    
    print("\n=== 学習完了 ===")
    print("\n次のステップ:")
    print("1. AI学習結果ページで精度を確認")
    print("2. 新しい特徴量の効果を評価")
    print("3. 必要に応じて追加の改良を実施")

if __name__ == "__main__":
    main()