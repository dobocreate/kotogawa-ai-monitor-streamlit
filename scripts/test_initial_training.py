#!/usr/bin/env python3
"""
初期学習のテストスクリプト（手動実行用）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# 必要な場合のみコメントアウトを外す
# import subprocess
# subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "river", "--user"])

from create_training_data_from_demo_csv import train_models_with_demo_csv

if __name__ == "__main__":
    print("初期学習を実行します...")
    train_models_with_demo_csv()