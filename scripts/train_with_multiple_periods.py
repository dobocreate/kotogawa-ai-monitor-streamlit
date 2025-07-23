#!/usr/bin/env python3
"""
複数期間のデータを使用した学習スクリプト
2023年と2024年のデータを組み合わせて学習
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import json
from datetime import datetime
import os

def load_csv_data(dam_path, water_path, year_label):
    """CSVデータを読み込んで前処理"""
    print(f"\n{year_label}のデータを読み込み中...")
    
    # ダムデータの読み込み
    dam_df = pd.read_csv(dam_path, encoding='shift_jis', skiprows=7)
    dam_df.columns = ['timestamp', 'hourly_rain', 'cumulative_rain', 'water_level', 
                     'storage_rate', 'inflow', 'outflow', 'storage_change']
    
    # 河川水位データの読み込み
    water_df = pd.read_csv(water_path, encoding='shift_jis', skiprows=6)
    if len(water_df.columns) == 4:
        water_df.columns = ['timestamp', 'water_level', 'level_change', 'empty']
        water_df = water_df.drop('empty', axis=1)
    else:
        water_df.columns = ['timestamp', 'water_level', 'level_change']
    
    # タイムスタンプのクリーニング
    dam_df['clean_timestamp'] = dam_df['timestamp'].astype(str).str.replace('　', '').str.strip()
    water_df['clean_timestamp'] = water_df['timestamp'].astype(str).str.replace('　', '').str.strip()
    
    # 数値データのクリーニング
    for col in ['hourly_rain', 'cumulative_rain', 'water_level', 'storage_rate', 'inflow', 'outflow']:
        if col in dam_df.columns:
            dam_df[col] = pd.to_numeric(dam_df[col], errors='coerce')
    
    water_df['water_level'] = pd.to_numeric(water_df['water_level'], errors='coerce')
    water_df['level_change'] = pd.to_numeric(water_df['level_change'], errors='coerce').fillna(0)
    
    print(f"  ダムデータ: {len(dam_df)}行")
    print(f"  河川データ: {len(water_df)}行")
    
    return dam_df, water_df

def create_combined_training_data():
    """複数期間のデータを組み合わせて学習データを作成"""
    
    # 全ての学習レコードを格納
    all_records = []
    
    # 2023年データ
    dam_2023 = Path(__file__).parent.parent / "sample" / "dam_20230625-20230701.csv"
    water_2023 = Path(__file__).parent.parent / "sample" / "water-level_20230625-20230701.csv"
    
    if dam_2023.exists() and water_2023.exists():
        dam_df_2023, water_df_2023 = load_csv_data(dam_2023, water_2023, "2023年")
        records_2023 = create_records_from_dataframes(dam_df_2023, water_df_2023, "2023")
        all_records.extend(records_2023)
        print(f"  2023年の学習レコード数: {len(records_2023)}")
    
    # 2024年データ
    dam_2024 = Path(__file__).parent.parent / "sample" / "dam_20240625-20240702.csv"
    water_2024 = Path(__file__).parent.parent / "sample" / "water-level_20240625-20240702.csv"
    
    if dam_2024.exists() and water_2024.exists():
        dam_df_2024, water_df_2024 = load_csv_data(dam_2024, water_2024, "2024年")
        records_2024 = create_records_from_dataframes(dam_df_2024, water_df_2024, "2024")
        all_records.extend(records_2024)
        print(f"  2024年の学習レコード数: {len(records_2024)}")
    
    print(f"\n合計学習レコード数: {len(all_records)}")
    
    # 学習データを保存
    training_data = {
        "description": "厚東川実データ（2023年・2024年の梅雨時期）",
        "created_at": datetime.now().isoformat(),
        "source": "multiple_csv_files",
        "periods": ["2023-06-25 to 2023-07-01", "2024-06-25 to 2024-07-02"],
        "records": all_records
    }
    
    output_path = Path(__file__).parent.parent / "sample" / "combined_training_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"学習データを保存しました: {output_path}")
    
    # 統計情報を表示
    if all_records:
        river_levels = [r['current']['river']['water_level'] for r in all_records 
                       if r['current']['river']['water_level'] is not None]
        dam_levels = [r['current']['dam']['water_level'] for r in all_records 
                     if r['current']['dam']['water_level'] is not None]
        
        print(f"\n河川水位統計:")
        print(f"  最小値: {min(river_levels):.2f}m")
        print(f"  最大値: {max(river_levels):.2f}m")
        print(f"  平均値: {sum(river_levels)/len(river_levels):.2f}m")
        
        print(f"\nダム水位統計:")
        print(f"  最小値: {min(dam_levels):.2f}m")
        print(f"  最大値: {max(dam_levels):.2f}m")
        print(f"  平均値: {sum(dam_levels)/len(dam_levels):.2f}m")
    
    return output_path

def create_records_from_dataframes(dam_df, water_df, year_label):
    """DataFrameから学習レコードを作成"""
    records = []
    
    # 全データをリスト化
    all_data = []
    
    for idx, row in dam_df.iterrows():
        clean_timestamp = row['clean_timestamp']
        if pd.isna(clean_timestamp) or clean_timestamp == '' or clean_timestamp == 'nan':
            continue
        
        # タイムスタンプの解析
        dt = None
        for fmt in ['%Y/%m/%d %H:%M', '%Y/%m/%d %H:%M:%S']:
            try:
                dt = datetime.strptime(clean_timestamp, fmt)
                break
            except ValueError:
                continue
        
        if dt is None:
            continue
        
        # ISO形式のタイムスタンプ
        iso_timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S+09:00')
        
        # 対応する河川データを探す
        water_row = water_df[water_df['clean_timestamp'] == clean_timestamp]
        
        # データポイントを作成
        data_point = {
            'timestamp': iso_timestamp,
            'data_time': iso_timestamp,
            'dam': {
                'water_level': float(row['water_level']) if pd.notna(row['water_level']) else None,
                'storage_rate': float(row['storage_rate']) if pd.notna(row['storage_rate']) else None,
                'inflow': float(row['inflow']) if pd.notna(row['inflow']) else None,
                'outflow': float(row['outflow']) if pd.notna(row['outflow']) else None,
            },
            'river': {
                'water_level': None,
                'level_change': None
            },
            'rainfall': {
                'hourly': float(row['hourly_rain']) if pd.notna(row['hourly_rain']) else 0,
                'cumulative': float(row['cumulative_rain']) if pd.notna(row['cumulative_rain']) else 0
            }
        }
        
        # 河川データがある場合は追加
        if not water_row.empty:
            water_level = water_row.iloc[0]['water_level']
            level_change = water_row.iloc[0]['level_change']
            if pd.notna(water_level):
                data_point['river']['water_level'] = float(water_level)
                data_point['river']['level_change'] = float(level_change) if pd.notna(level_change) else 0
        
        # 有効なデータのみ追加
        if (data_point['dam']['water_level'] is not None and 
            data_point['river']['water_level'] is not None):
            all_data.append(data_point)
    
    # データを時系列順にソート
    all_data.sort(key=lambda x: x['timestamp'])
    
    # 学習レコードを作成（3時間先までの実測値を含む）
    for i in range(len(all_data) - 18):
        current = all_data[i]
        future = all_data[i+1:i+19]  # 次の18個（10分刻みで3時間分）
        
        # 将来データが十分にあるか確認
        if len(future) == 18:
            record = {
                'current': current,
                'future': future
            }
            records.append(record)
    
    return records

def train_with_combined_data():
    """組み合わせたデータで学習"""
    print("=== 複数期間データによる学習 ===\n")
    
    # 学習データを作成
    data_path = create_combined_training_data()
    
    # 既存モデルをバックアップ
    model_file = "models/river_streaming_model_v2.pkl"
    if os.path.exists(model_file):
        backup_name = model_file.replace('.pkl', '_before_combined.pkl')
        os.rename(model_file, backup_name)
        print(f"\n既存モデルをバックアップ: {backup_name}")
    
    print("\n組み合わせたデータで学習を開始します...")
    print("=" * 50)
    
    # create_training_data_from_demo_csv.pyの関数を使用
    from create_training_data_from_demo_csv import train_models_with_json
    
    # JSONファイルから学習
    with open(data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    train_models_with_json(training_data)
    
    # バックアップを削除
    if 'backup_name' in locals() and os.path.exists(backup_name):
        os.remove(backup_name)
        print(f"\nバックアップ削除: {backup_name}")
    
    print("\n=== 学習完了 ===")
    print("\n期待される効果:")
    print("- より多様なパターンの学習")
    print("- 年度間の変動への対応")
    print("- 予測精度のさらなる向上")

if __name__ == "__main__":
    train_with_combined_data()