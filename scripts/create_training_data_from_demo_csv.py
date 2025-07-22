#!/usr/bin/env python3
"""
デモモードのCSVファイルから学習用データを作成するスクリプト
実際の厚東川データ（2023年6月25日〜7月1日）を使用
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))


def create_training_data_from_demo_csv():
    """デモモードのCSVファイルから学習用データを作成"""
    
    # CSVファイルのパス
    dam_csv_path = Path(__file__).parent.parent / "sample" / "dam_20230625-20230701.csv"
    water_csv_path = Path(__file__).parent.parent / "sample" / "water-level_20230625-20230701.csv"
    
    if not dam_csv_path.exists() or not water_csv_path.exists():
        print("エラー: デモモードのCSVファイルが見つかりません")
        return None
    
    print("デモモードのCSVファイルから学習用データを作成中...")
    
    # ダムデータの読み込み（Shift-JISエンコーディング）
    dam_df = pd.read_csv(dam_csv_path, encoding='shift_jis', skiprows=7)
    dam_df.columns = ['timestamp', 'hourly_rain', 'cumulative_rain', 'water_level', 
                     'storage_rate', 'inflow', 'outflow', 'storage_change']
    
    # 河川水位データの読み込み
    water_df = pd.read_csv(water_csv_path, encoding='shift_jis', skiprows=6)
    if len(water_df.columns) == 4:
        water_df.columns = ['timestamp', 'water_level', 'level_change', 'empty']
        water_df = water_df.drop('empty', axis=1)
    else:
        water_df.columns = ['timestamp', 'water_level', 'level_change']
    
    # タイムスタンプのクリーニング
    dam_df['clean_timestamp'] = dam_df['timestamp'].astype(str).str.replace('　', '').str.strip()
    water_df['clean_timestamp'] = water_df['timestamp'].astype(str).str.replace('　', '').str.strip()
    
    # 数値データのクリーニング
    dam_df['hourly_rain'] = pd.to_numeric(dam_df['hourly_rain'], errors='coerce').fillna(0)
    dam_df['cumulative_rain'] = pd.to_numeric(dam_df['cumulative_rain'], errors='coerce').fillna(0)
    dam_df['water_level'] = pd.to_numeric(dam_df['water_level'], errors='coerce')
    dam_df['storage_rate'] = pd.to_numeric(dam_df['storage_rate'], errors='coerce')
    dam_df['inflow'] = pd.to_numeric(dam_df['inflow'], errors='coerce')
    dam_df['outflow'] = pd.to_numeric(dam_df['outflow'], errors='coerce')
    
    water_df['water_level'] = pd.to_numeric(water_df['water_level'], errors='coerce')
    water_df['level_change'] = pd.to_numeric(water_df['level_change'], errors='coerce').fillna(0)
    
    # データを結合して学習用データを作成
    training_data = {
        "description": "厚東川実データ（2023年6月25日〜7月1日）に基づく学習用データ",
        "created_at": datetime.now().isoformat(),
        "source": "demo_csv_files",
        "period": "2023-06-25 to 2023-07-01",
        "records": []
    }
    
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
                'storage_change': float(row['storage_change']) if pd.notna(row['storage_change']) else None
            },
            'river': {
                'water_level': float(water_row['water_level'].iloc[0]) if not water_row.empty and pd.notna(water_row['water_level'].iloc[0]) else None,
                'level_change': float(water_row['level_change'].iloc[0]) if not water_row.empty and pd.notna(water_row['level_change'].iloc[0]) else 0.0,
                'status': '正常'
            },
            'rainfall': {
                'hourly': int(row['hourly_rain']) if pd.notna(row['hourly_rain']) else 0,
                'cumulative': int(row['cumulative_rain']) if pd.notna(row['cumulative_rain']) else 0,
                'rainfall_10min': int(row['hourly_rain'] / 6) if pd.notna(row['hourly_rain']) else 0,  # 時間雨量から推定
                'rainfall_1h': int(row['hourly_rain']) if pd.notna(row['hourly_rain']) else 0
            }
        }
        
        all_data.append(data_point)
    
    print(f"読み込んだデータ数: {len(all_data)}")
    
    # 学習用データの作成（18ステップ先まで必要）
    for i in range(len(all_data) - 18):
        current = all_data[i]
        
        # 現在のデータが有効か確認
        if not current.get('river', {}).get('water_level'):
            continue
        
        # 学習用レコードを作成
        record = {
            "timestamp": current['timestamp'],
            "current": {
                "data_time": current['timestamp'],
                "river": current.get('river', {}),
                "dam": current.get('dam', {}),
                "rainfall": current.get('rainfall', {})
            },
            "future": []
        }
        
        # 未来18ステップ分のデータ（10分～180分先）
        valid_record = True
        for j in range(1, 19):
            if i + j >= len(all_data):
                valid_record = False
                break
            
            future_data = all_data[i + j]
            if not future_data.get('river', {}).get('water_level'):
                valid_record = False
                break
            
            record["future"].append({
                "timestamp": future_data['timestamp'],
                "river": future_data.get('river', {})
            })
        
        if valid_record and len(record["future"]) == 18:
            training_data["records"].append(record)
    
    print(f"作成した学習レコード数: {len(training_data['records'])}")
    
    # 学習データを保存
    output_path = Path(__file__).parent.parent / "sample" / "demo_csv_training_data.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"学習データを保存しました: {output_path}")
    
    # データの統計情報を表示
    if training_data["records"]:
        # 水位の範囲を計算
        river_levels = []
        dam_levels = []
        for record in training_data["records"]:
            river_level = record["current"].get("river", {}).get("water_level")
            dam_level = record["current"].get("dam", {}).get("water_level")
            if river_level:
                river_levels.append(river_level)
            if dam_level:
                dam_levels.append(dam_level)
        
        if river_levels:
            print(f"\n河川水位統計:")
            print(f"  最小値: {min(river_levels):.2f}m")
            print(f"  最大値: {max(river_levels):.2f}m")
            print(f"  平均値: {sum(river_levels)/len(river_levels):.2f}m")
        
        if dam_levels:
            print(f"\nダム水位統計:")
            print(f"  最小値: {min(dam_levels):.2f}m")
            print(f"  最大値: {max(dam_levels):.2f}m")
            print(f"  平均値: {sum(dam_levels)/len(dam_levels):.2f}m")
    
    return output_path


def train_models_with_demo_csv():
    """デモCSVデータでモデルを学習"""
    from river_streaming_prediction import RiverStreamingPredictor
    
    # 学習データを作成
    data_path = create_training_data_from_demo_csv()
    if not data_path:
        return
    
    # 学習データを読み込み
    with open(data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    if not training_data.get("records"):
        print("エラー: 学習データが空です")
        return
    
    print(f"\nデモCSVデータでモデルを学習中... ({len(training_data['records'])}レコード)")
    
    # 新しいモデルを作成
    model = RiverStreamingPredictor()
    
    # 学習実行
    total_records = len(training_data['records'])
    for i, record in enumerate(training_data['records']):
        if i % 100 == 0:
            print(f"  進捗: {i}/{total_records} ({i/total_records*100:.1f}%)")
        
        # 現在のデータから予測
        current_data = record['current']
        predictions = model.predict_one(current_data)
        
        # 学習実行
        model.learn_one(current_data, record['future'])
    
    print(f"  進捗: {total_records}/{total_records} (100.0%)")
    print("学習完了")
    
    # モデルを保存
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # 基本モデルとして保存
    base_model_path = models_dir / "river_base_model_v2.pkl"
    model.model_path = str(base_model_path)
    model.save_model()
    print(f"\n基本モデルを保存しました: {base_model_path}")
    
    # 適応モデルの初期状態としても保存
    adaptive_model_path = models_dir / "river_adaptive_model_v2.pkl"
    model.model_path = str(adaptive_model_path)
    model.save_model()
    print(f"適応モデル（初期状態）を保存しました: {adaptive_model_path}")
    
    print(f"\n学習サンプル数: {model.n_samples}")
    
    # 精度情報を表示
    if hasattr(model.mae_metric, 'get'):
        mae = model.mae_metric.get()
        if mae > 0:
            print(f"平均絶対誤差 (MAE): {mae:.3f}m")


if __name__ == "__main__":
    train_models_with_demo_csv()