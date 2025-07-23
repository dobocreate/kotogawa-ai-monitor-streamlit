#!/usr/bin/env python3
"""
デモモードのCSVファイルから学習用データを作成するスクリプト
実際の厚東川データ（2023年6月25日〜7月1日）を使用
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, Any

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


def create_diagnostics_for_initial_model(model, training_records: list) -> Dict[str, Any]:
    """初期モデル学習の診断データを生成"""
    from river import metrics
    
    print("\n診断データを生成中...")
    
    # 各ステップのメトリクスを初期化
    step_metrics = {}
    for minutes in range(10, 190, 10):
        step_metrics[f"{minutes}min"] = {
            'mae': metrics.MAE(),
            'rmse': metrics.RMSE()
        }
    
    # 最後の100レコードで精度を評価
    # ただし、未来データがある分だけ前から取る（18ステップ先まで必要）
    eval_start = max(0, len(training_records) - 100 - 18)
    eval_end = len(training_records) - 18
    eval_records = training_records[eval_start:eval_end] if eval_end > eval_start else []
    
    print(f"評価レコード数: {len(eval_records)}")
    evaluation_count = 0
    prediction_samples = []
    
    for i, record in enumerate(eval_records):
        current_data = record['current']
        predictions = model.predict_one(current_data)
        
        # 最初の数個の予測をサンプリング
        if i < 3 and predictions:
            prediction_samples.append({
                'index': i,
                'predictions': predictions[:3] if isinstance(predictions, list) else predictions
            })
        
        # predictionsがリストの場合の処理
        if isinstance(predictions, list) and len(predictions) >= 18:
            # 各ステップの予測と実測値を比較
            for j in range(18):
                pred = predictions[j]
                future_data = record['future'][j]
                minutes = (j + 1) * 10
                step_key = f"{minutes}min"
                
                if step_key in step_metrics:
                    # 予測値を取得（辞書の場合は'level'キー、そうでなければ直接の値）
                    if isinstance(pred, dict):
                        pred_value = pred.get('level', pred.get('water_level'))
                    else:
                        pred_value = pred
                    
                    true_value = future_data['river'].get('water_level')
                    
                    if pred_value is not None and true_value is not None:
                        step_metrics[step_key]['mae'].update(true_value, pred_value)
                        step_metrics[step_key]['rmse'].update(true_value, pred_value)
                        evaluation_count += 1
                        
                        # デバッグ情報を最初の数回だけ出力
                        if evaluation_count <= 10 and j < 3:
                            print(f"  Step {j+1} ({minutes}min): pred={pred_value:.3f}, true={true_value:.3f}, diff={abs(pred_value-true_value):.3f}")
        else:
            print(f"警告: 予測結果の形式が不正です（record {i}）")
    
    print(f"評価データポイント数: {evaluation_count}")
    
    # サンプル予測を表示
    if prediction_samples:
        print("\n予測サンプル:")
        for sample in prediction_samples[:2]:
            print(f"  Record {sample['index']}:")
            if isinstance(sample['predictions'], list):
                for j, pred in enumerate(sample['predictions']):
                    if isinstance(pred, dict):
                        print(f"    Step {j+1}: {pred.get('level', 'N/A')}")
                    else:
                        print(f"    Step {j+1}: {pred}")
    
    # メトリクスを辞書形式に変換
    metrics_by_step = {}
    for step_key, metrics_dict in step_metrics.items():
        metrics_by_step[step_key] = {
            'mae': metrics_dict['mae'].get() if hasattr(metrics_dict['mae'], 'get') else None,
            'rmse': metrics_dict['rmse'].get() if hasattr(metrics_dict['rmse'], 'get') else None
        }
    
    # モデル自体のMAEメトリクスも取得
    model_mae = None
    if hasattr(model, 'mae_metric') and hasattr(model.mae_metric, 'get'):
        model_mae = model.mae_metric.get()
        print(f"\nモデル内部のMAE: {model_mae:.3f}m" if model_mae else "モデル内部のMAE: N/A")
    
    # 10分先のMAEを優先的に使用
    mae_10min = metrics_by_step.get('10min', {}).get('mae')
    if mae_10min and mae_10min > 0:
        primary_mae = mae_10min
    elif model_mae and model_mae > 0:
        primary_mae = model_mae
    else:
        primary_mae = None
    
    # 診断データの構築
    diagnostics_data = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'initial_training',
        'execution_info': {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'demo_csv_training',
            'description': 'デモCSVデータによる初期学習'
        },
        'training_stats': {
            'total_samples': len(training_records),
            'evaluation_samples': evaluation_count,
            'data_source': 'demo_csv',
            'period': '2023-06-25 to 2023-07-01'
        },
        'metrics_by_step': metrics_by_step,
        'initial_metrics': {
            'mae': primary_mae,
            'rmse': metrics_by_step.get('10min', {}).get('rmse')
        },
        'final_metrics': {
            'mae': primary_mae,
            'rmse': metrics_by_step.get('10min', {}).get('rmse')
        }
    }
    
    return diagnostics_data


def save_diagnostics(diagnostics_data: Dict[str, Any]):
    """診断データをファイルに保存"""
    diagnostics_dir = Path(__file__).parent.parent / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    
    # ファイル名を生成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = diagnostics_dir / f"initial_training_{timestamp}.json"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(diagnostics_data, f, ensure_ascii=False, indent=2)
    
    print(f"診断データを保存しました: {file_path}")
    return file_path


def train_models_with_json(training_data):
    """JSONデータを使用してモデルを学習"""
    from river_streaming_prediction import RiverStreamingPredictor
    
    if not training_data.get("records"):
        print("エラー: 学習データが空です")
        return
    
    print(f"\nモデルを学習中... ({len(training_data['records'])}レコード)")
    
    # 新しいモデルを作成
    model = RiverStreamingPredictor()
    
    # 学習とテストデータを分割（80%学習、20%テスト）
    split_index = int(len(training_data['records']) * 0.8)
    train_records = training_data['records'][:split_index]
    test_records = training_data['records'][split_index:]
    
    print(f"学習データ: {len(train_records)}レコード")
    print(f"テストデータ: {len(test_records)}レコード")
    
    # 学習実行
    for i, record in enumerate(train_records):
        if i % 100 == 0:
            print(f"  学習進捗: {i}/{len(train_records)} ({i/len(train_records)*100:.1f}%)")
        
        # 現在のデータから予測
        current_data = record['current']
        predictions = model.predict_one(current_data)
        
        # 学習実行（learn_oneが履歴バッファを自動更新）
        model.learn_one(current_data, record['future'])
    
    print(f"  学習進捗: {len(train_records)}/{len(train_records)} (100.0%)")
    print("学習完了")
    
    # テストデータで評価
    print("\nテストデータで評価中...")
    for i, record in enumerate(test_records):
        if i % 20 == 0:
            print(f"  評価進捗: {i}/{len(test_records)} ({i/len(test_records)*100:.1f}%)")
        
        # 現在のデータから予測
        current_data = record['current']
        predictions = model.predict_one(current_data)
        
        # 学習実行（これによりMAEが計算される）
        model.learn_one(current_data, record['future'])
    
    print(f"  評価進捗: {len(test_records)}/{len(test_records)} (100.0%)")
    print("評価完了")
    
    # 診断データの生成と保存
    create_diagnostics_for_initial_model(model, test_records, training_data)
    
    # モデルの保存
    model.save_model()
    base_model_path = Path(__file__).parent.parent / "models" / "river_base_model_v2.pkl"
    adaptive_model_path = Path(__file__).parent.parent / "models" / "river_adaptive_model_v2.pkl"
    
    # 基本モデルとして保存
    with open(base_model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n基本モデルを保存しました: {base_model_path}")
    
    # 適応モデル（初期状態）として保存
    with open(adaptive_model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"適応モデル（初期状態）を保存しました: {adaptive_model_path}")
    
    # モデル情報を表示
    model_info = model.get_model_info()
    print(f"\n学習サンプル数: {model_info['n_samples']}")
    print(f"平均絶対誤差 (MAE): {model_info['mae_10min']:.3f}m" if model_info['mae_10min'] else "MAE: N/A")
    
    # 主要時間ポイントの精度を表示
    if model_info.get('metrics_by_step'):
        print("\n主要時間ポイントの予測精度:")
        for time_point in ['30min', '60min', '120min', '180min']:
            if time_point in model_info['metrics_by_step']:
                mae = model_info['metrics_by_step'][time_point]['mae']
                if mae:
                    print(f"  {time_point}: MAE = {mae:.3f}m")

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
    
    # 学習とテストデータを分割（80%学習、20%テスト）
    split_index = int(len(training_data['records']) * 0.8)
    train_records = training_data['records'][:split_index]
    test_records = training_data['records'][split_index:]
    
    print(f"学習データ: {len(train_records)}レコード")
    print(f"テストデータ: {len(test_records)}レコード")
    
    # 学習実行
    for i, record in enumerate(train_records):
        if i % 100 == 0:
            print(f"  学習進捗: {i}/{len(train_records)} ({i/len(train_records)*100:.1f}%)")
        
        # 現在のデータから予測
        current_data = record['current']
        predictions = model.predict_one(current_data)
        
        # 学習実行（learn_oneが履歴バッファを自動更新）
        model.learn_one(current_data, record['future'])
    
    print(f"  学習進捗: {len(train_records)}/{len(train_records)} (100.0%)")
    print("学習完了")
    
    # テストデータで評価（これによりMAEが計算される）
    print("\nテストデータで評価中...")
    for i, record in enumerate(test_records):
        if i % 20 == 0:
            print(f"  評価進捗: {i}/{len(test_records)} ({i/len(test_records)*100:.1f}%)")
        
        # 現在のデータから予測
        current_data = record['current']
        predictions = model.predict_one(current_data)
        
        # 予測結果を実際の値と比較して学習（これによりメトリクスが更新される）
        model.learn_one(current_data, record['future'])
    
    print(f"  評価進捗: {len(test_records)}/{len(test_records)} (100.0%)")
    print("評価完了")
    
    # 診断データを生成（すべての学習済みレコードを使用）
    diagnostics_data = create_diagnostics_for_initial_model(model, training_data['records'])
    
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
    
    # 診断データを保存
    save_diagnostics(diagnostics_data)
    
    # 主要時間ポイントの精度を表示
    print("\n主要時間ポイントの予測精度:")
    for time_point in ['30min', '60min', '120min', '180min']:
        mae = diagnostics_data['metrics_by_step'].get(time_point, {}).get('mae')
        if mae:
            print(f"  {time_point}: MAE = {mae:.3f}m")


if __name__ == "__main__":
    train_models_with_demo_csv()