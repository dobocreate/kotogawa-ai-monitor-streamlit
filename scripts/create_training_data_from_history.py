#!/usr/bin/env python3
"""
履歴データから学習用データを作成するスクリプト
実際の厚東川データを使用してモデルを学習
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))


def create_training_data_from_history():
    """履歴データから学習用データを作成"""
    
    # データディレクトリ
    data_dir = Path(__file__).parent.parent / "data"
    history_dir = data_dir / "history"
    
    if not history_dir.exists():
        print("エラー: 履歴データディレクトリが見つかりません")
        return None
    
    print("履歴データから学習用データを作成中...")
    
    training_data = {
        "description": "厚東川実データに基づく学習用データ",
        "created_at": datetime.now().isoformat(),
        "source": "historical_data",
        "records": []
    }
    
    # 過去30日分のデータを収集
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    all_data = []
    current_date = start_date
    
    # 日付ごとにデータを収集
    while current_date <= end_date:
        year_dir = history_dir / current_date.strftime("%Y")
        month_dir = year_dir / current_date.strftime("%m")
        day_dir = month_dir / current_date.strftime("%d")
        
        if day_dir.exists():
            # その日のJSONファイルを読み込み
            for json_file in sorted(day_dir.glob("*.json")):
                if json_file.name == "daily_summary.json":
                    continue
                    
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data and 'timestamp' in data:
                            all_data.append(data)
                except:
                    continue
        
        current_date += timedelta(days=1)
    
    # 時系列順にソート
    all_data.sort(key=lambda x: x.get('timestamp', ''))
    
    print(f"収集したデータ数: {len(all_data)}")
    
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
    output_path = Path(__file__).parent.parent / "sample" / "real_training_data.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"学習データを保存しました: {output_path}")
    
    # データの統計情報を表示
    if training_data["records"]:
        # 水位の範囲を計算
        water_levels = []
        for record in training_data["records"]:
            level = record["current"].get("river", {}).get("water_level")
            if level:
                water_levels.append(level)
        
        if water_levels:
            print(f"\n水位統計:")
            print(f"  最小値: {min(water_levels):.2f}m")
            print(f"  最大値: {max(water_levels):.2f}m")
            print(f"  平均値: {sum(water_levels)/len(water_levels):.2f}m")
    
    return output_path


def train_models_with_real_data():
    """実データでモデルを学習"""
    from river_streaming_prediction import RiverStreamingPredictor
    
    # 学習データを作成
    data_path = create_training_data_from_history()
    if not data_path:
        return
    
    # 学習データを読み込み
    with open(data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    if not training_data.get("records"):
        print("エラー: 学習データが空です")
        return
    
    print(f"\n実データでモデルを学習中... ({len(training_data['records'])}レコード)")
    
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
    
    # 実データモデルとして保存
    model_path = models_dir / "river_realdata_model.pkl"
    model.model_path = str(model_path)
    model.save_model()
    
    print(f"\nモデルを保存しました: {model_path}")
    print(f"学習サンプル数: {model.n_samples}")
    
    # 精度情報を表示
    if hasattr(model.mae_metric, 'get'):
        mae = model.mae_metric.get()
        if mae > 0:
            print(f"平均絶対誤差 (MAE): {mae:.3f}m")


if __name__ == "__main__":
    train_models_with_real_data()