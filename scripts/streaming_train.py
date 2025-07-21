"""
ストリーミング学習スクリプト
データ収集直後に実行され、新しいデータで即座に学習
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.river_streaming_prediction_v2 import RiverStreamingPredictor


def get_latest_data():
    """最新のデータポイントを取得"""
    data_dir = Path('data')
    
    # まずlatest.jsonを試す
    latest_file = data_dir / 'latest.json'
    if latest_file.exists():
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data:
                return data
        except:
            pass
    
    # 次に今日のファイルを読み込み（フォールバック）
    today = datetime.now().strftime('%Y%m%d')
    today_file = data_dir / f'{today}.json'
    
    if not today_file.exists():
        return None
    
    with open(today_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        return None
    
    # データが配列の場合は最新を返す、そうでなければそのまま返す
    if isinstance(data, list):
        return data[-1]
    return data


def streaming_learn():
    """新しいデータでストリーミング学習を実行"""
    print(f"[{datetime.now()}] ストリーミング学習を開始")
    
    # モデルの読み込み（既存のモデルがあれば継続）
    predictor = RiverStreamingPredictor()
    
    # 最新データを取得
    latest_data = get_latest_data()
    if not latest_data:
        print("新しいデータがありません")
        return
    
    print(f"最新データ: {latest_data.get('data_time', 'unknown')}")
    
    # 既に学習済みかチェック
    if hasattr(predictor, '_processed_timestamps'):
        if latest_data.get('data_time') in predictor._processed_timestamps:
            print(f"このデータは既に学習済みです: {latest_data.get('data_time')}")
            return
    else:
        predictor._processed_timestamps = set()
    
    # 将来の実測値を確認するため、過去3時間分のデータを読み込み
    data_dir = Path('data')
    history_dir = data_dir / 'history'
    all_recent_data = []
    
    # 今日と昨日のデータを履歴ディレクトリから読み込み
    for days_ago in range(2):
        date = datetime.now()
        if days_ago > 0:
            date = date - timedelta(days=days_ago)
        
        # 履歴ディレクトリの日付パス
        date_path = history_dir / date.strftime("%Y") / date.strftime("%m") / date.strftime("%d")
        
        if date_path.exists():
            # その日のすべての時刻ファイルを読み込み
            for time_file in sorted(date_path.glob("*.json")):
                if time_file.name.startswith('error_') or time_file.name == 'daily_summary.json':
                    continue
                    
                try:
                    with open(time_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, dict):
                            all_recent_data.append(file_data)
                except Exception:
                    continue
    
    # 時系列順にソート
    all_recent_data.sort(key=lambda x: x.get('data_time', ''))
    
    # 現在のデータのインデックスを見つける
    current_time = latest_data.get('data_time')
    current_idx = None
    
    for i, data in enumerate(all_recent_data):
        if data.get('data_time') == current_time:
            current_idx = i
            break
    
    if current_idx is not None and current_idx + 18 < len(all_recent_data):
        # 3時間後までの実測値が利用可能
        future_data = all_recent_data[current_idx + 1:current_idx + 19]
        
        # 予測を実行
        predictions = predictor.predict_one(latest_data)
        if predictions:
            print(f"予測実行: 10分先 = {predictions[0]['level']:.2f}m")
        
        # 学習を実行
        predictor.learn_one(latest_data, future_data)
        print("学習完了")
        
        # モデルを保存
        predictor.save_model()
        
        # モデル情報を表示
        info = predictor.get_model_info()
        print(f"\nモデル状態:")
        print(f"- 総学習サンプル数: {info['n_samples']}")
        print(f"- MAE (10分先): {info.get('mae_10min', 'N/A')}")
        print(f"- ドリフト検出回数: {info['drift_count']}")
    else:
        print("将来の実測値がまだ利用できません（3時間後に再実行してください）")


if __name__ == '__main__':
    streaming_learn()