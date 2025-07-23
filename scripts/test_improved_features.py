#!/usr/bin/env python3
"""
改良された特徴量抽出のテストスクリプト
Phase 2実装の動作確認
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from river_streaming_prediction import RiverStreamingPredictor
import json
from datetime import datetime, timedelta

def create_test_data(timestamp_str, water_level=2.5, dam_outflow=50, rainfall=10):
    """テスト用データを作成"""
    return {
        'timestamp': timestamp_str,
        'data_time': timestamp_str,
        'river': {
            'water_level': water_level,
            'status': '正常'
        },
        'dam': {
            'outflow': dam_outflow,
            'inflow': dam_outflow * 0.9,
            'water_level': 100.0,
            'storage_rate': 70.0
        },
        'rainfall': {
            'hourly': rainfall,
            'cumulative': rainfall * 3,
            'rainfall_10min': rainfall / 6,
            'rainfall_1h': rainfall
        }
    }

def main():
    print("=== 改良された特徴量抽出のテスト ===\n")
    
    # テスト用の予測器を作成
    predictor = RiverStreamingPredictor()
    
    # 基準時刻
    base_time = datetime(2025, 1, 23, 15, 0, 0)
    
    # 履歴データを作成（90分分）
    print("1. 履歴データの作成（90分分）")
    history_data = []
    for i in range(10):  # 0-90分前
        time_offset = -(90 - i * 10)  # -90, -80, ..., 0
        timestamp = base_time + timedelta(minutes=time_offset)
        timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S+09:00')
        
        # 時間により変化するデータ
        water_level = 2.0 + 0.05 * i  # 徐々に上昇
        dam_outflow = 30 + 10 * (i // 3)  # 段階的に増加
        rainfall = 5 if i < 5 else 15  # 途中から雨が強くなる
        
        data = create_test_data(timestamp_str, water_level, dam_outflow, rainfall)
        history_data.append(data)
        
        # 履歴バッファに追加
        if i < 9:  # 最後のデータは現在データとして使用
            predictor.update_history_buffer(data)
    
    print(f"履歴バッファサイズ: {len(predictor.history_buffer)}")
    print()
    
    # 現在のデータで特徴量を抽出
    print("2. 特徴量の抽出")
    current_data = history_data[-1]
    features = predictor.extract_features(current_data)
    
    # 特徴量を分類して表示
    print("\n【基本特徴量】")
    basic_keys = ['water_level', 'dam_outflow', 'rainfall', 'elapsed_min']
    for key in basic_keys:
        if key in features:
            print(f"  {key}: {features[key]:.2f}")
    
    print("\n【Phase 1: 時系列生データ】")
    timeseries_keys = [k for k in features.keys() if k.startswith(('dam_outflow_t-', 'rainfall_t-', 'water_level_t-'))]
    for key in sorted(timeseries_keys):
        print(f"  {key}: {features[key]:.2f}")
    
    print("\n【Phase 1: 基本集約統計量】")
    basic_agg_keys = ['dam_sum_recent', 'dam_max_recent', 'rain_sum_recent', 
                      'dam_sum_90min', 'dam_max_90min', 'rain_sum_90min']
    for key in basic_agg_keys:
        if key in features:
            print(f"  {key}: {features[key]:.2f}")
    
    print("\n【Phase 2: 時間窓集約特徴量】")
    window_keys = [k for k in features.keys() if any(pattern in k for pattern in ['_0_30min', '_30_60min', '_60_90min'])]
    for key in sorted(window_keys):
        if key in features:
            print(f"  {key}: {features[key]:.2f}")
    
    print("\n【Phase 2: トレンド・パターン特徴量】")
    trend_keys = ['water_level_change_10min', 'water_level_change_30min', 'water_level_change_60min',
                  'dam_trend_recent', 'dam_trend_older', 'dam_peak_90min', 'dam_peak_timing',
                  'dam_std_90min', 'rain_std_90min']
    for key in trend_keys:
        if key in features:
            print(f"  {key}: {features[key]:.2f}")
    
    print(f"\n合計特徴量数: {len(features) - 1}")  # _timestampを除く
    
    # 予測のテスト
    print("\n3. 予測のテスト")
    predictions = predictor.predict_one(current_data)
    if predictions:
        print(f"予測数: {len(predictions)}")
        print("最初の3つの予測:")
        for i, pred in enumerate(predictions[:3]):
            print(f"  {i+1}. {pred['datetime']}: {pred['level']:.2f}m")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    main()