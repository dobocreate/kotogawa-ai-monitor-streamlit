#!/usr/bin/env python3
"""
デモ学習用データ作成スクリプト
様々な水位パターンを含むトレーニングデータを生成
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import random
import numpy as np


def create_demo_training_data():
    """
    様々なパターンを含むデモ学習データを作成
    """
    demo_data = {
        "description": "厚東川水位予測モデル学習用デモデータ",
        "created_at": datetime.now().isoformat(),
        "patterns": [
            "安定した晴天時",
            "小雨時の緩やかな上昇",
            "大雨時の急激な上昇",
            "ダム放流による変化",
            "放流停止後の下降"
        ],
        "records": []
    }
    
    # パターン1: 安定した晴天時（200レコード）
    print("パターン1: 安定した晴天時のデータを生成中...")
    base_time = datetime(2024, 10, 1, 0, 0)
    base_level = 2.5
    
    for i in range(200):
        # 微小な変動を加える
        noise = random.uniform(-0.02, 0.02)
        current_level = base_level + noise + 0.05 * np.sin(i * 0.1)  # 緩やかな周期変動
        
        record = {
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "current": {
                "data_time": (base_time + timedelta(minutes=i*10)).isoformat(),
                "river": {"water_level": round(current_level, 2)},
                "dam": {
                    "outflow": 5.0 + random.uniform(-0.5, 0.5),
                    "inflow": 5.0 + random.uniform(-0.5, 0.5),
                    "storage_rate": 70.0 + random.uniform(-2, 2)
                },
                "rainfall": {
                    "rainfall_10min": 0,
                    "rainfall_1h": 0
                }
            },
            "future": []
        }
        
        # 未来18ステップ分のデータ（10分～180分先）
        for j in range(1, 19):
            future_noise = random.uniform(-0.02, 0.02)
            future_level = base_level + future_noise + 0.05 * np.sin((i+j) * 0.1)
            record["future"].append({
                "timestamp": (base_time + timedelta(minutes=(i+j)*10)).isoformat(),
                "river": {"water_level": round(future_level, 2)}
            })
        
        demo_data["records"].append(record)
    
    # パターン2: 小雨時の緩やかな上昇（100レコード）
    print("パターン2: 小雨時の緩やかな上昇データを生成中...")
    base_time = datetime(2024, 10, 5, 0, 0)
    current_level = 2.5
    
    for i in range(100):
        # 降雨による緩やかな上昇
        rainfall = 2.0 + random.uniform(-0.5, 0.5)
        level_increase = 0.005 * rainfall  # 雨量に応じた上昇
        current_level += level_increase
        
        record = {
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "current": {
                "data_time": (base_time + timedelta(minutes=i*10)).isoformat(),
                "river": {"water_level": round(current_level, 2)},
                "dam": {
                    "outflow": 5.0 + random.uniform(-0.5, 0.5),
                    "inflow": 7.0 + random.uniform(-1, 1),  # 流入増加
                    "storage_rate": 72.0 + i * 0.02  # 貯水率上昇
                },
                "rainfall": {
                    "rainfall_10min": round(rainfall, 1),
                    "rainfall_1h": round(rainfall * 6, 1)
                }
            },
            "future": []
        }
        
        # 未来予測
        future_level = current_level
        for j in range(1, 19):
            future_level += 0.005 * max(0, rainfall - j * 0.1)  # 徐々に影響減少
            record["future"].append({
                "timestamp": (base_time + timedelta(minutes=(i+j)*10)).isoformat(),
                "river": {"water_level": round(future_level, 2)}
            })
        
        demo_data["records"].append(record)
    
    # パターン3: 大雨時の急激な上昇（50レコード）
    print("パターン3: 大雨時の急激な上昇データを生成中...")
    base_time = datetime(2024, 10, 10, 0, 0)
    current_level = 2.6
    
    for i in range(50):
        # 強い降雨
        if i < 20:
            rainfall = 10.0 + random.uniform(-2, 2)
        else:
            rainfall = max(0, 10.0 - (i-20) * 0.3)  # 徐々に弱まる
        
        level_increase = 0.02 * rainfall
        current_level += level_increase
        
        record = {
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "current": {
                "data_time": (base_time + timedelta(minutes=i*10)).isoformat(),
                "river": {"water_level": round(current_level, 2)},
                "dam": {
                    "outflow": min(50, 5.0 + i * 0.5),  # 放流量増加
                    "inflow": 20.0 + random.uniform(-2, 2),
                    "storage_rate": min(90, 75.0 + i * 0.3)
                },
                "rainfall": {
                    "rainfall_10min": round(rainfall, 1),
                    "rainfall_1h": round(rainfall * 6, 1)
                }
            },
            "future": []
        }
        
        # 未来予測
        future_level = current_level
        for j in range(1, 19):
            if i < 20:  # 雨が続く場合
                future_level += 0.02 * max(0, rainfall - j * 0.5)
            else:  # 雨が弱まった後
                future_level -= 0.01  # 徐々に下降
            record["future"].append({
                "timestamp": (base_time + timedelta(minutes=(i+j)*10)).isoformat(),
                "river": {"water_level": round(max(2.5, future_level), 2)}
            })
        
        demo_data["records"].append(record)
    
    # パターン4: ダム放流による変化（100レコード）
    print("パターン4: ダム放流による変化データを生成中...")
    base_time = datetime(2024, 10, 15, 0, 0)
    current_level = 2.5
    
    for i in range(100):
        # 放流パターン
        if i < 30:
            outflow = 5.0  # 通常
        elif i < 60:
            outflow = 50.0 + random.uniform(-5, 5)  # 大量放流
        else:
            outflow = 20.0 + random.uniform(-2, 2)  # 中程度
        
        # 放流量に応じた水位変化（30分遅延を考慮）
        if i >= 3:  # 30分後から影響
            if i < 33:
                level_change = 0
            elif i < 63:
                level_change = 0.03  # 上昇
            else:
                level_change = -0.01  # 下降
            current_level += level_change
        
        record = {
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "current": {
                "data_time": (base_time + timedelta(minutes=i*10)).isoformat(),
                "river": {"water_level": round(current_level, 2)},
                "dam": {
                    "outflow": round(outflow, 1),
                    "inflow": 15.0 + random.uniform(-2, 2),
                    "storage_rate": 80.0 - i * 0.05  # 放流により減少
                },
                "rainfall": {
                    "rainfall_10min": 0,
                    "rainfall_1h": 0
                }
            },
            "future": []
        }
        
        # 未来予測（放流影響を考慮）
        future_level = current_level
        for j in range(1, 19):
            future_i = i + j
            if future_i >= 33 and future_i < 63:
                future_level += 0.03
            elif future_i >= 63 and future_i < 93:
                future_level -= 0.01
            
            record["future"].append({
                "timestamp": (base_time + timedelta(minutes=(i+j)*10)).isoformat(),
                "river": {"water_level": round(max(2.5, min(4.0, future_level)), 2)}
            })
        
        demo_data["records"].append(record)
    
    # パターン5: 複合パターン（50レコード）
    print("パターン5: 複合パターンデータを生成中...")
    base_time = datetime(2024, 10, 20, 0, 0)
    current_level = 2.6
    
    for i in range(50):
        # 雨と放流の複合
        rainfall = 5.0 * np.sin(i * 0.1) + 5.0 + random.uniform(-1, 1)
        outflow = 20.0 + 10.0 * np.sin(i * 0.05 + 1) + random.uniform(-2, 2)
        
        # 複合的な水位変化
        level_change = 0.01 * rainfall + 0.002 * (outflow - 20)
        current_level += level_change
        
        record = {
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "current": {
                "data_time": (base_time + timedelta(minutes=i*10)).isoformat(),
                "river": {"water_level": round(current_level, 2)},
                "dam": {
                    "outflow": round(outflow, 1),
                    "inflow": round(outflow + rainfall, 1),
                    "storage_rate": 75.0 + 5.0 * np.sin(i * 0.02)
                },
                "rainfall": {
                    "rainfall_10min": round(max(0, rainfall), 1),
                    "rainfall_1h": round(max(0, rainfall * 6), 1)
                }
            },
            "future": []
        }
        
        # 未来予測
        future_level = current_level
        for j in range(1, 19):
            future_rainfall = max(0, 5.0 * np.sin((i+j) * 0.1) + 5.0)
            future_outflow = 20.0 + 10.0 * np.sin((i+j) * 0.05 + 1)
            future_change = 0.01 * future_rainfall + 0.002 * (future_outflow - 20)
            future_level += future_change
            
            record["future"].append({
                "timestamp": (base_time + timedelta(minutes=(i+j)*10)).isoformat(),
                "river": {"water_level": round(max(2.5, min(4.5, future_level)), 2)}
            })
        
        demo_data["records"].append(record)
    
    # 保存
    output_path = Path(__file__).parent.parent / "sample" / "demo_training_data.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nデモデータを作成しました: {output_path}")
    print(f"総レコード数: {len(demo_data['records'])}")
    print("パターン別レコード数:")
    print("  - 安定した晴天時: 200")
    print("  - 小雨時の緩やかな上昇: 100")
    print("  - 大雨時の急激な上昇: 50")
    print("  - ダム放流による変化: 100")
    print("  - 複合パターン: 50")
    print(f"  合計: 500レコード")


if __name__ == "__main__":
    create_demo_training_data()