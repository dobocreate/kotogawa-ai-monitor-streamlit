"""
厚東川水位の簡易予測モジュール
線形回帰と移動平均を使用した3時間先の水位予測
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class SimpleRiverLevelPredictor:
    """河川水位の簡易予測クラス"""
    
    def __init__(self):
        """初期化"""
        self.prediction_horizon = 18  # 3時間先まで（10分間隔）
        
    def predict(self, history_data: List[Dict]) -> Optional[List[Dict]]:
        """
        3時間先の水位を予測
        
        Args:
            history_data: 履歴データのリスト（最低18件必要）
            
        Returns:
            predictions: 予測結果のリスト（10分間隔で18ポイント）
        """
        if len(history_data) < 18:
            return None
            
        # 最新18個のデータから水位と放流量を抽出
        recent_data = history_data[-18:]
        
        # データ抽出
        water_levels = []
        outflows = []
        timestamps = []
        
        for data in recent_data:
            if 'river' in data and data['river'].get('water_level') is not None:
                water_levels.append(data['river']['water_level'])
                timestamps.append(data.get('data_time', data.get('timestamp')))
                
                # ダム放流量も考慮
                if 'dam' in data and data['dam'].get('outflow') is not None:
                    outflows.append(data['dam']['outflow'])
                else:
                    outflows.append(0)
                    
        if len(water_levels) < 3:
            return None
            
        # 予測の実行
        predictions = self._predict_water_levels(
            water_levels, 
            outflows, 
            timestamps[-1] if timestamps else datetime.now().isoformat()
        )
        
        return predictions
        
    def _predict_water_levels(self, water_levels: List[float], outflows: List[float], 
                            last_timestamp: str) -> List[Dict]:
        """
        水位予測の実行
        
        Args:
            water_levels: 過去の水位データ
            outflows: 過去の放流量データ
            last_timestamp: 最後のタイムスタンプ
            
        Returns:
            予測結果のリスト
        """
        predictions = []
        
        # 最後の観測時刻を解析
        try:
            current_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
        except:
            current_time = datetime.now()
            
        # トレンドの計算（最近1時間と3時間）
        if len(water_levels) >= 6:
            # 1時間のトレンド
            trend_1h = (water_levels[-1] - water_levels[-6]) / 6
        else:
            trend_1h = 0
            
        # 3時間のトレンド
        trend_3h = (water_levels[-1] - water_levels[0]) / len(water_levels)
        
        # 放流量の影響を考慮
        avg_outflow_recent = np.mean(outflows[-6:]) if len(outflows) >= 6 else 0
        avg_outflow_all = np.mean(outflows)
        outflow_change_rate = (avg_outflow_recent - avg_outflow_all) / (avg_outflow_all + 1e-6)
        
        # 基準となる水位
        base_level = water_levels[-1]
        
        # 予測の実行
        for i in range(self.prediction_horizon):
            # 予測時刻
            pred_time = current_time + timedelta(minutes=10 * (i + 1))
            
            # 時間経過による減衰を考慮したトレンド
            time_factor = 1.0 - (i / self.prediction_horizon) * 0.5  # 時間とともに影響を減衰
            
            # 複合トレンドの計算
            # 短期トレンドと長期トレンドの加重平均
            combined_trend = (trend_1h * 0.7 + trend_3h * 0.3) * time_factor
            
            # 放流量の影響（放流量が増えると水位も上昇する傾向）
            outflow_effect = outflow_change_rate * 0.1 * time_factor
            
            # 予測値の計算
            pred_level = base_level + combined_trend * (i + 1) + outflow_effect * (i + 1)
            
            # 予測値の妥当性チェック
            # 急激な変化を抑制（1時間で最大0.5mの変化）
            max_change_per_hour = 0.5
            max_change = max_change_per_hour * ((i + 1) / 6)
            
            if abs(pred_level - base_level) > max_change:
                pred_level = base_level + np.sign(pred_level - base_level) * max_change
                
            # 負の値を防ぐ
            pred_level = max(0, pred_level)
            
            # 結果を追加
            predictions.append({
                'datetime': pred_time.isoformat(),
                'level': round(pred_level, 2),
                'confidence': round(1.0 - (i / self.prediction_horizon) * 0.3, 2)  # 信頼度
            })
            
        return predictions
        
    def get_prediction_info(self) -> Dict:
        """予測情報の取得"""
        return {
            'method': '線形トレンド予測',
            'horizon_hours': 3,
            'interval_minutes': 10,
            'features': ['河川水位', 'ダム放流量'],
            'note': 'AI学習中につき参考値としてご利用ください'
        }