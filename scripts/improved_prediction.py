"""
厚東川水位の改善された予測モジュール
ダム放流量との相関を強化し、時間遅延を考慮した予測
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class ImprovedRiverLevelPredictor:
    """改善された河川水位予測クラス"""
    
    def __init__(self):
        """初期化"""
        self.prediction_horizon = 18  # 3時間先まで（10分間隔）
        self.time_lag_minutes = 40  # ダムから観測地点までの時間遅延（分）
        
    def predict(self, history_data: List[Dict]) -> Optional[List[Dict]]:
        """
        3時間先の水位を予測（改善版）
        
        Args:
            history_data: 履歴データのリスト（最低18件必要）
            
        Returns:
            predictions: 予測結果のリスト（10分間隔で18ポイント）
        """
        if len(history_data) < 18:
            return None
            
        # 最新24個のデータから水位、放流量、雨量を抽出
        recent_data = history_data[-24:] if len(history_data) >= 24 else history_data
        
        # データ抽出
        water_levels = []
        outflows = []
        rainfalls = []
        timestamps = []
        
        for data in recent_data:
            if 'river' in data and data['river'].get('water_level') is not None:
                water_levels.append(data['river']['water_level'])
                timestamps.append(data.get('data_time', data.get('timestamp')))
                
                # ダム放流量
                if 'dam' in data and data['dam'].get('outflow') is not None:
                    outflows.append(data['dam']['outflow'])
                else:
                    outflows.append(0)
                    
                # 雨量
                if 'rainfall' in data and data['rainfall'].get('hourly') is not None:
                    rainfalls.append(data['rainfall']['hourly'])
                else:
                    rainfalls.append(0)
                    
        if len(water_levels) < 3:
            return None
            
        # 予測の実行
        predictions = self._predict_water_levels_improved(
            water_levels, 
            outflows,
            rainfalls,
            timestamps[-1] if timestamps else datetime.now().isoformat()
        )
        
        return predictions
        
    def _predict_water_levels_improved(self, water_levels: List[float], 
                                     outflows: List[float], 
                                     rainfalls: List[float],
                                     last_timestamp: str) -> List[Dict]:
        """
        改善された水位予測の実行
        
        Args:
            water_levels: 過去の水位データ
            outflows: 過去の放流量データ
            rainfalls: 過去の雨量データ
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
            
        # 基準となる水位
        base_level = water_levels[-1]
        
        # 水位トレンドの計算（短期・中期・長期）
        trend_30min = 0
        trend_1h = 0
        trend_3h = 0
        
        if len(water_levels) >= 3:
            trend_30min = (water_levels[-1] - water_levels[-3]) / 3
        if len(water_levels) >= 6:
            trend_1h = (water_levels[-1] - water_levels[-6]) / 6
        if len(water_levels) >= 18:
            trend_3h = (water_levels[-1] - water_levels[-18]) / 18
            
        # 放流量の分析（時間遅延を考慮）
        # 40分前（4データポイント前）の放流量が現在の水位に影響
        lag_index = self.time_lag_minutes // 10
        
        # 現在影響している放流量（遅延考慮）
        if len(outflows) > lag_index:
            current_impact_outflow = outflows[-lag_index]
        else:
            current_impact_outflow = outflows[0] if outflows else 0
            
        # 最近の放流量の変化
        recent_outflow = np.mean(outflows[-6:]) if len(outflows) >= 6 else outflows[-1]
        outflow_trend = 0
        if len(outflows) >= 6:
            outflow_trend = (outflows[-1] - outflows[-6]) / 6
            
        # 放流量の絶対的な影響を計算
        # 100 m³/sの放流量変化で約0.3mの水位変化と仮定
        outflow_to_level_factor = 0.003  # m³/s → m の変換係数
        
        # 雨量の影響
        recent_rainfall = np.sum(rainfalls[-6:]) if len(rainfalls) >= 6 else 0
        rainfall_impact = recent_rainfall * 0.01  # 雨量1mmあたり0.01mの影響
        
        # 予測の実行
        for i in range(self.prediction_horizon):
            minutes_ahead = (i + 1) * 10
            pred_time = current_time + timedelta(minutes=minutes_ahead)
            
            # 将来の放流量の影響を推定（時間遅延考慮）
            # i+lag_index先の放流量が影響する
            future_impact_index = min(i + lag_index, len(outflows) - 1)
            if future_impact_index >= 0 and future_impact_index < len(outflows):
                future_outflow = outflows[future_impact_index]
            else:
                # 放流量のトレンドから推定
                future_outflow = recent_outflow + outflow_trend * (i + lag_index)
                future_outflow = max(0, future_outflow)  # 負の値を防ぐ
                
            # 放流量による水位変化
            outflow_impact = (future_outflow - current_impact_outflow) * outflow_to_level_factor
            
            # 水位トレンドの統合（時間とともに重みを調整）
            if minutes_ahead <= 30:
                # 30分以内：短期トレンドを重視
                combined_trend = trend_30min * 0.6 + trend_1h * 0.3 + trend_3h * 0.1
            elif minutes_ahead <= 60:
                # 1時間以内：中期トレンドを重視
                combined_trend = trend_30min * 0.3 + trend_1h * 0.5 + trend_3h * 0.2
            else:
                # 1時間以上：長期トレンドを重視
                combined_trend = trend_30min * 0.1 + trend_1h * 0.3 + trend_3h * 0.6
                
            # 水位が高い時は放流量の影響を増幅（非線形性）
            if base_level > 4.0:  # 4m以上で影響増大
                level_amplifier = 1.0 + (base_level - 4.0) * 0.3
                outflow_impact *= level_amplifier
                
            # 予測値の計算
            pred_level = base_level + (combined_trend * (i + 1)) + outflow_impact + (rainfall_impact * 0.5)
            
            # 放流量が急増している場合の補正
            if outflow_trend > 50:  # 50 m³/s/h以上の増加
                # 急激な変化を許容
                rapid_change_factor = 1.0 + (outflow_trend / 100) * 0.5
                pred_level = base_level + ((pred_level - base_level) * rapid_change_factor)
            else:
                # 通常時の変化制限（1時間で最大1.0mに緩和）
                max_change_per_hour = 1.0
                max_change = max_change_per_hour * (minutes_ahead / 60)
                
                if abs(pred_level - base_level) > max_change:
                    pred_level = base_level + np.sign(pred_level - base_level) * max_change
                    
            # 負の値を防ぐ
            pred_level = max(0, pred_level)
            
            # 信頼度の計算（放流量の変動が大きいほど低下）
            outflow_volatility = np.std(outflows[-6:]) if len(outflows) >= 6 else 0
            confidence = max(0.5, 1.0 - (i / self.prediction_horizon) * 0.3 - (outflow_volatility / 1000))
            
            # 結果を追加
            predictions.append({
                'datetime': pred_time.isoformat(),
                'level': round(pred_level, 2),
                'confidence': round(confidence, 2),
                'outflow_impact': round(outflow_impact, 3),  # デバッグ用
                'trend_impact': round(combined_trend * (i + 1), 3)  # デバッグ用
            })
            
        return predictions
        
    def get_prediction_info(self) -> Dict:
        """予測情報の取得"""
        return {
            'method': '改善型予測モデル',
            'horizon_hours': 3,
            'interval_minutes': 10,
            'features': ['河川水位', 'ダム放流量', '時間雨量'],
            'improvements': [
                'ダム放流量の時間遅延考慮（40分）',
                '放流量の絶対的影響を評価',
                '水位による非線形増幅',
                '急激な変化への対応'
            ],
            'note': '放流量急増時の予測精度を改善'
        }