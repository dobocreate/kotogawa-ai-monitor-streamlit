"""
厚東川水位の高度な予測モジュール
動的な時間遅延、連続的な重み付け、放流量の加速度を考慮
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple


class AdvancedRiverLevelPredictor:
    """高度な河川水位予測クラス"""
    
    def __init__(self):
        """初期化"""
        self.prediction_horizon = 18  # 3時間先まで（10分間隔）
        self.base_time_lag_minutes = 40  # 基本的な時間遅延（分）
        
    def predict(self, history_data: List[Dict]) -> Optional[List[Dict]]:
        """
        3時間先の水位を予測（高度版）
        
        Args:
            history_data: 履歴データのリスト（最低18件必要）
            
        Returns:
            predictions: 予測結果のリスト（10分間隔で18ポイント）
        """
        if len(history_data) < 18:
            return None
            
        # 最新36個のデータから水位、放流量、雨量を抽出（6時間分）
        recent_data = history_data[-36:] if len(history_data) >= 36 else history_data
        
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
        predictions = self._predict_water_levels_advanced(
            water_levels, 
            outflows,
            rainfalls,
            timestamps[-1] if timestamps else datetime.now().isoformat()
        )
        
        return predictions
        
    def _calculate_dynamic_time_lag(self, outflow: float) -> int:
        """
        動的な時間遅延を計算
        流量が多いほど到達時間が短くなる
        
        Args:
            outflow: 放流量 (m³/s)
            
        Returns:
            time_lag: 時間遅延（分）
        """
        # 基本遅延時間から、流量に応じて短縮
        # 100 m³/sごとに2分短縮、最小20分
        reduction = min(20, (outflow / 100) * 2)
        time_lag = max(20, self.base_time_lag_minutes - reduction)
        return int(time_lag)
        
    def _calculate_smooth_weights(self, minutes_ahead: int) -> Tuple[float, float, float]:
        """
        時間に応じた滑らかな重み付けを計算
        
        Args:
            minutes_ahead: 予測時間（分）
            
        Returns:
            (short_weight, medium_weight, long_weight): 重みのタプル
        """
        # シグモイド関数を使用して滑らかに遷移
        x = minutes_ahead / 180  # 3時間を1に正規化
        
        # 短期重み：最初は高く、徐々に減少
        short_weight = 0.7 * np.exp(-3 * x)
        
        # 中期重み：中間で最大
        medium_weight = 0.8 * np.exp(-4 * (x - 0.3)**2)
        
        # 長期重み：後半で増加
        long_weight = 0.6 * (1 - np.exp(-3 * x))
        
        # 正規化
        total = short_weight + medium_weight + long_weight
        if total > 0:
            return (short_weight/total, medium_weight/total, long_weight/total)
        else:
            return (0.33, 0.33, 0.34)
            
    def _calculate_outflow_acceleration(self, outflows: List[float]) -> float:
        """
        放流量の加速度（変化の変化）を計算
        
        Args:
            outflows: 放流量データ
            
        Returns:
            acceleration: 加速度 (m³/s/10min²)
        """
        if len(outflows) < 3:
            return 0
            
        # 直近3点から2次微分を計算
        if len(outflows) >= 3:
            acc = outflows[-1] - 2 * outflows[-2] + outflows[-3]
            return acc
        return 0
        
    def _predict_future_outflow(self, outflows: List[float], steps_ahead: int) -> float:
        """
        将来の放流量を予測（加速度を考慮）
        
        Args:
            outflows: 放流量データ
            steps_ahead: 予測ステップ数
            
        Returns:
            predicted_outflow: 予測放流量
        """
        if len(outflows) < 3:
            return outflows[-1] if outflows else 0
            
        # 現在の値、速度、加速度
        current = outflows[-1]
        velocity = (outflows[-1] - outflows[-2]) if len(outflows) >= 2 else 0
        acceleration = self._calculate_outflow_acceleration(outflows)
        
        # 2次の運動方程式で予測
        # x = x0 + v*t + 0.5*a*t²
        predicted = current + velocity * steps_ahead + 0.5 * acceleration * (steps_ahead ** 2)
        
        # 負の値を防ぐ
        return max(0, predicted)
        
    def _predict_water_levels_advanced(self, water_levels: List[float], 
                                      outflows: List[float], 
                                      rainfalls: List[float],
                                      last_timestamp: str) -> List[Dict]:
        """
        高度な水位予測の実行
        
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
        
        # 水位トレンドの計算
        trends = self._calculate_water_trends(water_levels)
        
        # 放流量の統計
        current_outflow = outflows[-1] if outflows else 0
        outflow_acceleration = self._calculate_outflow_acceleration(outflows)
        
        # 雨量の影響
        recent_rainfall = np.sum(rainfalls[-6:]) if len(rainfalls) >= 6 else 0
        rainfall_impact = recent_rainfall * 0.01
        
        # 各時点での予測
        for i in range(self.prediction_horizon):
            minutes_ahead = (i + 1) * 10
            pred_time = current_time + timedelta(minutes=minutes_ahead)
            
            # 現在の放流量による動的な時間遅延
            time_lag = self._calculate_dynamic_time_lag(current_outflow)
            lag_steps = time_lag // 10
            
            # 影響する放流量を特定（動的遅延考慮）
            impact_time_steps = i - lag_steps
            
            if impact_time_steps >= 0:
                # 未来の予測放流量が影響
                future_outflow = self._predict_future_outflow(outflows, impact_time_steps)
            else:
                # 過去の実測放流量が影響
                past_index = len(outflows) + impact_time_steps
                if past_index >= 0:
                    future_outflow = outflows[past_index]
                else:
                    future_outflow = outflows[0] if outflows else 0
                    
            # 放流量による水位変化（改善版）
            # 基本係数 + 流量依存の非線形性
            base_factor = 0.003
            if future_outflow > 500:  # 大流量時は影響増大
                base_factor = 0.004
            elif future_outflow > 800:
                base_factor = 0.005
                
            outflow_impact = (future_outflow - current_outflow) * base_factor
            
            # 水位が高い時の増幅効果（改善版）
            if base_level > 4.0:
                # 4m以上で段階的に増幅
                if base_level > 5.0:
                    level_amplifier = 1.5
                else:
                    level_amplifier = 1.0 + (base_level - 4.0) * 0.5
                outflow_impact *= level_amplifier
                
            # 滑らかな重み付けでトレンドを統合
            short_w, medium_w, long_w = self._calculate_smooth_weights(minutes_ahead)
            combined_trend = (
                trends['short'] * short_w +
                trends['medium'] * medium_w +
                trends['long'] * long_w
            )
            
            # 予測値の計算
            pred_level = base_level + (combined_trend * minutes_ahead / 10) + outflow_impact + rainfall_impact
            
            # 放流量の加速度による補正
            if outflow_acceleration > 10:  # 急加速中
                acceleration_factor = 1.0 + (outflow_acceleration / 100) * 0.3
                pred_level += (pred_level - base_level) * acceleration_factor * 0.1
                
            # 変化制限（動的）
            # 放流量が多いほど、または加速度が大きいほど制限を緩和
            base_max_change = 1.0  # 基本は1.0m/h
            if current_outflow > 500 or abs(outflow_acceleration) > 20:
                base_max_change = 1.5
                
            max_change = base_max_change * (minutes_ahead / 60)
            
            if abs(pred_level - base_level) > max_change:
                pred_level = base_level + np.sign(pred_level - base_level) * max_change
                
            # 負の値を防ぐ
            pred_level = max(0, pred_level)
            
            # 信頼度の計算（改善版）
            confidence = self._calculate_confidence(
                i, outflows, outflow_acceleration, time_lag
            )
            
            # 結果を追加
            predictions.append({
                'datetime': pred_time.isoformat(),
                'level': round(pred_level, 2),
                'confidence': round(confidence, 2),
                'outflow_impact': round(outflow_impact, 3),
                'trend_impact': round(combined_trend * minutes_ahead / 10, 3),
                'time_lag': time_lag  # デバッグ用
            })
            
        return predictions
        
    def _calculate_water_trends(self, water_levels: List[float]) -> Dict[str, float]:
        """水位トレンドを計算"""
        trends = {'short': 0, 'medium': 0, 'long': 0}
        
        # 短期トレンド（30分）
        if len(water_levels) >= 3:
            trends['short'] = (water_levels[-1] - water_levels[-3]) / 3
            
        # 中期トレンド（1時間）
        if len(water_levels) >= 6:
            trends['medium'] = (water_levels[-1] - water_levels[-6]) / 6
            
        # 長期トレンド（3時間）
        if len(water_levels) >= 18:
            trends['long'] = (water_levels[-1] - water_levels[-18]) / 18
            
        return trends
        
    def _calculate_confidence(self, step: int, outflows: List[float], 
                            acceleration: float, time_lag: int) -> float:
        """予測信頼度を計算"""
        # 基本信頼度（時間とともに減少）
        base_confidence = 1.0 - (step / self.prediction_horizon) * 0.3
        
        # 放流量の変動性
        if len(outflows) >= 6:
            volatility = np.std(outflows[-6:]) / (np.mean(outflows[-6:]) + 1e-6)
            volatility_penalty = min(0.2, volatility * 0.1)
        else:
            volatility_penalty = 0.1
            
        # 加速度による不確実性
        acceleration_penalty = min(0.2, abs(acceleration) / 1000)
        
        # 時間遅延の不確実性
        lag_penalty = max(0, (time_lag - 30) / 100) * 0.1
        
        confidence = base_confidence - volatility_penalty - acceleration_penalty - lag_penalty
        
        return max(0.3, confidence)
        
    def get_prediction_info(self) -> Dict:
        """予測情報の取得"""
        return {
            'method': '高度予測モデル',
            'horizon_hours': 3,
            'interval_minutes': 10,
            'features': ['河川水位', 'ダム放流量', '時間雨量'],
            'improvements': [
                '動的な時間遅延（流量依存）',
                '滑らかな重み付け遷移',
                '放流量の加速度考慮',
                '流量依存の係数調整',
                '予測の連続性確保'
            ],
            'note': '急激な変化にも対応した高精度予測'
        }