"""
厚東川水位のストリーミング予測モジュール（仕様書準拠版）
River 0.22.0対応 - ARFRegressorとADWINドリフト検出実装
"""

import numpy as np
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path
import warnings

# River 0.21.0のインポート
from river import (
    compose, 
    feature_extraction as fx,
    preprocessing as pp,
    forest,
    drift,
    metrics,
    stats,
    linear_model,
    optim,
    utils,
    anomaly
)
from river.stats import Mean
from scipy import signal


class DynamicDelayEstimator:
    """放流量に基づいて遅延時間を動的に推定"""
    
    def __init__(self, initial_delays: Dict[str, float] = None):
        # 初期遅延設定
        self.initial_delays = initial_delays or {
            'high': 30,    # 大量放流時（>100 m³/s）
            'medium': 60,  # 中程度（50-100 m³/s）
            'low': 90,     # 少量（0-50 m³/s）
            'none': 120    # 放流なし
        }
        
        # 遅延時間予測モデル（ARFRegressor使用）
        self.delay_model = compose.Pipeline(
            pp.StandardScaler(),
            forest.ARFRegressor(
                n_models=10,
                max_depth=10,
                drift_detector=drift.ADWIN(delta=0.002),
                seed=42
            )
        )
        
        # 観測された遅延データ
        self.observed_delays = deque(maxlen=200)
        self.n_learned = 0
        
    def get_initial_delay(self, outflow: float) -> float:
        """初期遅延時間の推定"""
        if outflow > 100:
            return self.initial_delays['high']
        elif outflow > 50:
            return self.initial_delays['medium']
        elif outflow > 0:
            return self.initial_delays['low']
        else:
            return self.initial_delays['none']
    
    def estimate_delay(self, outflow: float, context: Dict = None) -> Tuple[float, float]:
        """
        遅延時間を推定
        Returns: (推定遅延時間（分）, 信頼度)
        """
        if context is None:
            context = {}
            
        # 特徴量構築
        features = {
            'outflow': outflow,
            'outflow_squared': outflow ** 2,
            'outflow_log': np.log1p(outflow),
            'river_level': context.get('river_level', 2.0),
            'recent_rainfall': context.get('rainfall_3h', 0),
            'outflow_change': context.get('outflow_change', 0)
        }
        
        if self.n_learned < 10:
            # 学習初期は初期値を使用
            base_delay = self.get_initial_delay(outflow)
            confidence = 0.3
        else:
            # 学習済みモデルで予測
            try:
                predicted_delay = self.delay_model.predict_one(features)
                if predicted_delay is not None:
                    base_delay = max(10, min(180, predicted_delay))  # 10-180分に制限
                else:
                    base_delay = self.get_initial_delay(outflow)
                confidence = min(0.9, 0.3 + self.n_learned / 100)
            except:
                base_delay = self.get_initial_delay(outflow)
                confidence = 0.5
                
        return base_delay, confidence
    
    def learn_from_observation(self, outflow: float, delay_minutes: float, context: Dict = None):
        """観測された遅延から学習"""
        if context is None:
            context = {}
            
        features = {
            'outflow': outflow,
            'outflow_squared': outflow ** 2,
            'outflow_log': np.log1p(outflow),
            'river_level': context.get('river_level', 2.0),
            'recent_rainfall': context.get('rainfall_3h', 0),
            'outflow_change': context.get('outflow_change', 0)
        }
        
        # モデル更新
        self.delay_model.learn_one(features, delay_minutes)
        self.observed_delays.append((outflow, delay_minutes))
        self.n_learned += 1


class RiverStreamingPredictor:
    """River 0.21.0を使用したストリーミング予測モデル（仕様書準拠）"""
    
    def __init__(self, model_path='models/river_streaming_model_v2.pkl'):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 履歴データバッファ（90分分 = 9ステップ + 現在）
        self.history_buffer = deque(maxlen=10)
        
        # 外れ値検出器
        self.outlier_detector = anomaly.HalfSpaceTrees(
            n_trees=10,
            height=8,
            window_size=100,
            seed=42
        )
        
        # 指数移動平均（EMA）計算用
        self.ema_alpha = {
            'short': 0.5,   # 短期（約20分）
            'medium': 0.2,  # 中期（約50分）
            'long': 0.1     # 長期（約100分）
        }
        self.ema_values = {
            'water_level': {'short': None, 'medium': None, 'long': None},
            'dam_outflow': {'short': None, 'medium': None, 'long': None}
        }
        
        # 遅延推定器
        self.delay_estimator = DynamicDelayEstimator()
        
        # 仕様書準拠のパイプライン構築
        self._build_pipeline()
        
        # 評価メトリクス
        self.mae_metric = metrics.MAE()
        self.rmse_metric = metrics.RMSE()
        self.mae_by_step = {f'step_{i}': metrics.MAE() for i in range(1, 19)}
        self.rmse_by_step = {f'step_{i}': metrics.RMSE() for i in range(1, 19)}
        
        # ドリフト検出器（独立）
        self.drift_detector = drift.ADWIN(delta=0.001)
        self.drift_count = 0
        self.drift_history = deque(maxlen=100)
        
        # 統計量追跡（River 0.22.0対応）
        self.level_stats = utils.Rolling(stats.Mean(), window_size=100)
        self.mae_rolling = utils.Rolling(stats.Mean(), window_size=100)
        
        # 学習回数
        self.n_samples = 0
        
        # モデル読み込み
        if self.model_path.exists():
            self.load_model()
    
    def _build_pipeline(self):
        """高精度予測のための最適化されたパイプライン"""
        # メインパイプライン（10分先予測用）
        self.pipeline = compose.Pipeline(
            # 標準化
            pp.StandardScaler(),
            # より複雑なパターンを学習できるように最適化
            forest.ARFRegressor(
                n_models=50,  # モデル数を大幅増加（15→50）
                max_depth=30,  # より深い木を許可（15→30）
                drift_detector=drift.ADWIN(delta=1e-4),  # より敏感なドリフト検出
                leaf_prediction='adaptive',  # 適応的な葉の予測
                grace_period=50,  # 初期学習期間
                seed=42
            )
        )
        
        # 各ステップ用のモデル（長期予測用）
        self.models = {}
        for step in range(1, 19):
            # 予測時間に応じてモデルの複雑さを調整
            if step <= 6:  # 60分以内
                n_models = 40
                max_depth = 25
            elif step <= 12:  # 120分以内
                n_models = 30
                max_depth = 20
            else:  # 180分まで
                n_models = 25
                max_depth = 20
            
            self.models[f'step_{step}'] = compose.Pipeline(
                pp.StandardScaler(),
                forest.ARFRegressor(
                    n_models=n_models,
                    max_depth=max_depth,
                    drift_detector=drift.ADWIN(delta=1e-4),
                    leaf_prediction='adaptive',
                    grace_period=30,
                    seed=42 + step
                )
            )
    
    def update_history_buffer(self, data: Dict):
        """履歴バッファを更新"""
        self.history_buffer.append(data)
        
    def update_ema(self, key: str, value: float):
        """指数移動平均を更新"""
        for period in ['short', 'medium', 'long']:
            if self.ema_values[key][period] is None:
                self.ema_values[key][period] = value
            else:
                alpha = self.ema_alpha[period]
                self.ema_values[key][period] = alpha * value + (1 - alpha) * self.ema_values[key][period]
    
    def detect_outlier(self, features: Dict) -> bool:
        """外れ値検出"""
        # 主要な特徴量のみを使用
        outlier_features = {
            'water_level': features.get('water_level', 0),
            'dam_outflow': features.get('dam_outflow', 0),
            'water_level_change_10min': features.get('water_level_change_10min', 0),
            'dam_trend_recent': features.get('dam_trend_recent', 0)
        }
        
        score = self.outlier_detector.score_one(outlier_features)
        self.outlier_detector.learn_one(outlier_features)
        
        # スコアが異常に高い場合は外れ値
        return score > 0.8 if score is not None else False
    
    def extract_features(self, data: Dict) -> Dict:
        """データから特徴量を抽出（Phase 2実装）"""
        # 基本データ取得
        timestamp = data.get('data_time', data.get('timestamp', ''))
        
        # 安全な値取得（Noneの場合はデフォルト値を使用）
        river_data = data.get('river') or {}
        level = river_data.get('water_level') if river_data.get('water_level') is not None else 0
        
        dam_data = data.get('dam') or {}
        outflow = dam_data.get('outflow') if dam_data.get('outflow') is not None else 0
        
        rainfall_data = data.get('rainfall') or {}
        rainfall = rainfall_data.get('hourly') if rainfall_data.get('hourly') is not None else 0
        
        # 前回観測からの経過時間計算
        elapsed_min = 10  # デフォルト10分
        if hasattr(self, '_last_timestamp'):
            try:
                current = datetime.fromisoformat(timestamp)
                last = datetime.fromisoformat(self._last_timestamp)
                elapsed_min = (current - last).total_seconds() / 60
            except:
                pass
        self._last_timestamp = timestamp
        
        # 基本特徴量
        features = {
            'water_level': float(level),
            'dam_outflow': float(outflow),
            'rainfall': float(rainfall),
            'elapsed_min': float(elapsed_min),
            # 平常時との差分（重要な特徴量）
            'water_level_diff_from_normal': float(level) - 2.75,  # 平常時水位との差
            'dam_outflow_diff_from_normal': float(outflow) - 4.5,  # 平常時放流量との差
            'is_above_normal': 1.0 if float(level) > 2.75 else 0.0,  # 平常時より高いか
            'outflow_ratio': float(outflow) / 4.5 if outflow > 0 else 0.0  # 平常時の何倍か
        }
        
        # Phase 1: 時系列生データ（過去90分）
        history = list(self.history_buffer)
        for i in range(1, min(10, len(history) + 1)):
            if i <= len(history):
                hist_data = history[-i]
                # ダム放流量の時系列
                features[f'dam_outflow_t-{i*10}'] = float(hist_data.get('dam', {}).get('outflow', 0))
                # 降雨量の時系列
                features[f'rainfall_t-{i*10}'] = float(hist_data.get('rainfall', {}).get('hourly', 0))
                # 水位の時系列（10, 30, 60分前のみ）
                if i in [1, 3, 6]:
                    features[f'water_level_t-{i*10}'] = float(hist_data.get('river', {}).get('water_level', 0))
            else:
                # データがない場合はデフォルト値
                features[f'dam_outflow_t-{i*10}'] = 0.0
                features[f'rainfall_t-{i*10}'] = 0.0
                if i in [1, 3, 6]:
                    features[f'water_level_t-{i*10}'] = level  # 現在値で補完
        
        # Phase 1: 基本集約統計量
        if len(history) >= 3:
            recent = history[-3:]
            features['dam_sum_recent'] = sum(d.get('dam', {}).get('outflow', 0) for d in recent)
            features['dam_max_recent'] = max(d.get('dam', {}).get('outflow', 0) for d in recent)
            features['rain_sum_recent'] = sum(d.get('rainfall', {}).get('hourly', 0) for d in recent)
        
        if len(history) >= 9:
            all_data = history[-9:]
            features['dam_sum_90min'] = sum(d.get('dam', {}).get('outflow', 0) for d in all_data)
            features['dam_max_90min'] = max(d.get('dam', {}).get('outflow', 0) for d in all_data)
            features['rain_sum_90min'] = sum(d.get('rainfall', {}).get('hourly', 0) for d in all_data)
        
        # Phase 2: 時間窓集約特徴量
        if len(history) >= 3:
            recent_30 = history[-3:]
            features['dam_sum_0_30min'] = sum(d.get('dam', {}).get('outflow', 0) for d in recent_30)
            features['dam_max_0_30min'] = max(d.get('dam', {}).get('outflow', 0) for d in recent_30)
            features['dam_avg_0_30min'] = np.mean([d.get('dam', {}).get('outflow', 0) for d in recent_30])
            features['rain_sum_0_30min'] = sum(d.get('rainfall', {}).get('hourly', 0) for d in recent_30)
        
        if len(history) >= 6:
            middle_30 = history[-6:-3]
            features['dam_sum_30_60min'] = sum(d.get('dam', {}).get('outflow', 0) for d in middle_30)
            features['dam_max_30_60min'] = max(d.get('dam', {}).get('outflow', 0) for d in middle_30)
            features['dam_avg_30_60min'] = np.mean([d.get('dam', {}).get('outflow', 0) for d in middle_30])
            features['rain_sum_30_60min'] = sum(d.get('rainfall', {}).get('hourly', 0) for d in middle_30)
        
        if len(history) >= 9:
            older_30 = history[-9:-6]
            features['dam_sum_60_90min'] = sum(d.get('dam', {}).get('outflow', 0) for d in older_30)
            features['dam_max_60_90min'] = max(d.get('dam', {}).get('outflow', 0) for d in older_30)
            features['dam_avg_60_90min'] = np.mean([d.get('dam', {}).get('outflow', 0) for d in older_30])
            features['rain_sum_60_90min'] = sum(d.get('rainfall', {}).get('hourly', 0) for d in older_30)
        
        # Phase 2: トレンド・パターン特徴量（変化率を重視）
        # 水位変化率
        if len(history) >= 1:
            level_10min_ago = float(history[-1].get('river', {}).get('water_level', level))
            features['water_level_change_10min'] = level - level_10min_ago
            features['water_level_change_rate_10min'] = (level - level_10min_ago) * 6  # 時間あたり変化率
            
        if len(history) >= 3:
            level_30min_ago = float(history[-3].get('river', {}).get('water_level', level))
            features['water_level_change_30min'] = level - level_30min_ago
            features['water_level_change_rate_30min'] = (level - level_30min_ago) * 2  # 時間あたり変化率
            
            # 加速度（変化率の変化）
            if len(history) >= 1:
                features['water_level_acceleration'] = features['water_level_change_10min'] - \
                    (level_30min_ago - float(history[-2].get('river', {}).get('water_level', level)))
            
        if len(history) >= 6:
            level_60min_ago = float(history[-6].get('river', {}).get('water_level', level))
            features['water_level_change_60min'] = level - level_60min_ago
            features['water_level_change_rate_60min'] = level - level_60min_ago  # 時間あたり変化率
        
        # ダム放流のトレンド（変化の影響を強調）
        if len(history) >= 3:
            dam_recent = float(history[-1].get('dam', {}).get('outflow', 0))
            dam_30min = float(history[-3].get('dam', {}).get('outflow', 0))
            features['dam_trend_recent'] = dam_recent - dam_30min
            features['dam_change_impact'] = (dam_recent - dam_30min) * 0.01  # 放流変化の影響係数
        
        if len(history) >= 6:
            dam_30min = float(history[-3].get('dam', {}).get('outflow', 0))
            dam_60min = float(history[-6].get('dam', {}).get('outflow', 0))
            features['dam_trend_older'] = dam_30min - dam_60min
            features['dam_trend_acceleration'] = features.get('dam_trend_recent', 0) - (dam_30min - dam_60min)
        
        # ピーク検出と急激な変化の検出
        if len(history) >= 9:
            dam_values = [d.get('dam', {}).get('outflow', 0) for d in history[-9:]]
            features['dam_peak_90min'] = max(dam_values)
            features['dam_peak_timing'] = (dam_values.index(max(dam_values)) + 1) * 10  # 何分前がピークか
            
            # 変動性
            features['dam_std_90min'] = np.std(dam_values)
            rain_values = [d.get('rainfall', {}).get('hourly', 0) for d in history[-9:]]
            features['rain_std_90min'] = np.std(rain_values)
            
            # 急激な変化の検出
            water_levels = [d.get('river', {}).get('water_level', level) for d in history[-9:]] + [level]
            max_change = max(water_levels[i] - water_levels[i-1] for i in range(1, len(water_levels)))
            features['max_water_change_90min'] = max_change
            features['is_rapid_rise'] = 1.0 if max_change > 0.1 else 0.0  # 10cm以上の上昇
            
            # 異常状態の検出
            max_level_90min = max(water_levels)
            features['is_flood_risk'] = 1.0 if max_level_90min > 5.0 else 0.0  # 5m以上は洪水リスク
            features['is_high_water'] = 1.0 if max_level_90min > 4.0 else 0.0  # 4m以上は高水位
            
            # 放流量の異常検出
            max_outflow = max(dam_values)
            features['is_emergency_discharge'] = 1.0 if max_outflow > 50.0 else 0.0  # 50m³/s以上は緊急放流
            features['is_high_discharge'] = 1.0 if max_outflow > 20.0 else 0.0  # 20m³/s以上は高放流
        
        # 指数移動平均の更新と特徴量追加
        self.update_ema('water_level', float(level))
        self.update_ema('dam_outflow', float(outflow))
        
        # EMA特徴量を追加
        if self.ema_values['water_level']['short'] is not None:
            features['water_level_ema_short'] = self.ema_values['water_level']['short']
            features['water_level_ema_medium'] = self.ema_values['water_level']['medium']
            features['water_level_ema_long'] = self.ema_values['water_level']['long']
            
            # EMAとの差分（トレンドからの乖離）
            features['water_level_ema_diff_short'] = float(level) - self.ema_values['water_level']['short']
            features['water_level_ema_diff_medium'] = float(level) - self.ema_values['water_level']['medium']
            features['water_level_ema_diff_long'] = float(level) - self.ema_values['water_level']['long']
            
            # EMAクロスオーバー（トレンド転換の検出）
            features['ema_crossover_short_medium'] = 1.0 if self.ema_values['water_level']['short'] > self.ema_values['water_level']['medium'] else 0.0
            features['ema_crossover_medium_long'] = 1.0 if self.ema_values['water_level']['medium'] > self.ema_values['water_level']['long'] else 0.0
        
        if self.ema_values['dam_outflow']['short'] is not None:
            features['dam_outflow_ema_short'] = self.ema_values['dam_outflow']['short']
            features['dam_outflow_ema_medium'] = self.ema_values['dam_outflow']['medium']
            features['dam_outflow_ema_long'] = self.ema_values['dam_outflow']['long']
            
            # ダム放流のEMA差分
            features['dam_outflow_ema_diff_short'] = float(outflow) - self.ema_values['dam_outflow']['short']
        
        # 移動平均（Simple Moving Average）
        if len(history) >= 3:
            water_levels_3 = [d.get('river', {}).get('water_level', level) for d in history[-3:]] + [level]
            features['water_level_sma_3'] = np.mean(water_levels_3)
            features['water_level_sma_diff'] = float(level) - features['water_level_sma_3']
            
            dam_values_3 = [d.get('dam', {}).get('outflow', 0) for d in history[-3:]] + [outflow]
            features['dam_outflow_sma_3'] = np.mean(dam_values_3)
        
        if len(history) >= 6:
            water_levels_6 = [d.get('river', {}).get('water_level', level) for d in history[-6:]] + [level]
            features['water_level_sma_6'] = np.mean(water_levels_6)
            
            # ボリンジャーバンド的な特徴（変動幅の検出）
            water_std = np.std(water_levels_6)
            features['water_level_bb_upper'] = features['water_level_sma_6'] + 2 * water_std
            features['water_level_bb_lower'] = features['water_level_sma_6'] - 2 * water_std
            features['water_level_bb_position'] = (float(level) - features['water_level_bb_lower']) / (features['water_level_bb_upper'] - features['water_level_bb_lower'] + 1e-6)
        
        # 自己相関特徴量（周期性の検出）
        if len(history) >= 9:
            water_levels_all = [d.get('river', {}).get('water_level', level) for d in history[-9:]] + [level]
            
            # ラグ1, 3, 6の自己相関（NaN処理付き）
            try:
                if len(water_levels_all) >= 7:
                    corr1 = np.corrcoef(water_levels_all[:-1], water_levels_all[1:])[0, 1]
                    features['water_autocorr_lag1'] = corr1 if not np.isnan(corr1) else 0.0
                if len(water_levels_all) >= 4:
                    corr3 = np.corrcoef(water_levels_all[:-3], water_levels_all[3:])[0, 1]
                    features['water_autocorr_lag3'] = corr3 if not np.isnan(corr3) else 0.0
                if len(water_levels_all) >= 7:
                    corr6 = np.corrcoef(water_levels_all[:-6], water_levels_all[6:])[0, 1]
                    features['water_autocorr_lag6'] = corr6 if not np.isnan(corr6) else 0.0
            except:
                # エラー時はデフォルト値を設定
                features['water_autocorr_lag1'] = 0.0
                features['water_autocorr_lag3'] = 0.0
                features['water_autocorr_lag6'] = 0.0
            
            # 簡易的な周波数特徴（FFTの代わり）
            # 変化の回数をカウント（ゼロクロッシング的な特徴）
            changes = np.diff(water_levels_all)
            features['water_direction_changes'] = sum(1 for i in range(1, len(changes)) if changes[i] * changes[i-1] < 0)
            
            # 累積変化量（モメンタム）
            features['water_momentum_30min'] = sum(changes[-3:]) if len(changes) >= 3 else 0
            features['water_momentum_60min'] = sum(changes[-6:]) if len(changes) >= 6 else 0
        
        # 外れ値スコア
        outlier_score = self.detect_outlier(features)
        features['is_outlier'] = 1.0 if outlier_score else 0.0
        
        # データ品質の特徴量
        features['data_quality_score'] = 1.0  # デフォルトは高品質
        
        # 物理的に不可能な値のチェック
        if float(level) < 0 or float(level) > 10:  # 水位が負または10m以上
            features['data_quality_score'] = 0.5
        if float(outflow) < 0 or float(outflow) > 200:  # 放流量が負または200m³/s以上
            features['data_quality_score'] = min(features['data_quality_score'], 0.5)
        
        # 急激すぎる変化のチェック
        if 'water_level_change_10min' in features and abs(features['water_level_change_10min']) > 0.5:  # 10分で50cm以上の変化
            features['data_quality_score'] = min(features['data_quality_score'], 0.7)
        
        # タイムスタンプは別途保存（特徴量には含めない）
        features['_timestamp'] = timestamp
        
        return features
    
    def predict_one(self, data: Dict) -> Optional[List[Dict]]:
        """
        単一データポイントから3時間先まで予測（仕様書準拠）
        
        Args:
            data: 現在の観測データ
            
        Returns:
            predictions: 10分刻みの予測リスト
        """
        # 履歴バッファを更新
        self.update_history_buffer(data)
        
        # 特徴量抽出
        features = self.extract_features(data)
        
        predictions = []
        base_time = datetime.fromisoformat(features['_timestamp'])
        current_level = features['water_level']
        
        # 基本予測（メインパイプライン使用）
        try:
            # 10分先予測（タイムスタンプを除外）
            prediction_features = {k: v for k, v in features.items() if not k.startswith('_')}
            base_prediction = self.pipeline.predict_one(prediction_features)
            if base_prediction is not None:
                base_change = base_prediction - current_level
            else:
                base_change = 0
        except:
            base_change = 0
        
        # 各ステップの予測
        accumulated_change = 0
        
        for step in range(1, 19):
            # ステップ固有の特徴を追加
            step_features = features.copy()
            step_features['prediction_step'] = step
            step_features['accumulated_change'] = accumulated_change
            
            # 遅延を考慮
            delay_context = {
                'river_level': current_level + accumulated_change,
                'rainfall_3h': features.get('rainfall', 0) * 3,
                'outflow_change': 0
            }
            estimated_delay, delay_confidence = self.delay_estimator.estimate_delay(
                features['dam_outflow'], delay_context
            )
            
            # 予測
            try:
                model_key = f'step_{step}'
                if step == 1:
                    # 最初のステップはメインパイプラインの予測を使用
                    pred_change = base_change
                else:
                    # タイムスタンプを除外して予測
                    step_prediction_features = {k: v for k, v in step_features.items() if not k.startswith('_')}
                    pred_change = self.models[model_key].predict_one(step_prediction_features)
                    # predict_oneがNoneを返す場合の対処
                    if pred_change is None:
                        pred_change = 0
                
                # 物理的制約の適用
                # 1. 基本的な制約
                pred_level = current_level + accumulated_change + pred_change
                pred_level = max(0, pred_level)  # 負値を防ぐ
                pred_level = min(10, pred_level)  # 10m以上は非現実的
                
                # 2. 変化率の制約（10分あたり最大30cmの変化）
                max_change_per_10min = 0.3
                if step == 1:
                    actual_change = pred_level - current_level
                else:
                    prev_level = current_level + accumulated_change
                    actual_change = pred_level - prev_level
                
                if abs(actual_change) > max_change_per_10min:
                    # 変化を制限
                    limited_change = max_change_per_10min if actual_change > 0 else -max_change_per_10min
                    pred_level = (current_level + accumulated_change) + limited_change
                
                # 3. トレンドベースの制約
                if 'water_level_change_rate_30min' in features:
                    trend_rate = features['water_level_change_rate_30min']
                    expected_change = trend_rate * (step * 10 / 60)  # 時間あたり変化率から期待値を計算
                    
                    # 予測がトレンドから大きく外れる場合は調整
                    predicted_total_change = pred_level - current_level
                    if abs(predicted_total_change - expected_change) > 0.5:  # 50cm以上の乖離
                        # トレンドと予測の重み付け平均
                        pred_level = 0.7 * pred_level + 0.3 * (current_level + expected_change)
                
                # 4. データ品質による調整
                if features.get('data_quality_score', 1.0) < 0.8:
                    # データ品質が低い場合は保守的な予測に調整
                    pred_level = 0.8 * pred_level + 0.2 * current_level
                
                # 累積変化を更新
                accumulated_change = pred_level - current_level
                
            except Exception as e:
                # エラー時は単純な外挿
                pred_level = current_level + base_change * step
                pred_level = max(0, pred_level)
            
            # 予測時刻
            pred_time = base_time + timedelta(minutes=10 * step)
            
            # MAE計算（履歴がある場合）
            mae_metric = self.mae_by_step[f'step_{step}']
            mae_value = mae_metric.get() if mae_metric.get() > 0 else None
            
            predictions.append({
                'datetime': pred_time.isoformat(),
                'level': round(pred_level, 2),
                'confidence': round(0.5 + min(0.4, self.n_samples / 1000), 2),
                'model_type': 'river_streaming_v2',
                'considered_delay': round(estimated_delay, 1),
                'mae_last_100': round(mae_value, 3) if mae_value else None,
                'drift_detected': self.drift_count > 0,
                'model_version': f'river-0.21.0-{datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%dT%H%M")}'
            })
        
        return predictions
    
    def learn_one(self, data: Dict, future_data: Optional[List[Dict]] = None):
        """
        単一データポイントから学習（仕様書準拠）
        
        Args:
            data: 現在の観測データ
            future_data: 将来の実測データ（利用可能な場合）
        """
        # 履歴バッファを更新
        self.update_history_buffer(data)
        
        # 特徴量抽出
        features = self.extract_features(data)
        current_level = features['water_level']
        
        # 統計量更新
        self.level_stats.update(current_level)
        
        # 将来データがある場合は各ステップで学習
        if future_data:
            accumulated_change = 0
            
            for step, future in enumerate(future_data[:18], 1):
                if 'river' in future and future['river'].get('water_level') is not None:
                    actual_level = future['river']['water_level']
                    
                    if step == 1:
                        # メインパイプラインで学習（タイムスタンプを除外）
                        learning_features = {k: v for k, v in features.items() if not k.startswith('_')}
                        
                        # 急激な変化を重視した学習
                        change_magnitude = abs(actual_level - current_level)
                        if change_magnitude > 0.1:  # 10cm以上の変化
                            # 重要なパターンは複数回学習（過学習を避けるため最大3回）
                            repeat_times = min(3, int(change_magnitude * 10))
                            for _ in range(repeat_times):
                                self.pipeline.learn_one(learning_features, actual_level)
                        else:
                            # 通常の学習
                            self.pipeline.learn_one(learning_features, actual_level)
                        
                        # ドリフト検出
                        pred = self.pipeline.predict_one(learning_features)
                        if pred is not None:
                            error = abs(actual_level - pred)
                        else:
                            error = abs(actual_level - current_level)  # フォールバック
                        self.drift_detector.update(error)
                        if self.drift_detector.drift_detected:
                            self.drift_count += 1
                            self.drift_history.append({
                                'timestamp': features['_timestamp'],
                                'error': error
                            })
                    
                    # 各ステップモデルも更新
                    actual_change = actual_level - current_level - accumulated_change
                    
                    step_features = features.copy()
                    step_features['prediction_step'] = step
                    step_features['accumulated_change'] = accumulated_change
                    
                    model_key = f'step_{step}'
                    # タイムスタンプを除外して学習
                    step_learning_features = {k: v for k, v in step_features.items() if not k.startswith('_')}
                    
                    # 長期予測でも急激な変化を重視
                    if abs(actual_change) > 0.05:  # 5cm以上の変化
                        repeat_times = min(2, int(abs(actual_change) * 20))
                        for _ in range(repeat_times):
                            self.models[model_key].learn_one(step_learning_features, actual_change)
                    else:
                        self.models[model_key].learn_one(step_learning_features, actual_change)
                    
                    # メトリクス更新
                    pred_change = self.models[model_key].predict_one(step_learning_features)
                    if pred_change is None:
                        pred_change = 0
                    pred_level = current_level + accumulated_change + pred_change
                    
                    self.mae_by_step[model_key].update(actual_level, pred_level)
                    self.rmse_by_step[model_key].update(actual_level, pred_level)
                    
                    # 全体メトリクス（10分先のみ）
                    if step == 1:
                        self.mae_metric.update(actual_level, pred_level)
                        self.rmse_metric.update(actual_level, pred_level)
                        self.mae_rolling.update(abs(actual_level - pred_level))
                    
                    accumulated_change += actual_change
        
        self.n_samples += 1
        
        # 定期的にモデル保存（100サンプルごと）
        if self.n_samples % 100 == 0:
            self.save_model()
    
    def predict(self, history_data: List[Dict]) -> Optional[List[Dict]]:
        """
        履歴データから予測（互換性のため）
        """
        if not history_data:
            return None
        
        # 最新データを使用
        latest_data = history_data[-1]
        return self.predict_one(latest_data)
    
    def get_model_info(self) -> Dict:
        """モデル情報を取得（仕様書準拠）"""
        performance = {
            'n_samples': self.n_samples,
            'model_type': 'River Streaming Model v2 (ARF + ADWIN)',
            'mae_10min': round(self.mae_metric.get(), 3) if self.mae_metric.get() > 0 else None,
            'rmse_10min': round(self.rmse_metric.get(), 3) if self.rmse_metric.get() > 0 else None,
            'mae_rolling_avg': round(self.mae_rolling.get(), 3) if hasattr(self.mae_rolling, 'get') and self.mae_rolling.get() > 0 else None,
            'drift_count': self.drift_count,
            'drift_rate': round(self.drift_count / max(1, self.n_samples) * 100, 2),
            'metrics_by_step': {}
        }
        
        # 各ステップのメトリクス
        for step in range(1, 19):
            model_key = f'step_{step}'
            mae_val = self.mae_by_step[model_key].get() if self.mae_by_step[model_key].get() > 0 else None
            rmse_val = self.rmse_by_step[model_key].get() if self.rmse_by_step[model_key].get() > 0 else None
            
            performance['metrics_by_step'][f'{step*10}min'] = {
                'mae': round(mae_val, 3) if mae_val else None,
                'rmse': round(rmse_val, 3) if rmse_val else None
            }
        
        # 最近のドリフト情報
        if self.drift_history:
            recent_drifts = list(self.drift_history)[-5:]
            performance['recent_drifts'] = recent_drifts
        
        return performance
    
    def save_model(self):
        """モデルの保存"""
        model_data = {
            'pipeline': self.pipeline,
            'models': self.models,
            'mae_metric': self.mae_metric,
            'rmse_metric': self.rmse_metric,
            'mae_by_step': self.mae_by_step,
            'rmse_by_step': self.rmse_by_step,
            'drift_detector': self.drift_detector,
            'drift_count': self.drift_count,
            'drift_history': self.drift_history,
            'n_samples': self.n_samples,
            'delay_estimator': self.delay_estimator,
            'level_stats': self.level_stats,
            'mae_rolling': self.mae_rolling,
            'history_buffer': self.history_buffer,  # 履歴バッファも保存
            'ema_values': self.ema_values,  # EMA値を保存
            'outlier_detector': self.outlier_detector  # 外れ値検出器を保存
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """モデルの読み込み"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                
                # パイプラインとモデル
                self.pipeline = model_data.get('pipeline', self.pipeline)
                self.models = model_data.get('models', self.models)
                
                # メトリクス
                self.mae_metric = model_data.get('mae_metric', self.mae_metric)
                self.rmse_metric = model_data.get('rmse_metric', self.rmse_metric)
                self.mae_by_step = model_data.get('mae_by_step', self.mae_by_step)
                self.rmse_by_step = model_data.get('rmse_by_step', self.rmse_by_step)
                
                # ドリフト検出
                self.drift_detector = model_data.get('drift_detector', self.drift_detector)
                self.drift_count = model_data.get('drift_count', 0)
                self.drift_history = model_data.get('drift_history', deque(maxlen=100))
                
                # その他
                self.n_samples = model_data.get('n_samples', 0)
                self.delay_estimator = model_data.get('delay_estimator', self.delay_estimator)
                self.level_stats = model_data.get('level_stats', self.level_stats)
                self.mae_rolling = model_data.get('mae_rolling', self.mae_rolling)
                
                # 履歴バッファ（新規追加）
                self.history_buffer = model_data.get('history_buffer', deque(maxlen=10))
                
                # EMA値と外れ値検出器
                self.ema_values = model_data.get('ema_values', self.ema_values)
                self.outlier_detector = model_data.get('outlier_detector', self.outlier_detector)
                
            print(f"モデル読み込み成功: {self.n_samples}サンプル学習済み")
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")