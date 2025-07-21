"""
厚東川水位のストリーミング予測モジュール（仕様書準拠版）
River 0.22.0対応 - ARFRegressorとADWINドリフト検出実装
"""

import numpy as np
from datetime import datetime, timedelta
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
    utils
)
from river.stats import Mean


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
        """仕様書準拠のパイプライン構築"""
        # 特徴量エンジニアリングパイプライン
        # River 0.22.0対応の構成
        self.pipeline = compose.Pipeline(
            # 標準化のみ（StatImputerは削除）
            pp.StandardScaler(),
            # ARFRegressor（仕様書準拠）
            forest.ARFRegressor(
                n_models=15,
                max_depth=15,
                drift_detector=drift.ADWIN(delta=1e-3),
                seed=42
            )
        )
        
        # 各ステップ用のモデル（互換性のため維持）
        self.models = {}
        for step in range(1, 19):
            self.models[f'step_{step}'] = compose.Pipeline(
                pp.StandardScaler(),
                forest.ARFRegressor(
                    n_models=10,
                    max_depth=12,
                    drift_detector=drift.ADWIN(delta=1e-3),
                    seed=42 + step
                )
            )
    
    def extract_features(self, data: Dict) -> Dict:
        """データから特徴量を抽出（仕様書準拠）"""
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
        
        # 仕様書準拠の特徴量
        features = {
            'timestamp': timestamp,
            'water_level': level,
            'dam_outflow': outflow,
            'rainfall': rainfall,
            'elapsed_min': elapsed_min
        }
        
        return features
    
    def predict_one(self, data: Dict) -> Optional[List[Dict]]:
        """
        単一データポイントから3時間先まで予測（仕様書準拠）
        
        Args:
            data: 現在の観測データ
            
        Returns:
            predictions: 10分刻みの予測リスト
        """
        # 特徴量抽出
        features = self.extract_features(data)
        
        predictions = []
        base_time = datetime.fromisoformat(features['timestamp'])
        current_level = features['water_level']
        
        # 基本予測（メインパイプライン使用）
        try:
            # 10分先予測
            base_prediction = self.pipeline.predict_one(features)
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
                    pred_change = self.models[model_key].predict_one(step_features)
                    # predict_oneがNoneを返す場合の対処
                    if pred_change is None:
                        pred_change = 0
                
                pred_level = current_level + accumulated_change + pred_change
                pred_level = max(0, pred_level)  # 負値を防ぐ
                
                accumulated_change += pred_change
                
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
                'model_version': f'river-0.21.0-{datetime.now().strftime("%Y%m%dT%H%M")}'
            })
        
        return predictions
    
    def learn_one(self, data: Dict, future_data: Optional[List[Dict]] = None):
        """
        単一データポイントから学習（仕様書準拠）
        
        Args:
            data: 現在の観測データ
            future_data: 将来の実測データ（利用可能な場合）
        """
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
                        # メインパイプラインで学習
                        self.pipeline.learn_one(features, actual_level)
                        
                        # ドリフト検出
                        pred = self.pipeline.predict_one(features)
                        if pred is not None:
                            error = abs(actual_level - pred)
                        else:
                            error = abs(actual_level - current_level)  # フォールバック
                        self.drift_detector.update(error)
                        if self.drift_detector.drift_detected:
                            self.drift_count += 1
                            self.drift_history.append({
                                'timestamp': features['timestamp'],
                                'error': error
                            })
                    
                    # 各ステップモデルも更新
                    actual_change = actual_level - current_level - accumulated_change
                    
                    step_features = features.copy()
                    step_features['prediction_step'] = step
                    step_features['accumulated_change'] = accumulated_change
                    
                    model_key = f'step_{step}'
                    self.models[model_key].learn_one(step_features, actual_change)
                    
                    # メトリクス更新
                    pred_change = self.models[model_key].predict_one(step_features)
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
            'mae_rolling': self.mae_rolling
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
                
            print(f"モデル読み込み成功: {self.n_samples}サンプル学習済み")
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")