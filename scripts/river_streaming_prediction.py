"""
厚東川水位のストリーミング予測モジュール
River 0.21.0対応の動的遅延モデル実装
"""

import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

# River 0.21.0のインポート
from river import linear_model, preprocessing, metrics, stats, tree
from river.compose import Pipeline, TransformerUnion, Select
import river.optim as optim


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
        
        # 遅延時間予測モデル（River 0.21.0）
        self.delay_model = Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(
                optimizer=optim.SGD(lr=0.01),
                l2=0.001
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
                base_delay = max(10, min(180, predicted_delay))  # 10-180分に制限
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


class StreamingFeatureExtractor:
    """ストリーミングデータから特徴量を抽出"""
    
    def __init__(self):
        # 統計量計算器（River 0.21.0）
        self.level_mean_1h = stats.RollingMean(window_size=6)
        self.level_std_1h = stats.RollingVar(window_size=6)
        self.outflow_mean_1h = stats.RollingMean(window_size=6)
        self.outflow_sum_3h = stats.RollingSum(window_size=18)
        self.rainfall_sum_3h = stats.RollingSum(window_size=18)
        
        # 過去データバッファ（最大3時間分）
        self.level_buffer = deque(maxlen=18)
        self.outflow_buffer = deque(maxlen=18)
        self.rainfall_buffer = deque(maxlen=18)
        
        # 前回値
        self.prev_level = None
        self.prev_outflow = None
        
    def extract_features(self, data: Dict) -> Dict:
        """単一データポイントから特徴量を抽出"""
        # 現在値
        level = data.get('river', {}).get('water_level', 0)
        outflow = data.get('dam', {}).get('outflow', 0)
        rainfall = data.get('rainfall', {}).get('hourly', 0)
        
        # バッファに追加
        self.level_buffer.append(level)
        self.outflow_buffer.append(outflow)
        self.rainfall_buffer.append(rainfall)
        
        # 統計量更新
        self.level_mean_1h.update(level)
        self.level_std_1h.update(level)
        self.outflow_mean_1h.update(outflow)
        self.outflow_sum_3h.update(outflow)
        self.rainfall_sum_3h.update(rainfall)
        
        # 特徴量構築
        features = {
            # 現在値
            'level': level,
            'outflow': outflow,
            'rainfall': rainfall,
            
            # 変化量
            'level_change': level - self.prev_level if self.prev_level is not None else 0,
            'outflow_change': outflow - self.prev_outflow if self.prev_outflow is not None else 0,
            
            # 統計量
            'level_mean_1h': self.level_mean_1h.get() if self.level_mean_1h.n > 0 else level,
            'level_std_1h': np.sqrt(self.level_std_1h.get()) if self.level_std_1h.n > 0 else 0,
            'outflow_mean_1h': self.outflow_mean_1h.get() if self.outflow_mean_1h.n > 0 else outflow,
            'outflow_sum_3h': self.outflow_sum_3h.get() if self.outflow_sum_3h.n > 0 else outflow,
            'rainfall_sum_3h': self.rainfall_sum_3h.get() if self.rainfall_sum_3h.n > 0 else rainfall,
            
            # 放流イベント
            'outflow_started': 1 if outflow > 0 and (self.prev_outflow or 0) == 0 else 0,
            'outflow_stopped': 1 if outflow == 0 and (self.prev_outflow or 0) > 0 else 0,
        }
        
        # 動的遅延特徴量
        if len(self.outflow_buffer) > 0:
            delay_estimator = DynamicDelayEstimator()
            estimated_delay, _ = delay_estimator.estimate_delay(
                outflow, 
                context={
                    'river_level': level,
                    'rainfall_3h': features['rainfall_sum_3h'],
                    'outflow_change': features['outflow_change']
                }
            )
            
            # 推定遅延に基づく過去の放流量
            delay_steps = int(estimated_delay / 10)  # 10分単位
            if delay_steps < len(self.outflow_buffer):
                features['delayed_outflow'] = self.outflow_buffer[-delay_steps-1] if delay_steps > 0 else outflow
                features['delayed_outflow_mean'] = np.mean(list(self.outflow_buffer)[-delay_steps-3:-delay_steps+3] if delay_steps > 3 else list(self.outflow_buffer))
            else:
                features['delayed_outflow'] = 0
                features['delayed_outflow_mean'] = 0
        
        # 更新
        self.prev_level = level
        self.prev_outflow = outflow
        
        return features


class RiverStreamingPredictor:
    """River 0.21.0を使用したストリーミング予測モデル"""
    
    def __init__(self, model_path='models/river_streaming_model.pkl'):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 特徴量抽出器
        self.feature_extractor = StreamingFeatureExtractor()
        
        # 遅延推定器
        self.delay_estimator = DynamicDelayEstimator()
        
        # 予測モデル（各ステップ用）
        self.models = {}
        self.metrics_dict = {}
        
        for step in range(1, 19):  # 10分後から3時間後まで
            # River 0.21.0対応モデル
            self.models[f'step_{step}'] = Pipeline(
                preprocessing.StandardScaler(),
                tree.HoeffdingTreeRegressor(
                    grace_period=100,
                    delta=0.01,
                    leaf_prediction='adaptive',
                    model_selector_decay=0.95
                )
            )
            
            self.metrics_dict[f'step_{step}'] = {
                'mae': metrics.MAE(),
                'rmse': metrics.RMSE()
            }
        
        # 学習回数
        self.n_samples = 0
        
        # モデル読み込み
        if self.model_path.exists():
            self.load_model()
    
    def predict_one(self, data: Dict) -> Optional[List[Dict]]:
        """
        単一データポイントから3時間先まで予測
        
        Args:
            data: 現在の観測データ
            
        Returns:
            predictions: 10分刻みの予測リスト
        """
        # 特徴量抽出
        features = self.feature_extractor.extract_features(data)
        
        predictions = []
        base_time = datetime.fromisoformat(data.get('data_time', data.get('timestamp')))
        current_level = features['level']
        
        # 各ステップの予測
        accumulated_change = 0
        
        for step in range(1, 19):
            # ステップ固有の特徴を追加
            step_features = features.copy()
            step_features['prediction_step'] = step
            step_features['accumulated_change'] = accumulated_change
            
            # 遅延を考慮した特徴
            future_minutes = step * 10
            delay_context = {
                'river_level': current_level + accumulated_change,
                'rainfall_3h': features['rainfall_sum_3h'],
                'outflow_change': features['outflow_change']
            }
            
            # 予測
            try:
                model_key = f'step_{step}'
                pred_change = self.models[model_key].predict_one(step_features)
                pred_level = current_level + accumulated_change + pred_change
                pred_level = max(0, pred_level)  # 負値を防ぐ
                
                accumulated_change += pred_change
                
            except Exception as e:
                # エラー時は単純な外挿
                pred_level = current_level + features.get('level_change', 0) * step
                pred_level = max(0, pred_level)
            
            # 予測時刻
            pred_time = base_time + timedelta(minutes=10 * step)
            
            predictions.append({
                'datetime': pred_time.isoformat(),
                'level': round(pred_level, 2),
                'confidence': round(0.5 + min(0.4, self.n_samples / 1000), 2),
                'model_type': 'streaming',
                'considered_delay': round(self.delay_estimator.estimate_delay(
                    features['outflow'], delay_context)[0], 1)
            })
        
        return predictions
    
    def learn_one(self, data: Dict, future_data: Optional[List[Dict]] = None):
        """
        単一データポイントから学習
        
        Args:
            data: 現在の観測データ
            future_data: 将来の実測データ（利用可能な場合）
        """
        # 特徴量抽出
        features = self.feature_extractor.extract_features(data)
        current_level = features['level']
        
        # 将来データがある場合は各ステップで学習
        if future_data:
            accumulated_change = 0
            
            for step, future in enumerate(future_data[:18], 1):
                if 'river' in future and future['river'].get('water_level') is not None:
                    actual_level = future['river']['water_level']
                    actual_change = actual_level - current_level - accumulated_change
                    
                    # ステップ固有の特徴
                    step_features = features.copy()
                    step_features['prediction_step'] = step
                    step_features['accumulated_change'] = accumulated_change
                    
                    # モデル更新
                    model_key = f'step_{step}'
                    self.models[model_key].learn_one(step_features, actual_change)
                    
                    # メトリクス更新
                    pred_change = self.models[model_key].predict_one(step_features)
                    pred_level = current_level + accumulated_change + pred_change
                    self.metrics_dict[model_key]['mae'].update(actual_level, pred_level)
                    self.metrics_dict[model_key]['rmse'].update(actual_level, pred_level)
                    
                    accumulated_change += actual_change
        
        self.n_samples += 1
    
    def get_model_info(self) -> Dict:
        """モデル情報を取得"""
        performance = {
            'n_samples': self.n_samples,
            'model_type': 'River Streaming Model',
            'metrics_by_step': {}
        }
        
        for step in range(1, 19):
            model_key = f'step_{step}'
            if model_key in self.metrics_dict:
                mae_val = self.metrics_dict[model_key]['mae'].get()
                rmse_val = self.metrics_dict[model_key]['rmse'].get()
                
                performance['metrics_by_step'][f'{step*10}min'] = {
                    'mae': round(mae_val, 3) if mae_val else None,
                    'rmse': round(rmse_val, 3) if rmse_val else None
                }
        
        return performance
    
    def save_model(self):
        """モデルの保存"""
        model_data = {
            'models': self.models,
            'metrics': self.metrics_dict,
            'n_samples': self.n_samples,
            'feature_extractor': self.feature_extractor,
            'delay_estimator': self.delay_estimator
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """モデルの読み込み"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.models = model_data.get('models', self.models)
                self.metrics_dict = model_data.get('metrics', self.metrics_dict)
                self.n_samples = model_data.get('n_samples', 0)
                self.feature_extractor = model_data.get('feature_extractor', self.feature_extractor)
                self.delay_estimator = model_data.get('delay_estimator', self.delay_estimator)
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")