"""
厚東川水位のオンライン学習予測モジュール
Riverライブラリを使用した適応的な予測
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from river import linear_model, preprocessing, metrics, optim
from river import time_series, tree, ensemble, rules
from river.compose import Pipeline
import json
import os
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Tuple


class RiverOnlinePredictor:
    """Riverライブラリを使用したオンライン学習予測クラス"""
    
    def __init__(self, model_path='models/river_online_model.pkl'):
        """
        初期化
        
        Args:
            model_path: モデル保存パス
        """
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 予測ホライズン
        self.prediction_horizon = 18  # 3時間先まで（10分間隔）
        
        # モデルの初期化または読み込み
        if self.model_path.exists():
            self.load_model()
        else:
            self.initialize_model()
            
        # 学習履歴の初期化
        self.learning_history = []
        
    def initialize_model(self):
        """モデルの初期化"""
        # 各予測ステップ用のモデルを作成（マルチステップ予測）
        self.models = {}
        self.ensemble_models = {}
        self.scalers = {}
        self.metrics_dict = {}
        
        for step in range(1, self.prediction_horizon + 1):
            # 特徴量の正規化
            scaler = preprocessing.StandardScaler()
            
            # 複数のモデルを作成してアンサンブル
            # 1. Hoeffding Adaptive Tree（非線形パターン対応）
            hat_model = tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=50,
                split_confidence=0.01,
                leaf_prediction='adaptive',
                model_selector_decay=0.95
            )
            
            # 2. AMRules（ルールベース学習）
            amrules_model = rules.AMRules(
                grace_period=50,
                tie_threshold=0.05,
                expand_confidence=0.01
            )
            
            # 3. 線形回帰モデル（ベースライン）
            linear_model_instance = linear_model.LinearRegression(
                optimizer=optim.SGD(lr=0.01),
                l2=0.001
            )
            
            # アンサンブルモデル（加重平均）
            ensemble_model = ensemble.AdaptiveRandomForestRegressor(
                n_models=3,
                seed=42,
                grace_period=50,
                delta=0.01
            )
            
            # メインモデルはアンサンブル
            self.models[f'step_{step}'] = Pipeline(
                ('scaler', scaler),
                ('model', ensemble_model)
            )
            
            # 個別モデルも保存（分析用）
            self.ensemble_models[f'step_{step}'] = {
                'hat': Pipeline(('scaler', scaler.clone()), ('model', hat_model)),
                'amrules': Pipeline(('scaler', scaler.clone()), ('model', amrules_model)),
                'linear': Pipeline(('scaler', scaler.clone()), ('model', linear_model_instance))
            }
            
            # メトリクス
            self.metrics_dict[f'step_{step}'] = {
                'mae': metrics.MAE(),
                'rmse': metrics.RMSE()
            }
            
        # 学習回数カウンター
        self.n_learned = 0
        
    def prepare_features(self, history_data: List[Dict], target_step: int = 1) -> Optional[Dict]:
        """
        特徴量の準備
        
        Args:
            history_data: 履歴データのリスト
            target_step: 予測ステップ（1-18）
            
        Returns:
            features: 特徴量辞書
        """
        if len(history_data) < 72:  # 12時間分のデータが必要
            return None
            
        # 最新72個のデータを使用（12時間分）
        recent_data = history_data[-72:]
        
        # データ抽出
        features = {}
        
        # 水位データ
        water_levels = []
        for d in recent_data:
            if 'river' in d and d['river'].get('water_level') is not None:
                water_levels.append(d['river']['water_level'])
                
        if len(water_levels) < 18:
            return None
            
        # 基本的な水位特徴量
        features['current_level'] = water_levels[-1]
        features['level_lag_1'] = water_levels[-2] if len(water_levels) >= 2 else water_levels[-1]
        features['level_lag_3'] = water_levels[-4] if len(water_levels) >= 4 else water_levels[-1]
        features['level_lag_6'] = water_levels[-7] if len(water_levels) >= 7 else water_levels[-1]
        features['level_lag_12'] = water_levels[-13] if len(water_levels) >= 13 else water_levels[-1]
        
        # 水位変化率
        features['level_change_10min'] = water_levels[-1] - water_levels[-2] if len(water_levels) >= 2 else 0
        features['level_change_30min'] = water_levels[-1] - water_levels[-4] if len(water_levels) >= 4 else 0
        features['level_change_1h'] = water_levels[-1] - water_levels[-7] if len(water_levels) >= 7 else 0
        features['level_change_3h'] = water_levels[-1] - water_levels[-19] if len(water_levels) >= 19 else 0
        features['level_change_6h'] = water_levels[-1] - water_levels[-37] if len(water_levels) >= 37 else 0
        features['level_change_12h'] = water_levels[-1] - water_levels[-73] if len(water_levels) >= 73 else 0
        features['level_change_24h'] = water_levels[-1] - water_levels[-144] if len(water_levels) >= 144 else 0
        
        # 統計的特徴量（短期・中期・長期）
        features['level_mean_1h'] = np.mean(water_levels[-6:]) if len(water_levels) >= 6 else water_levels[-1]
        features['level_std_1h'] = np.std(water_levels[-6:]) if len(water_levels) >= 6 else 0
        features['level_max_1h'] = np.max(water_levels[-6:]) if len(water_levels) >= 6 else water_levels[-1]
        features['level_min_1h'] = np.min(water_levels[-6:]) if len(water_levels) >= 6 else water_levels[-1]
        
        features['level_mean_6h'] = np.mean(water_levels[-36:]) if len(water_levels) >= 36 else water_levels[-1]
        features['level_std_6h'] = np.std(water_levels[-36:]) if len(water_levels) >= 36 else 0
        features['level_max_6h'] = np.max(water_levels[-36:]) if len(water_levels) >= 36 else water_levels[-1]
        
        features['level_mean_12h'] = np.mean(water_levels) if len(water_levels) >= 72 else water_levels[-1]
        features['level_std_12h'] = np.std(water_levels) if len(water_levels) >= 72 else 0
        
        # ダム放流量特徴量
        outflows = []
        for d in recent_data:
            if 'dam' in d and d['dam'].get('outflow') is not None:
                outflows.append(d['dam']['outflow'])
                
        if outflows:
            features['current_outflow'] = outflows[-1]
            
            # 時間遅延特徴量（30分、60分、90分、120分）
            features['outflow_lag_3'] = outflows[-4] if len(outflows) >= 4 else outflows[-1]  # 30分前
            features['outflow_lag_6'] = outflows[-7] if len(outflows) >= 7 else outflows[-1]  # 60分前
            features['outflow_lag_9'] = outflows[-10] if len(outflows) >= 10 else outflows[-1]  # 90分前
            features['outflow_lag_12'] = outflows[-13] if len(outflows) >= 13 else outflows[-1]  # 120分前
            
            # 累積効果（重み付き合計）
            # 最近の放流量ほど影響が大きいと仮定
            weights_1h = np.exp(-np.arange(6) * 0.2)[::-1]  # 指数的減衰
            weights_1h = weights_1h / weights_1h.sum()
            features['outflow_weighted_1h'] = np.sum(weights_1h * np.array(outflows[-6:])) if len(outflows) >= 6 else outflows[-1]
            
            weights_3h = np.exp(-np.arange(18) * 0.1)[::-1]
            weights_3h = weights_3h / weights_3h.sum()
            features['outflow_weighted_3h'] = np.sum(weights_3h * np.array(outflows[-18:])) if len(outflows) >= 18 else outflows[-1]
            
            # 基本統計量
            features['outflow_mean_1h'] = np.mean(outflows[-6:]) if len(outflows) >= 6 else outflows[-1]
            features['outflow_mean_3h'] = np.mean(outflows[-18:]) if len(outflows) >= 18 else outflows[-1]
            features['outflow_mean_6h'] = np.mean(outflows[-36:]) if len(outflows) >= 36 else outflows[-1]
            
            features['outflow_change_1h'] = outflows[-1] - outflows[-7] if len(outflows) >= 7 else 0
            features['outflow_change_3h'] = outflows[-1] - outflows[-19] if len(outflows) >= 19 else 0
            
            features['outflow_max_1h'] = np.max(outflows[-6:]) if len(outflows) >= 6 else outflows[-1]
            features['outflow_max_3h'] = np.max(outflows[-18:]) if len(outflows) >= 18 else outflows[-1]
            
            # 放流開始/停止の検出
            if len(outflows) >= 2:
                features['outflow_started'] = 1 if outflows[-1] > 0 and outflows[-2] == 0 else 0
                features['outflow_stopped'] = 1 if outflows[-1] == 0 and outflows[-2] > 0 else 0
            else:
                features['outflow_started'] = 0
                features['outflow_stopped'] = 0
                
        else:
            # デフォルト値
            features['current_outflow'] = 0
            features['outflow_lag_3'] = 0
            features['outflow_lag_6'] = 0
            features['outflow_lag_9'] = 0
            features['outflow_lag_12'] = 0
            features['outflow_weighted_1h'] = 0
            features['outflow_weighted_3h'] = 0
            features['outflow_mean_1h'] = 0
            features['outflow_mean_3h'] = 0
            features['outflow_mean_6h'] = 0
            features['outflow_change_1h'] = 0
            features['outflow_change_3h'] = 0
            features['outflow_max_1h'] = 0
            features['outflow_max_3h'] = 0
            features['outflow_started'] = 0
            features['outflow_stopped'] = 0
            
        # 雨量特徴量
        rainfalls = []
        for d in recent_data:
            if 'rainfall' in d and d['rainfall'].get('hourly') is not None:
                rainfalls.append(d['rainfall']['hourly'])
                
        if rainfalls:
            features['rainfall_current'] = rainfalls[-1]
            features['rainfall_sum_1h'] = sum(rainfalls[-6:]) if len(rainfalls) >= 6 else sum(rainfalls)
            features['rainfall_sum_3h'] = sum(rainfalls[-18:]) if len(rainfalls) >= 18 else sum(rainfalls)
            features['rainfall_sum_6h'] = sum(rainfalls[-36:]) if len(rainfalls) >= 36 else sum(rainfalls)
            features['rainfall_sum_12h'] = sum(rainfalls[-72:]) if len(rainfalls) >= 72 else sum(rainfalls)
            features['rainfall_sum_24h'] = sum(rainfalls[-144:]) if len(rainfalls) >= 144 else sum(rainfalls)
            
            features['rainfall_max_3h'] = max(rainfalls[-18:]) if len(rainfalls) >= 18 else max(rainfalls)
            features['rainfall_max_6h'] = max(rainfalls[-36:]) if len(rainfalls) >= 36 else max(rainfalls)
            features['rainfall_max_24h'] = max(rainfalls) if rainfalls else 0
            
            # 降雨イベント検出
            # 連続降雨時間（降雨があった期間）
            rain_duration = 0
            for i in range(len(rainfalls) - 1, -1, -1):
                if rainfalls[i] > 0:
                    rain_duration += 1
                else:
                    break
            features['rain_duration'] = rain_duration
            
            # 降雨開始からの経過時間（降雨がない場合は大きな値）
            rain_start_idx = -1
            for i in range(len(rainfalls) - 1, -1, -1):
                if rainfalls[i] > 0:
                    rain_start_idx = i
            
            if rain_start_idx >= 0:
                features['time_since_rain_start'] = (len(rainfalls) - 1 - rain_start_idx) * 10  # 分単位
            else:
                features['time_since_rain_start'] = 999  # 降雨なし
                
            # 降雨強度の変化
            if len(rainfalls) >= 6:
                recent_rain = np.mean(rainfalls[-6:])
                older_rain = np.mean(rainfalls[-12:-6]) if len(rainfalls) >= 12 else 0
                features['rainfall_intensity_change'] = recent_rain - older_rain
            else:
                features['rainfall_intensity_change'] = 0
                
        else:
            features['rainfall_current'] = 0
            features['rainfall_sum_1h'] = 0
            features['rainfall_sum_3h'] = 0
            features['rainfall_sum_6h'] = 0
            features['rainfall_sum_12h'] = 0
            features['rainfall_sum_24h'] = 0
            features['rainfall_max_3h'] = 0
            features['rainfall_max_6h'] = 0
            features['rainfall_max_24h'] = 0
            features['rain_duration'] = 0
            features['time_since_rain_start'] = 999
            features['rainfall_intensity_change'] = 0
            
        # 時間的特徴量（周期性）
        try:
            last_time = datetime.fromisoformat(recent_data[-1].get('data_time', recent_data[-1].get('timestamp')))
            features['hour'] = last_time.hour
            features['hour_sin'] = np.sin(2 * np.pi * last_time.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * last_time.hour / 24)
        except:
            features['hour'] = 12
            features['hour_sin'] = 0
            features['hour_cos'] = 1
            
        # 予測ステップに応じた追加特徴量
        features['prediction_step'] = target_step
        features['prediction_minutes'] = target_step * 10
        
        return features
        
    def predict(self, history_data: List[Dict]) -> Optional[List[Dict]]:
        """
        3時間先の水位を予測（オンライン学習モデル）
        
        Args:
            history_data: 履歴データのリスト
            
        Returns:
            predictions: 予測結果のリスト
        """
        features = self.prepare_features(history_data)
        if features is None:
            return None
            
        predictions = []
        base_time = datetime.fromisoformat(
            history_data[-1].get('data_time', history_data[-1].get('timestamp'))
        )
        
        # 各ステップの予測
        for step in range(1, self.prediction_horizon + 1):
            # ステップ固有の特徴量を準備
            step_features = features.copy()
            step_features['prediction_step'] = step
            step_features['prediction_minutes'] = step * 10
            
            # モデルで予測
            model_key = f'step_{step}'
            if model_key in self.models:
                try:
                    # 予測実行
                    pred_change = self.models[model_key].predict_one(step_features)
                    
                    # 基準水位に変化量を加算
                    pred_level = features['current_level'] + pred_change
                    
                    # 負の値を防ぐ
                    pred_level = max(0, pred_level)
                    
                except Exception as e:
                    # エラー時は単純な線形外挿
                    if 'level_change_10min' in features:
                        pred_level = features['current_level'] + features['level_change_10min'] * step
                    else:
                        pred_level = features['current_level']
            else:
                pred_level = features['current_level']
                
            # 予測時刻
            pred_time = base_time + timedelta(minutes=10 * step)
            
            # 予測信頼度（学習回数に基づく）
            confidence = min(0.9, 0.5 + (self.n_learned / 1000) * 0.4)
            
            predictions.append({
                'datetime': pred_time.isoformat(),
                'level': round(pred_level, 2),
                'confidence': round(confidence, 2),
                'model_type': 'online_learning',
                'n_learned': self.n_learned
            })
            
        return predictions
        
    def learn(self, history_data: List[Dict]):
        """
        履歴データから学習（オンライン学習）
        
        Args:
            history_data: 履歴データのリスト
        """
        if len(history_data) < 72 + self.prediction_horizon:  # 12時間分のデータが必要
            return
            
        # 学習データの準備
        for i in range(len(history_data) - 72 - self.prediction_horizon):
            # 訓練用の履歴データ（12時間分）
            train_history = history_data[i:i+72]
            
            # 各予測ステップで学習
            for step in range(1, self.prediction_horizon + 1):
                # 特徴量準備
                features = self.prepare_features(train_history, step)
                if features is None:
                    continue
                    
                # 実際の水位（target）
                target_idx = i + 72 + step - 1
                if target_idx < len(history_data):
                    target_data = history_data[target_idx]
                    if 'river' in target_data and target_data['river'].get('water_level') is not None:
                        actual_level = target_data['river']['water_level']
                        # 変化量として学習
                        target_change = actual_level - features['current_level']
                        
                        # モデル更新
                        model_key = f'step_{step}'
                        if model_key in self.models:
                            # 予測
                            pred_change = self.models[model_key].predict_one(features)
                            
                            # 学習
                            self.models[model_key].learn_one(features, target_change)
                            
                            # メトリクス更新
                            pred_level = features['current_level'] + pred_change
                            self.metrics_dict[model_key]['mae'].update(actual_level, pred_level)
                            self.metrics_dict[model_key]['rmse'].update(actual_level, pred_level)
                            
        self.n_learned += 1
        
    def get_model_performance(self) -> Dict:
        """モデルの性能指標を取得"""
        performance = {
            'n_learned': self.n_learned,
            'model_type': 'River Online Learning',
            'metrics_by_step': {}
        }
        
        for step in range(1, self.prediction_horizon + 1):
            model_key = f'step_{step}'
            if model_key in self.metrics_dict:
                mae_val = self.metrics_dict[model_key]['mae'].get()
                rmse_val = self.metrics_dict[model_key]['rmse'].get()
                
                performance['metrics_by_step'][f'{step*10}min'] = {
                    'mae': round(mae_val, 3) if mae_val else None,
                    'rmse': round(rmse_val, 3) if rmse_val else None
                }
                
        # 平均性能
        all_mae = [m['mae'] for m in performance['metrics_by_step'].values() if m['mae']]
        all_rmse = [m['rmse'] for m in performance['metrics_by_step'].values() if m['rmse']]
        
        if all_mae:
            performance['avg_mae'] = round(np.mean(all_mae), 3)
        if all_rmse:
            performance['avg_rmse'] = round(np.mean(all_rmse), 3)
            
        return performance
        
    def save_model(self):
        """モデルの保存"""
        model_data = {
            'models': self.models,
            'metrics': self.metrics_dict,
            'n_learned': self.n_learned,
            'learning_history': self.learning_history[-100:]  # 最新100件のみ
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self):
        """モデルの読み込み"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.models = model_data.get('models', {})
                self.metrics_dict = model_data.get('metrics', {})
                self.n_learned = model_data.get('n_learned', 0)
                self.learning_history = model_data.get('learning_history', [])
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            self.initialize_model()
            
    def get_prediction_info(self) -> Dict:
        """予測情報の取得"""
        performance = self.get_model_performance()
        
        return {
            'method': 'River オンライン学習（改良版）',
            'algorithm': 'Adaptive Random Forest + Hoeffding Tree + AMRules',
            'horizon_hours': 3,
            'interval_minutes': 10,
            'features': [
                '水位（現在値、遅延値、変化率、12時間統計）',
                'ダム放流量（時間遅延30-120分、累積効果、開始/停止検出）',
                '雨量（累積1-12時間、降雨イベント検出、強度変化）',
                '時間的特徴（時刻の周期性）'
            ],
            'learning': {
                'type': 'オンライン学習',
                'n_samples': self.n_learned,
                'performance': performance
            },
            'improvements': [
                '12時間分のデータを活用',
                '降雨イベント特徴量を追加',
                '放流量の時間遅延効果をモデル化',
                'アンサンブル学習による精度向上'
            ],
            'note': '降雨時の予測精度を重点的に改善'
        }