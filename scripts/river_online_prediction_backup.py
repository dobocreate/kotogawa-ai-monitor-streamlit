"""
厚東川水位のオンライン学習予測モジュール
Riverライブラリを使用した適応的な予測
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from river import linear_model, preprocessing, metrics, optim
from river import time_series
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
        self.scalers = {}
        self.metrics_dict = {}
        
        for step in range(1, self.prediction_horizon + 1):
            # 線形回帰モデル（SGD最適化）
            model = linear_model.LinearRegression(
                optimizer=optim.SGD(lr=0.01),
                l2=0.001  # 正則化
            )
            
            # 特徴量の正規化
            scaler = preprocessing.StandardScaler()
            
            # パイプライン作成
            self.models[f'step_{step}'] = Pipeline(
                ('scaler', scaler),
                ('model', model)
            )
            
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
        if len(history_data) < 24:  # 4時間分のデータが必要
            return None
            
        # 最新24個のデータを使用
        recent_data = history_data[-24:]
        
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
        
        # 水位変化率
        features['level_change_10min'] = water_levels[-1] - water_levels[-2] if len(water_levels) >= 2 else 0
        features['level_change_30min'] = water_levels[-1] - water_levels[-4] if len(water_levels) >= 4 else 0
        features['level_change_1h'] = water_levels[-1] - water_levels[-7] if len(water_levels) >= 7 else 0
        
        # 統計的特徴量
        features['level_mean_1h'] = np.mean(water_levels[-6:]) if len(water_levels) >= 6 else water_levels[-1]
        features['level_std_1h'] = np.std(water_levels[-6:]) if len(water_levels) >= 6 else 0
        features['level_max_1h'] = np.max(water_levels[-6:]) if len(water_levels) >= 6 else water_levels[-1]
        features['level_min_1h'] = np.min(water_levels[-6:]) if len(water_levels) >= 6 else water_levels[-1]
        
        # ダム放流量特徴量
        outflows = []
        for d in recent_data:
            if 'dam' in d and d['dam'].get('outflow') is not None:
                outflows.append(d['dam']['outflow'])
                
        if outflows:
            features['current_outflow'] = outflows[-1]
            features['outflow_lag_4'] = outflows[-5] if len(outflows) >= 5 else outflows[-1]  # 40分前
            features['outflow_mean_1h'] = np.mean(outflows[-6:]) if len(outflows) >= 6 else outflows[-1]
            features['outflow_change_1h'] = outflows[-1] - outflows[-7] if len(outflows) >= 7 else 0
            features['outflow_max_1h'] = np.max(outflows[-6:]) if len(outflows) >= 6 else outflows[-1]
        else:
            # デフォルト値
            features['current_outflow'] = 0
            features['outflow_lag_4'] = 0
            features['outflow_mean_1h'] = 0
            features['outflow_change_1h'] = 0
            features['outflow_max_1h'] = 0
            
        # 雨量特徴量
        rainfalls = []
        for d in recent_data:
            if 'rainfall' in d and d['rainfall'].get('hourly') is not None:
                rainfalls.append(d['rainfall']['hourly'])
                
        if rainfalls:
            features['rainfall_current'] = rainfalls[-1]
            features['rainfall_sum_3h'] = sum(rainfalls[-18:]) if len(rainfalls) >= 18 else sum(rainfalls)
            features['rainfall_max_3h'] = max(rainfalls[-18:]) if len(rainfalls) >= 18 else max(rainfalls)
        else:
            features['rainfall_current'] = 0
            features['rainfall_sum_3h'] = 0
            features['rainfall_max_3h'] = 0
            
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
        if len(history_data) < 24 + self.prediction_horizon:
            return
            
        # 学習データの準備
        for i in range(len(history_data) - 24 - self.prediction_horizon):
            # 訓練用の履歴データ
            train_history = history_data[i:i+24]
            
            # 各予測ステップで学習
            for step in range(1, self.prediction_horizon + 1):
                # 特徴量準備
                features = self.prepare_features(train_history, step)
                if features is None:
                    continue
                    
                # 実際の水位（target）
                target_idx = i + 24 + step - 1
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
            'method': 'River オンライン学習',
            'algorithm': 'Linear Regression with SGD',
            'horizon_hours': 3,
            'interval_minutes': 10,
            'features': [
                '水位（現在値、遅延値、変化率）',
                'ダム放流量（現在値、遅延値、変化率）',
                '雨量（現在値、累積値）',
                '時間的特徴（時刻の周期性）'
            ],
            'learning': {
                'type': 'オンライン学習',
                'n_samples': self.n_learned,
                'performance': performance
            },
            'note': '継続的に学習して精度が向上します'
        }