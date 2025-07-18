"""
厚東川水位予測モジュール
Riverライブラリを使用した3時間先の水位予測
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
try:
    from river import time_series, preprocessing, metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    # Riverが利用できない場合の簡易実装
    class SimpleSNARIMAX:
        def __init__(self, **kwargs):
            self.params = kwargs
            self.history = []
            
        def learn_one(self, x, y):
            self.history.append((x, y))
            if len(self.history) > 100:  # 最新100件のみ保持
                self.history = self.history[-100:]
                
        def forecast(self, horizon, xs):
            # 簡易的な線形回帰による予測
            if len(self.history) < 2:
                return 0.0
            
            # 最近のトレンドを計算
            recent = [h[1] for h in self.history[-10:]]
            if len(recent) >= 2:
                trend = (recent[-1] - recent[0]) / len(recent)
                return trend * horizon
            return 0.0
    
    class SimpleMAE:
        def __init__(self):
            self.sum_error = 0
            self.n = 0
            
        def update(self, y_true, y_pred):
            self.sum_error += abs(y_true - y_pred)
            self.n += 1
            
        def get(self):
            return self.sum_error / self.n if self.n > 0 else 0
    
    class SimpleRMSE:
        def __init__(self):
            self.sum_squared_error = 0
            self.n = 0
            
        def update(self, y_true, y_pred):
            self.sum_squared_error += (y_true - y_pred) ** 2
            self.n += 1
            
        def get(self):
            return (self.sum_squared_error / self.n) ** 0.5 if self.n > 0 else 0
    
    class SimpleStandardScaler:
        def __init__(self):
            pass
            
        def learn_one(self, x):
            pass
            
        def transform_one(self, x):
            return x
import json
import os
from pathlib import Path
import pickle


class RiverLevelPredictor:
    """河川水位予測クラス"""
    
    def __init__(self, model_path='models/river_level_model.pkl'):
        """
        初期化
        
        Args:
            model_path: モデル保存パス
        """
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # モデルの初期化または読み込み
        if self.model_path.exists():
            self.load_model()
        else:
            self.initialize_model()
            
        # 評価メトリクス
        if RIVER_AVAILABLE:
            self.mae = metrics.MAE()
            self.rmse = metrics.RMSE()
        else:
            self.mae = SimpleMAE()
            self.rmse = SimpleRMSE()
        
    def initialize_model(self):
        """モデルの初期化"""
        if RIVER_AVAILABLE:
            # SNARIMAXモデルの初期化
            # p=2, d=1, q=1: 基本的なARIMAパラメータ
            # m=6: 1時間周期（10分×6）
            self.model = time_series.SNARIMAX(
                p=2,  # 自己回帰項
                d=1,  # 差分の階数
                q=1,  # 移動平均項
                m=6,  # 季節性の周期
                sp=1, # 季節性自己回帰項
                sd=0, # 季節性差分
                sq=1, # 季節性移動平均項
            )
            
            # 特徴量の正規化
            self.scaler = preprocessing.StandardScaler()
        else:
            # 簡易実装を使用
            self.model = SimpleSNARIMAX(p=2, d=1, q=1, m=6, sp=1, sd=0, sq=1)
            self.scaler = SimpleStandardScaler()
        
    def prepare_features(self, history_data):
        """
        特徴量の準備
        
        Args:
            history_data: 履歴データのリスト（時系列順）
            
        Returns:
            features: 特徴量辞書
        """
        if len(history_data) < 18:  # 3時間分のデータが必要
            return None
            
        # 最新18個のデータを使用
        recent_data = history_data[-18:]
        
        # 基本統計量の計算
        river_levels = [d['river']['water_level'] for d in recent_data if d['river']['water_level'] is not None]
        outflows = [d['dam']['outflow'] for d in recent_data if d['dam']['outflow'] is not None]
        inflows = [d['dam']['inflow'] for d in recent_data if d['dam']['inflow'] is not None]
        rainfalls = [d['rainfall']['hourly'] for d in recent_data if d['rainfall']['hourly'] is not None]
        
        if not river_levels:
            return None
            
        features = {
            # 現在の水位
            'current_level': river_levels[-1],
            
            # 水位の変化傾向
            'level_change_1h': river_levels[-1] - river_levels[-6] if len(river_levels) >= 6 else 0,
            'level_change_3h': river_levels[-1] - river_levels[0] if len(river_levels) >= 18 else 0,
            
            # 平均値
            'avg_level_1h': np.mean(river_levels[-6:]) if len(river_levels) >= 6 else river_levels[-1],
            'avg_outflow_3h': np.mean(outflows) if outflows else 0,
            'avg_inflow_3h': np.mean(inflows) if inflows else 0,
            
            # 累積雨量
            'total_rainfall_3h': sum(rainfalls) if rainfalls else 0,
            
            # 流入出量の差（貯水量変化の指標）
            'flow_diff_3h': np.mean([i - o for i, o in zip(inflows, outflows)]) if inflows and outflows else 0,
            
            # 降水強度の最大値（利用可能な場合）
            'max_precipitation_intensity': self._get_max_precipitation_intensity(recent_data),
        }
        
        return features
        
    def _get_max_precipitation_intensity(self, data):
        """降水強度の最大値を取得"""
        max_intensity = 0
        for d in data:
            if 'precipitation_intensity' in d and d['precipitation_intensity']:
                obs = d['precipitation_intensity'].get('observation', [])
                fore = d['precipitation_intensity'].get('forecast', [])
                
                for item in obs + fore:
                    if isinstance(item, dict) and 'intensity' in item:
                        max_intensity = max(max_intensity, item['intensity'])
                        
        return max_intensity
        
    def predict(self, history_data):
        """
        3時間先の水位を予測
        
        Args:
            history_data: 履歴データのリスト
            
        Returns:
            predictions: 予測結果のリスト（10分間隔で18ポイント）
        """
        features = self.prepare_features(history_data)
        if features is None:
            return None
            
        # 予測結果を格納するリスト
        predictions = []
        current_time = datetime.fromisoformat(history_data[-1]['data_time'])
        current_level = features['current_level']
        
        # 3時間先まで10分間隔で予測
        for i in range(18):
            # 予測時刻
            pred_time = current_time + timedelta(minutes=10 * (i + 1))
            
            # SNARIMAXは単一値の予測を行うため、
            # 各ステップで予測を更新
            if i == 0:
                # 最初の予測
                pred = self.model.forecast(horizon=1, xs=[features])
                pred_level = current_level + pred
            else:
                # 前の予測値を使って次を予測
                # 簡易的な実装として、トレンドを維持
                trend = (predictions[-1]['level'] - current_level) / (i)
                pred_level = predictions[-1]['level'] + trend
                
            # 予測値の妥当性チェック（負の値を防ぐ）
            pred_level = max(0, pred_level)
            
            predictions.append({
                'datetime': pred_time.isoformat(),
                'level': round(pred_level, 2)
            })
            
        return predictions
        
    def update_model(self, actual_data, predicted_data):
        """
        実測値でモデルを更新（オンライン学習）
        
        Args:
            actual_data: 実測値
            predicted_data: 予測値
        """
        if actual_data and predicted_data:
            # 予測誤差の計算
            actual = actual_data['river']['water_level']
            predicted = predicted_data[0]['level']  # 最初の予測値
            
            # メトリクスの更新
            self.mae.update(actual, predicted)
            self.rmse.update(actual, predicted)
            
            # モデルの更新
            features = self.prepare_features([actual_data])
            if features:
                self.model.learn_one(features, actual)
                
    def save_model(self):
        """モデルの保存"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'mae': self.mae.get(),
            'rmse': self.rmse.get()
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self):
        """モデルの読み込み"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            self.initialize_model()
            
    def get_metrics(self):
        """評価メトリクスの取得"""
        return {
            'mae': round(self.mae.get(), 3) if self.mae.n > 0 else None,
            'rmse': round(self.rmse.get(), 3) if self.rmse.n > 0 else None,
            'n_samples': self.mae.n
        }


def load_history_data(data_dir='data/history', hours=24):
    """
    履歴データの読み込み
    
    Args:
        data_dir: データディレクトリ
        hours: 読み込む時間数
        
    Returns:
        history_data: 履歴データのリスト
    """
    history_data = []
    current_time = datetime.now()
    
    for i in range(hours * 6):  # 10分間隔なので1時間に6データ
        target_time = current_time - timedelta(minutes=10 * i)
        year = target_time.strftime('%Y')
        month = target_time.strftime('%m')
        day = target_time.strftime('%d')
        time_str = target_time.strftime('%H%M')
        
        file_path = Path(data_dir) / year / month / day / f"{time_str}.json"
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history_data.append(data)
            except Exception as e:
                continue
                
    return sorted(history_data, key=lambda x: x['data_time'])


if __name__ == "__main__":
    # テスト実行
    predictor = RiverLevelPredictor()
    
    # 履歴データの読み込み
    history_data = load_history_data(hours=6)
    
    if len(history_data) >= 18:
        # 予測実行
        predictions = predictor.predict(history_data)
        
        if predictions:
            print("3時間先の水位予測:")
            for pred in predictions[:6]:  # 最初の1時間分を表示
                print(f"{pred['datetime']}: {pred['level']}m")
                
            # メトリクスの表示
            metrics = predictor.get_metrics()
            print(f"\n予測精度: MAE={metrics['mae']}, RMSE={metrics['rmse']}")
    else:
        print("予測に必要なデータが不足しています。")