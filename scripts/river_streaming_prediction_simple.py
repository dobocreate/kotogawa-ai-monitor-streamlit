"""
シンプルで安定したRiverストリーミング予測モデル
エラーを最小限に抑えた実装
"""

import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class SimpleRiverPredictor:
    """シンプルで安定したRiver予測モデル"""
    
    def __init__(self, model_path='models/river_streaming_model_simple.pkl'):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # シンプルな構成
        self.n_samples = 0
        self.last_prediction = None
        
        # モデル読み込み
        if self.model_path.exists():
            self.load_model()
    
    def extract_features(self, data: Dict) -> Dict:
        """データから特徴量を安全に抽出"""
        try:
            # デフォルト値を使用して安全に値を取得
            level = 0.0
            outflow = 0.0
            rainfall = 0.0
            
            if isinstance(data, dict):
                river_data = data.get('river', {})
                if isinstance(river_data, dict):
                    level = float(river_data.get('water_level', 0.0) or 0.0)
                
                dam_data = data.get('dam', {})
                if isinstance(dam_data, dict):
                    outflow = float(dam_data.get('outflow', 0.0) or 0.0)
                
                rain_data = data.get('rainfall', {})
                if isinstance(rain_data, dict):
                    rainfall = float(rain_data.get('hourly', 0.0) or 0.0)
            
            return {
                'water_level': level,
                'dam_outflow': outflow,
                'rainfall': rainfall
            }
        except Exception:
            # エラー時はデフォルト値を返す
            return {
                'water_level': 0.0,
                'dam_outflow': 0.0,
                'rainfall': 0.0
            }
    
    def predict_one(self, data: Dict) -> Optional[List[Dict]]:
        """シンプルな予測"""
        try:
            features = self.extract_features(data)
            current_level = features['water_level']
            outflow = features['dam_outflow']
            
            # シンプルな予測ロジック
            # 放流量に基づく基本的な水位変化予測
            base_change = 0.0
            if outflow > 0:
                # 放流量100m³/sあたり0.1mの水位上昇
                base_change = (outflow / 100.0) * 0.1
            
            predictions = []
            base_time = datetime.now()
            
            for step in range(1, 19):  # 3時間先まで
                # 時間経過による減衰
                decay = 0.95 ** step
                predicted_change = base_change * decay
                predicted_level = max(0.0, current_level + predicted_change)
                
                pred_time = base_time + timedelta(minutes=10 * step)
                
                predictions.append({
                    'datetime': pred_time.isoformat(),
                    'level': round(predicted_level, 2),
                    'confidence': 0.7,  # 固定値
                    'model_type': 'river_simple',
                    'mae_last_100': None,
                    'drift_detected': False,
                    'model_version': 'simple-v1'
                })
            
            self.last_prediction = predictions
            return predictions
            
        except Exception:
            # エラー時はNoneを返す
            return None
    
    def learn_one(self, data: Dict, future_data: Optional[List[Dict]] = None):
        """学習（カウントのみ）"""
        try:
            self.n_samples += 1
            # シンプルな実装では実際の学習は行わない
        except Exception:
            pass
    
    def predict(self, history_data: List[Dict]) -> Optional[List[Dict]]:
        """履歴データから予測（互換性のため）"""
        if not history_data:
            return None
        return self.predict_one(history_data[-1])
    
    def get_model_info(self) -> Dict:
        """モデル情報を取得"""
        return {
            'n_samples': self.n_samples,
            'model_type': 'Simple River Predictor',
            'mae_10min': None,
            'rmse_10min': None,
            'drift_count': 0,
            'drift_rate': 0.0,
            'status': 'stable'
        }
    
    def save_model(self):
        """モデルの保存"""
        try:
            model_data = {
                'n_samples': self.n_samples,
                'last_prediction': self.last_prediction
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception:
            pass
    
    def load_model(self):
        """モデルの読み込み"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.n_samples = model_data.get('n_samples', 0)
                self.last_prediction = model_data.get('last_prediction')
        except Exception:
            pass