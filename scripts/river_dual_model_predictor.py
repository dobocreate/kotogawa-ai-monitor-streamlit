#!/usr/bin/env python3
"""
基本モデルと適応モデルを組み合わせたデュアルモデル予測器
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from river_streaming_prediction_v2 import RiverStreamingPredictor
from datetime import datetime
import json
from typing import Dict, List, Optional, Any


class RiverDualModelPredictor:
    """基本モデルと適応モデルを組み合わせた予測器"""
    
    def __init__(self):
        self.base_model = None
        self.adaptive_model = None
        self.adaptive_weight = 0.3  # 初期重み（30%）
        self.models_loaded = False
        
        # モデルパス
        self.base_model_path = Path(__file__).parent.parent / "models" / "river_base_model_v2.pkl"
        self.adaptive_model_path = Path(__file__).parent.parent / "models" / "river_adaptive_model_v2.pkl"
        
    def load_models(self) -> bool:
        """両モデルを読み込み"""
        try:
            # 基本モデルの読み込み
            if not self.base_model_path.exists():
                print(f"エラー: 基本モデルが見つかりません: {self.base_model_path}")
                print("initialize_models.py を実行してモデルを初期化してください。")
                return False
                
            self.base_model = RiverStreamingPredictor()
            self.base_model.model_path = str(self.base_model_path)
            self.base_model.load_model()
            print(f"基本モデルを読み込みました (サンプル数: {self.base_model.n_samples})")
            
            # 適応モデルの読み込み
            if not self.adaptive_model_path.exists():
                print(f"エラー: 適応モデルが見つかりません: {self.adaptive_model_path}")
                print("initialize_models.py を実行してモデルを初期化してください。")
                return False
                
            self.adaptive_model = RiverStreamingPredictor()
            self.adaptive_model.model_path = str(self.adaptive_model_path)
            self.adaptive_model.load_model()
            print(f"適応モデルを読み込みました (サンプル数: {self.adaptive_model.n_samples})")
            
            # 適応モデルの重みを調整
            self._adjust_adaptive_weight()
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False
    
    def _adjust_adaptive_weight(self):
        """適応モデルの重みを学習サンプル数に基づいて調整"""
        if not self.adaptive_model:
            return
            
        # 基本モデルのサンプル数との差分を計算
        base_samples = self.base_model.n_samples if self.base_model else 0
        adaptive_samples = self.adaptive_model.n_samples
        additional_samples = adaptive_samples - base_samples
        
        # 追加学習サンプル数に基づいて重みを設定
        if additional_samples <= 0:
            self.adaptive_weight = 0.1  # 追加学習なし: 10%
        elif additional_samples < 100:
            self.adaptive_weight = 0.15  # 少量: 15%
        elif additional_samples < 500:
            self.adaptive_weight = 0.25  # 中量: 25%
        elif additional_samples < 1000:
            self.adaptive_weight = 0.30  # 多量: 30%
        else:
            self.adaptive_weight = 0.35  # 大量: 35%（最大）
            
        print(f"適応モデルの重み: {self.adaptive_weight:.0%} (追加学習: {additional_samples}サンプル)")
    
    def predict_one(self, data: Dict[str, Any], model_type: str = 'combined') -> List[Dict[str, Any]]:
        """両モデルで予測し、重み付け統合
        
        Args:
            data: 予測用データ
            model_type: 'combined' (統合), 'base' (基本のみ), 'adaptive' (適応のみ)
        """
        if not self.models_loaded:
            if not self.load_models():
                return []
        
        # モデルタイプに応じて予測
        if model_type == 'base':
            # 基本モデルのみ使用
            return self.base_model.predict_one(data)
        elif model_type == 'adaptive':
            # 適応モデルのみ使用
            return self.adaptive_model.predict_one(data)
        
        # 統合モデル（デフォルト）
        # 各モデルで予測
        base_predictions = self.base_model.predict_one(data)
        adaptive_predictions = self.adaptive_model.predict_one(data)
        
        if not base_predictions or not adaptive_predictions:
            print("警告: 予測結果が取得できませんでした")
            return []
        
        # 予測結果を統合
        combined_predictions = []
        
        for base, adaptive in zip(base_predictions, adaptive_predictions):
            # 重み付け平均で水位を計算
            combined_level = (
                base["level"] * (1 - self.adaptive_weight) + 
                adaptive["level"] * self.adaptive_weight
            )
            
            # 信頼度も重み付け平均
            combined_confidence = (
                base["confidence"] * (1 - self.adaptive_weight) + 
                adaptive["confidence"] * self.adaptive_weight
            )
            
            # 統合結果を作成
            combined = {
                "datetime": base["datetime"],
                "level": round(combined_level, 2),
                "confidence": round(combined_confidence, 3),
                "model_type": "dual_model_v2",
                "base_level": base["level"],
                "adaptive_level": adaptive["level"],
                "adaptive_weight": round(self.adaptive_weight, 2),
                "considered_delay": base.get("considered_delay", 90),
                "mae_last_100": adaptive.get("mae_last_100"),  # 適応モデルの精度を使用
                "drift_detected": adaptive.get("drift_detected", False),
                "model_version": f"dual-{datetime.now():%Y%m%d}"
            }
            
            # その他の情報もコピー
            for key in ["mae_10min", "mae_30min", "mae_60min"]:
                if key in adaptive:
                    combined[key] = adaptive[key]
                    
            combined_predictions.append(combined)
        
        return combined_predictions
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """特徴量を抽出（適応モデルのメソッドを使用）"""
        if not self.adaptive_model:
            return {}
        return self.adaptive_model.extract_features(data)
    
    def learn_one(self, features: Dict[str, float], target: float, step: int = 0):
        """適応モデルのみ学習"""
        if not self.models_loaded:
            if not self.load_models():
                return
                
        # 適応モデルのみ学習
        self.adaptive_model.learn_one(features, target, step)
        
        # 100サンプルごとに重みを再調整
        if self.adaptive_model.n_samples % 100 == 0:
            self._adjust_adaptive_weight()
    
    def save_model(self, path: Optional[str] = None):
        """適応モデルのみ保存（基本モデルは変更しない）"""
        if not self.adaptive_model:
            return
            
        # pathが指定されていない場合はデフォルトパスを使用
        if path is None:
            path = str(self.adaptive_model_path)
            
        self.adaptive_model.save_model(path)
        print(f"適応モデルを保存しました: {path}")
    
    def save_models(self):
        """両モデルを保存（実際は適応モデルのみ）"""
        self.save_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        if not self.models_loaded:
            self.load_models()
            
        info = {
            "model_type": "dual_model_v2",
            "adaptive_weight": round(self.adaptive_weight, 2),
            "base_model": {
                "loaded": self.base_model is not None,
                "samples": self.base_model.n_samples if self.base_model else 0,
                "mae_10min": None
            },
            "adaptive_model": {
                "loaded": self.adaptive_model is not None,
                "samples": self.adaptive_model.n_samples if self.adaptive_model else 0,
                "mae_10min": None,
                "additional_samples": 0
            }
        }
        
        # 基本モデルの精度情報
        if self.base_model and hasattr(self.base_model.mae_metric, 'get'):
            mae = self.base_model.mae_metric.get()
            if mae > 0:
                info["base_model"]["mae_10min"] = round(mae, 3)
                
        # 適応モデルの精度情報
        if self.adaptive_model:
            if hasattr(self.adaptive_model.mae_metric, 'get'):
                mae = self.adaptive_model.mae_metric.get()
                if mae > 0:
                    info["adaptive_model"]["mae_10min"] = round(mae, 3)
                    
            # 追加学習サンプル数
            if self.base_model:
                info["adaptive_model"]["additional_samples"] = (
                    self.adaptive_model.n_samples - self.base_model.n_samples
                )
        
        # 全体の精度（重み付け平均）
        base_mae = info["base_model"]["mae_10min"]
        adaptive_mae = info["adaptive_model"]["mae_10min"]
        
        if base_mae is not None and adaptive_mae is not None:
            info["combined_mae_10min"] = round(
                base_mae * (1 - self.adaptive_weight) + 
                adaptive_mae * self.adaptive_weight, 
                3
            )
        
        return info
    
    def print_status(self):
        """モデルの状態を表示"""
        info = self.get_model_info()
        
        print("\n=== デュアルモデル状態 ===")
        print(f"モデルタイプ: {info['model_type']}")
        print(f"適応モデル重み: {info['adaptive_weight']:.0%}")
        
        print("\n基本モデル:")
        print(f"  読み込み: {'✓' if info['base_model']['loaded'] else '✗'}")
        print(f"  学習サンプル数: {info['base_model']['samples']:,}")
        if info['base_model']['mae_10min'] is not None:
            print(f"  MAE (10分): {info['base_model']['mae_10min']:.3f}")
            
        print("\n適応モデル:")
        print(f"  読み込み: {'✓' if info['adaptive_model']['loaded'] else '✗'}")
        print(f"  学習サンプル数: {info['adaptive_model']['samples']:,}")
        print(f"  追加学習数: {info['adaptive_model']['additional_samples']:,}")
        if info['adaptive_model']['mae_10min'] is not None:
            print(f"  MAE (10分): {info['adaptive_model']['mae_10min']:.3f}")
            
        if 'combined_mae_10min' in info:
            print(f"\n統合MAE (10分): {info['combined_mae_10min']:.3f}")


# テスト用
if __name__ == "__main__":
    print("デュアルモデル予測器のテスト")
    print("-" * 50)
    
    predictor = RiverDualModelPredictor()
    
    # モデルの読み込み
    if predictor.load_models():
        # 状態表示
        predictor.print_status()
        
        # テストデータで予測
        test_data = {
            "data_time": datetime.now().isoformat(),
            "river": {"water_level": 2.75},
            "dam": {
                "outflow": 10.0,
                "inflow": 12.0,
                "storage_rate": 75.0
            },
            "rainfall": {
                "rainfall_10min": 0,
                "rainfall_1h": 0
            }
        }
        
        print("\n予測テスト:")
        predictions = predictor.predict_one(test_data)
        if predictions:
            print(f"予測数: {len(predictions)}")
            print("\n最初の3つの予測:")
            for i, pred in enumerate(predictions[:3]):
                print(f"  {i+1}. {pred['datetime']}: {pred['level']:.2f}m")
                print(f"     基本: {pred['base_level']:.2f}m, 適応: {pred['adaptive_level']:.2f}m")
                print(f"     重み: 基本{100-pred['adaptive_weight']*100:.0f}% + 適応{pred['adaptive_weight']*100:.0f}%")