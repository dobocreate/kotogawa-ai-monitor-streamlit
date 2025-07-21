"""
River MLベースのモデル評価モジュール
ストリーミング環境での継続的な評価を実現
"""

from river import metrics, utils, stats
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict


class StreamingModelEvaluator:
    """ストリーミング予測モデルの評価クラス"""
    
    def __init__(self):
        # 基本メトリクス
        self.mae = metrics.MAE()
        self.rmse = metrics.RMSE()
        self.mse = metrics.MSE()
        self.r2 = metrics.R2()
        self.mape = metrics.MAPE()
        
        # 時間別メトリクス（10分〜180分）
        self.metrics_by_step = {}
        for minutes in range(10, 181, 10):
            self.metrics_by_step[f"{minutes}min"] = {
                "mae": metrics.MAE(),
                "rmse": metrics.RMSE(),
                "mse": metrics.MSE(),
                "count": 0
            }
        
        # ローリングウィンドウメトリクス
        self.rolling_mae = utils.Rolling(metrics.MAE(), window_size=100)
        self.rolling_rmse = utils.Rolling(metrics.RMSE(), window_size=100)
        self.rolling_r2 = utils.Rolling(metrics.R2(), window_size=100)
        
        # 時間帯別メトリクス
        self.hourly_metrics = defaultdict(lambda: {
            "mae": metrics.MAE(),
            "count": 0
        })
        
        # エラー分布統計
        self.error_stats = stats.Var()  # 誤差の分散
        self.error_quantiles = stats.Quantile([0.25, 0.5, 0.75, 0.95])
        
        # パフォーマンス履歴
        self.performance_history = []
        self.jst_tz = timezone(timedelta(hours=9))
        
    def update(self, y_true: float, y_pred: float, 
               prediction_time: str = None, step_minutes: int = None):
        """
        メトリクスを更新
        
        Args:
            y_true: 実測値
            y_pred: 予測値
            prediction_time: 予測時刻（ISO形式）
            step_minutes: 予測ステップ（分）
        """
        # 基本メトリクスの更新
        self.mae.update(y_true, y_pred)
        self.rmse.update(y_true, y_pred)
        self.mse.update(y_true, y_pred)
        self.r2.update(y_true, y_pred)
        
        # MAPEは0でない実測値のみ
        if y_true != 0:
            self.mape.update(y_true, y_pred)
        
        # ローリングメトリクスの更新
        self.rolling_mae.update(y_true, y_pred)
        self.rolling_rmse.update(y_true, y_pred)
        self.rolling_r2.update(y_true, y_pred)
        
        # エラー統計の更新
        error = y_pred - y_true
        self.error_stats.update(error)
        self.error_quantiles.update(abs(error))
        
        # ステップ別メトリクスの更新
        if step_minutes:
            step_key = f"{step_minutes}min"
            if step_key in self.metrics_by_step:
                self.metrics_by_step[step_key]["mae"].update(y_true, y_pred)
                self.metrics_by_step[step_key]["rmse"].update(y_true, y_pred)
                self.metrics_by_step[step_key]["mse"].update(y_true, y_pred)
                self.metrics_by_step[step_key]["count"] += 1
        
        # 時間帯別メトリクスの更新
        if prediction_time:
            try:
                dt = datetime.fromisoformat(prediction_time)
                hour = dt.hour
                self.hourly_metrics[hour]["mae"].update(y_true, y_pred)
                self.hourly_metrics[hour]["count"] += 1
            except:
                pass
    
    def get_current_metrics(self) -> Dict:
        """現在のメトリクスを取得"""
        return {
            "overall": {
                "mae": self.mae.get(),
                "rmse": self.rmse.get(),
                "mse": self.mse.get(),
                "r2": self.r2.get(),
                "mape": self.mape.get() if hasattr(self.mape, 'n') and self.mape.n > 0 else None,
                "sample_count": self.mae.n if hasattr(self.mae, 'n') else 0
            },
            "rolling": {
                "mae": self.rolling_mae.get() if hasattr(self.rolling_mae, 'get') else None,
                "rmse": self.rolling_rmse.get() if hasattr(self.rolling_rmse, 'get') else None,
                "r2": self.rolling_r2.get() if hasattr(self.rolling_r2, 'get') else None,
                "window_size": 100
            },
            "error_distribution": {
                "mean": self.error_stats.get()['mean'] if self.error_stats.n > 0 else None,
                "std": self.error_stats.get()['variance'] ** 0.5 if self.error_stats.n > 0 else None,
                "quantiles": {
                    "q25": self.error_quantiles.get()[0.25] if self.error_quantiles.n > 0 else None,
                    "q50": self.error_quantiles.get()[0.5] if self.error_quantiles.n > 0 else None,
                    "q75": self.error_quantiles.get()[0.75] if self.error_quantiles.n > 0 else None,
                    "q95": self.error_quantiles.get()[0.95] if self.error_quantiles.n > 0 else None
                }
            }
        }
    
    def get_step_metrics(self) -> Dict:
        """ステップ別メトリクスを取得"""
        result = {}
        for step, metrics in self.metrics_by_step.items():
            if metrics["count"] > 0:
                result[step] = {
                    "mae": metrics["mae"].get(),
                    "rmse": metrics["rmse"].get(),
                    "mse": metrics["mse"].get(),
                    "count": metrics["count"]
                }
        return result
    
    def get_hourly_metrics(self) -> Dict:
        """時間帯別メトリクスを取得"""
        result = {}
        for hour, metrics in self.hourly_metrics.items():
            if metrics["count"] > 0:
                result[f"{hour:02d}:00"] = {
                    "mae": metrics["mae"].get(),
                    "count": metrics["count"]
                }
        return result
    
    def progressive_validate(self, predictions: List[Tuple[str, float]], 
                           actuals: List[Tuple[str, float]], 
                           delay_minutes: int = 0) -> Dict:
        """
        プログレッシブ検証を実行
        
        Args:
            predictions: [(timestamp, value), ...] の予測リスト
            actuals: [(timestamp, value), ...] の実測値リスト
            delay_minutes: 遅延時間（分）
            
        Returns:
            検証結果の辞書
        """
        # タイムスタンプでマッチング
        pred_dict = {p[0]: p[1] for p in predictions}
        results = []
        
        for actual_time, actual_value in actuals:
            # 遅延を考慮した予測時刻
            pred_time = (datetime.fromisoformat(actual_time) - 
                        timedelta(minutes=delay_minutes)).isoformat()
            
            if pred_time in pred_dict:
                pred_value = pred_dict[pred_time]
                self.update(actual_value, pred_value, actual_time, delay_minutes)
                
                results.append({
                    "time": actual_time,
                    "actual": actual_value,
                    "predicted": pred_value,
                    "error": abs(actual_value - pred_value),
                    "delay_minutes": delay_minutes
                })
        
        return {
            "validated_count": len(results),
            "metrics": self.get_current_metrics(),
            "details": results
        }
    
    def save_evaluation_report(self, filepath: str):
        """評価レポートを保存"""
        report = {
            "timestamp": datetime.now(self.jst_tz).isoformat(),
            "overall_metrics": self.get_current_metrics(),
            "step_metrics": self.get_step_metrics(),
            "hourly_metrics": self.get_hourly_metrics(),
            "summary": {
                "best_step": self._find_best_step(),
                "worst_step": self._find_worst_step(),
                "best_hour": self._find_best_hour(),
                "performance_trend": self._analyze_trend()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def _find_best_step(self) -> Optional[str]:
        """最も精度の良い予測ステップを見つける"""
        step_metrics = self.get_step_metrics()
        if not step_metrics:
            return None
        
        return min(step_metrics.items(), 
                  key=lambda x: x[1]["mae"])[0]
    
    def _find_worst_step(self) -> Optional[str]:
        """最も精度の悪い予測ステップを見つける"""
        step_metrics = self.get_step_metrics()
        if not step_metrics:
            return None
        
        return max(step_metrics.items(), 
                  key=lambda x: x[1]["mae"])[0]
    
    def _find_best_hour(self) -> Optional[str]:
        """最も精度の良い時間帯を見つける"""
        hourly_metrics = self.get_hourly_metrics()
        if not hourly_metrics:
            return None
        
        return min(hourly_metrics.items(), 
                  key=lambda x: x[1]["mae"])[0]
    
    def _analyze_trend(self) -> str:
        """パフォーマンスのトレンドを分析"""
        if self.rolling_mae.n < 20:
            return "データ不足"
        
        # 簡単なトレンド分析（実装は簡略化）
        current_mae = self.rolling_mae.get()
        overall_mae = self.mae.get()
        
        if current_mae < overall_mae * 0.9:
            return "改善傾向"
        elif current_mae > overall_mae * 1.1:
            return "悪化傾向"
        else:
            return "安定"


class ModelComparator:
    """複数モデルの比較評価クラス"""
    
    def __init__(self):
        self.models = {}
        
    def add_model(self, model_name: str, evaluator: StreamingModelEvaluator):
        """モデルを追加"""
        self.models[model_name] = evaluator
    
    def compare_models(self) -> pd.DataFrame:
        """モデルを比較"""
        comparison_data = []
        
        for name, evaluator in self.models.items():
            metrics = evaluator.get_current_metrics()
            comparison_data.append({
                "model": name,
                "mae": metrics["overall"]["mae"],
                "rmse": metrics["overall"]["rmse"],
                "r2": metrics["overall"]["r2"],
                "samples": metrics["overall"]["sample_count"]
            })
        
        return pd.DataFrame(comparison_data).sort_values("mae")
    
    def plot_comparison(self, metric: str = "mae"):
        """モデル比較をプロット（実装は省略）"""
        pass