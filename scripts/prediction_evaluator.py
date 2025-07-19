"""
AI予測精度評価モジュール
各予測モデルの精度を評価し、統計情報を提供
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
from collections import defaultdict

class PredictionEvaluator:
    """予測精度評価クラス"""
    
    def __init__(self):
        """初期化"""
        self.evaluation_results = {
            'expert_rule': defaultdict(list),
            'river_online': defaultdict(list)
        }
        self.evaluation_history = []
        
    def evaluate_prediction(self, predictions: List[Dict], actual_data: List[Dict], 
                          model_type: str) -> Dict:
        """
        予測精度を評価
        
        Args:
            predictions: 予測データのリスト
            actual_data: 実際のデータのリスト
            model_type: モデルタイプ ('expert_rule' or 'river_online')
            
        Returns:
            evaluation_metrics: 評価指標の辞書
        """
        if not predictions or not actual_data:
            return None
            
        # 予測と実測値を時刻でマッチング
        matched_data = self._match_prediction_actual(predictions, actual_data)
        
        if not matched_data:
            return None
            
        # 各種メトリクスを計算
        metrics = {}
        
        # 予測ステップごとの評価
        step_metrics = defaultdict(lambda: {'errors': [], 'count': 0})
        
        for match in matched_data:
            pred_value = match['predicted']
            actual_value = match['actual']
            minutes_ahead = match['minutes_ahead']
            
            error = actual_value - pred_value
            abs_error = abs(error)
            squared_error = error ** 2
            percentage_error = abs(error / actual_value) * 100 if actual_value != 0 else 0
            
            # ステップごとに集計
            step_key = f"{minutes_ahead}min"
            step_metrics[step_key]['errors'].append({
                'absolute': abs_error,
                'squared': squared_error,
                'percentage': percentage_error,
                'raw': error
            })
            step_metrics[step_key]['count'] += 1
            
        # 全体メトリクスの計算
        all_errors = []
        for step_data in step_metrics.values():
            all_errors.extend(step_data['errors'])
            
        if all_errors:
            metrics['overall'] = {
                'mae': np.mean([e['absolute'] for e in all_errors]),
                'rmse': np.sqrt(np.mean([e['squared'] for e in all_errors])),
                'mape': np.mean([e['percentage'] for e in all_errors]),
                'bias': np.mean([e['raw'] for e in all_errors]),
                'std': np.std([e['raw'] for e in all_errors]),
                'count': len(all_errors)
            }
            
            # ステップごとのメトリクス
            metrics['by_step'] = {}
            for step, data in step_metrics.items():
                if data['errors']:
                    metrics['by_step'][step] = {
                        'mae': np.mean([e['absolute'] for e in data['errors']]),
                        'rmse': np.sqrt(np.mean([e['squared'] for e in data['errors']])),
                        'mape': np.mean([e['percentage'] for e in data['errors']]),
                        'count': data['count']
                    }
                    
        # 精度スコア（0-100）の計算
        if 'overall' in metrics:
            # MAPEベースのスコア（誤差率が低いほど高スコア）
            mape = metrics['overall']['mape']
            score = max(0, 100 - mape)  # 100%誤差で0点
            metrics['overall']['score'] = score
            
        # 評価時刻を記録
        metrics['evaluated_at'] = datetime.now().isoformat()
        metrics['model_type'] = model_type
        
        # 履歴に保存
        self._save_evaluation(metrics, model_type)
        
        return metrics
        
    def _match_prediction_actual(self, predictions: List[Dict], 
                               actual_data: List[Dict]) -> List[Dict]:
        """
        予測値と実測値を時刻でマッチング
        
        Returns:
            matched_data: マッチングされたデータのリスト
        """
        matched = []
        
        # 実測データを時刻でインデックス化
        actual_by_time = {}
        for data in actual_data:
            if 'river' in data and data['river'].get('water_level') is not None:
                timestamp = data.get('data_time', data.get('timestamp'))
                if timestamp:
                    actual_by_time[timestamp] = data['river']['water_level']
                    
        # 予測データとマッチング
        base_time = None
        for i, pred in enumerate(predictions):
            pred_time = pred.get('datetime')
            if not pred_time:
                continue
                
            # 基準時刻の設定（最初の予測の10分前）
            if base_time is None and i == 0:
                base_dt = datetime.fromisoformat(pred_time)
                base_time = (base_dt - timedelta(minutes=10)).isoformat()
                
            # 実測値を探す
            if pred_time in actual_by_time:
                matched.append({
                    'predicted': pred['level'],
                    'actual': actual_by_time[pred_time],
                    'datetime': pred_time,
                    'minutes_ahead': (i + 1) * 10,
                    'confidence': pred.get('confidence', 1.0)
                })
                
        return matched
        
    def _save_evaluation(self, metrics: Dict, model_type: str):
        """評価結果を保存"""
        # モデルタイプ別に保存
        if model_type in self.evaluation_results:
            self.evaluation_results[model_type]['history'].append(metrics)
            
            # 最新の評価を更新
            self.evaluation_results[model_type]['latest'] = metrics
            
            # 統計情報を更新
            self._update_statistics(model_type)
            
    def _update_statistics(self, model_type: str):
        """統計情報を更新"""
        history = self.evaluation_results[model_type].get('history', [])
        
        if not history:
            return
            
        # 最近のN件の平均を計算
        recent_n = min(10, len(history))
        recent_history = history[-recent_n:]
        
        # 全体メトリクスの平均
        mae_values = [h['overall']['mae'] for h in recent_history if 'overall' in h]
        rmse_values = [h['overall']['rmse'] for h in recent_history if 'overall' in h]
        score_values = [h['overall']['score'] for h in recent_history if 'overall' in h]
        
        if mae_values:
            self.evaluation_results[model_type]['average'] = {
                'mae': np.mean(mae_values),
                'rmse': np.mean(rmse_values),
                'score': np.mean(score_values),
                'sample_count': recent_n
            }
            
    def get_comparison_summary(self) -> Dict:
        """
        モデル間の比較サマリーを取得
        """
        summary = {}
        
        for model_type in ['expert_rule', 'river_online']:
            if model_type in self.evaluation_results:
                latest = self.evaluation_results[model_type].get('latest')
                average = self.evaluation_results[model_type].get('average')
                
                summary[model_type] = {
                    'latest': latest,
                    'average': average,
                    'evaluation_count': len(self.evaluation_results[model_type].get('history', []))
                }
                
        return summary
        
    def get_detailed_comparison(self) -> pd.DataFrame:
        """
        詳細な比較データフレームを取得
        """
        comparison_data = []
        
        for model_type, data in self.evaluation_results.items():
            latest = data.get('latest')
            if latest and 'by_step' in latest:
                for step, metrics in latest['by_step'].items():
                    comparison_data.append({
                        'モデル': 'エキスパートルール' if model_type == 'expert_rule' else 'Riverオンライン学習',
                        '予測時間': step,
                        'MAE (m)': round(metrics['mae'], 3),
                        'RMSE (m)': round(metrics['rmse'], 3),
                        'MAPE (%)': round(metrics['mape'], 1),
                        'サンプル数': metrics['count']
                    })
                    
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            # 予測時間でソート
            df['sort_key'] = df['予測時間'].str.extract('(\d+)').astype(int)
            df = df.sort_values(['モデル', 'sort_key']).drop('sort_key', axis=1)
            return df
        
        return pd.DataFrame()
        
    def save_to_file(self, filepath: str):
        """評価結果をファイルに保存"""
        save_data = {
            'evaluation_results': self.evaluation_results,
            'saved_at': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
    def load_from_file(self, filepath: str):
        """評価結果をファイルから読み込み"""
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.evaluation_results = data.get('evaluation_results', {})
                
    def get_performance_trend(self, model_type: str, metric: str = 'mae') -> pd.DataFrame:
        """
        性能トレンドデータを取得
        
        Args:
            model_type: モデルタイプ
            metric: 評価指標 ('mae', 'rmse', 'score')
            
        Returns:
            trend_df: トレンドデータフレーム
        """
        if model_type not in self.evaluation_results:
            return pd.DataFrame()
            
        history = self.evaluation_results[model_type].get('history', [])
        
        trend_data = []
        for eval_data in history:
            if 'overall' in eval_data and metric in eval_data['overall']:
                trend_data.append({
                    'datetime': datetime.fromisoformat(eval_data['evaluated_at']),
                    metric: eval_data['overall'][metric],
                    'count': eval_data['overall']['count']
                })
                
        if trend_data:
            return pd.DataFrame(trend_data).sort_values('datetime')
            
        return pd.DataFrame()