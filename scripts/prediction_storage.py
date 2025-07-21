"""
予測結果の保存・読み込みモジュール
ストリーム学習のための予測結果管理
"""

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import os


class PredictionStorage:
    """予測結果の保存・管理クラス"""
    
    def __init__(self, storage_dir: str = "predictions"):
        # スクリプトディレクトリの親ディレクトリを基準にする
        base_dir = Path(__file__).parent.parent
        self.storage_dir = base_dir / storage_dir
        self.storage_dir.mkdir(exist_ok=True)
        self.jst_tz = timezone(timedelta(hours=9))
        
    def save_predictions(self, base_time: str, features: Dict, predictions: List[Dict]) -> bool:
        """
        予測結果を保存
        
        Args:
            base_time: 予測基準時刻（ISO形式）
            features: 予測に使用した特徴量
            predictions: 予測結果リスト
            
        Returns:
            保存成功の可否
        """
        try:
            # 日付別ディレクトリ作成
            base_dt = datetime.fromisoformat(base_time)
            date_dir = self.storage_dir / base_dt.strftime("%Y%m%d")
            date_dir.mkdir(exist_ok=True)
            
            # ファイル名は時刻を含める
            filename = f"pred_{base_dt.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = date_dir / filename
            
            # 予測データの構造
            prediction_data = {
                "base_time": base_time,
                "features": features,
                "predictions": predictions,
                "created_at": datetime.now(self.jst_tz).isoformat()
            }
            
            # 保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(prediction_data, f, ensure_ascii=False, indent=2)
                
            return True
            
        except Exception as e:
            print(f"予測結果の保存エラー: {e}")
            return False
    
    def get_predictions_for_learning(self, target_time: str, tolerance_minutes: int = 5) -> List[Tuple[Dict, Dict]]:
        """
        指定時刻に対する予測結果を取得（学習用）
        
        Args:
            target_time: 対象時刻（ISO形式）
            tolerance_minutes: 時刻の許容誤差（分）
            
        Returns:
            [(予測データ, 該当する予測結果), ...] のリスト
        """
        results = []
        target_dt = datetime.fromisoformat(target_time)
        
        try:
            # 10分前から3時間前までの予測ファイルを探索（10分前のデータが現在を予測している）
            for minutes_ago in range(10, 181, 10):  # 10分〜180分前を10分刻みで
                check_dt = target_dt - timedelta(minutes=minutes_ago)
                date_dir = self.storage_dir / check_dt.strftime("%Y%m%d")
                
                if not date_dir.exists():
                    continue
                    
                # その時刻付近の予測ファイルをチェック
                # ファイル名: pred_YYYYMMDD_HHMMSS.json (SSは00固定)
                for minute_offset in range(-5, 6):  # ±5分の範囲で探す
                    file_dt = check_dt + timedelta(minutes=minute_offset)
                    filename = f"pred_{file_dt.strftime('%Y%m%d_%H%M00')}.json"
                    pred_file = date_dir / filename
                    
                    if not pred_file.exists():
                        continue
                        
                    try:
                        with open(pred_file, 'r', encoding='utf-8') as f:
                            pred_data = json.load(f)
                            
                        # 各予測結果をチェック
                        for prediction in pred_data['predictions']:
                            pred_time = datetime.fromisoformat(prediction['datetime'])
                            
                            # 時刻が許容範囲内か確認
                            time_diff = abs((pred_time - target_dt).total_seconds() / 60)
                            if time_diff <= tolerance_minutes:
                                results.append((pred_data, prediction))
                                
                    except Exception as e:
                        print(f"予測ファイル読み込みエラー {pred_file}: {e}")
                        continue
                        
        except Exception as e:
            print(f"予測結果取得エラー: {e}")
            
        return results
    
    def cleanup_old_predictions(self, days_to_keep: int = 7):
        """
        古い予測結果を削除
        
        Args:
            days_to_keep: 保持する日数
        """
        try:
            cutoff_date = datetime.now(self.jst_tz) - timedelta(days=days_to_keep)
            
            for date_dir in self.storage_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                    
                # ディレクトリ名から日付を取得
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y%m%d")
                    dir_date = dir_date.replace(tzinfo=self.jst_tz)
                    
                    if dir_date < cutoff_date:
                        # 古いディレクトリを削除
                        for file in date_dir.iterdir():
                            file.unlink()
                        date_dir.rmdir()
                        print(f"古い予測結果を削除: {date_dir.name}")
                        
                except ValueError:
                    # 日付形式でないディレクトリは無視
                    continue
                    
        except Exception as e:
            print(f"クリーンアップエラー: {e}")
    
    def get_recent_predictions_count(self) -> Dict[str, int]:
        """直近の予測数を取得"""
        counts = {
            "last_hour": 0,
            "last_24h": 0,
            "total": 0
        }
        
        now = datetime.now(self.jst_tz)
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        try:
            for date_dir in self.storage_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                    
                for pred_file in date_dir.glob("pred_*.json"):
                    counts["total"] += 1
                    
                    # ファイル名から時刻を取得
                    try:
                        time_str = pred_file.stem.replace("pred_", "")
                        file_dt = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                        file_dt = file_dt.replace(tzinfo=self.jst_tz)
                        
                        if file_dt >= one_hour_ago:
                            counts["last_hour"] += 1
                        if file_dt >= one_day_ago:
                            counts["last_24h"] += 1
                            
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"予測数カウントエラー: {e}")
            
        return counts