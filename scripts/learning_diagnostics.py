"""
学習プロセスの診断機能
各ステップの成功/失敗を詳細に記録し、問題の特定を支援
"""

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import traceback
from enum import Enum

class StepStatus(Enum):
    """ステップの実行状態"""
    PENDING = "⏸️ 未実行"
    RUNNING = "⏳ 実行中"
    SUCCESS = "✅ 成功"
    FAILED = "❌ 失敗"
    SKIPPED = "⏭️ スキップ"
    WARNING = "⚠️ 警告"

class LearningDiagnostics:
    """学習プロセスの診断とトラッキング"""
    
    def __init__(self):
        self.steps = {}
        self.start_time = None
        self.end_time = None
        
    def start_diagnostics(self):
        """診断の開始"""
        # JSTで記録
        jst_offset = timedelta(hours=9)
        jst_tz = timezone(jst_offset)
        self.start_time = datetime.now(jst_tz)
        # ステップを初期化（status_textも含める）
        def create_step(name: str, status: StepStatus = StepStatus.PENDING):
            return {
                "name": name,
                "status": status,
                "status_text": status.value,
                "details": {}
            }
        
        self.steps = {
            # 1. データ取得フェーズ
            "1.1_file_check": create_step("最新データファイルの確認"),
            "1.2_file_read": create_step("データファイルの読み込み"),
            "1.3_format_validation": create_step("データ形式の検証"),
            "1.4_timestamp_check": create_step("タイムスタンプの妥当性確認"),
            
            # 2. データ前処理フェーズ
            "2.1_missing_values": create_step("欠損値の検出と処理"),
            "2.2_outlier_detection": create_step("異常値の検出"),
            "2.3_continuity_check": create_step("時系列の連続性確認"),
            "2.4_feature_extraction": create_step("特徴量の抽出"),
            
            # 3. 将来データの確認フェーズ
            "3.1_future_data_check": create_step("将来データの存在確認"),
            "3.2_future_completeness": create_step("将来データの完全性チェック"),
            "3.3_target_data_check": create_step("予測対象時刻のデータ確認"),
            
            # 4. モデル初期化フェーズ
            "4.1_model_load": create_step("既存モデルの読み込み"),
            "4.2_model_integrity": create_step("モデルの整合性確認"),
            "4.3_river_import": create_step("Riverモジュールのインポート"),
            "4.4_pipeline_build": create_step("パイプラインの構築"),
            
            # 5. 予測実行フェーズ
            "5.1_prediction_run": create_step("現在データでの予測実行"),
            "5.2_step_predictions": create_step("各ステップの予測値生成"),
            "5.3_prediction_validation": create_step("予測値の妥当性確認"),
            "5.4_confidence_calc": create_step("信頼度スコアの計算"),
            
            # 6. 学習実行フェーズ
            "6.1_online_learning": create_step("オンライン学習の実行"),
            "6.2_step_learning": create_step("各時間ステップでの学習"),
            "6.3_drift_detection": create_step("ドリフト検出の実行"),
            "6.4_error_handling": create_step("学習エラーのハンドリング"),
            
            # 7. 評価メトリクス更新フェーズ
            "7.1_mae_update": create_step("MAEの計算と更新"),
            "7.2_rmse_update": create_step("RMSEの計算と更新"),
            "7.3_step_metrics": create_step("ステップ別精度の記録"),
            "7.4_rolling_stats": create_step("ローリング統計の更新"),
            
            # 8. モデル保存フェーズ
            "8.1_model_serialize": create_step("モデルのシリアライズ"),
            "8.2_save_directory": create_step("保存先ディレクトリの確認"),
            "8.3_file_write": create_step("ファイル書き込みの実行"),
            "8.4_save_verify": create_step("保存成功の確認"),
            
            # 9. 学習履歴記録フェーズ
            "9.1_time_record": create_step("学習実行時刻の記録"),
            "9.2_data_count": create_step("処理データ数の記録"),
            "9.3_error_record": create_step("エラー発生箇所の記録"),
            "9.4_status_record": create_step("成功/失敗ステータスの記録"),
            
            # 10. 診断情報生成フェーズ
            "10.1_performance_metrics": create_step("モデル性能指標の生成"),
            "10.2_drift_history": create_step("ドリフト検出履歴の生成"),
            "10.3_data_quality": create_step("データ品質統計の生成"),
            "10.4_next_schedule": create_step("次回学習推奨時刻の計算"),
        }
    
    def update_step(self, step_id: str, status: StepStatus, details: Dict = None, error: Exception = None):
        """ステップの状態を更新"""
        if step_id in self.steps:
            self.steps[step_id]["status"] = status
            self.steps[step_id]["status_text"] = status.value  # JSON用にテキストも保存
            # JSTで記録
            jst_offset = timedelta(hours=9)
            jst_tz = timezone(jst_offset)
            self.steps[step_id]["timestamp"] = datetime.now(jst_tz).isoformat()
            
            if details:
                self.steps[step_id]["details"].update(details)
            
            if error:
                self.steps[step_id]["error"] = {
                    "message": str(error),
                    "type": type(error).__name__,
                    "traceback": traceback.format_exc()
                }
    
    def complete_diagnostics(self):
        """診断の完了"""
        # JSTで記録
        jst_offset = timedelta(hours=9)
        jst_tz = timezone(jst_offset)
        self.end_time = datetime.now(jst_tz)
        
    def get_summary(self) -> Dict:
        """診断結果のサマリーを取得"""
        if not self.start_time:
            return {"status": "未実行", "message": "診断がまだ開始されていません"}
        
        # ステータス別の集計
        status_counts = {
            StepStatus.SUCCESS: 0,
            StepStatus.FAILED: 0,
            StepStatus.WARNING: 0,
            StepStatus.SKIPPED: 0,
            StepStatus.PENDING: 0,
            StepStatus.RUNNING: 0
        }
        
        failed_steps = []
        warning_steps = []
        
        for step_id, step_info in self.steps.items():
            status = step_info["status"]
            status_counts[status] += 1
            
            if status == StepStatus.FAILED:
                failed_steps.append({
                    "id": step_id,
                    "name": step_info["name"],
                    "error": step_info.get("error", {}).get("message", "詳細なし")
                })
            elif status == StepStatus.WARNING:
                warning_steps.append({
                    "id": step_id,
                    "name": step_info["name"],
                    "details": step_info.get("details", {})
                })
        
        # 全体のステータス判定
        if status_counts[StepStatus.FAILED] > 0:
            overall_status = "失敗"
        elif status_counts[StepStatus.WARNING] > 0:
            overall_status = "警告あり"
        elif status_counts[StepStatus.PENDING] > 0 or status_counts[StepStatus.RUNNING] > 0:
            overall_status = "実行中"
        else:
            overall_status = "成功"
        
        duration = None
        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        # タイムゾーン情報を確実に含めるため、明示的にフォーマット
        start_time_str = None
        end_time_str = None
        
        if self.start_time:
            # Python 3.8以前でもタイムゾーン情報を含めるようにする
            if self.start_time.tzinfo:
                start_time_str = self.start_time.isoformat()
            else:
                # タイムゾーンがない場合はJSTを付加
                jst_offset = timedelta(hours=9)
                jst_tz = timezone(jst_offset)
                start_time_str = self.start_time.replace(tzinfo=jst_tz).isoformat()
        
        if self.end_time:
            if self.end_time.tzinfo:
                end_time_str = self.end_time.isoformat()
            else:
                jst_offset = timedelta(hours=9)
                jst_tz = timezone(jst_offset)
                end_time_str = self.end_time.replace(tzinfo=jst_tz).isoformat()
        
        return {
            "overall_status": overall_status,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "duration_seconds": duration,
            "status_counts": {k.value: v for k, v in status_counts.items()},
            "failed_steps": failed_steps,
            "warning_steps": warning_steps,
            "total_steps": len(self.steps)
        }
    
    def get_detailed_results(self) -> Dict:
        """詳細な診断結果を取得"""
        phases = {
            "1_data_acquisition": {
                "name": "データ取得フェーズ",
                "steps": ["1.1_file_check", "1.2_file_read", "1.3_format_validation", "1.4_timestamp_check"]
            },
            "2_preprocessing": {
                "name": "データ前処理フェーズ",
                "steps": ["2.1_missing_values", "2.2_outlier_detection", "2.3_continuity_check", "2.4_feature_extraction"]
            },
            "3_future_data": {
                "name": "将来データ確認フェーズ",
                "steps": ["3.1_future_data_check", "3.2_future_completeness", "3.3_target_data_check"]
            },
            "4_model_init": {
                "name": "モデル初期化フェーズ",
                "steps": ["4.1_model_load", "4.2_model_integrity", "4.3_river_import", "4.4_pipeline_build"]
            },
            "5_prediction": {
                "name": "予測実行フェーズ",
                "steps": ["5.1_prediction_run", "5.2_step_predictions", "5.3_prediction_validation", "5.4_confidence_calc"]
            },
            "6_learning": {
                "name": "学習実行フェーズ",
                "steps": ["6.1_online_learning", "6.2_step_learning", "6.3_drift_detection", "6.4_error_handling"]
            },
            "7_metrics": {
                "name": "評価メトリクス更新フェーズ",
                "steps": ["7.1_mae_update", "7.2_rmse_update", "7.3_step_metrics", "7.4_rolling_stats"]
            },
            "8_save": {
                "name": "モデル保存フェーズ",
                "steps": ["8.1_model_serialize", "8.2_save_directory", "8.3_file_write", "8.4_save_verify"]
            },
            "9_history": {
                "name": "学習履歴記録フェーズ",
                "steps": ["9.1_time_record", "9.2_data_count", "9.3_error_record", "9.4_status_record"]
            },
            "10_diagnostics": {
                "name": "診断情報生成フェーズ",
                "steps": ["10.1_performance_metrics", "10.2_drift_history", "10.3_data_quality", "10.4_next_schedule"]
            }
        }
        
        results = {
            "summary": self.get_summary(),
            "phases": {}
        }
        
        for phase_id, phase_info in phases.items():
            phase_steps = []
            phase_status = StepStatus.SUCCESS
            
            for step_id in phase_info["steps"]:
                if step_id in self.steps:
                    step_data = self.steps[step_id].copy()
                    step_data["id"] = step_id
                    # status_textが既に設定されていない場合のみ追加
                    if "status_text" not in step_data and isinstance(step_data.get("status"), StepStatus):
                        step_data["status_text"] = step_data["status"].value
                    # StepStatusオブジェクトを削除（JSON化できないため）
                    if "status" in step_data and isinstance(step_data["status"], StepStatus):
                        del step_data["status"]
                    phase_steps.append(step_data)
                    
                    # フェーズ全体のステータスを判定（元のstepデータから）
                    original_status = self.steps[step_id]["status"]
                    if original_status == StepStatus.FAILED:
                        phase_status = StepStatus.FAILED
                    elif original_status == StepStatus.WARNING and phase_status != StepStatus.FAILED:
                        phase_status = StepStatus.WARNING
                    elif original_status in [StepStatus.PENDING, StepStatus.RUNNING] and phase_status not in [StepStatus.FAILED, StepStatus.WARNING]:
                        phase_status = StepStatus.RUNNING
            
            results["phases"][phase_id] = {
                "name": phase_info["name"],
                "status": phase_status.value,
                "steps": phase_steps
            }
        
        return results
    
    def save_results(self, filepath: Path):
        """診断結果をファイルに保存"""
        results = self.get_detailed_results()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_results(filepath: Path) -> Optional[Dict]:
        """保存された診断結果を読み込み"""
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


def check_data_availability(diagnostics: LearningDiagnostics) -> Tuple[bool, Optional[Dict]]:
    """データの利用可能性をチェック"""
    try:
        diagnostics.update_step("1.1_file_check", StepStatus.RUNNING)
        
        data_dir = Path('data')
        
        # まずlatest.jsonを確認
        latest_file = data_dir / 'latest.json'
        if latest_file.exists():
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data:
                    diagnostics.update_step("1.1_file_check", StepStatus.SUCCESS, 
                                          {"file_path": str(latest_file), 
                                           "file_size": latest_file.stat().st_size,
                                           "data_time": data.get('data_time', 'unknown')})
                    
                    diagnostics.update_step("1.2_file_read", StepStatus.SUCCESS, 
                                          {"data_source": "latest.json"})
                    
                    return True, data
            except json.JSONDecodeError as e:
                diagnostics.update_step("1.2_file_read", StepStatus.FAILED, error=e)
                return False, None
        
        # latest.jsonがない場合は従来の方法も試す
        # JSTで今日の日付を取得
        jst_offset = timedelta(hours=9)
        jst_tz = timezone(jst_offset)
        today = datetime.now(jst_tz).strftime('%Y%m%d')
        today_file = data_dir / f'{today}.json'
        
        if not today_file.exists():
            diagnostics.update_step("1.1_file_check", StepStatus.FAILED, 
                                  {"message": f"データファイルが見つかりません: {latest_file} および {today_file}"})
            return False, None
        
        diagnostics.update_step("1.1_file_check", StepStatus.SUCCESS, 
                              {"file_path": str(today_file), "file_size": today_file.stat().st_size})
        
        # ファイル読み込み
        diagnostics.update_step("1.2_file_read", StepStatus.RUNNING)
        try:
            with open(today_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                diagnostics.update_step("1.2_file_read", StepStatus.WARNING, 
                                      {"message": "データが空です"})
                return False, None
            
            diagnostics.update_step("1.2_file_read", StepStatus.SUCCESS, 
                                  {"data_count": len(data)})
            
            return True, data[-1]  # 最新データを返す
            
        except json.JSONDecodeError as e:
            diagnostics.update_step("1.2_file_read", StepStatus.FAILED, error=e)
            return False, None
            
    except Exception as e:
        diagnostics.update_step("1.1_file_check", StepStatus.FAILED, error=e)
        return False, None


def validate_data_format(data: Dict, diagnostics: LearningDiagnostics) -> bool:
    """データ形式の検証"""
    diagnostics.update_step("1.3_format_validation", StepStatus.RUNNING)
    
    required_fields = ['data_time']
    missing_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        diagnostics.update_step("1.3_format_validation", StepStatus.FAILED, 
                              {"missing_fields": missing_fields})
        return False
    
    # 詳細な検証
    validation_details = {
        "has_river_data": 'river' in data and data['river'] is not None,
        "has_dam_data": 'dam' in data and data['dam'] is not None,
        "has_rainfall_data": 'rainfall' in data and data['rainfall'] is not None,
        "has_water_level": 'river' in data and data['river'] and 'water_level' in data['river'],
        "has_outflow": 'dam' in data and data['dam'] and 'outflow' in data['dam']
    }
    
    if not validation_details["has_river_data"] or not validation_details["has_water_level"]:
        diagnostics.update_step("1.3_format_validation", StepStatus.WARNING, validation_details)
    else:
        diagnostics.update_step("1.3_format_validation", StepStatus.SUCCESS, validation_details)
    
    return True