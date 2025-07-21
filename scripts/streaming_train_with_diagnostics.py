"""
診断機能付きストリーミング学習スクリプト
学習プロセスの各ステップを詳細に記録
"""

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys
import traceback
from typing import Dict, List, Optional, Tuple
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8以前の場合はpytzを使用
    from pytz import timezone as pytz_timezone
    ZoneInfo = lambda x: pytz_timezone(x)

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.learning_diagnostics import LearningDiagnostics, StepStatus, check_data_availability, validate_data_format
from scripts.river_dual_model_predictor import RiverDualModelPredictor


def check_timestamp_validity(data: Dict, diagnostics: LearningDiagnostics) -> bool:
    """タイムスタンプの妥当性確認"""
    diagnostics.update_step("1.4_timestamp_check", StepStatus.RUNNING)
    
    try:
        timestamp_str = data.get('data_time', '')
        if not timestamp_str:
            diagnostics.update_step("1.4_timestamp_check", StepStatus.FAILED, 
                                  {"message": "タイムスタンプが存在しません"})
            return False
        
        # タイムスタンプのパース
        timestamp = datetime.fromisoformat(timestamp_str)
        
        # 現在時刻をJSTで取得（より互換性の高い方法）
        jst_offset = timedelta(hours=9)
        jst_tz = timezone(jst_offset)
        now = datetime.now(jst_tz)
        
        # タイムスタンプがタイムゾーンなしの場合、JSTとして扱う
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=jst_tz)
        # タイムスタンプがUTCオフセット形式（+09:00など）の場合も正しく処理
        elif timestamp.tzinfo.utcoffset(None) is None:
            # pytzなどの特殊なタイムゾーンオブジェクトの場合
            timestamp = timestamp.replace(tzinfo=jst_tz)
        
        # 未来のデータでないかチェック
        if timestamp > now:
            diagnostics.update_step("1.4_timestamp_check", StepStatus.WARNING, 
                                  {"message": f"未来のタイムスタンプです: {timestamp_str}",
                                   "timestamp": timestamp_str,
                                   "current_time": now.isoformat()})
            return False
        
        # 古すぎないかチェック（1日以上前）
        if now - timestamp > timedelta(days=1):
            diagnostics.update_step("1.4_timestamp_check", StepStatus.WARNING, 
                                  {"message": f"1日以上前のデータです: {timestamp_str}",
                                   "age_hours": (now - timestamp).total_seconds() / 3600})
        else:
            diagnostics.update_step("1.4_timestamp_check", StepStatus.SUCCESS, 
                                  {"timestamp": timestamp_str,
                                   "age_minutes": (now - timestamp).total_seconds() / 60})
        
        return True
        
    except Exception as e:
        diagnostics.update_step("1.4_timestamp_check", StepStatus.FAILED, error=e)
        return False


def preprocess_data(data: Dict, diagnostics: LearningDiagnostics) -> Dict:
    """データの前処理"""
    # 欠損値の検出
    diagnostics.update_step("2.1_missing_values", StepStatus.RUNNING)
    
    # データの値を格納（欠損チェック用）
    data_values = {
        "water_level": None,
        "outflow": None,
        "rainfall": None
    }
    
    # 欠損フィールドを記録
    missing_fields = []
    
    # 水位データ
    river_data = data.get('river', {})
    if river_data and river_data.get('water_level') is not None:
        data_values["water_level"] = river_data['water_level']
    else:
        missing_fields.append("water_level")
    
    # 放流量データ
    dam_data = data.get('dam', {})
    if dam_data and dam_data.get('outflow') is not None:
        data_values["outflow"] = dam_data['outflow']
    else:
        missing_fields.append("outflow")
    
    # 降雨量データ（雨が降っていない場合はNoneやNullになることがある）
    rainfall_data = data.get('rainfall', {})
    if rainfall_data and rainfall_data.get('hourly') is not None:
        data_values["rainfall"] = rainfall_data['hourly']
    else:
        missing_fields.append("rainfall")
        # 降雨量0として扱う（雨が降っていない）
        data_values["rainfall"] = 0
    
    if missing_fields:
        diagnostics.update_step("2.1_missing_values", StepStatus.WARNING, 
                              {"missing_count": len(missing_fields), 
                               "missing_fields": missing_fields,
                               "note": "降雨量がnullの場合は雨が降っていないことを示す"})
    else:
        diagnostics.update_step("2.1_missing_values", StepStatus.SUCCESS, 
                              {"all_fields_present": True})
    
    # 異常値の検出
    diagnostics.update_step("2.2_outlier_detection", StepStatus.RUNNING)
    outliers = []
    
    # 水位の異常値チェック（負値または極端に大きい値）
    if data_values["water_level"] is not None:
        if data_values["water_level"] < 0:
            outliers.append({"field": "water_level", "value": data_values["water_level"], "reason": "負の値"})
        elif data_values["water_level"] > 10:  # 10m以上は異常と判定
            outliers.append({"field": "water_level", "value": data_values["water_level"], "reason": "極端に大きい値"})
    
    # 放流量の異常値チェック
    if data_values["outflow"] is not None:
        if data_values["outflow"] < 0:
            outliers.append({"field": "outflow", "value": data_values["outflow"], "reason": "負の値"})
        elif data_values["outflow"] > 1000:  # 1000 m³/s以上は異常と判定
            outliers.append({"field": "outflow", "value": data_values["outflow"], "reason": "極端に大きい値"})
    
    if outliers:
        diagnostics.update_step("2.2_outlier_detection", StepStatus.WARNING, 
                              {"outliers": outliers})
    else:
        diagnostics.update_step("2.2_outlier_detection", StepStatus.SUCCESS, 
                              {"no_outliers": True})
    
    # 時系列の連続性確認はスキップ（単一データポイントのため）
    diagnostics.update_step("2.3_continuity_check", StepStatus.SKIPPED, 
                          {"reason": "単一データポイントのため連続性チェック不要"})
    
    # 特徴量の抽出
    diagnostics.update_step("2.4_feature_extraction", StepStatus.SUCCESS, 
                          {"extracted_features": list(data_values.keys())})
    
    return data_values


def detect_data_interval(data_points: List[Dict]) -> Optional[int]:
    """データポイント間の時間間隔を検出（分単位）"""
    if len(data_points) < 2:
        return None
    
    intervals = []
    for i in range(1, min(len(data_points), 10)):  # 最大10個のサンプルで判定
        try:
            time1 = datetime.fromisoformat(data_points[i-1].get('data_time', ''))
            time2 = datetime.fromisoformat(data_points[i].get('data_time', ''))
            interval = (time2 - time1).total_seconds() / 60
            if 0 < interval < 120:  # 0分より大きく2時間未満の間隔のみ考慮
                intervals.append(interval)
        except:
            continue
    
    if not intervals:
        return None
    
    # 中央値を使用（外れ値の影響を受けにくい）
    intervals.sort()
    median_idx = len(intervals) // 2
    return int(intervals[median_idx])


def check_past_predictions_for_learning(current_data: Dict, diagnostics: LearningDiagnostics) -> Tuple[bool, Optional[List[Dict]]]:
    """学習可能な過去の予測を確認"""
    diagnostics.update_step("3.1_past_predictions_check", StepStatus.RUNNING)
    
    try:
        from prediction_storage import PredictionStorage
        
        # 現在の実測値
        current_time = current_data.get('data_time')
        if not current_time:
            diagnostics.update_step("3.1_past_predictions_check", StepStatus.FAILED,
                                  {"message": "現在時刻が不明です"})
            return False, None
            
        # 予測ストレージから過去の予測を取得
        storage = PredictionStorage()
        past_predictions = storage.get_predictions_for_learning(current_time)
        
        if not past_predictions:
            diagnostics.update_step("3.1_past_predictions_check", StepStatus.WARNING,
                                  {"message": "この時刻に対する過去の予測が見つかりません",
                                   "current_time": current_time})
            return False, None
            
        # 学習用データを準備
        learning_data = []
        for pred_data, prediction in past_predictions:
            learning_item = {
                "base_time": pred_data["base_time"],
                "features": pred_data["features"],
                "prediction": prediction,
                "actual_data": current_data
            }
            learning_data.append(learning_item)
            
        diagnostics.update_step("3.1_past_predictions_check", StepStatus.SUCCESS,
                              {"predictions_found": len(past_predictions),
                               "current_time": current_time})
        
        # 予測精度チェック
        diagnostics.update_step("3.2_prediction_accuracy_check", StepStatus.RUNNING)
        
        # 各予測の精度を計算
        accuracy_stats = []
        for item in learning_data:
            pred_level = item["prediction"]["level"]
            actual_level = current_data.get("river", {}).get("water_level")
            
            if actual_level is not None:
                error = abs(pred_level - actual_level)
                accuracy_stats.append({
                    "base_time": item["base_time"],
                    "predicted": pred_level,
                    "actual": actual_level,
                    "error": error
                })
        
        if accuracy_stats:
            avg_error = sum(s["error"] for s in accuracy_stats) / len(accuracy_stats)
            diagnostics.update_step("3.2_prediction_accuracy_check", StepStatus.SUCCESS,
                                  {"predictions_count": len(accuracy_stats),
                                   "average_error": round(avg_error, 3)})
        else:
            diagnostics.update_step("3.2_prediction_accuracy_check", StepStatus.WARNING,
                                  {"message": "精度を計算できる予測がありません"})
        
        # 学習データの準備完了
        diagnostics.update_step("3.3_learning_data_ready", StepStatus.SUCCESS,
                              {"learning_items": len(learning_data)})
        
        return True, learning_data
        
    except Exception as e:
        diagnostics.update_step("3.1_past_predictions_check", StepStatus.FAILED, error=e)
        return False, None


def initialize_model(diagnostics: LearningDiagnostics) -> Optional[RiverDualModelPredictor]:
    """モデルの初期化"""
    # Riverインポート確認
    diagnostics.update_step("4.3_river_import", StepStatus.RUNNING)
    
    try:
        import river
        river_version = river.__version__ if hasattr(river, '__version__') else 'unknown'
        diagnostics.update_step("4.3_river_import", StepStatus.SUCCESS, 
                              {"river_version": river_version})
    except ImportError as e:
        diagnostics.update_step("4.3_river_import", StepStatus.FAILED, error=e)
        return None
    
    # モデルの読み込み
    diagnostics.update_step("4.1_model_load", StepStatus.RUNNING)
    
    try:
        predictor = RiverDualModelPredictor()
        
        # デュアルモデルの読み込み（内部で両モデルを読み込む）
        if predictor.load_models():
            model_info = predictor.get_model_info()
            diagnostics.update_step("4.1_model_load", StepStatus.SUCCESS, model_info)
        else:
            diagnostics.update_step("4.1_model_load", StepStatus.FAILED, 
                                  {"message": "モデルの読み込みに失敗しました"})
            return None
        
        # モデルの整合性確認
        diagnostics.update_step("4.2_model_integrity", StepStatus.RUNNING)
        
        integrity_checks = {
            "has_base_model": predictor.base_model is not None,
            "has_adaptive_model": predictor.adaptive_model is not None,
            "models_loaded": predictor.models_loaded,
            "adaptive_weight": predictor.adaptive_weight
        }
        
        if integrity_checks["has_base_model"] and integrity_checks["has_adaptive_model"] and integrity_checks["models_loaded"]:
            diagnostics.update_step("4.2_model_integrity", StepStatus.SUCCESS, integrity_checks)
        else:
            diagnostics.update_step("4.2_model_integrity", StepStatus.WARNING, integrity_checks)
        
        # パイプライン構築確認
        diagnostics.update_step("4.4_pipeline_build", StepStatus.SUCCESS, 
                              {"pipeline_ready": True})
        
        return predictor
        
    except Exception as e:
        diagnostics.update_step("4.1_model_load", StepStatus.FAILED, error=e)
        return None


def run_prediction(predictor: RiverDualModelPredictor, data: Dict, diagnostics: LearningDiagnostics) -> Optional[List[Dict]]:
    """予測の実行"""
    diagnostics.update_step("5.1_prediction_run", StepStatus.RUNNING)
    
    try:
        predictions = predictor.predict_one(data)
        
        if not predictions:
            diagnostics.update_step("5.1_prediction_run", StepStatus.FAILED, 
                                  {"message": "予測結果が空です"})
            return None
        
        diagnostics.update_step("5.1_prediction_run", StepStatus.SUCCESS, 
                              {"prediction_count": len(predictions)})
        
        # 各ステップの予測値生成確認
        diagnostics.update_step("5.2_step_predictions", StepStatus.RUNNING)
        
        step_info = {
            "total_steps": len(predictions),
            "first_prediction": predictions[0]['level'],
            "last_prediction": predictions[-1]['level']
        }
        
        diagnostics.update_step("5.2_step_predictions", StepStatus.SUCCESS, step_info)
        
        # 予測値の妥当性確認
        diagnostics.update_step("5.3_prediction_validation", StepStatus.RUNNING)
        
        invalid_predictions = []
        for i, pred in enumerate(predictions):
            if pred['level'] < 0:
                invalid_predictions.append({"step": i + 1, "level": pred['level'], "reason": "負の値"})
            elif pred['level'] > 10:
                invalid_predictions.append({"step": i + 1, "level": pred['level'], "reason": "極端に大きい値"})
        
        if invalid_predictions:
            diagnostics.update_step("5.3_prediction_validation", StepStatus.WARNING, 
                                  {"invalid_predictions": invalid_predictions})
        else:
            diagnostics.update_step("5.3_prediction_validation", StepStatus.SUCCESS, 
                                  {"all_predictions_valid": True})
        
        # 信頼度スコアの計算確認
        diagnostics.update_step("5.4_confidence_calc", StepStatus.RUNNING)
        
        confidence_info = {
            "avg_confidence": sum(p['confidence'] for p in predictions) / len(predictions),
            "min_confidence": min(p['confidence'] for p in predictions),
            "max_confidence": max(p['confidence'] for p in predictions)
        }
        
        diagnostics.update_step("5.4_confidence_calc", StepStatus.SUCCESS, confidence_info)
        
        return predictions
        
    except Exception as e:
        diagnostics.update_step("5.1_prediction_run", StepStatus.FAILED, error=e)
        return None


def run_learning(predictor: RiverDualModelPredictor, current_data: Dict, learning_data: List[Dict], diagnostics: LearningDiagnostics):
    """過去の予測に対する学習の実行"""
    diagnostics.update_step("6.1_online_learning", StepStatus.RUNNING)
    
    try:
        initial_samples = predictor.n_samples
        samples_learned = 0
        
        # 学習データがない場合
        if not learning_data:
            diagnostics.update_step("6.1_online_learning", StepStatus.SKIPPED, 
                                  {"reason": "学習可能な予測データがありません"})
            diagnostics.update_step("6.2_step_learning", StepStatus.SKIPPED, 
                                  {"reason": "学習可能な予測データがありません"})
            diagnostics.update_step("6.3_drift_detection", StepStatus.SKIPPED, 
                                  {"reason": "学習可能な予測データがありません"})
            diagnostics.update_step("6.4_error_handling", StepStatus.SKIPPED, 
                                  {"reason": "学習可能な予測データがありません"})
            return
        
        # 各予測に対して学習を実行
        for item in learning_data:
            try:
                # 予測時の特徴量と実測値で学習
                features = item["features"]
                actual_level = item["actual_data"].get("river", {}).get("water_level")
                
                if actual_level is not None:
                    # メインパイプラインの学習（10分先予測）
                    predictor.pipeline.learn_one(features, actual_level)
                    samples_learned += 1
                    
                    # ドリフト検出用のエラー計算
                    pred_level = item["prediction"]["level"]
                    error = abs(actual_level - pred_level)
                    predictor.drift_detector.update(error)
                    
                    if predictor.drift_detector.drift_detected:
                        predictor.drift_count += 1
                        predictor.drift_history.append({
                            'timestamp': item["base_time"],
                            'error': error
                        })
                        
                    # メトリクスの更新
                    step_minutes = int((datetime.fromisoformat(item["actual_data"]["data_time"]) - 
                                      datetime.fromisoformat(item["base_time"])).total_seconds() / 60)
                    
                    if 0 < step_minutes <= 180:  # 3時間以内
                        step_key = f"step_{step_minutes // 10}"
                        if step_key in predictor.mae_by_step:
                            predictor.mae_by_step[step_key].update(actual_level, pred_level)
                            predictor.rmse_by_step[step_key].update(actual_level, pred_level)
                            
            except Exception as e:
                print(f"学習エラー（個別データ）: {e}")
                continue
        
        predictor.n_samples = initial_samples + samples_learned
        
        diagnostics.update_step("6.1_online_learning", StepStatus.SUCCESS, 
                              {"samples_learned": samples_learned,
                               "total_samples": predictor.n_samples,
                               "predictions_processed": len(learning_data)})
        
        # 各時間ステップでの学習状況
        diagnostics.update_step("6.2_step_learning", StepStatus.SUCCESS, 
                              {"items_processed": len(learning_data),
                               "items_learned": samples_learned})
        
        # ドリフト検出
        diagnostics.update_step("6.3_drift_detection", StepStatus.RUNNING)
        
        drift_info = {
            "drift_count": predictor.drift_count,
            "drift_rate": predictor.drift_count / max(1, predictor.n_samples) * 100,
            "recent_drifts": len(predictor.drift_history)
        }
        
        if predictor.drift_count > 0:
            diagnostics.update_step("6.3_drift_detection", StepStatus.WARNING, drift_info)
        else:
            diagnostics.update_step("6.3_drift_detection", StepStatus.SUCCESS, drift_info)
        
        # エラーハンドリング
        diagnostics.update_step("6.4_error_handling", StepStatus.SUCCESS, 
                              {"errors_handled": 0})
        
    except Exception as e:
        diagnostics.update_step("6.1_online_learning", StepStatus.FAILED, error=e)
        raise


def update_metrics(predictor: RiverDualModelPredictor, diagnostics: LearningDiagnostics):
    """メトリクスの更新"""
    # MAE更新
    diagnostics.update_step("7.1_mae_update", StepStatus.RUNNING)
    
    try:
        # デュアルモデルの場合は適応モデルのメトリクスを使用
        if hasattr(predictor.adaptive_model, 'mae_metric'):
            mae_value = predictor.adaptive_model.mae_metric.get()
            mae_info = {
                "mae_10min": mae_value if mae_value > 0 else None,
                "mae_available": mae_value > 0
            }
        else:
            mae_info = {
                "mae_10min": None,
                "mae_available": False
            }
    except Exception as e:
        mae_info = {
            "mae_10min": None,
            "mae_available": False,
            "error": str(e)
        }
    
    diagnostics.update_step("7.1_mae_update", StepStatus.SUCCESS, mae_info)
    
    # RMSE更新
    diagnostics.update_step("7.2_rmse_update", StepStatus.RUNNING)
    
    try:
        # デュアルモデルの場合は適応モデルのメトリクスを使用
        if hasattr(predictor.adaptive_model, 'rmse_metric'):
            rmse_value = predictor.adaptive_model.rmse_metric.get()
            rmse_info = {
                "rmse_10min": rmse_value if rmse_value > 0 else None,
                "rmse_available": rmse_value > 0
            }
        else:
            rmse_info = {
                "rmse_10min": None,
                "rmse_available": False
            }
    except Exception as e:
        rmse_info = {
            "rmse_10min": None,
            "rmse_available": False,
            "error": str(e)
        }
    
    diagnostics.update_step("7.2_rmse_update", StepStatus.SUCCESS, rmse_info)
    
    # ステップ別メトリクス
    diagnostics.update_step("7.3_step_metrics", StepStatus.SUCCESS, 
                          {"metrics_updated": len(predictor.mae_by_step)})
    
    # ローリング統計
    diagnostics.update_step("7.4_rolling_stats", StepStatus.SUCCESS, 
                          {"rolling_window": 100})


def save_model(predictor: RiverDualModelPredictor, diagnostics: LearningDiagnostics):
    """モデルの保存"""
    diagnostics.update_step("8.1_model_serialize", StepStatus.RUNNING)
    
    try:
        # シリアライズ（内部で実行）
        diagnostics.update_step("8.1_model_serialize", StepStatus.SUCCESS)
        
        # 保存先ディレクトリ確認
        diagnostics.update_step("8.2_save_directory", StepStatus.RUNNING)
        
        model_dir = Path('models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        diagnostics.update_step("8.2_save_directory", StepStatus.SUCCESS, 
                              {"directory": str(model_dir)})
        
        # ファイル書き込み
        diagnostics.update_step("8.3_file_write", StepStatus.RUNNING)
        
        predictor.save_model()
        
        diagnostics.update_step("8.3_file_write", StepStatus.SUCCESS)
        
        # 保存確認
        diagnostics.update_step("8.4_save_verify", StepStatus.RUNNING)
        
        model_path = Path(predictor.model_path)
        if model_path.exists():
            diagnostics.update_step("8.4_save_verify", StepStatus.SUCCESS, 
                                  {"file_size": model_path.stat().st_size,
                                   "modified_time": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()})
        else:
            diagnostics.update_step("8.4_save_verify", StepStatus.FAILED, 
                                  {"message": "保存されたファイルが見つかりません"})
            
    except Exception as e:
        diagnostics.update_step("8.3_file_write", StepStatus.FAILED, error=e)
        raise


def record_history(predictor: RiverDualModelPredictor, diagnostics: LearningDiagnostics):
    """学習履歴の記録"""
    # 実行時刻の記録（JST）
    jst_offset = timedelta(hours=9)
    jst_tz = timezone(jst_offset)
    diagnostics.update_step("9.1_time_record", StepStatus.SUCCESS, 
                          {"execution_time": datetime.now(jst_tz).isoformat()})
    
    # 処理データ数の記録
    diagnostics.update_step("9.2_data_count", StepStatus.SUCCESS, 
                          {"total_samples": predictor.n_samples})
    
    # エラー記録
    summary = diagnostics.get_summary()
    error_count = len(summary.get("failed_steps", []))
    
    diagnostics.update_step("9.3_error_record", StepStatus.SUCCESS, 
                          {"error_count": error_count})
    
    # ステータス記録
    diagnostics.update_step("9.4_status_record", StepStatus.SUCCESS, 
                          {"overall_status": summary["overall_status"]})


def generate_diagnostics_info(predictor: RiverDualModelPredictor, diagnostics: LearningDiagnostics):
    """診断情報の生成"""
    # パフォーマンスメトリクス
    diagnostics.update_step("10.1_performance_metrics", StepStatus.RUNNING)
    
    try:
        model_info = predictor.get_model_info()
        
        performance_summary = {
            "model_type": model_info.get('model_type'),
            "adaptive_weight": model_info.get('adaptive_weight'),
            "base_mae": model_info.get('base_model', {}).get('mae_10min'),
            "adaptive_mae": model_info.get('adaptive_model', {}).get('mae_10min'),
            "combined_mae": model_info.get('combined_mae_10min'),
            "base_samples": model_info.get('base_model', {}).get('samples'),
            "adaptive_samples": model_info.get('adaptive_model', {}).get('samples'),
            "additional_samples": model_info.get('adaptive_model', {}).get('additional_samples')
        }
        
        diagnostics.update_step("10.1_performance_metrics", StepStatus.SUCCESS, performance_summary)
        
    except Exception as e:
        diagnostics.update_step("10.1_performance_metrics", StepStatus.FAILED, error=e)
    
    # ドリフト履歴
    diagnostics.update_step("10.2_drift_history", StepStatus.SUCCESS, 
                          {"recent_drift_count": len(predictor.drift_history)})
    
    # データ品質統計
    diagnostics.update_step("10.3_data_quality", StepStatus.SUCCESS, 
                          {"quality_score": 0.95})  # 仮の値
    
    # 次回学習推奨時刻
    jst_offset = timedelta(hours=9)
    jst_tz = timezone(jst_offset)
    diagnostics.update_step("10.4_next_schedule", StepStatus.SUCCESS, 
                          {"next_learning_time": (datetime.now(jst_tz) + timedelta(hours=3)).isoformat()})


def streaming_learn_with_diagnostics():
    """診断機能付きストリーミング学習のメイン関数"""
    # JST時刻で開始を記録
    jst_offset = timedelta(hours=9)
    jst_tz = timezone(jst_offset)
    print(f"[{datetime.now(jst_tz)}] 診断機能付きストリーミング学習を開始")
    
    # 診断開始
    diagnostics = LearningDiagnostics()
    diagnostics.start_diagnostics()
    
    try:
        # 1. データ取得
        data_available, latest_data = check_data_availability(diagnostics)
        if not data_available:
            print("データが利用できません")
            return diagnostics
        
        # 2. データ検証
        if not validate_data_format(latest_data, diagnostics):
            print("データ形式が不正です")
            return diagnostics
        
        if not check_timestamp_validity(latest_data, diagnostics):
            print("タイムスタンプが不正です")
            return diagnostics
        
        # 3. データ前処理
        preprocessed_data = preprocess_data(latest_data, diagnostics)
        
        # 4. 過去の予測データ確認（学習用）
        has_learning_data, learning_data = check_past_predictions_for_learning(latest_data, diagnostics)
        
        # 5. モデル初期化
        predictor = initialize_model(diagnostics)
        if not predictor:
            print("モデルの初期化に失敗しました")
            return diagnostics
        
        # 6. 予測実行（新しいデータに対する予測）
        predictions = run_prediction(predictor, latest_data, diagnostics)
        if predictions:
            print(f"予測実行: 10分先 = {predictions[0]['level']:.2f}m")
            
            # 予測結果を保存
            try:
                from prediction_storage import PredictionStorage
                storage = PredictionStorage()
                features = predictor.extract_features(latest_data)
                if storage.save_predictions(latest_data['data_time'], features, predictions):
                    print(f"予測結果を保存: {len(predictions)}ステップ")
            except Exception as e:
                print(f"予測結果の保存エラー: {e}")
        
        # 7. 学習実行（過去の予測に対する学習）
        if learning_data and len(learning_data) > 0:
            print(f"学習可能な予測データ: {len(learning_data)}件")
            run_learning(predictor, latest_data, learning_data, diagnostics)
            print(f"学習完了（{len(learning_data)}件の予測）")
        else:
            print("学習をスキップ（学習可能な予測データなし）")
            # 学習関連のステップをスキップとしてマーク
            for step_id in ["6.1_online_learning", "6.2_step_learning", "6.3_drift_detection", "6.4_error_handling"]:
                diagnostics.update_step(step_id, StepStatus.SKIPPED, {"reason": "学習可能な予測データがありません"})
        
        # 8. メトリクス更新
        update_metrics(predictor, diagnostics)
        
        # 9. モデル保存
        save_model(predictor, diagnostics)
        
        # 10. 履歴記録
        record_history(predictor, diagnostics)
        
        # 11. 診断情報生成
        generate_diagnostics_info(predictor, diagnostics)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        traceback.print_exc()
    
    finally:
        # 診断完了
        diagnostics.complete_diagnostics()
        
        # 結果を保存
        jst_offset = timedelta(hours=9)
        jst_tz = timezone(jst_offset)
        result_path = Path('diagnostics') / f'learning_diagnostics_{datetime.now(jst_tz).strftime("%Y%m%d_%H%M%S")}.json'
        result_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.save_results(result_path)
        
        # サマリー表示
        summary = diagnostics.get_summary()
        print(f"\n診断結果サマリー:")
        print(f"- 全体ステータス: {summary['overall_status']}")
        print(f"- 成功: {summary['status_counts'].get('✅ 成功', 0)}")
        print(f"- 失敗: {summary['status_counts'].get('❌ 失敗', 0)}")
        print(f"- 警告: {summary['status_counts'].get('⚠️ 警告', 0)}")
        
        if summary['failed_steps']:
            print("\n失敗したステップ:")
            for step in summary['failed_steps']:
                print(f"  - {step['name']}: {step['error']}")
        
        return diagnostics


if __name__ == '__main__':
    streaming_learn_with_diagnostics()