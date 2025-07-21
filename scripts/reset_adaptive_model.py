#!/usr/bin/env python3
"""
適応モデルを初期状態にリセットするスクリプト
週次実行を想定
"""

import shutil
from pathlib import Path
from datetime import datetime
import json


def reset_adaptive_model():
    """適応モデルを初期状態にリセット"""
    
    # パスの設定
    models_dir = Path(__file__).parent.parent / "models"
    initial_path = models_dir / "river_adaptive_model_initial.pkl"
    adaptive_path = models_dir / "river_adaptive_model_v2.pkl"
    backup_dir = models_dir / "backup"
    
    print("=== 適応モデルリセット処理 ===")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初期モデルの存在確認
    if not initial_path.exists():
        print(f"エラー: 初期モデルが見つかりません: {initial_path}")
        print("initialize_models.py を実行してモデルを初期化してください。")
        return False
    
    # バックアップディレクトリの作成
    backup_dir.mkdir(exist_ok=True)
    
    # 現在の適応モデルの情報を取得
    current_info = None
    if adaptive_path.exists():
        try:
            # RiverStreamingPredictorをインポートして情報を取得
            import sys
            sys.path.append(str(Path(__file__).parent))
            from river_streaming_prediction_v2 import RiverStreamingPredictor
            
            current_model = RiverStreamingPredictor()
            current_model.load_model(str(adaptive_path))
            
            current_info = {
                "samples": current_model.n_samples,
                "mae_10min": None,
                "last_update": None
            }
            
            if hasattr(current_model.mae_metric, 'get'):
                mae = current_model.mae_metric.get()
                if mae > 0:
                    current_info["mae_10min"] = round(mae, 3)
                    
            print(f"\n現在の適応モデル情報:")
            print(f"  学習サンプル数: {current_info['samples']:,}")
            if current_info["mae_10min"]:
                print(f"  MAE (10分): {current_info['mae_10min']:.3f}")
                
        except Exception as e:
            print(f"警告: 現在のモデル情報を取得できませんでした: {e}")
    
    # バックアップの作成
    if adaptive_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"adaptive_model_before_reset_{timestamp}.pkl"
        
        print(f"\nバックアップを作成中...")
        shutil.copy(adaptive_path, backup_path)
        print(f"バックアップ完了: {backup_path.name}")
        
        # バックアップ情報を記録
        backup_info = {
            "reset_time": datetime.now().isoformat(),
            "backup_file": backup_path.name,
            "model_info": current_info
        }
        
        # バックアップログに追記
        log_path = backup_dir / "reset_log.json"
        reset_log = []
        
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    reset_log = json.load(f)
            except:
                reset_log = []
                
        reset_log.append(backup_info)
        
        # 最新の20件のみ保持
        reset_log = reset_log[-20:]
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(reset_log, f, ensure_ascii=False, indent=2)
    
    # リセット実行
    print(f"\n適応モデルをリセット中...")
    shutil.copy(initial_path, adaptive_path)
    print(f"リセット完了: 初期状態に戻しました")
    
    # リセット後の情報を確認
    try:
        reset_model = RiverStreamingPredictor()
        reset_model.load_model(str(adaptive_path))
        
        print(f"\nリセット後の適応モデル情報:")
        print(f"  学習サンプル数: {reset_model.n_samples:,}")
        
        if hasattr(reset_model.mae_metric, 'get'):
            mae = reset_model.mae_metric.get()
            if mae > 0:
                print(f"  MAE (10分): {mae:.3f}")
                
    except Exception as e:
        print(f"警告: リセット後のモデル情報を確認できませんでした: {e}")
    
    # 古いバックアップのクリーンアップ
    print(f"\n古いバックアップをクリーンアップ中...")
    cleanup_old_backups(backup_dir, days_to_keep=30)
    
    print("\n=== リセット処理完了 ===")
    return True


def cleanup_old_backups(backup_dir: Path, days_to_keep: int = 30):
    """古いバックアップファイルを削除"""
    cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    removed_count = 0
    
    for backup_file in backup_dir.glob("adaptive_model_before_reset_*.pkl"):
        if backup_file.stat().st_mtime < cutoff_date:
            backup_file.unlink()
            removed_count += 1
            
    if removed_count > 0:
        print(f"{removed_count}個の古いバックアップを削除しました")


def show_reset_history():
    """リセット履歴を表示"""
    log_path = Path(__file__).parent.parent / "models" / "backup" / "reset_log.json"
    
    if not log_path.exists():
        print("リセット履歴がありません")
        return
        
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            reset_log = json.load(f)
            
        print("\n=== 適応モデルリセット履歴 ===")
        print(f"記録数: {len(reset_log)}件")
        
        for i, entry in enumerate(reversed(reset_log[-5:]), 1):  # 最新5件
            print(f"\n{i}. {entry['reset_time']}")
            print(f"   バックアップ: {entry['backup_file']}")
            
            if entry.get('model_info'):
                info = entry['model_info']
                print(f"   リセット前: {info['samples']:,}サンプル", end="")
                if info.get('mae_10min'):
                    print(f", MAE: {info['mae_10min']:.3f}", end="")
                print()
                
    except Exception as e:
        print(f"履歴の読み込みエラー: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="適応モデルのリセット管理")
    parser.add_argument("--history", action="store_true", help="リセット履歴を表示")
    args = parser.parse_args()
    
    if args.history:
        show_reset_history()
    else:
        reset_adaptive_model()