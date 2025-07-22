"""
AI学習結果ページ
モデルの学習状況と予測精度を可視化（学習プロセス診断を統合）
"""

import streamlit as st
from pathlib import Path
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import numpy as np
import json
from typing import Dict, List, Optional
import subprocess
import sys

# ページ設定
st.set_page_config(
    page_title="AI学習結果",
    page_icon="📊",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    /* サイドバーのページナビゲーションボタンのフォントサイズを大きく */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        font-size: 18px !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        line-height: 1.5 !important;
    }
    
    /* サイドバーのページナビゲーションボタンのテキスト */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a span {
        font-size: 18px !important;
    }
    
    /* サイドバーのページナビゲーションボタンのアイコン */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a [data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
        margin-right: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI学習結果")
st.markdown("機械学習モデルの学習状況、プロセス診断、予測精度を確認します。")


def load_dual_model_info():
    """デュアルモデル情報を読み込み"""
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from scripts.river_dual_model_predictor import RiverDualModelPredictor
        
        predictor = RiverDualModelPredictor()
        if predictor.load_models():
            return predictor.get_model_info()
        return None
    except Exception as e:
        st.error(f"モデル情報の読み込みエラー: {e}")
        return None


def load_prediction_stats():
    """予測統計情報を読み込み"""
    try:
        from scripts.prediction_storage import PredictionStorage
        storage = PredictionStorage()
        return storage.get_recent_predictions_count()
    except:
        return {"last_hour": 0, "last_24h": 0, "total": 0}


def load_recent_diagnostics():
    """最新の診断結果を読み込み"""
    diagnostics_dir = Path('diagnostics')
    if not diagnostics_dir.exists():
        return None
    
    # 最新のファイルを探す
    json_files = list(diagnostics_dir.glob('*.json'))
    if not json_files:
        return None
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None


def load_diagnostics_history(days=7):
    """指定期間の診断履歴を読み込み"""
    diagnostics_dir = Path('diagnostics')
    if not diagnostics_dir.exists():
        return []
    
    history = []
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for json_file in diagnostics_dir.glob('*.json'):
        # ファイル名から日時を抽出
        try:
            file_date = datetime.fromtimestamp(json_file.stat().st_mtime)
            if file_date > cutoff_date:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history.append({
                        'timestamp': file_date,
                        'data': data
                    })
        except:
            continue
    
    return sorted(history, key=lambda x: x['timestamp'])


def format_mae(mae_value):
    """MAE値をフォーマット"""
    if mae_value is None:
        return "データなし"
    return f"±{mae_value:.3f}m"


def get_accuracy_emoji(mae_value):
    """精度に応じた絵文字を返す"""
    if mae_value is None:
        return "⚫"
    elif mae_value < 0.05:
        return "🟢"  # 優秀
    elif mae_value < 0.10:
        return "🟡"  # 良好
    else:
        return "🔴"  # 要改善


def plot_step_accuracy(metrics_by_step):
    """ステップ別精度のグラフを作成"""
    steps = []
    mae_values = []
    rmse_values = []
    
    for step_label, metrics in sorted(metrics_by_step.items(), 
                                    key=lambda x: int(x[0].replace('min', ''))):
        time_minutes = int(step_label.replace('min', ''))
        steps.append(time_minutes)
        mae_values.append(metrics.get('mae'))
        rmse_values.append(metrics.get('rmse'))
    
    fig = go.Figure()
    
    # MAEのライン
    fig.add_trace(go.Scatter(
        x=steps,
        y=mae_values,
        mode='lines+markers',
        name='MAE（平均絶対誤差）',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # RMSEのライン
    fig.add_trace(go.Scatter(
        x=steps,
        y=rmse_values,
        mode='lines+markers',
        name='RMSE（二乗平均平方根誤差）',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # 精度基準線
    fig.add_hline(y=0.05, line_dash="dot", line_color="green", 
                  annotation_text="優秀レベル（±5cm）")
    fig.add_hline(y=0.10, line_dash="dot", line_color="orange", 
                  annotation_text="良好レベル（±10cm）")
    
    fig.update_layout(
        title="予測時間別の精度",
        xaxis_title="予測時間（分）",
        yaxis_title="誤差（m）",
        height=400,
        hovermode='x unified',
        xaxis=dict(
            tickmode='linear',
            tick0=10,
            dtick=20
        )
    )
    
    return fig


def plot_adaptive_learning_timeline(diagnostics_history):
    """適応モデルの学習タイムラインを表示"""
    if not diagnostics_history:
        return None
    
    fig = go.Figure()
    
    timestamps = []
    mae_values = []
    sample_counts = []
    
    for entry in diagnostics_history:
        timestamps.append(entry['timestamp'])
        data = entry['data']
        mae_values.append(data.get('final_metrics', {}).get('mae'))
        sample_counts.append(data.get('training_stats', {}).get('total_samples', 0))
    
    # MAEの推移
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=mae_values,
        mode='lines+markers',
        name='MAE推移',
        yaxis='y',
        line=dict(color='blue', width=2)
    ))
    
    # サンプル数の推移（第2軸）
    fig.add_trace(go.Bar(
        x=timestamps,
        y=sample_counts,
        name='学習サンプル数',
        yaxis='y2',
        opacity=0.3,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title="適応モデルの学習推移",
        xaxis_title="日時",
        yaxis=dict(
            title="MAE (m)",
            side='left'
        ),
        yaxis2=dict(
            title="学習サンプル数",
            overlaying='y',
            side='right'
        ),
        height=400,
        hovermode='x unified'
    )
    
    return fig


def show_adaptive_model_execution_steps(latest_diagnostics):
    """適応モデルの詳細な実行ステップを表示"""
    st.subheader("🔄 適応モデルの実行ステップ詳細")
    
    if not latest_diagnostics:
        st.info("診断データがありません")
        return
    
    # 実行情報
    execution_info = latest_diagnostics.get('execution_info', {})
    training_stats = latest_diagnostics.get('training_stats', {})
    
    # タブで各フェーズを表示
    tabs = st.tabs(["📥 データ収集", "🔧 前処理", "📚 学習", "✅ 検証", "📋 実行ログ"])
    
    with tabs[0]:  # データ収集
        st.markdown("### データ収集フェーズ")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("実行時刻", execution_info.get('start_time', 'N/A'))
        with col2:
            st.metric("収集元", execution_info.get('source', 'GitHub Actions'))
        with col3:
            st.metric("収集データ数", training_stats.get('total_samples', 0))
        
        # データ品質
        data_quality = training_stats.get('data_quality', {})
        if data_quality:
            st.markdown("#### データ品質チェック")
            quality_df = pd.DataFrame({
                '項目': ['有効データ', 'スキップ', 'エラー'],
                '件数': [
                    data_quality.get('valid', 0),
                    data_quality.get('skipped', 0),
                    data_quality.get('errors', 0)
                ]
            })
            st.bar_chart(quality_df.set_index('項目'))
    
    with tabs[1]:  # 前処理
        st.markdown("### 前処理フェーズ")
        
        preprocessing = training_stats.get('preprocessing', {})
        if preprocessing:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 欠損値処理")
                st.info(f"処理済み: {preprocessing.get('missing_handled', 0)}件")
                
            with col2:
                st.markdown("#### 異常値検出")
                st.info(f"検出数: {preprocessing.get('outliers_detected', 0)}件")
        
        # 特徴量情報
        st.markdown("#### 使用特徴量")
        features = [
            "河川水位（現在値）",
            "ダム放流量",
            "ダム流入量",
            "10分間降雨量",
            "1時間降雨量",
            "貯水率"
        ]
        for feature in features:
            st.write(f"• {feature}")
    
    with tabs[2]:  # 学習
        st.markdown("### 学習フェーズ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 学習前")
            initial_metrics = latest_diagnostics.get('initial_metrics', {})
            st.metric("MAE", format_mae(initial_metrics.get('mae')))
            st.metric("RMSE", format_mae(initial_metrics.get('rmse')))
        
        with col2:
            st.markdown("#### 学習後")
            final_metrics = latest_diagnostics.get('final_metrics', {})
            mae_improvement = None
            if initial_metrics.get('mae') and final_metrics.get('mae'):
                mae_improvement = (initial_metrics['mae'] - final_metrics['mae']) / initial_metrics['mae'] * 100
            
            st.metric("MAE", format_mae(final_metrics.get('mae')), 
                     f"{mae_improvement:.1f}%" if mae_improvement else None)
            st.metric("RMSE", format_mae(final_metrics.get('rmse')))
        
        # 学習詳細
        st.markdown("#### 学習統計")
        stats_df = pd.DataFrame({
            '項目': ['処理時間', '学習レート', 'バッチサイズ'],
            '値': [
                f"{training_stats.get('processing_time', 0):.2f}秒",
                "自動調整",
                "1（オンライン学習）"
            ]
        })
        st.table(stats_df)
    
    with tabs[3]:  # 検証
        st.markdown("### 検証フェーズ")
        
        # ドリフト検出
        drift_info = latest_diagnostics.get('drift_detection', {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ドリフト検出")
            drift_detected = drift_info.get('detected', False)
            if drift_detected:
                st.error("⚠️ ドリフト検出")
            else:
                st.success("✅ 正常")
            
            st.metric("ドリフト率", f"{drift_info.get('rate', 0):.1f}%")
        
        with col2:
            st.markdown("#### 精度評価")
            accuracy_status = "改善" if mae_improvement and mae_improvement > 0 else "悪化"
            st.info(f"前回比: {accuracy_status}")
            
            # 推奨事項
            st.markdown("#### 推奨事項")
            if drift_detected:
                st.warning("モデルの再初期化を検討してください")
            elif mae_improvement and mae_improvement < -10:
                st.warning("精度が悪化しています。データ品質を確認してください")
            else:
                st.success("正常に学習が進行しています")
    
    with tabs[4]:  # 実行ログ
        st.markdown("### 実行ログ")
        
        # ログメッセージ
        logs = latest_diagnostics.get('logs', [])
        if logs:
            log_text = "\n".join([f"[{log.get('time', '')}] {log.get('level', '')}: {log.get('message', '')}" 
                                 for log in logs[-20:]])  # 最新20件
            st.text_area("ログ出力", log_text, height=300)
        else:
            st.info("ログデータがありません")
        
        # エラー/警告サマリー
        error_count = sum(1 for log in logs if log.get('level') == 'ERROR')
        warning_count = sum(1 for log in logs if log.get('level') == 'WARNING')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("エラー数", error_count)
        with col2:
            st.metric("警告数", warning_count)


# メイン処理
# 1. モデル概要セクション
st.header("1️⃣ モデル概要")

model_info = load_dual_model_info()

if model_info:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔷 基本モデル（固定）")
        base_info = model_info.get('base_model', {})
        
        if base_info.get('loaded'):
            st.success("✅ モデル読み込み成功")
            st.info("📌 初期学習済み・更新なし")
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("学習サンプル数", f"{base_info.get('samples', 0):,}")
            with metrics_col2:
                mae = base_info.get('mae_10min')
                emoji = get_accuracy_emoji(mae)
                st.metric(f"MAE (10分先) {emoji}", format_mae(mae))
        else:
            st.error("❌ モデル読み込み失敗")
    
    with col2:
        st.subheader("🔶 適応モデル（継続学習）")
        adaptive_info = model_info.get('adaptive_model', {})
        
        if adaptive_info.get('loaded'):
            st.success("✅ モデル読み込み成功")
            st.info("📈 リアルタイムで継続学習中")
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("学習サンプル数", f"{adaptive_info.get('samples', 0):,}")
                st.metric("追加学習数", f"+{adaptive_info.get('additional_samples', 0):,}")
            with metrics_col2:
                mae = adaptive_info.get('mae_10min')
                emoji = get_accuracy_emoji(mae)
                st.metric(f"MAE (10分先) {emoji}", format_mae(mae))
        else:
            st.error("❌ モデル読み込み失敗")
else:
    st.warning("モデル情報を読み込めませんでした")

# 2. 学習プロセス診断セクション（統合）
st.header("2️⃣ 学習プロセス診断")

# 最新の診断結果を読み込み
latest_diagnostics = load_recent_diagnostics()
diagnostics_history = load_diagnostics_history(days=7)

# 基本モデルと適応モデルのタブ
model_tabs = st.tabs(["🔷 基本モデル診断", "🔶 適応モデル診断"])

with model_tabs[0]:  # 基本モデル
    st.info("基本モデルは初期学習のみで、継続学習は行いません。")
    
    if model_info and model_info.get('base_model', {}).get('loaded'):
        st.markdown("### 初期学習情報")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("学習データソース", "デモCSVファイル")
        with col2:
            st.metric("学習期間", "2023/6/25-7/1")
        with col3:
            st.metric("初期MAE", format_mae(model_info['base_model'].get('mae_10min')))
        
        # 学習データの特性
        st.markdown("### 学習データ特性")
        st.write("""
        - 実際の厚東川データを使用
        - 10分間隔のデータ
        - 約1週間分のデータ
        - 晴天時、雨天時、ダム放流時のパターンを含む
        """)

with model_tabs[1]:  # 適応モデル
    if latest_diagnostics:
        # 詳細な実行ステップを表示
        show_adaptive_model_execution_steps(latest_diagnostics)
        
        # 学習履歴グラフ
        if diagnostics_history:
            st.markdown("### 📈 学習履歴")
            timeline_fig = plot_adaptive_learning_timeline(diagnostics_history)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.info("診断データがありません。学習プロセスが実行されるのをお待ちください。")
        
        # 手動実行ボタン
        if st.button("🔄 今すぐ学習を実行", type="primary"):
            with st.spinner("学習プロセスを実行中..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "scripts/streaming_train_with_diagnostics.py"],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ 学習が完了しました！")
                        st.experimental_rerun()
                    else:
                        st.error("❌ 学習中にエラーが発生しました")
                        st.text(result.stderr)
                except subprocess.TimeoutExpired:
                    st.error("⏱️ 学習がタイムアウトしました（5分）")
                except Exception as e:
                    st.error(f"エラー: {str(e)}")

# 3. 予測精度分析セクション
st.header("3️⃣ 予測精度分析")

# 最新の診断結果から精度情報を表示
if latest_diagnostics:
    # ステップ別精度
    metrics_by_step = latest_diagnostics.get('metrics_by_step', {})
    if metrics_by_step:
        st.subheader("📊 予測時間別の精度")
        step_fig = plot_step_accuracy(metrics_by_step)
        st.plotly_chart(step_fig, use_container_width=True)
        
        # 精度サマリー
        st.subheader("📋 精度サマリー")
        summary_data = []
        for step_label, metrics in sorted(metrics_by_step.items(), 
                                        key=lambda x: int(x[0].replace('min', ''))):
            mae = metrics.get('mae')
            emoji = get_accuracy_emoji(mae)
            summary_data.append({
                '予測時間': step_label,
                '精度評価': emoji,
                'MAE': format_mae(mae),
                'RMSE': format_mae(metrics.get('rmse'))
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

# 予測統計
st.header("4️⃣ 予測統計")
pred_stats = load_prediction_stats()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("過去1時間の予測回数", pred_stats.get('last_hour', 0))
with col2:
    st.metric("過去24時間の予測回数", pred_stats.get('last_24h', 0))
with col3:
    st.metric("総予測回数", f"{pred_stats.get('total', 0):,}")

# フッター情報
st.markdown("---")
st.caption("💡 ヒント: 適応モデルは6時間ごとに自動的に学習されます。手動実行も可能です。")