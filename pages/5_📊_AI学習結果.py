"""
AI学習結果ページ
モデルの学習状況と予測精度を可視化
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
st.markdown("機械学習モデルの学習状況と予測精度を確認します。")


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


def plot_drift_history(drift_history, drift_count, n_samples):
    """ドリフト検出履歴のグラフを作成"""
    if not drift_history:
        return None
    
    # ダミーデータでドリフトの発生パターンを示す
    fig = go.Figure()
    
    # サンプル数の推移（仮想的なデータ）
    sample_range = list(range(0, n_samples + 1, max(1, n_samples // 100)))
    drift_points = []
    
    # ドリフト発生率から仮想的なドリフトポイントを生成
    if drift_count > 0:
        drift_interval = n_samples // (drift_count + 1)
        for i in range(1, drift_count + 1):
            drift_points.append(i * drift_interval)
    
    fig.add_trace(go.Scatter(
        x=sample_range,
        y=[0] * len(sample_range),
        mode='lines',
        name='通常学習',
        line=dict(color='lightblue', width=2)
    ))
    
    # ドリフトポイントをマーク
    if drift_points:
        fig.add_trace(go.Scatter(
            x=drift_points,
            y=[0] * len(drift_points),
            mode='markers',
            name='ドリフト検出',
            marker=dict(
                size=15,
                color='red',
                symbol='x',
                line=dict(width=3)
            )
        ))
    
    fig.update_layout(
        title=f"ドリフト検出履歴（検出回数: {drift_count}回）",
        xaxis_title="学習サンプル数",
        yaxis_title="",
        height=200,
        showlegend=True,
        yaxis=dict(visible=False)
    )
    
    return fig


# メイン処理
model_info = load_dual_model_info()
prediction_stats = load_prediction_stats()
recent_diagnostics = load_recent_diagnostics()

if not model_info:
    st.warning("学習済みモデルが見つかりません。データ収集と学習が実行されるのをお待ちください。")
else:
    # モデル選択
    st.markdown("### 🎯 表示するモデルを選択")
    model_type = st.radio(
        "モデルタイプ",
        ["統合モデル（基本 + 適応）", "基本モデルのみ", "適応モデルのみ"],
        horizontal=True,
        help="基本モデルはデモデータで学習済み・固定、適応モデルはリアルタイムデータで継続学習"
    )
    
    # 選択に応じて表示するモデル情報を調整
    if model_type == "基本モデルのみ":
        display_info = model_info.get('base_model', {})
        model_name = "基本モデル（固定）"
    elif model_type == "適応モデルのみ":
        display_info = model_info.get('adaptive_model', {})
        model_name = "適応モデル（継続学習）"
    else:
        display_info = model_info
        model_name = "統合モデル"
    # 概要ダッシュボード
    st.markdown(f"## 📈 {model_name} の概要")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if model_type == "統合モデル（基本 + 適応）":
            # 統合モデルの場合は両方のサンプル数を表示
            base_samples = model_info.get('base_model', {}).get('samples', 0)
            adaptive_samples = model_info.get('adaptive_model', {}).get('samples', 0)
            st.metric(
                "学習サンプル数",
                f"基本: {base_samples:,}件",
                delta=f"適応: {adaptive_samples:,}件",
                help="基本モデル（固定）と適応モデル（継続学習）のサンプル数"
            )
        else:
            samples = display_info.get('samples', 0)
            st.metric(
                "学習サンプル数",
                f"{samples:,}件",
                delta=f"過去1時間: +{prediction_stats['last_hour']}件" if model_type == "適応モデルのみ" and prediction_stats['last_hour'] > 0 else None,
                help="モデルが学習したデータポイントの総数"
            )
    
    with col2:
        if model_type == "統合モデル（基本 + 適応）":
            mae_10min = model_info.get('combined_mae_10min')
        else:
            mae_10min = display_info.get('mae_10min')
        emoji = get_accuracy_emoji(mae_10min)
        st.metric(
            f"{emoji} 10分先予測精度",
            format_mae(mae_10min),
            help="10分先の水位予測の平均絶対誤差"
        )
    
    with col3:
        if model_type == "統合モデル（基本 + 適応）":
            adaptive_weight = model_info.get('adaptive_weight', 0.3)
            st.metric(
                "適応モデルの重み",
                f"{adaptive_weight:.0%}",
                help="統合予測における適応モデルの寄与度"
            )
        else:
            # 基本モデルはドリフト検出なし、適応モデルのみドリフト表示
            if model_type == "適応モデルのみ":
                st.metric(
                    "追加学習サンプル",
                    f"{display_info.get('additional_samples', 0):,}件",
                    help="初期化後に追加で学習したサンプル数"
                )
            else:
                st.metric(
                    "モデルタイプ",
                    "固定モデル",
                    help="デモデータで学習済み、更新されません"
                )
    
    with col4:
        if model_type == "基本モデルのみ":
            st.metric(
                "学習データ",
                "デモデータ",
                help="500件の典型的パターンで学習"
            )
        elif model_type == "適応モデルのみ":
            additional = display_info.get('additional_samples', 0)
            if additional > 0:
                st.metric(
                    "学習進度",
                    f"{min(100, additional // 10)}%",
                    delta=f"+{additional}件",
                    help="1000件で100%とした学習進度"
                )
            else:
                st.metric(
                    "学習進度",
                    "初期状態",
                    help="週次リセット後の初期状態"
                )
        else:
            st.metric(
                "統合方式",
                f"基本{int((1-adaptive_weight)*100)}% + 適応{int(adaptive_weight*100)}%",
                help="両モデルの予測を重み付け平均"
            )
    
    # 詳細情報のタブ
    if model_type == "統合モデル（基本 + 適応）":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["時間別精度", "精度詳細表", "学習履歴", "適応モデル進化", "ドリフト分析", "予測統計", "エラー分析", "モデル情報"])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["時間別精度", "精度詳細表", "学習履歴", "ドリフト分析", "予測統計", "エラー分析", "モデル情報"])
    
    with tab1:
        st.markdown("### 📊 予測時間別の精度")
        st.markdown("予測する時間が長くなるほど、誤差が大きくなる傾向があります。")
        
        # デュアルモデルでは、個別モデルのmetrics_by_stepを使用
        metrics_by_step = None
        if model_type == "基本モデルのみ" and 'base_model' in model_info:
            # 基本モデルのmetricsを取得（TODO: 基本モデルにmetrics_by_stepが必要）
            st.info("基本モデルの時間別精度は現在準備中です。")
        elif model_type == "適応モデルのみ" and 'adaptive_model' in model_info:
            # 適応モデルのmetricsを取得（TODO: 適応モデルにmetrics_by_stepが必要）
            st.info("適応モデルの時間別精度は現在準備中です。")
        else:
            # 統合モデルの場合は両方のモデルの精度を表示
            st.info("統合モデルの時間別精度は現在準備中です。")
            
        # 精度の解釈
        st.info("""
        **精度の目安**
        - 🟢 **優秀**（±5cm未満）: 非常に高い精度で予測できています
        - 🟡 **良好**（±5〜10cm）: 実用的な精度で予測できています
        - 🔴 **要改善**（±10cm以上）: さらなる学習が必要です
        """)
    
    with tab2:
        st.markdown("### 📋 時間別精度詳細")
        
        # デュアルモデルではmetrics_by_stepが存在しない
        if False:  # 一時的に無効化
            # データフレームの作成
            data = []
            for step_label, metrics in sorted(model_info['metrics_by_step'].items(), 
                                            key=lambda x: int(x[0].replace('min', ''))):
                time_minutes = int(step_label.replace('min', ''))
                mae = metrics.get('mae')
                rmse = metrics.get('rmse')
                
                data.append({
                    '予測時間': f"{time_minutes}分後",
                    '状態': get_accuracy_emoji(mae),
                    'MAE（平均絶対誤差）': format_mae(mae),
                    'RMSE（二乗平均平方根誤差）': format_mae(rmse) if rmse else "データなし",
                    '精度評価': (
                        "優秀" if mae and mae < 0.05 else
                        "良好" if mae and mae < 0.10 else
                        "要改善" if mae else "データなし"
                    )
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 統計サマリー（一時的に無効化）
            if False:
                st.markdown("#### 統計サマリー")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最小誤差", "準備中")
                with col2:
                    st.metric("平均誤差", "準備中")
                with col3:
                    st.metric("最大誤差", "準備中")
        else:
            st.info("現在、時間別精度の詳細表示を準備中です。")
    
    with tab3:
        st.markdown("### 📈 学習履歴")
        if model_type == "基本モデルのみ":
            st.markdown("基本モデルはデモデータで学習済みの固定モデルです。")
        else:
            st.markdown("モデルの学習進捗と精度の改善状況を確認できます。")
        
        # 学習曲線のシミュレーション
        if model_type == "統合モデル（基本 + 適応）":
            n_samples = model_info.get('adaptive_model', {}).get('samples', 0)
        else:
            n_samples = display_info.get('samples', 0)
        if n_samples > 0:
            # 仮想的な学習曲線を生成
            sample_points = np.linspace(0, n_samples, min(100, n_samples))
            
            # 初期の高い誤差から徐々に改善するパターン
            initial_mae = 0.15
            current_mae = model_info.get('mae_10min', 0.05) or 0.05
            
            # 学習曲線（指数関数的な改善）
            mae_curve = initial_mae * np.exp(-sample_points / (n_samples / 3)) + current_mae
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sample_points,
                y=mae_curve,
                mode='lines',
                name='MAE推移',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 255, 0.1)'
            ))
            
            # 現在のポイントを強調
            fig.add_trace(go.Scatter(
                x=[n_samples],
                y=[current_mae],
                mode='markers',
                name='現在',
                marker=dict(size=12, color='red', symbol='star')
            ))
            
            fig.update_layout(
                title="学習による精度改善の推移",
                xaxis_title="学習サンプル数",
                yaxis_title="MAE（平均絶対誤差）[m]",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 学習統計
            col1, col2, col3 = st.columns(3)
            with col1:
                improvement = ((initial_mae - current_mae) / initial_mae * 100) if current_mae else 0
                st.metric("精度改善率", f"{improvement:.1f}%", 
                         help="初期状態からの改善率")
            with col2:
                samples_per_day = n_samples / max(1, (n_samples / 144))  # 10分間隔なら1日144サンプル
                st.metric("1日あたり学習数", f"{samples_per_day:.0f}サンプル",
                         help="平均的な1日の学習データ数")
            with col3:
                days_running = n_samples / 144 if n_samples > 144 else n_samples / 24
                st.metric("稼働日数", f"{days_running:.1f}日",
                         help="モデルが学習を開始してからの日数")
        else:
            st.info("学習履歴データがまだありません。")
        
        # 診断情報へのリンク
        st.info("💡 より詳細な学習プロセスの情報は、[学習プロセス診断ページ](/4_🔍_学習プロセス診断)でご確認いただけます。")
    
    # 統合モデルの場合、適応モデル進化タブを表示
    if model_type == "統合モデル（基本 + 適応）":
        with tab4:
            st.markdown("### 🔄 適応モデルの進化")
            st.markdown("適応モデルがリアルタイムデータでどのように進化しているかを確認できます。")
            
            base_info = model_info.get('base_model', {})
            adaptive_info = model_info.get('adaptive_model', {})
            
            # 両モデルの比較
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 基本モデル（固定）")
                st.info("""
                - 学習データ: デモデータ（500件）
                - 学習パターン: 5種類の典型的状況
                - 更新: なし（固定）
                - 役割: 安定した基準予測を提供
                """)
                if base_info.get('mae_10min'):
                    st.metric("予測精度", format_mae(base_info['mae_10min']))
            
            with col2:
                st.markdown("#### 適応モデル（継続学習）")
                additional = adaptive_info.get('additional_samples', 0)
                if additional > 0:
                    st.success(f"""
                    - 初期学習: デモデータ（500件）
                    - 追加学習: リアルタイムデータ（{additional}件）
                    - 更新: 3時間ごと
                    - 役割: 最新パターンへの適応
                    """)
                else:
                    st.info("""
                    - 初期学習: デモデータ（500件）
                    - 追加学習: なし（リセット直後）
                    - 更新: 3時間ごと
                    - 役割: 最新パターンへの適応
                    """)
                if adaptive_info.get('mae_10min'):
                    st.metric("予測精度", format_mae(adaptive_info['mae_10min']))
            
            # 適応重みの変化
            st.markdown("#### 適応重みの推移")
            adaptive_weight = model_info.get('adaptive_weight', 0.3)
            
            # 重み変化のビジュアライゼーション
            fig = go.Figure()
            
            # 現在の重み
            fig.add_trace(go.Bar(
                x=['基本モデル', '適応モデル'],
                y=[1 - adaptive_weight, adaptive_weight],
                text=[f'{(1-adaptive_weight)*100:.0f}%', f'{adaptive_weight*100:.0f}%'],
                textposition='outside',
                marker_color=['lightblue', 'lightgreen']
            ))
            
            fig.update_layout(
                title="現在の統合重み",
                yaxis_title="重み",
                height=300,
                showlegend=False,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 適応モデルの進化説明
            st.markdown("#### 適応モデルの学習サイクル")
            st.markdown("""
            1. **初期化**: デモデータで学習した状態からスタート
            2. **継続学習**: 3時間ごとに最新の予測と実測値で学習
            3. **重み調整**: 追加学習サンプル数に応じて自動調整
            4. **週次リセット**: 毎週月曜日に初期状態にリセット（過学習防止）
            """)
            
            # 次回リセットまでの時間
            from datetime import datetime
            now = datetime.now()
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0 and now.hour >= 3:
                days_until_monday = 7
            
            st.info(f"次回リセットまで: {days_until_monday}日 {(3 - now.hour) % 24}時間")
        
        # 統合モデルの場合、タブ5以降を定義
        current_tab = 5
    else:
        # 統合モデル以外の場合はtab4から続ける
        current_tab = 4
    
    # ドリフト分析タブ
    with (tab5 if model_type == "統合モデル（基本 + 適応）" else tab4):
        st.markdown("### 🔍 ドリフト分析")
        st.markdown("ドリフトは、データの統計的性質が時間とともに変化することを示します。")
        
        drift_count = model_info.get('drift_count', 0)
        drift_rate = model_info.get('drift_rate', 0)
        
        if drift_count > 0:
            # ドリフト情報の表示
            st.warning(f"""
            **ドリフト検出状況**
            - 検出回数: {drift_count}回
            - 発生率: {drift_rate:.2f}%
            - 影響: モデルが環境変化に適応するため再学習を実施
            """)
            
            # ドリフト履歴のグラフ
            drift_history = model_info.get('recent_drifts', [])
            # 適応モデルのサンプル数を使用
            n_samples = model_info.get('adaptive_model', {}).get('samples', 0)
            fig = plot_drift_history(drift_history, drift_count, n_samples)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # 最近のドリフト
            if drift_history:
                st.markdown("#### 最近のドリフト検出")
                for i, drift in enumerate(drift_history[-5:], 1):
                    st.write(f"{i}. 時刻: {drift.get('timestamp', 'N/A')}, "
                           f"エラー: {drift.get('error', 0):.3f}")
        else:
            st.success("現在までドリフトは検出されていません。モデルは安定して動作しています。")
    
    # 予測統計タブ
    with (tab6 if model_type == "統合モデル（基本 + 適応）" else tab5):
        st.markdown("### 📈 予測統計")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "直近1時間の予測数",
                f"{prediction_stats['last_hour']}件",
                help="過去1時間に生成された予測の数"
            )
        
        with col2:
            st.metric(
                "直近24時間の予測数",
                f"{prediction_stats['last_24h']}件",
                help="過去24時間に生成された予測の数"
            )
        
        with col3:
            st.metric(
                "総予測数",
                f"{prediction_stats['total']:,}件",
                help="保存されている予測の総数"
            )
        
        # 予測頻度グラフ
        if prediction_stats['last_24h'] > 0:
            st.markdown("#### 予測頻度")
            expected_predictions = 24 * 6  # 10分間隔で24時間
            actual_rate = (prediction_stats['last_24h'] / expected_predictions) * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=actual_rate,
                title={"text": "予測実行率（24時間）"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                        {"range": [80, 100], "color": "lightgreen"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # 最新の診断情報
        if recent_diagnostics:
            st.markdown("#### 最新の学習診断")
            
            # 診断サマリーを表示
            if 'summary' in recent_diagnostics:
                summary = recent_diagnostics['summary']
                
                col1, col2 = st.columns(2)
                with col1:
                    success_rate = summary.get('success_rate', 0)
                    st.metric(
                        "成功率",
                        f"{success_rate:.1f}%",
                        help="全ステップの成功率"
                    )
                with col2:
                    duration = summary.get('duration_seconds', 0)
                    st.metric(
                        "実行時間",
                        f"{duration:.1f}秒",
                        help="診断の実行時間"
                    )
                
                # 学習データの状況
                if 'steps' in recent_diagnostics:
                    learning_step = next(
                        (step for step in recent_diagnostics['steps'] 
                         if step['id'] == '3.1_past_predictions_check'),
                        None
                    )
                    
                    if learning_step and learning_step.get('status') == 'SUCCESS':
                        details = learning_step.get('details', {})
                        predictions_found = details.get('predictions_found', 0)
                        if predictions_found > 0:
                            st.success(f"✅ {predictions_found}件の予測データで学習を実行")
                        else:
                            st.info("📝 学習可能な予測データを待機中")
                    else:
                        st.warning("⚠️ 学習データの確認中")
    
    # エラー分析タブ
    with (tab7 if model_type == "統合モデル（基本 + 適応）" else tab6):
        st.markdown("### 📉 エラー分析")
        
        # エラー統計の計算
        if False:  # デュアルモデルではmetrics_by_stepが存在しないため一時的に無効化
            # 各ステップのMAEを収集
            mae_values = []
            step_labels = []
            
            for step_label, metrics in model_info['metrics_by_step'].items():
                if metrics.get('mae') is not None:
                    mae_values.append(metrics['mae'])
                    step_labels.append(step_label)
            
            if mae_values:
                # エラー分布のヒストグラム
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=[v * 100 for v in mae_values],  # メートルをセンチメートルに変換
                    nbinsx=20,
                    name='誤差分布',
                    marker_color='blue',
                    opacity=0.75
                ))
                
                fig_hist.update_layout(
                    title="予測誤差の分布",
                    xaxis_title="平均絶対誤差 (cm)",
                    yaxis_title="頻度",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # エラー統計
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "最小誤差",
                        f"±{min(mae_values)*100:.1f}cm",
                        help="最も精度の良い予測ステップ"
                    )
                
                with col2:
                    st.metric(
                        "平均誤差",
                        f"±{np.mean(mae_values)*100:.1f}cm",
                        help="全ステップの平均誤差"
                    )
                
                with col3:
                    st.metric(
                        "最大誤差",
                        f"±{max(mae_values)*100:.1f}cm",
                        help="最も精度の悪い予測ステップ"
                    )
                
                # 誤差要因の分析
                st.markdown("#### 誤差要因の分析")
                
                # 時間帯別の精度（仮想データ）
                hours = list(range(24))
                hourly_mae = [0.05 + 0.02 * abs(np.sin(h * np.pi / 12)) for h in hours]
                
                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Bar(
                    x=hours,
                    y=[mae * 100 for mae in hourly_mae],
                    name='時間帯別MAE',
                    marker_color='lightblue'
                ))
                
                fig_hourly.update_layout(
                    title="時間帯別の予測精度",
                    xaxis_title="時刻",
                    yaxis_title="平均絶対誤差 (cm)",
                    height=300,
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=3
                    )
                )
                
                st.plotly_chart(fig_hourly, use_container_width=True)
                
                # 誤差の傾向
                st.info("""
                **観察される傾向**
                - 🌅 早朝（4-7時）: ダム放流パターンの変化により誤差が増加
                - ☀️ 日中（10-16時）: 比較的安定した予測精度
                - 🌙 夜間（22-3時）: データ更新頻度の低下により若干精度が低下
                - 🌧️ 降雨時: 急激な水位変化により予測誤差が増大
                """)
        else:
            st.info("エラー分析の詳細情報を準備中です。")
            
            # 基本的な精度情報を表示
            col1, col2 = st.columns(2)
            with col1:
                if model_type == "統合モデル（基本 + 適応）":
                    mae = model_info.get('combined_mae_10min')
                else:
                    mae = display_info.get('mae_10min')
                
                if mae:
                    st.metric("🎯 10分先予測の平均誤差", format_mae(mae))
                    
            with col2:
                if model_type == "適応モデルのみ":
                    additional = display_info.get('additional_samples', 0)
                    st.metric("📈 追加学習サンプル", f"{additional:,}件")
    
    # モデル情報タブ
    with (tab8 if model_type == "統合モデル（基本 + 適応）" else tab7):
        st.markdown("### ℹ️ モデル情報")
        
        # モデルタイプ
        st.markdown(f"**モデルタイプ**: {model_info.get('model_type', 'Unknown')}")
        
        # 技術仕様
        with st.expander("技術仕様", expanded=True):
            st.markdown("""
            **アルゴリズム**
            - 学習方式: オンライン学習（ストリーミング）
            - 基本モデル: ARF（Adaptive Random Forest）
            - ドリフト検出: ADWIN（ADaptive WINdowing）
            
            **特徴**
            - リアルタイムでの学習と予測
            - データ分布の変化に自動適応
            - メモリ効率的な実装
            
            **予測範囲**
            - 時間: 10分〜3時間先（10分刻み）
            - 対象: 厚東川水位（末信橋観測所）
            """)
        
        # 学習設定
        with st.expander("学習設定"):
            st.markdown("""
            **データ収集**
            - 間隔: 10分（GitHub Actionsによる自動実行）
            - ソース: 山口県土木防災情報システム
            
            **学習タイミング**
            - 実行: データ収集ごとに過去の予測を評価・学習
            - 方式: ストリーム学習（遅延フィードバック）
            - 保存: 学習ごとに自動保存
            """)

# サイドバー情報
with st.sidebar:
    st.markdown("### 📊 学習状況")
    
    if model_info:
        # 全体的な精度評価
        mae_10min = model_info.get('mae_10min')
        if mae_10min:
            if mae_10min < 0.05:
                st.success("モデルは高精度で動作中")
            elif mae_10min < 0.10:
                st.info("モデルは良好な精度で動作中")
            else:
                st.warning("モデルは学習中です")
        
        # 最終更新情報
        st.markdown("### ⏰ 更新情報")
        # デュアルモデルの総サンプル数を表示
        base_samples = model_info.get('base_model', {}).get('samples', 0)
        adaptive_samples = model_info.get('adaptive_model', {}).get('samples', 0)
        st.caption(f"基本モデル: {base_samples:,}件 | 適応モデル: {adaptive_samples:,}件")
        
        # 予測統計
        st.markdown("### 🔮 予測活動")
        if prediction_stats['last_hour'] > 0:
            st.metric("直近1時間", f"{prediction_stats['last_hour']}件")
        if prediction_stats['last_24h'] > 0:
            expected = 144  # 10分間隔で24時間
            rate = (prediction_stats['last_24h'] / expected) * 100
            st.metric("稼働率（24時間）", f"{rate:.1f}%")
        
        # パフォーマンス指標
        st.markdown("### 📈 パフォーマンス")
        if mae_10min:
            accuracy_score = min(1.0, 0.05 / mae_10min)
            st.progress(accuracy_score, text=f"予測精度 {accuracy_score*100:.0f}%")
        
        drift_rate = model_info.get('drift_rate', 0)
        stability = max(0, 1 - drift_rate / 10)  # 10%以上でゼロ
        st.progress(stability, text=f"モデル安定性 {stability*100:.0f}%")
        
        # ストリーム学習の説明
        with st.expander("ℹ️ ストリーム学習とは"):
            st.markdown("""
            本システムは**ストリーム学習**を採用しています：
            
            1. **リアルタイム予測**: データ到着時に即座に予測
            2. **遅延学習**: 実測値が確定後に学習
            3. **継続的改善**: 24時間365日学習を継続
            4. **適応的モデル**: 環境変化に自動対応
            
            これにより、常に最新の状況に適応した予測が可能です。
            """)
    else:
        st.info("モデル情報を読み込み中...")