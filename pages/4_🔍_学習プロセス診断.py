"""
学習プロセス診断ページ
モデルの学習ステップを可視化し、問題の特定を支援
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8以前の場合
    from pytz import timezone as pytz_timezone
    ZoneInfo = lambda x: pytz_timezone(x)

# ページ設定
st.set_page_config(
    page_title="学習プロセス診断",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 学習プロセス診断")
st.markdown("モデルの学習プロセスを詳細に分析し、問題の特定を支援します。")


def to_jst(dt_str):
    """ISO形式の日時文字列を日本時間に変換"""
    if not dt_str:
        return None
    
    dt = datetime.fromisoformat(dt_str)
    
    # タイムゾーンがない場合はUTCとして扱う
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # JSTに変換
    jst_offset = timedelta(hours=9)
    jst_tz = timezone(jst_offset)
    dt_jst = dt.astimezone(jst_tz)
    
    return dt_jst


def load_latest_diagnostics():
    """最新の診断結果を読み込み"""
    diagnostics_dir = Path('diagnostics')
    if not diagnostics_dir.exists():
        return None
    
    # 最新のファイルを探す
    files = list(diagnostics_dir.glob('learning_diagnostics_*.json'))
    if not files:
        return None
    
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"診断結果ファイルの読み込みエラー: {latest_file.name}")
        st.error(f"エラー詳細: {str(e)}")
        # 破損したファイルを削除または移動することを検討
        return None
    except Exception as e:
        st.error(f"予期しないエラー: {str(e)}")
        return None


def load_all_diagnostics():
    """すべての診断履歴を読み込み"""
    diagnostics_dir = Path('diagnostics')
    if not diagnostics_dir.exists():
        return []
    
    results = []
    for file in diagnostics_dir.glob('learning_diagnostics_*.json'):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['filename'] = file.name
                results.append(data)
        except:
            continue
    
    return sorted(results, key=lambda x: x['summary']['start_time'], reverse=True)


def display_phase_status(phase_name, phase_data):
    """フェーズのステータスを表示"""
    status_colors = {
        "✅ 成功": "green",
        "❌ 失敗": "red",
        "⚠️ 警告": "orange",
        "⏳ 実行中": "blue",
        "⏸️ 未実行": "gray",
        "⏭️ スキップ": "gray"
    }
    
    # フェーズ全体のステータスを判定
    phase_status = phase_data.get('status', '⏸️ 未実行')
    color = status_colors.get(phase_status, "gray")
    
    with st.expander(f"{phase_status} **{phase_name}**", expanded=phase_status in ["❌ 失敗", "⚠️ 警告"]):
        # ステップのテーブルを作成
        steps_data = []
        for step in phase_data.get('steps', []):
            steps_data.append({
                'ステップ': step['name'],
                'ステータス': step['status_text'],
                '詳細': json.dumps(step.get('details', {}), ensure_ascii=False)[:100] + '...' if step.get('details') else '',
                'エラー': step.get('error', {}).get('message', '') if 'error' in step else ''
            })
        
        if steps_data:
            df = pd.DataFrame(steps_data)
            st.dataframe(df, use_container_width=True, hide_index=True)


def display_diagnostics_summary(diagnostics):
    """診断結果のサマリーを表示"""
    summary = diagnostics['summary']
    
    # メトリクスの表示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("全体ステータス", summary['overall_status'])
    
    with col2:
        success_count = summary['status_counts'].get('✅ 成功', 0)
        total_steps = summary['total_steps']
        success_rate = (success_count / total_steps * 100) if total_steps > 0 else 0
        st.metric("成功率", f"{success_rate:.1f}%", f"{success_count}/{total_steps}")
    
    with col3:
        duration = summary.get('duration_seconds')
        if duration:
            st.metric("実行時間", f"{duration:.1f}秒")
        else:
            st.metric("実行時間", "実行中")
    
    with col4:
        failed_count = summary['status_counts'].get('❌ 失敗', 0)
        warning_count = summary['status_counts'].get('⚠️ 警告', 0)
        st.metric("問題数", f"{failed_count + warning_count}件", f"失敗:{failed_count} 警告:{warning_count}")
    
    # ステータスの円グラフ
    st.markdown("### ステップ実行状況")
    
    fig = go.Figure(data=[go.Pie(
        labels=list(summary['status_counts'].keys()),
        values=list(summary['status_counts'].values()),
        hole=.3,
        marker_colors=['green', 'red', 'orange', 'lightblue', 'gray', 'darkgray']
    )])
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # 失敗ステップの詳細
    if summary.get('failed_steps'):
        st.error(f"**失敗したステップ ({len(summary['failed_steps'])}件)**")
        for step in summary['failed_steps']:
            st.write(f"- **{step['name']}**: {step['error']}")
    
    # 警告ステップの詳細
    if summary.get('warning_steps'):
        st.warning(f"**警告のあるステップ ({len(summary['warning_steps'])}件)**")
        for step in summary['warning_steps']:
            st.write(f"- **{step['name']}**")


def display_performance_trends():
    """パフォーマンストレンドを表示"""
    all_diagnostics = load_all_diagnostics()
    
    if len(all_diagnostics) < 2:
        st.info("トレンド表示には2回以上の実行履歴が必要です")
        return
    
    # 成功率の推移
    timestamps = []
    success_rates = []
    durations = []
    
    for diag in all_diagnostics[-20:]:  # 最新20件
        summary = diag['summary']
        timestamps.append(to_jst(summary['start_time']))
        
        total = summary['total_steps']
        success = summary['status_counts'].get('✅ 成功', 0)
        success_rates.append((success / total * 100) if total > 0 else 0)
        
        duration = summary.get('duration_seconds')
        durations.append(duration if duration else None)
    
    # 成功率グラフ
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=timestamps,
        y=success_rates,
        mode='lines+markers',
        name='成功率',
        line=dict(color='green', width=2)
    ))
    
    fig1.update_layout(
        title="学習成功率の推移",
        xaxis_title="実行時刻",
        yaxis_title="成功率 (%)",
        yaxis=dict(range=[0, 105]),
        height=300
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # 実行時間グラフ
    if any(d is not None for d in durations):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=timestamps,
            y=durations,
            mode='lines+markers',
            name='実行時間',
            line=dict(color='blue', width=2)
        ))
        
        fig2.update_layout(
            title="実行時間の推移",
            xaxis_title="実行時刻",
            yaxis_title="実行時間 (秒)",
            height=300
        )
        
        st.plotly_chart(fig2, use_container_width=True)


def run_diagnostics_now():
    """今すぐ診断を実行"""
    with st.spinner("診断を実行中..."):
        import subprocess
        import sys
        
        # 診断付き学習スクリプトを実行
        result = subprocess.run(
            [sys.executable, "scripts/streaming_train_with_diagnostics.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            st.success("診断が完了しました")
            st.experimental_rerun()
        else:
            st.error("診断中にエラーが発生しました")
            st.code(result.stderr)


# メイン表示
tab1, tab2, tab3 = st.tabs(["最新の診断結果", "診断履歴", "手動実行"])

with tab1:
    diagnostics = load_latest_diagnostics()
    
    if not diagnostics:
        st.warning("診断結果がありません。学習が実行されるのを待つか、手動で診断を実行してください。")
    else:
        # 実行時刻
        start_time = to_jst(diagnostics['summary']['start_time'])
        st.info(f"最終実行: {start_time.strftime('%Y年%m月%d日 %H:%M:%S')} (JST)")
        
        # サマリー表示
        display_diagnostics_summary(diagnostics)
        
        # 各フェーズの詳細
        st.markdown("### 詳細な実行ステップ")
        
        phases = diagnostics.get('phases', {})
        for phase_id, phase_data in phases.items():
            display_phase_status(phase_data['name'], phase_data)

with tab2:
    st.markdown("### 実行履歴とトレンド")
    
    # パフォーマンストレンド
    display_performance_trends()
    
    # 履歴テーブル
    st.markdown("### 過去の実行履歴")
    
    all_diagnostics = load_all_diagnostics()
    
    if not all_diagnostics:
        st.info("履歴データがありません")
    else:
        history_data = []
        for diag in all_diagnostics[:50]:  # 最新50件
            summary = diag['summary']
            history_data.append({
                '実行時刻': to_jst(summary['start_time']).strftime('%Y-%m-%d %H:%M'),
                'ステータス': summary['overall_status'],
                '成功率': f"{summary['status_counts'].get('✅ 成功', 0) / summary['total_steps'] * 100:.1f}%",
                '実行時間': f"{summary.get('duration_seconds', 0):.1f}秒" if summary.get('duration_seconds') else "N/A",
                '失敗数': summary['status_counts'].get('❌ 失敗', 0),
                '警告数': summary['status_counts'].get('⚠️ 警告', 0)
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 手動診断実行")
    st.markdown("学習プロセスの診断を手動で実行します。")
    
    st.warning("注意: 診断実行には数秒〜数分かかる場合があります。")
    
    if st.button("🔍 診断を実行", type="primary"):
        run_diagnostics_now()
    
    # 次回の自動実行予定
    st.markdown("### 自動実行スケジュール")
    st.info("学習と診断は、GitHub Actionsにより1時間ごとに自動実行されます。")
    
    # JSTで次回実行時刻を計算
    jst_offset = timedelta(hours=9)
    jst_tz = timezone(jst_offset)
    next_run = datetime.now(jst_tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    st.write(f"次回実行予定: {next_run.strftime('%Y年%m月%d日 %H:%M')} (JST)")


# サイドバーに簡易ステータス表示
with st.sidebar:
    st.markdown("### 🔍 診断ステータス")
    
    latest = load_latest_diagnostics()
    if latest:
        summary = latest['summary']
        
        # ステータスバッジ
        status_emoji = {
            "成功": "✅",
            "失敗": "❌",
            "警告あり": "⚠️",
            "実行中": "⏳"
        }
        
        status = summary['overall_status']
        emoji = status_emoji.get(status, "❓")
        
        st.markdown(f"### {emoji} {status}")
        
        # 簡易メトリクス
        st.metric("成功ステップ", summary['status_counts'].get('✅ 成功', 0))
        st.metric("失敗ステップ", summary['status_counts'].get('❌ 失敗', 0))
        st.metric("警告ステップ", summary['status_counts'].get('⚠️ 警告', 0))
        
        # 最終更新
        start_time = to_jst(summary['start_time'])
        jst_offset = timedelta(hours=9)
        jst_tz = timezone(jst_offset)
        elapsed = datetime.now(jst_tz) - start_time
        
        if elapsed.total_seconds() < 3600:
            time_str = f"{int(elapsed.total_seconds() / 60)}分前"
        elif elapsed.total_seconds() < 86400:
            time_str = f"{int(elapsed.total_seconds() / 3600)}時間前"
        else:
            time_str = f"{int(elapsed.total_seconds() / 86400)}日前"
        
        st.caption(f"最終更新: {time_str}")
    else:
        st.info("診断データなし")