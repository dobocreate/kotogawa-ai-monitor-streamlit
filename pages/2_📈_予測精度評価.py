"""
AI予測精度評価ページ
各予測モデルの精度を評価・比較します
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json

# スクリプトのディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

# モジュールのインポート
try:
    from scripts.prediction_evaluator import PredictionEvaluator
    from scripts.advanced_prediction import AdvancedRiverLevelPredictor
    from scripts.river_streaming_prediction import RiverStreamingPredictor
    from scripts.collect_data import DataCollector
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    
# ページ設定
st.set_page_config(
    page_title="予測精度評価 - 厚東川監視システム",
    page_icon="📈",
    layout="wide"
)

def load_evaluation_data():
    """評価データの読み込み"""
    evaluator = PredictionEvaluator()
    
    # 保存済みデータがあれば読み込み
    eval_file = "data/evaluation_results.json"
    if Path(eval_file).exists():
        evaluator.load_from_file(eval_file)
        
    return evaluator

def run_backtest(evaluator: PredictionEvaluator, history_data: list, 
                start_time: datetime, end_time: datetime):
    """バックテストの実行"""
    
    # 進捗表示
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 予測器の初期化
    expert_predictor = AdvancedRiverLevelPredictor()
    river_predictor = RiverStreamingPredictor()
    
    # テスト期間のデータを抽出
    test_data = []
    for data in history_data:
        data_time = datetime.fromisoformat(data.get('data_time', data.get('timestamp')))
        if start_time <= data_time <= end_time:
            test_data.append(data)
            
    if len(test_data) < 50:
        st.error("テストデータが不足しています（最低50件必要）")
        return
        
    # 10分間隔でテスト
    total_tests = len(test_data) - 30  # 予測に必要な履歴を確保
    completed_tests = 0
    
    for i in range(30, len(test_data) - 18):  # 3時間先までの予測が可能な範囲
        # 履歴データ
        history = test_data[:i]
        
        # 実測データ（3時間先まで）
        actual = test_data[i:i+18]
        
        # エキスパートルール予測
        status_text.text(f"エキスパートルール予測を評価中... ({i-29}/{total_tests})")
        expert_predictions = expert_predictor.predict(history)
        if expert_predictions:
            evaluator.evaluate_prediction(expert_predictions, actual, 'expert_rule')
            
        # Riverオンライン学習予測
        status_text.text(f"Riverオンライン学習予測を評価中... ({i-29}/{total_tests})")
        # 学習を実行
        if i > 50:  # 十分なデータがある場合のみ学習
            river_predictor.learn(history[:-20])  # 最新20件を除いて学習
            
        river_predictions = river_predictor.predict(history)
        if river_predictions:
            evaluator.evaluate_prediction(river_predictions, actual, 'river_online')
            
        # 進捗更新
        completed_tests += 1
        progress_bar.progress(completed_tests / total_tests)
        
    # 評価結果を保存
    evaluator.save_to_file("data/evaluation_results.json")
    
    progress_bar.empty()
    status_text.text("評価完了！")
    
    return evaluator

def main():
    """メイン処理"""
    st.title("📈 AI予測精度評価")
    st.markdown("各予測モデルの精度を評価・比較します。")
    
    if not MODULES_AVAILABLE:
        st.error("必要なモジュールが読み込めません。")
        return
        
    # 評価データの読み込み
    evaluator = load_evaluation_data()
    
    # タブ構成
    tab1, tab2, tab3, tab4 = st.tabs(["リアルタイム評価", "バックテスト", "詳細比較", "性能トレンド"])
    
    with tab1:
        show_realtime_evaluation(evaluator)
        
    with tab2:
        show_backtest(evaluator)
        
    with tab3:
        show_detailed_comparison(evaluator)
        
    with tab4:
        show_performance_trend(evaluator)
        
def show_realtime_evaluation(evaluator: PredictionEvaluator):
    """リアルタイム評価の表示"""
    st.header("リアルタイム評価")
    
    # 最新の評価結果を取得
    summary = evaluator.get_comparison_summary()
    
    if not summary or not any(data.get('latest') for data in summary.values()):
        st.info("まだ評価データがありません。バックテストを実行するか、システムを稼働させてデータを蓄積してください。")
        return
        
    # モデル比較カード
    col1, col2 = st.columns(2)
    
    with col1:
        show_model_card("エキスパートルール予測", summary.get('expert_rule', {}), "blue")
        
    with col2:
        show_model_card("Riverオンライン学習予測", summary.get('river_online', {}), "green")
        
    # 詳細メトリクス
    st.subheader("📊 詳細メトリクス")
    
    # 比較表
    comparison_df = evaluator.get_detailed_comparison()
    if not comparison_df.empty:
        # ピボットテーブルで見やすく表示
        pivot_df = comparison_df.pivot(index='予測時間', columns='モデル', values=['MAE (m)', 'RMSE (m)', 'MAPE (%)'])
        st.dataframe(pivot_df, use_container_width=True)
        
        # グラフ表示
        fig = px.line(comparison_df, x='予測時間', y='MAE (m)', 
                     color='モデル', markers=True,
                     title="予測時間別の平均絶対誤差（MAE）")
        st.plotly_chart(fig, use_container_width=True)
        
def show_model_card(model_name: str, model_data: dict, color: str):
    """モデル評価カードの表示"""
    latest = model_data.get('latest', {})
    average = model_data.get('average', {})
    
    # カードのスタイル
    card_style = f"""
    <div style="
        background-color: {'#e3f2fd' if color == 'blue' else '#e8f5e9'};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {color};
    ">
    """
    
    st.markdown(card_style, unsafe_allow_html=True)
    st.subheader(model_name)
    
    if latest and 'overall' in latest:
        overall = latest['overall']
        
        # スコア表示（大きく）
        score = overall.get('score', 0)
        score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
        st.markdown(f"### 精度スコア: <span style='color: {score_color}; font-size: 36px;'>{score:.1f}</span> / 100", 
                   unsafe_allow_html=True)
        
        # メトリクス表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", f"{overall['mae']:.3f} m", 
                     help="平均絶対誤差（低いほど良い）")
            
        with col2:
            st.metric("RMSE", f"{overall['rmse']:.3f} m",
                     help="二乗平均平方根誤差（低いほど良い）")
            
        with col3:
            st.metric("MAPE", f"{overall['mape']:.1f} %",
                     help="平均絶対パーセント誤差（低いほど良い）")
            
        # 平均値との比較
        if average:
            st.markdown("**最近10回の平均:**")
            st.text(f"MAE: {average['mae']:.3f} m, RMSE: {average['rmse']:.3f} m, スコア: {average['score']:.1f}")
            
        # 評価情報
        eval_time = datetime.fromisoformat(latest['evaluated_at'])
        st.caption(f"最終評価: {eval_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"評価回数: {model_data.get('evaluation_count', 0)}回")
        
    else:
        st.info("評価データがありません")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
def show_backtest(evaluator: PredictionEvaluator):
    """バックテストの実行と表示"""
    st.header("バックテスト")
    st.markdown("""
    過去のデータを使用して、各モデルの予測精度を評価します。
    デモモードのデータ（2023年6月25日〜7月2日）でテストすることができます。
    """)
    
    # バックテスト設定
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_start = st.date_input(
            "開始日",
            value=datetime(2023, 6, 26).date(),
            min_value=datetime(2023, 6, 25).date(),
            max_value=datetime(2023, 7, 1).date()
        )
        
    with col2:
        test_end = st.date_input(
            "終了日",
            value=datetime(2023, 6, 28).date(),
            min_value=datetime(2023, 6, 25).date(),
            max_value=datetime(2023, 7, 1).date()
        )
        
    with col3:
        st.write("")  # スペーサー
        if st.button("バックテスト実行", type="primary"):
            # データ読み込み
            with st.spinner("過去データを読み込み中..."):
                history_data = load_demo_data()
                
            if history_data:
                # バックテスト実行
                start_dt = datetime.combine(test_start, datetime.min.time())
                end_dt = datetime.combine(test_end, datetime.max.time())
                
                updated_evaluator = run_backtest(evaluator, history_data, start_dt, end_dt)
                
                # 結果を表示
                st.success("バックテスト完了！")
                st.rerun()
            else:
                st.error("デモデータの読み込みに失敗しました")
                
def load_demo_data():
    """デモデータの読み込み"""
    collector = DataCollector()
    
    # CSVファイルから読み込み
    water_df = pd.read_csv('sample/water-level_20230625-20230701.csv')
    dam_df = pd.read_csv('sample/dam_20230625-20230701.csv')
    
    # データ形式を変換
    history_data = []
    
    for idx, row in water_df.iterrows():
        timestamp = pd.to_datetime(row['日時']).isoformat()
        
        # 対応するダムデータを探す
        dam_row = dam_df[dam_df['日時'] == row['日時']]
        
        data = {
            'data_time': timestamp,
            'river': {
                'water_level': float(row['水位(m)']) if pd.notna(row['水位(m)']) else None
            }
        }
        
        if not dam_row.empty:
            data['dam'] = {
                'outflow': float(dam_row.iloc[0]['全放流量(m3/s)']) if pd.notna(dam_row.iloc[0]['全放流量(m3/s)']) else None
            }
            
        history_data.append(data)
        
    return history_data
    
def show_detailed_comparison(evaluator: PredictionEvaluator):
    """詳細比較の表示"""
    st.header("詳細比較")
    
    summary = evaluator.get_comparison_summary()
    
    if not summary:
        st.info("評価データがありません")
        return
        
    # 予測ステップごとの比較
    st.subheader("予測ステップ別精度")
    
    # データ準備
    step_data = []
    for model_type, model_name in [('expert_rule', 'エキスパートルール'), 
                                   ('river_online', 'Riverオンライン学習')]:
        if model_type in summary and summary[model_type].get('latest'):
            latest = summary[model_type]['latest']
            if 'by_step' in latest:
                for step, metrics in latest['by_step'].items():
                    minutes = int(step.replace('min', ''))
                    step_data.append({
                        'モデル': model_name,
                        '予測時間（分）': minutes,
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse'],
                        'MAPE': metrics['mape']
                    })
                    
    if step_data:
        step_df = pd.DataFrame(step_data)
        
        # MAEの比較グラフ
        fig1 = px.line(step_df, x='予測時間（分）', y='MAE', color='モデル',
                      title='予測時間別 平均絶対誤差（MAE）',
                      markers=True)
        fig1.update_yaxis(title='MAE (m)')
        st.plotly_chart(fig1, use_container_width=True)
        
        # RMSEの比較グラフ
        fig2 = px.line(step_df, x='予測時間（分）', y='RMSE', color='モデル',
                      title='予測時間別 二乗平均平方根誤差（RMSE）',
                      markers=True)
        fig2.update_yaxis(title='RMSE (m)')
        st.plotly_chart(fig2, use_container_width=True)
        
        # MAPEの比較グラフ
        fig3 = px.line(step_df, x='予測時間（分）', y='MAPE', color='モデル',
                      title='予測時間別 平均絶対パーセント誤差（MAPE）',
                      markers=True)
        fig3.update_yaxis(title='MAPE (%)')
        st.plotly_chart(fig3, use_container_width=True)
        
    # 統計的分析
    st.subheader("統計的分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'expert_rule' in summary and summary['expert_rule'].get('latest'):
            latest = summary['expert_rule']['latest']['overall']
            st.markdown("**エキスパートルール予測**")
            st.write(f"- バイアス: {latest.get('bias', 0):.3f} m")
            st.write(f"- 標準偏差: {latest.get('std', 0):.3f} m")
            st.write(f"- サンプル数: {latest.get('count', 0)}")
            
    with col2:
        if 'river_online' in summary and summary['river_online'].get('latest'):
            latest = summary['river_online']['latest']['overall']
            st.markdown("**Riverオンライン学習予測**")
            st.write(f"- バイアス: {latest.get('bias', 0):.3f} m")
            st.write(f"- 標準偏差: {latest.get('std', 0):.3f} m")
            st.write(f"- サンプル数: {latest.get('count', 0)}")
            
def show_performance_trend(evaluator: PredictionEvaluator):
    """性能トレンドの表示"""
    st.header("性能トレンド")
    
    # メトリクス選択
    metric = st.selectbox(
        "評価指標",
        ["mae", "rmse", "score"],
        format_func=lambda x: {"mae": "MAE", "rmse": "RMSE", "score": "精度スコア"}[x]
    )
    
    # トレンドデータ取得
    trend_data = []
    for model_type, model_name in [('expert_rule', 'エキスパートルール'), 
                                   ('river_online', 'Riverオンライン学習')]:
        df = evaluator.get_performance_trend(model_type, metric)
        if not df.empty:
            df['モデル'] = model_name
            trend_data.append(df)
            
    if trend_data:
        # データ結合
        trend_df = pd.concat(trend_data, ignore_index=True)
        
        # グラフ作成
        fig = px.line(trend_df, x='datetime', y=metric, color='モデル',
                     title=f'{metric.upper()}の推移',
                     markers=True)
        
        # Y軸のラベル
        if metric == 'mae':
            fig.update_yaxis(title='MAE (m)')
        elif metric == 'rmse':
            fig.update_yaxis(title='RMSE (m)')
        else:
            fig.update_yaxis(title='精度スコア')
            
        fig.update_xaxis(title='評価日時')
        st.plotly_chart(fig, use_container_width=True)
        
        # 統計サマリー
        st.subheader("統計サマリー")
        summary_df = trend_df.groupby('モデル')[metric].agg(['mean', 'std', 'min', 'max']).round(3)
        summary_df.columns = ['平均', '標準偏差', '最小', '最大']
        st.dataframe(summary_df, use_container_width=True)
        
    else:
        st.info("トレンドデータがありません。継続的に評価を実行してデータを蓄積してください。")

if __name__ == "__main__":
    main()