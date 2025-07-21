"""
AI予測モデル解説ページ
厚東川水位予測システムで使用される2つのAIモデルについて詳しく説明します
"""

import streamlit as st
from pathlib import Path
import sys

# スクリプトのディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

# 予測モジュールのインポート（情報取得用）
try:
    from scripts.advanced_prediction import AdvancedRiverLevelPredictor
    EXPERT_AVAILABLE = True
except ImportError:
    EXPERT_AVAILABLE = False

try:
    from scripts.river_streaming_prediction_v2 import RiverStreamingPredictor
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

MODELS_AVAILABLE = EXPERT_AVAILABLE or STREAMING_AVAILABLE

# ページ設定
st.set_page_config(
    page_title="AI予測モデル解説 - 厚東川監視システム",
    page_icon="🤖",
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

def main():
    """メイン処理"""
    st.title("🤖 AI予測モデル解説")
    st.markdown("厚東川水位予測システムで使用される2つのAIモデルについて詳しく説明します。")
    
    # タブで2つのモデルを切り替え
    tab1, tab2, tab3 = st.tabs(["エキスパートルール予測", "Riverストリーミング予測", "モデル比較"])
    
    # 現在の利用可能状況を表示
    with st.sidebar:
        st.markdown("### 📊 モデル利用可能状況")
        st.markdown(f"エキスパートルール: {'✅' if EXPERT_AVAILABLE else '❌'}")
        st.markdown(f"Riverストリーミング予測: {'✅' if STREAMING_AVAILABLE else '❌'}")
        
        if STREAMING_AVAILABLE:
            st.success("✅ Riverストリーミング予測が利用可能です")
            st.caption("動的遅延推定とストリーミング処理で高精度予測")
    
    with tab1:
        show_expert_rule_explanation()
    
    with tab2:
        show_river_streaming_explanation()
    
    with tab3:
        show_model_comparison()

def show_expert_rule_explanation():
    """エキスパートルール予測の解説"""
    st.header("エキスパートルール予測")
    
    st.markdown("""
    エキスパートルール予測は、河川工学の専門知識と物理法則に基づいて設計された予測モデルです。
    ダムと河川の関係性を詳細にモデル化し、高精度な予測を実現しています。
    """)
    
    # 主な特徴
    with st.expander("🎯 主な特徴", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **動的な時間遅延**
            - ダムから観測地点までの水の到達時間を考慮
            - 流量に応じて時間遅延を動的に調整（20〜40分）
            - 流量が多いほど到達時間が短くなる物理法則を反映
            """)
            
            st.markdown("""
            **滑らかな重み付け遷移**
            - 短期・中期・長期トレンドを時間に応じて滑らかに統合
            - シグモイド関数を使用した自然な遷移
            - 予測時間に応じて適切なトレンドを重視
            """)
        
        with col2:
            st.markdown("""
            **放流量の加速度考慮**
            - 放流量の変化率だけでなく、変化の変化（加速度）も考慮
            - 急激な放流量変化に対する応答性を向上
            - 2次の運動方程式を用いた物理的にな予測
            """)
            
            st.markdown("""
            **流量依存の係数調整**
            - 大流量時（500m³/s以上）は影響係数を増大
            - 水位が高い時の非線形な増幅効果を考慮
            - 実際の河川挙動に合わせた調整
            """)
    
    # アルゴリズムの詳細
    with st.expander("🔧 アルゴリズムの詳細", expanded=False):
        st.markdown("""
        ### 1. 時間遅延の計算
        ```python
        time_lag = max(20, 40 - (outflow / 100) * 2)
        ```
        - 基本遅延時間：40分
        - 100m³/sごとに2分短縮
        - 最小遅延時間：20分
        
        ### 2. 放流量による水位変化
        ```python
        outflow_impact = (future_outflow - current_outflow) * base_factor
        ```
        - 基本係数：0.003（通常時）
        - 大流量時（>500m³/s）：0.004
        - 超大流量時（>800m³/s）：0.005
        
        ### 3. 水位トレンドの統合
        - 30分以内：短期トレンド60%、中期30%、長期10%
        - 1時間以内：短期30%、中期50%、長期20%
        - 1時間以上：短期10%、中期30%、長期60%
        
        ### 4. 予測信頼度
        - 基本信頼度から以下を減算：
          - 時間経過による減少（最大30%）
          - 放流量の変動性によるペナルティ
          - 加速度によるペナルティ
          - 時間遅延の不確実性
        """)
    
    # 長所と短所
    col1, col2 = st.columns(2)
    with col1:
        st.success("**長所**")
        st.markdown("""
        - 物理法則に基づく解釈可能な予測
        - 学習データ不要で即座に利用可能
        - 急激な変化にも対応可能
        - 安定した予測性能
        """)
    
    with col2:
        st.warning("**短所**")
        st.markdown("""
        - 事前に定義されたルールに依存
        - 未知のパターンへの適応が困難
        - パラメータの手動調整が必要
        - 地域特性の学習ができない
        """)

def show_river_streaming_explanation():
    """Riverストリーミング学習予測の解説"""
    st.header("Riverストリーミング学習予測")
    
    st.markdown("""
    Riverストリーミング予測は、機械学習ライブラリ「River」を使用した適応型予測モデルです。
    データから継続的に学習し、時間とともに予測精度を向上させます。
    
    **最新バージョン（v2）では真のストリーム学習を実装：**
    - **遅延フィードバック学習**：予測時点と学習時点を分離し、実測値が確定後に学習
    - **予測結果の保存**：全ての予測を保存し、後で実測値と比較して学習
    - **ARFRegressor**：適応的ランダムフォレストによる高精度予測
    - **ADWINドリフト検出**：概念ドリフトを自動検知し、モデルを適応
    - **プログレッシブ検証**：各予測時点での精度を個別に評価
    """)
    
    
    # ストリーミング予測の特徴
    with st.expander("🎯 主な特徴", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **遅延フィードバック学習**
            - データ到着時に即座に予測を実行
            - 予測結果を保存（predictions/ディレクトリ）
            - 実測値が確定後（10分後）に学習
            - 過去の予測と実測値を比較して精度向上
            - River MLのベストプラクティスに準拠
            """)
            
            st.markdown("""
            **真のストリーミング処理**
            - 1件ずつのデータをリアルタイム処理
            - predict_one → learn_oneの順序を厳守
            - メモリ効率が非常に高い
            - 大量データでも安定動作
            """)
        
        with col2:
            st.markdown("""
            **シンプルな特徴量設計**
            - 基本特徴量：水位、放流量、雨量など
            - 時間特徴：時刻、曜日、昼夜フラグ
            - 変化率：各値の変化速度
            - 推定遅延時間も特徴量として使用
            """)
            
            st.markdown("""
            **評価と監視**
            - MAE/RMSEをリアルタイム追跡
            - 時間ステップ別の精度評価
            - ローリング窓での性能監視
            - ADWINによるドリフト検出
            """)
    
    
    # 学習プロセス
    with st.expander("📚 学習プロセス", expanded=False):
        st.markdown("""
        ### 遅延フィードバック学習の実装
        
        **ストリーム学習の正しい実装**
        1. **データ収集時に予測を実行**
           - 10分ごとにデータ収集（GitHub Actions）
           - 収集したデータで即座に予測
           - 予測結果をpredictions/に保存
        
        2. **遅延学習の実行**
           ```python
           # collect_data.pyでの予測
           data = collect_latest_data()
           predictions = predictor.predict_one(data)
           storage.save_predictions(time, features, predictions)
           
           # streaming_train_with_diagnostics.pyでの学習
           # 1. 現在のデータを取得
           current_data = get_latest_data()
           
           # 2. 過去の予測を検索（許容誤差5分）
           past_predictions = storage.get_predictions_for_learning(
               current_time, tolerance_minutes=5
           )
           
           # 3. 予測と実測値で学習
           for features, prediction in past_predictions:
               predictor.learn_one(features, actual_value)
           ```
        
        3. **Streamlit.ioとの連携**
           - Streamlit.ioは表示専用（読み取り専用）
           - 最新の学習済みモデルを自動的に読み込み
           - data_timestamp.pyの更新で自動リロード
        
        ### データフロー
        ```
        10:10 データ収集 → 予測実行 → 予測を保存
               ↓
        10:20 学習実行 → 10:10の予測を検索
               ↓         実測値(10:20)で学習
        モデル更新 → GitHubに保存
        ```
        
        ### 予測の保存と管理
        - predictions/YYYYMMDD/HH_predictions.json
        - 時刻、特徴量、予測値を保存
        - 3日以上古いファイルは自動削除
        - 高速な時刻ベース検索
        """)
    
    # 使用される特徴量
    with st.expander("📊 使用される特徴量", expanded=False):
        st.markdown("### シンプルで効果的な特徴量設計")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **基本情報**
            - water_level: 現在水位
            - dam_outflow: ダム放流量
            - dam_inflow: ダム流入量
            - storage_rate: 貯水率
            - rainfall_1h: 1時間降雨量
            """)
        
        with col2:
            st.markdown("""
            **時系列特徴**
            - water_level_diff: 水位変化量
            - outflow_diff: 放流量変化
            - hour: 時刻（0-23）
            - day_of_week: 曜日（0-6）
            - is_night: 夜間フラグ
            """)
    
    # 長所と短所
    col1, col2 = st.columns(2)
    with col1:
        st.success("**長所**")
        st.markdown("""
        - データから自動的に学習
        - 地域特性を捉えることが可能
        - 継続的な精度向上
        - 新しいパターンへの適応
        """)
    
    with col2:
        st.warning("**短所**")
        st.markdown("""
        - 初期は学習データが少なく精度が低い
        - 異常データの影響を受けやすい
        - 物理的な妥当性の保証がない
        - 解釈性が低い（ブラックボックス）
        """)

def show_model_comparison():
    """モデル比較"""
    st.header("モデル比較")
    
    st.markdown("""
    2つのモデルは異なるアプローチで水位予測を行います。
    それぞれの特性を理解して、状況に応じて適切なモデルを選択してください。
    """)
    
    # 比較表
    st.subheader("📊 特性比較")
    
    comparison_data = {
        "特性": ["予測手法", "初期精度", "長期精度", "適応性", "解釈性", "計算速度", "メモリ使用量", "異常値耐性"],
        "エキスパートルール": [
            "物理法則＋専門知識",
            "⭐⭐⭐⭐⭐ 高い",
            "⭐⭐⭐⭐ 安定",
            "⭐⭐ 固定的",
            "⭐⭐⭐⭐⭐ 非常に高い",
            "⭐⭐⭐⭐⭐ 高速",
            "⭐⭐⭐⭐⭐ 少ない",
            "⭐⭐⭐⭐ 高い"
        ],
        "Riverストリーミング予測": [
            "機械学習（ARFRegressor）",
            "⭐⭐ 低い→学習で向上",
            "⭐⭐⭐⭐⭐ 継続的に向上",
            "⭐⭐⭐⭐⭐ 適応的",
            "⭐⭐ 低い",
            "⭐⭐⭐⭐⭐ 非常に高速",
            "⭐⭐⭐⭐⭐ 極めて少ない",
            "⭐⭐⭐ 中程度"
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.table(df)
    
    # 使い分けガイド
    st.subheader("🎯 使い分けガイド")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**エキスパートルール予測を選ぶべき場合**")
        st.markdown("""
        - システム導入直後で学習データが少ない
        - 予測の根拠を明確に説明する必要がある
        - 安定した予測性能が求められる
        - 異常気象や特殊な状況での予測
        - リアルタイムの意思決定支援
        """)
    
    with col2:
        st.info("**Riverストリーミング予測を選ぶべき場合**")
        st.markdown("""
        - 十分な過去データが蓄積されている
        - 地域特有のパターンを学習させたい
        - 長期的な精度向上を期待する
        - 通常の気象条件での予測
        - 実験的な利用や精度検証
        """)
    
    # 併用の推奨
    st.subheader("💡 併用の推奨")
    st.success("""
    **両モデルの併用をお勧めします**
    
    - エキスパートルール予測を主として使用し、安定した予測を確保
    - Riverストリーミング予測を補助的に使用し、学習の進捗を確認
    - 両モデルの予測が大きく異なる場合は、慎重な判断が必要
    - 時間の経過とともにRiverモデルの信頼性が向上することを期待
    """)
    
    # 今後の展望
    with st.expander("🔮 今後の展望", expanded=False):
        st.markdown("""
        ### 実装済みの改良点
        
        **Riverストリーミング予測**
        - ✅ ARFRegressorによる高精度予測
        - ✅ ADWINドリフト検出器の実装
        - ✅ MAE/RMSEのリアルタイム追跡
        - ✅ **遅延フィードバック学習の実装**
        - ✅ **予測結果の保存と管理**
        - ✅ **プログレッシブ検証の実装**
        - ✅ **時間ステップ別精度評価**
        
        **システム全体**
        - ✅ 真のストリーム学習の実現
        - ✅ 学習プロセス診断機能
        - ✅ AI学習結果の詳細表示
        - ✅ 予測統計とエラー分析
        - ✅ **10分間隔での自動実行**
        
        ### 今後の機能拡張
        
        **エキスパートルール予測**
        - 季節変動の考慮
        - 潮位の影響モデル化
        - より詳細な雨量予測の統合
        
        **Riverストリーミング予測**
        - 深層学習モデルの統合
        - 不確実性の定量化
        - 異常検知機能の追加
        - 複数地点の相関学習
        
        **共通機能**
        - 予測の信頼区間表示
        - 複数地点の同時予測
        - 外部気象データとの連携
        - 長期予測（24時間以上）
        """)

if __name__ == "__main__":
    main()