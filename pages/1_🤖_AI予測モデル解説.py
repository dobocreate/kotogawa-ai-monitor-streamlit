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
    from scripts.river_online_prediction import RiverOnlinePredictor
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# ページ設定
st.set_page_config(
    page_title="AI予測モデル解説 - 厚東川監視システム",
    page_icon="🤖",
    layout="wide"
)

def main():
    """メイン処理"""
    st.title("🤖 AI予測モデル解説")
    st.markdown("厚東川水位予測システムで使用される2つのAIモデルについて詳しく説明します。")
    
    # タブで2つのモデルを切り替え
    tab1, tab2, tab3 = st.tabs(["エキスパートルール予測", "Riverオンライン学習予測", "モデル比較"])
    
    with tab1:
        show_expert_rule_explanation()
    
    with tab2:
        show_river_online_explanation()
    
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

def show_river_online_explanation():
    """Riverオンライン学習予測の解説"""
    st.header("Riverオンライン学習予測")
    
    st.markdown("""
    Riverオンライン学習予測は、機械学習ライブラリ「River」を使用した適応型予測モデルです。
    データから継続的に学習し、時間とともに予測精度を向上させます。
    """)
    
    # 主な特徴
    with st.expander("🎯 主な特徴", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **オンライン学習**
            - 新しいデータが来るたびに継続的に学習
            - 過去の全データを保持する必要がない
            - メモリ効率的な学習方式
            - 時間とともに精度が向上
            """)
            
            st.markdown("""
            **マルチステップ予測**
            - 各予測ステップ（10分間隔）に個別のモデル
            - 18個のモデルで3時間先まで予測
            - ステップごとに最適化された予測
            """)
        
        with col2:
            st.markdown("""
            **豊富な特徴量**
            - 水位：現在値、遅延値、変化率、統計量
            - 放流量：現在値、40分前、平均、変化率
            - 雨量：現在値、累積値、最大値
            - 時間：時刻の周期性（sin/cos変換）
            """)
            
            st.markdown("""
            **適応的な学習**
            - SGD（確率的勾配降下法）による最適化
            - L2正則化で過学習を防止
            - 学習率：0.01（適度な学習速度）
            - StandardScalerによる特徴量正規化
            """)
    
    # 学習プロセス
    with st.expander("📚 学習プロセス", expanded=False):
        st.markdown("""
        ### 1. 特徴量の準備
        - 過去24個（4時間分）のデータから特徴量を抽出
        - 水位、放流量、雨量の様々な統計量を計算
        - 時間的特徴を周期関数で表現
        
        ### 2. オンライン学習
        ```python
        for each new_data:
            features = prepare_features(history)
            prediction = model.predict_one(features)
            model.learn_one(features, actual_value)
        ```
        
        ### 3. モデルの保存
        - 学習済みモデルはpickle形式で保存
        - 次回起動時に自動的に読み込み
        - 継続的な学習履歴を保持
        
        ### 4. 性能評価
        - MAE（平均絶対誤差）
        - RMSE（二乗平均平方根誤差）
        - ステップごとの性能を個別に追跡
        """)
    
    # 使用される特徴量の詳細
    with st.expander("📊 使用される特徴量", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **水位関連**
            - current_level
            - level_lag_1, 3, 6
            - level_change_10min
            - level_change_30min
            - level_change_1h
            - level_mean_1h
            - level_std_1h
            - level_max_1h
            - level_min_1h
            """)
        
        with col2:
            st.markdown("""
            **放流量関連**
            - current_outflow
            - outflow_lag_4
            - outflow_mean_1h
            - outflow_change_1h
            - outflow_max_1h
            """)
        
        with col3:
            st.markdown("""
            **その他**
            - rainfall_current
            - rainfall_sum_3h
            - rainfall_max_3h
            - hour
            - hour_sin
            - hour_cos
            - prediction_step
            - prediction_minutes
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
        "Riverオンライン学習": [
            "機械学習（線形回帰）",
            "⭐⭐ 低い",
            "⭐⭐⭐⭐⭐ 向上する",
            "⭐⭐⭐⭐⭐ 適応的",
            "⭐⭐ 低い",
            "⭐⭐⭐⭐ 高速",
            "⭐⭐⭐ 中程度",
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
        st.info("**Riverオンライン学習予測を選ぶべき場合**")
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
    - Riverオンライン学習予測を補助的に使用し、学習の進捗を確認
    - 両モデルの予測が大きく異なる場合は、慎重な判断が必要
    - 時間の経過とともにRiverモデルの信頼性が向上することを期待
    """)
    
    # 今後の展望
    with st.expander("🔮 今後の展望", expanded=False):
        st.markdown("""
        ### 機能拡張の可能性
        
        **エキスパートルール予測**
        - 季節変動の考慮
        - 潮位の影響モデル化
        - より詳細な雨量予測の統合
        
        **Riverオンライン学習予測**
        - 深層学習モデルの導入
        - アンサンブル学習
        - 異常検知機能の追加
        
        **共通機能**
        - 予測の不確実性の可視化
        - 複数地点の同時予測
        - 外部気象データとの連携
        """)

if __name__ == "__main__":
    main()