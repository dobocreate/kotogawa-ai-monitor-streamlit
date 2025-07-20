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
    from scripts.river_streaming_prediction import RiverStreamingPredictor
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

def show_river_online_explanation():
    """Riverオンライン学習予測の解説"""
    st.header("Riverオンライン学習予測")
    
    st.markdown("""
    Riverオンライン学習予測は、機械学習ライブラリ「River」を使用した適応型予測モデルです。
    データから継続的に学習し、時間とともに予測精度を向上させます。
    
    現在、2つのバージョンが利用可能です：
    - **Riverストリーミング予測（新版、推奨）**: 動的遅延推定と真のストリーミング処理
    - **Riverオンライン学習（従来版）**: バッチ処理と多様な特徴量
    """)
    
    
    # ストリーミング予測の特徴
    with st.expander("🎯 主な特徴", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **動的遅延推定**
            - ダム放流量に応じて水の到達時間を動的に調整
            - 低流量時（<50m³/s）: 90分遅延
            - 中流量時（50-100m³/s）: 60分遅延
            - 高流量時（>100m³/s）: 30分遅延
            - 実測値に基づいて継続的に改善
            """)
            
            st.markdown("""
            **真のストリーミング処理**
            - 1件ずつのデータをリアルタイム処理
            - predict_oneメソッドで即座に予測
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
            **アンサンブル学習**
            - HoeffdingAdaptiveTreeRegressor：適応的決定木
            - LinearRegressor：線形回帰
            - BaggingRegressor：複数モデルの統合
            - 各モデルの強みを組み合わせ
            """)
    
    
    # 学習プロセス
    with st.expander("📚 学習プロセス", expanded=False):
        st.markdown("""
        ### ストリーミング処理の流れ
        ```python
        # 1. 新しいデータが到着
        current_data = get_latest_data()
        
        # 2. 特徴量を抽出（シンプル）
        features = {
            'water_level': current_data['river']['water_level'],
            'dam_outflow': current_data['dam']['outflow'],
            'rainfall': current_data['rainfall'],
            'hour': datetime.now().hour,
            # ...その他の特徴量
        }
        
        # 3. 遅延時間を動的に推定
        delay = delay_estimator.estimate_delay(features['dam_outflow'])
        features['estimated_delay'] = delay
        
        # 4. 予測を実行（1件ずつ）
        prediction = model.predict_one(features)
        
        # 5. 実測値が利用可能になったら学習
        if actual_value_available:
            model.learn_one(features, actual_value)
            delay_estimator.update(delay, actual_delay)
        ```
        
        ### モデルの保存
        - `models/river_streaming_model.pkl`として保存
        - 学習履歴、遅延推定パラメータも保存
        - システム再起動後も学習を継続
        """)
    
    # 使用される特徴量
    with st.expander("📊 使用される特徴量", expanded=False):
        st.markdown("### シンプルで効果的な特徴量設計")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **基本情報（現在値のみ）**
            - water_level: 現在水位
            - dam_outflow: ダム放流量
            - dam_inflow: ダム流入量
            - storage_rate: 貯水率
            - rainfall: 降雨量
            """)
        
        with col2:
            st.markdown("""
            **動的特徴**
            - estimated_delay: 推定遅延時間
            - level_change_rate: 水位変化率
            - hour: 時刻
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
            "機械学習（アンサンブル）",
            "⭐⭐ 低い",
            "⭐⭐⭐⭐⭐ 向上する",
            "⭐⭐⭐⭐⭐ 適応的",
            "⭐⭐ 低い",
            "⭐⭐⭐⭐⭐ 非常に高速",
            "⭐⭐⭐⭐ 少ない",
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
        - ✅ 動的遅延推定の実装
        - ✅ 真のストリーミング処理（predict_one）
        - ✅ River 0.21.0完全対応
        - ✅ アンサンブル学習の実装
        
        **システム全体**
        - ✅ モデル再初期化機能
        - ✅ 学習データクリア機能
        - ✅ モデル動作状態の可視化
        - ✅ システム診断情報の表示
        
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