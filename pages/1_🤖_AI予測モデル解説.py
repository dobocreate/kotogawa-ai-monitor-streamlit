"""
AI予測モデル解説ページ
厚東川水位予測システムで使用されるAIモデルについて詳しく説明します
"""

import streamlit as st
from pathlib import Path
import sys

# スクリプトのディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

# 予測モジュールのインポート（情報取得用）
try:
    from scripts.river_dual_model_predictor import RiverDualModelPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ページ設定
st.set_page_config(
    page_title="AI予測モデル解説",
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
    st.markdown("厚東川水位予測システムで使用されるAI予測モデルについて詳しく説明します。")
    
    # タブでモデルの説明を切り替え
    tab1, tab2, tab3 = st.tabs(["システム概要", "技術詳細", "運用ガイド"])
    
    # 現在の利用可能状況を表示
    with st.sidebar:
        st.markdown("### 📊 モデル利用可能状況")
        if MODEL_AVAILABLE:
            st.success("✅ AI予測モデルが利用可能です")
            st.caption("River MLによる高精度予測")
        else:
            st.error("❌ AI予測モデルが利用できません")
    
    with tab1:
        show_system_overview()
    
    with tab2:
        show_technical_details()
    
    with tab3:
        show_operation_guide()

def show_system_overview():
    """システム概要"""
    st.header("システム概要")
    
    st.markdown("""
    厚東川水位予測システムは、River MLライブラリを使用したオンライン機械学習モデルです。
    RiverDualModelPredictorクラスが内部で2つのモデルインスタンス（基本モデルと適応モデル）を管理し、
    それぞれ独立した予測を提供します。
    """)
    
    # システムアーキテクチャ
    st.subheader("🏗️ システムアーキテクチャ")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### データ収集と予測
        - 10分ごとにデータ収集（GitHub Actions）
        - 河川、ダム、降雨データを統合
        - 18ステップ先まで予測（10分〜180分）
        - 予測結果はpredictions/に保存
        """)
    
    with col2:
        st.markdown("""
        ### モデル学習と更新
        - 6時間ごとに自動学習（GitHub Actions）
        - 過去30日分のデータで学習
        - 適応モデルのみ更新
        - 診断結果はdiagnostics/に保存
        """)
    
    # モデル構成
    st.subheader("🎯 内部モデル構成")
    st.info("""
    **RiverDualModelPredictor**が以下の2つのモデルインスタンスを内部管理：
    
    1. **基本モデル** (`river_base_model_v2.pkl`)
       - 初期学習済み・固定パラメータ
       - デモCSVデータで学習（2023/6/25-7/1）
       - ベースラインとして安定した予測
    
    2. **適応モデル** (`river_adaptive_model_v2.pkl`)
       - 継続的に学習・パラメータ更新
       - 最新のデータパターンに適応
       - 地域特性を徐々に学習
    """)
    
    # 予測の流れ
    st.subheader("📊 予測の流れ")
    st.markdown("""
    1. **データ収集** - 10分ごとに最新データを取得
    2. **特徴量生成** - 水位、放流量、降雨量などから特徴を抽出
    3. **予測実行** - 両モデルで個別に予測（統合なし）
    4. **結果表示** - 各モデルの予測を別々に表示
    """)
    
    # 主な特徴
    with st.expander("✨ システムの主な特徴", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **オンライン学習**
            - データを1件ずつ処理
            - メモリ効率が高い
            - リアルタイム性に優れる
            """)
            
            st.markdown("""
            **適応的な学習**
            - ARFRegressorを使用
            - ドリフト検出機能
            - 自動的にモデルを調整
            """)
        
        with col2:
            st.markdown("""
            **高速な予測**
            - 推論時間 < 100ms
            - 18ステップを一度に予測
            - 低リソースで動作
            """)
            
            st.markdown("""
            **堅牢性**
            - 異常値への耐性
            - 欠損値の自動処理
            - エラーハンドリング
            """)

def show_technical_details():
    """技術詳細"""
    st.header("技術詳細")
    
    # アルゴリズム
    st.subheader("🤖 使用アルゴリズム")
    st.markdown("""
    ### ARFRegressor (Adaptive Random Forest Regressor)
    - **アンサンブル学習**: 複数の決定木を組み合わせ
    - **適応的**: データの変化に応じて木を更新
    - **ドリフト検出**: ADWINアルゴリズムでデータ変化を検知
    - **効率的**: オンライン学習に最適化
    """)
    
    # 特徴量
    st.subheader("📊 使用する特徴量")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **基本特徴量**
        - 河川水位（現在値）
        - ダム放流量
        - ダム流入量
        - 貯水率
        - 10分間降雨量
        - 1時間降雨量
        """)
    
    with col2:
        st.markdown("""
        **派生特徴量**
        - 水位変化率（10分間）
        - 放流量変化率
        - 時刻（0-23）
        - 曜日（0-6）
        - 昼夜フラグ
        - 推定遅延時間
        """)
    
    # モデルパラメータ
    with st.expander("⚙️ モデルパラメータ", expanded=False):
        st.markdown("""
        ### ARFRegressorの設定
        ```python
        model = ARFRegressor(
            n_models=10,  # 木の数
            seed=42,      # 乱数シード
            model_selector_decay=0.95,
            detector=ADWIN(),  # ドリフト検出
            warning_detector=ADWIN(),
            max_features='sqrt'
        )
        ```
        
        ### 学習設定
        - **バッチサイズ**: 1（オンライン学習）
        - **学習率**: 自動調整
        - **メモリ制限**: なし（ストリーミング）
        - **更新頻度**: データ到着時
        """)
    
    # パフォーマンス
    st.subheader("⚡ パフォーマンス")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("推論時間", "< 100ms", "高速")
        st.caption("18ステップの予測")
    
    with col2:
        st.metric("メモリ使用量", "< 100MB", "省メモリ")
        st.caption("モデル全体")
    
    with col3:
        st.metric("精度 (MAE)", "0.02-0.05m", "高精度")
        st.caption("10分先予測")
    
    # 学習プロセス
    st.subheader("📚 学習プロセス")
    st.markdown("""
    ### 継続学習の流れ
    1. **データ収集** - 最新の河川データを取得
    2. **予測照合** - 過去の予測と実測値を比較
    3. **誤差計算** - 予測誤差を算出
    4. **モデル更新** - 誤差に基づいてパラメータ調整
    5. **ドリフト検出** - データ分布の変化を監視
    6. **診断保存** - 学習結果を記録
    """)

def show_operation_guide():
    """運用ガイド"""
    st.header("運用ガイド")
    
    st.markdown("""
    AI予測モデルを効果的に活用するためのガイドラインです。
    """)
    
    # モデルの使い分け
    st.subheader("🎯 モデル出力の解釈")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**基本モデルの予測**")
        st.markdown("""
        - 安定した予測値
        - 学習データ期間の平均的な挙動
        - 異常値の影響を受けない
        - ベースラインとして利用
        """)
    
    with col2:
        st.info("**適応モデルの予測**")
        st.markdown("""
        - 最新のパターンを反映
        - 季節変動や地域特性を学習
        - 継続的に精度向上
        - より実態に即した予測
        """)
    
    # 予測の活用
    st.subheader("📈 予測の活用方法")
    
    with st.expander("通常時の運用", expanded=True):
        st.markdown("""
        ### 両モデルの予測が近い場合
        - 高い信頼性で予測を利用可能
        - 適応モデルの値を参考に
        - 基本モデルで妥当性を確認
        
        ### 予測精度の確認
        - AI学習結果ページでMAEを確認
        - 時間別の精度分布を参照
        - ドリフト検出状態をチェック
        """)
    
    with st.expander("異常時の対応", expanded=False):
        st.markdown("""
        ### 予測が大きく異なる場合
        - 現地の状況を優先確認
        - 基本モデルを参考に
        - 過去の類似状況と比較
        
        ### システムエラー時
        - GitHub Actionsの状態確認
        - エラーログの確認
        - 手動でモデル実行も可能
        """)
    
    # モニタリング
    st.subheader("📊 モニタリングポイント")
    
    st.markdown("""
    ### 定期的に確認すべき項目
    1. **予測精度（MAE）の推移**
       - 基本モデル：固定値の確認
       - 適応モデル：改善傾向の確認
    
    2. **学習状況**
       - サンプル数の増加
       - ドリフト検出の有無
       - エラー発生状況
    
    3. **システム稼働状況**
       - データ収集の成功率
       - 予測実行の完了率
       - モデル更新の実行状況
    """)
    
    # トラブルシューティング
    st.subheader("🔧 トラブルシューティング")
    
    trouble_data = {
        "症状": [
            "予測が表示されない",
            "精度が悪化している",
            "エラーが頻発する",
            "学習が実行されない"
        ],
        "原因": [
            "データ収集の失敗",
            "ドリフトの発生",
            "メモリ不足",
            "GitHub Actions停止"
        ],
        "対処法": [
            "データソースを確認",
            "診断結果を確認",
            "ログを確認",
            "手動実行を試行"
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(trouble_data)
    st.table(df)
    
    # まとめ
    st.subheader("📢 まとめ")
    st.success("""
    **AI予測モデルの特長**
    
    1. **継続的な改善**
       - オンライン学習で常に最新化
       - 地域特性を自動的に学習
       - 季節変動にも対応
    
    2. **高い信頼性**
       - 基本モデルで安定性確保
       - 適応モデルで精度向上
       - 異常値への耐性
    
    3. **運用の容易さ**
       - 自動実行・自動更新
       - 低リソースで動作
       - 詳細な診断情報
    """)

if __name__ == "__main__":
    main()