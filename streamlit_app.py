#!/usr/bin/env python3
"""
厚東川リアルタイム監視システム - Streamlitアプリケーション
山口県宇部市の厚東川ダムおよび厚東川（持世寺）の監視データを表示
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8以前の場合
    import pytz
    ZoneInfo = lambda x: pytz.timezone(x)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# AI予測モジュールのインポート
AI_PREDICTION_AVAILABLE = False
RIVER_LEARNING_AVAILABLE = False

try:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from scripts.advanced_prediction import AdvancedRiverLevelPredictor
    AI_PREDICTION_AVAILABLE = True
except ImportError:
    try:
        from scripts.improved_prediction import ImprovedRiverLevelPredictor as AdvancedRiverLevelPredictor
        AI_PREDICTION_AVAILABLE = True
    except ImportError:
        try:
            from scripts.simple_prediction import SimpleRiverLevelPredictor as AdvancedRiverLevelPredictor
            AI_PREDICTION_AVAILABLE = True
        except ImportError:
            AI_PREDICTION_AVAILABLE = False
            print("エキスパートルール予測が利用できません。")

# Riverオンライン学習モジュールのインポート
try:
    from scripts.river_streaming_prediction import RiverStreamingPredictor
    RIVER_LEARNING_AVAILABLE = True
    RIVER_STREAMING_AVAILABLE = True
except ImportError:
    RIVER_STREAMING_AVAILABLE = False
    try:
        from scripts.river_online_prediction import RiverOnlinePredictor
        RIVER_LEARNING_AVAILABLE = True
    except ImportError:
        RIVER_LEARNING_AVAILABLE = False
        print("Riverオンライン学習が利用できません。")

# River import helperのインポート
get_river_predictor = None
try:
    from scripts.river_import_helper import get_river_predictor
except ImportError:
    print("River import helperが利用できません。")

# 予測評価モジュールのインポート
try:
    from scripts.prediction_evaluator import PredictionEvaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("予測評価機能が利用できません。")

# ページ設定
st.set_page_config(
    page_title="厚東川監視システム",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed"  # モバイル・デスクトップ共に初期状態は閉じる
)

# サイドバー表示時のレスポンシブ対応CSS
st.markdown("""
<style>
    /* サイドバーが開いている時のメインコンテンツ幅調整 */
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        max-width: 100%;
    }
    
    /* Streamlitのデフォルト上部マージンを完全に除去 */
    .main .block-container > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* ページ全体の上部マージンを除去 */
    .stApp > header {
        display: none !important;
    }
    
    /* Streamlitのメインコンテナの上部スペースを除去 */
    .main {
        padding-top: 0 !important;
    }
    
    /* 自動更新コンポーネントの上部マージンを除去 */
    [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Streamlitの全体的な上部スペースを強制除去 */
    .stApp {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* 最初の要素の上部マージンを完全除去 */
    .main .block-container > div > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* 自動更新コンポーネントを完全に非表示 */
    iframe[title="st_autorefresh.autorefresh"] {
        display: none !important;
        height: 0 !important;
        width: 0 !important;
    }
    
    /* 自動更新コンポーネントのコンテナも非表示 */
    [data-testid="stIFrame"]:has(iframe[title*="autorefresh"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* サイドバーが開いている時のグラフコンテナ */
    [data-testid="stSidebar"][aria-expanded="true"] ~ .main .block-container {
        max-width: calc(100vw - 21rem);
    }
    
    /* Plotlyグラフのレスポンシブ対応 */
    .js-plotly-plot .plotly {
        width: 100% !important;
        height: auto !important;
    }
    
    /* Streamlitのグラフコンテナ */
    .stPlotlyChart {
        width: 100% !important;
    }
    
    /* メトリクス表示の調整 */
    [data-testid="metric-container"] {
        width: 100%;
        min-width: 0;
    }
    
    /* サイドバーの上部余白調整 */
    section[data-testid="stSidebar"] > div {
        padding-top: 0rem;
    }
    
    /* システム名を中央配置 */
    .main .block-container h1,
    .main h1,
    h1[data-testid="stMarkdown"] {
        text-align: center !important;
    }
    
    
</style>
""", unsafe_allow_html=True)

class KotogawaMonitor:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.history_dir = self.data_dir / "history"
        
        # アラート閾値（デフォルト値）
        self.default_thresholds = {
            'river_warning': 3.0,
            'river_danger': 5.0,
            'dam_warning': 90.0,
            'dam_danger': 95.0
        }
    
    def load_latest_data(_self) -> Optional[Dict[str, Any]]:
        """最新データを読み込む（ファイル更新時刻ベースのキャッシュ）"""
        latest_file = _self.data_dir / "latest.json"
        
        if not latest_file.exists():
            st.warning("■ データファイルが見つかりません。データ収集スクリプトを実行してください。")
            return None
        
        try:
            # ファイル更新時刻をキャッシュキーとして使用
            file_mtime = latest_file.stat().st_mtime
            return _self._load_latest_data_cached(str(latest_file), file_mtime)
        except Exception as e:
            st.error(f"× データ読み込みエラー: {e}")
            return None
    
    @st.cache_data(ttl=300)  # ファイル更新時刻が変わるまでキャッシュ
    def _load_latest_data_cached(_self, file_path: str, file_mtime: float) -> Optional[Dict[str, Any]]:
        """ファイル更新時刻をキーとするキャッシュされたデータ読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # データの整合性チェック
                if not data or 'timestamp' not in data:
                    st.error("× データファイルの形式が正しくありません")
                    return None
                
                return data
        except json.JSONDecodeError as e:
            st.error(f"× JSONファイルの形式エラー: {e}")
            return None
        except FileNotFoundError:
            st.warning("■ データファイルが見つかりません")
            return None
        except Exception as e:
            st.error(f"× データ読み込みエラー: {e}")
            return None
    
    def get_cache_key(self) -> str:
        """キャッシュキー用の最新ファイル時刻を取得"""
        try:
            # latest.jsonの更新時刻を取得
            latest_file = self.data_dir / "latest.json"
            if latest_file.exists():
                return str(latest_file.stat().st_mtime)
            return "no_file"
        except Exception:
            return "error"
    
    @st.cache_data(ttl=300)  # 5分間キャッシュ（短縮）
    def load_history_data(_self, hours: int = 72, cache_key: str = None) -> List[Dict[str, Any]]:
        """履歴データを読み込む（固定期間で全データを読み込み、表示はグラフ側で制御）"""
        history_data = []
        # JST（日本標準時）で現在時刻を取得
        end_time = datetime.now(ZoneInfo('Asia/Tokyo'))
        start_time = end_time - timedelta(hours=hours)
        
        if not _self.history_dir.exists():
            st.info("■ 履歴データディレクトリがありません。データが蓄積されるまでお待ちください。")
            return history_data
        
        error_count = 0
        processed_files = 0
        # 時間に応じて最大処理ファイル数を動的に調整（10分間隔データを想定）
        max_files = min(hours * 6 + 50, 500)  # 余裕を持って設定
        
        # JST時刻で日付ディレクトリを処理（新しいデータから逆順で処理）
        current_time = end_time
        while current_time >= start_time and processed_files < max_files:
            date_dir = (_self.history_dir / 
                       current_time.strftime("%Y") / 
                       current_time.strftime("%m") / 
                       current_time.strftime("%d"))
            
            if date_dir.exists():
                # ファイルを降順でソートして新しいものから処理
                json_files = sorted(date_dir.glob("*.json"), reverse=True)
                for file_path in json_files:
                    if processed_files >= max_files:
                        break
                    
                    # daily_summaryファイルはスキップ
                    if file_path.name == "daily_summary.json":
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # データの基本検証とJST時刻での範囲チェック
                            if data and 'timestamp' in data:
                                # タイムスタンプをJSTで解析
                                try:
                                    data_timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                                    if data_timestamp.tzinfo is None:
                                        data_timestamp = data_timestamp.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                                    else:
                                        data_timestamp = data_timestamp.astimezone(ZoneInfo('Asia/Tokyo'))
                                    
                                    # 全データを読み込み（表示範囲はグラフ側で制御）
                                    history_data.append(data)
                                    processed_files += 1
                                    
                                except Exception as e:
                                    # タイムスタンプ解析エラーの場合も追加（後方互換性）
                                    history_data.append(data)
                                    processed_files += 1
                            else:
                                error_count += 1
                                
                    except json.JSONDecodeError:
                        error_count += 1
                        # 個別のファイルエラーは表示しない（サマリーのみ）
                    except Exception as e:
                        error_count += 1
                        # 個別のファイルエラーは表示しない（サマリーのみ）
            
            current_time -= timedelta(days=1)
        
        # エラーサマリー表示（エラーが多い場合のみ表示）
        if error_count > 10:
            st.warning(f"■ 履歴データの読み込みで {error_count} 件のエラーがありました")
        
        # 時系列順にソート
        try:
            history_data.sort(key=lambda x: x.get('timestamp', ''))
        except Exception as e:
            st.error(f"× 履歴データソートエラー: {e}")
            
        return history_data
    
    def load_sample_csv_data(self) -> List[Dict[str, Any]]:
        """サンプルCSVファイルを読み込んで通常モードと同じJSON形式に変換"""
        import pandas as pd
        from datetime import datetime
        
        # CSVファイルのパス
        dam_csv_path = Path("sample/dam_20230625-20230701.csv")
        water_csv_path = Path("sample/water-level_20230625-20230701.csv")
        
        try:
            
            # ファイル存在確認
            if not dam_csv_path.exists():
                st.error(f"❌ ダムCSVファイルが見つかりません: {dam_csv_path}")
                return []
            if not water_csv_path.exists():
                st.error(f"❌ 河川CSVファイルが見つかりません: {water_csv_path}")
                return []
            
            
            # ダムデータの読み込み（Shift-JISエンコーディング）
            dam_df = pd.read_csv(dam_csv_path, encoding='shift_jis', skiprows=7)
            
            dam_df.columns = ['timestamp', 'hourly_rain', 'cumulative_rain', 'water_level', 
                             'storage_rate', 'inflow', 'outflow', 'storage_change']
            
            # 河川水位データの読み込み（Shift-JISエンコーディング）
            water_df = pd.read_csv(water_csv_path, encoding='shift_jis', skiprows=6)
            
            # 列数に応じて適切に列名を設定
            if len(water_df.columns) == 4:
                water_df.columns = ['timestamp', 'water_level', 'level_change', 'empty']
                # 不要な空列を削除
                water_df = water_df.drop('empty', axis=1)
            else:
                water_df.columns = ['timestamp', 'water_level', 'level_change']
            
            # 河川データのタイムスタンプもクリーニング（ダムデータと同じ形式に統一）
            water_df['clean_timestamp'] = water_df['timestamp'].astype(str).str.replace('　', '').str.strip()
            
            # データクリーニング：空の値を適切に処理
            dam_df['hourly_rain'] = pd.to_numeric(dam_df['hourly_rain'], errors='coerce').fillna(0)
            dam_df['cumulative_rain'] = pd.to_numeric(dam_df['cumulative_rain'], errors='coerce').fillna(0)
            dam_df['water_level'] = pd.to_numeric(dam_df['water_level'], errors='coerce')
            dam_df['storage_rate'] = pd.to_numeric(dam_df['storage_rate'], errors='coerce')
            dam_df['inflow'] = pd.to_numeric(dam_df['inflow'], errors='coerce')
            dam_df['outflow'] = pd.to_numeric(dam_df['outflow'], errors='coerce')
            
            water_df['water_level'] = pd.to_numeric(water_df['water_level'], errors='coerce')
            water_df['level_change'] = pd.to_numeric(water_df['level_change'], errors='coerce').fillna(0)
            
            # データの結合と変換
            sample_data = []
            processed_count = 0
            error_count = 0
            
            for idx, row in dam_df.iterrows():
                timestamp_str = str(row['timestamp']).strip()
                if pd.isna(timestamp_str) or timestamp_str == '' or timestamp_str == 'nan':
                    continue
                
                # 複数の形式を試行（全角スペースや半角スペースを考慮）
                # 先頭と末尾の全角スペースや半角スペースのみを削除して標準化
                clean_timestamp = timestamp_str.replace('　', '').strip()
                
                
                # タイムスタンプの解析とISO形式への変換
                dt = None
                formatted_timestamp = None
                
                timestamp_formats = [
                    '%Y/%m/%d %H:%M',    # 標準形式: '2023/06/25 00:20'
                    '%Y/%m/%d %H:%M:%S', # 秒あり: '2023/06/25 00:20:00'
                ]
                
                for fmt in timestamp_formats:
                    try:
                        dt = datetime.strptime(clean_timestamp, fmt)
                        formatted_timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S+09:00')
                        
                        break
                        
                    except ValueError:
                        continue
                
                if dt is None:
                    error_count += 1
                    if processed_count < 5:
                        st.error(f"❌ 全ての形式で解析失敗: '{timestamp_str}' (長さ: {len(timestamp_str)}文字)")
                        # 文字の詳細表示
                        char_info = [f"'{c}' ({ord(c)})" for c in timestamp_str[:20]]  # 最初の20文字
                        st.error(f"文字詳細: {', '.join(char_info)}")
                    continue
                
                # 対応する河川データを探す（クリーニング済みタイムスタンプでマッチング）
                water_row = water_df[water_df['clean_timestamp'] == clean_timestamp]
                
                if processed_count < 5:  # デバッグ出力
                    if not water_row.empty:
                        river_level = water_row['water_level'].iloc[0]
                    else:
                        st.warning(f"⚠️ 河川データマッチ失敗: '{clean_timestamp}'")
                
                # 通常モードと同じJSON形式のデータ構造に変換
                data_point = {
                    'timestamp': formatted_timestamp,
                    'data_time': formatted_timestamp,
                    'dam': {
                        'water_level': float(row['water_level']) if pd.notna(row['water_level']) else None,
                        'storage_rate': float(row['storage_rate']) if pd.notna(row['storage_rate']) else None,
                        'inflow': float(row['inflow']) if pd.notna(row['inflow']) else None,
                        'outflow': float(row['outflow']) if pd.notna(row['outflow']) else None,
                        'storage_change': None  # サンプルデータには含まれない
                    },
                    'river': {
                        'water_level': float(water_row['water_level'].iloc[0]) if not water_row.empty and pd.notna(water_row['water_level'].iloc[0]) else None,
                        'level_change': float(water_row['level_change'].iloc[0]) if not water_row.empty and pd.notna(water_row['level_change'].iloc[0]) else 0.0,
                        'status': '正常'  # サンプルデータでは常に正常とする
                    },
                    'rainfall': {
                        'hourly': int(row['hourly_rain']) if pd.notna(row['hourly_rain']) else 0,
                        'cumulative': int(row['cumulative_rain']) if pd.notna(row['cumulative_rain']) else 0,
                        'change': 0  # 通常データとの互換性のため
                    },
                    # ダミーの天気データ（グラフ描画に必要）
                    'weather': {
                        'today': {
                            'weather_code': '100',
                            'weather_text': 'サンプルデータ',
                            'temp_max': None,
                            'temp_min': None,
                            'precipitation_probability': [0],
                            'precipitation_times': ['']
                        },
                        'tomorrow': {
                            'weather_code': '100',
                            'weather_text': 'サンプルデータ',
                            'temp_max': None,
                            'temp_min': None,
                            'precipitation_probability': [0],
                            'precipitation_times': ['']
                        },
                        'update_time': formatted_timestamp,
                        'weekly_forecast': []
                    },
                    # ダミーの降水強度データ
                    'precipitation_intensity': {
                        'observation': [],
                        'forecast': [],
                        'update_time': formatted_timestamp
                    }
                }
                
                sample_data.append(data_point)
                processed_count += 1
            
            # 統計情報を表示
            
            if not sample_data:
                st.warning("⚠️ サンプルデータの読み込みに失敗しました")
            
            return sample_data
            
        except Exception as e:
            st.error(f"サンプルCSVファイルの読み込みエラー: {e}")
            import traceback
            st.error(f"詳細エラー: {traceback.format_exc()}")
            return []
    
    def check_alert_status(self, data: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, str]:
        """アラート状態をチェック"""
        alerts = {
            'river': '正常',
            'dam': '正常',
            'rainfall': '正常',
            'overall': '正常'
        }
        
        if not data:
            alerts['overall'] = 'データなし'
            return alerts
        
        alert_level = 0  # 0=正常, 1=注意, 2=警戒, 3=危険
        
        # 河川水位チェック（実際のステータスを使用）
        river_status = data.get('river', {}).get('status', '正常')
        river_level = data.get('river', {}).get('water_level')
        
        if river_status in ['氾濫危険']:
            alerts['river'] = '危険'
            alert_level = max(alert_level, 3)
        elif river_status in ['避難判断']:
            alerts['river'] = '避難判断'
            alert_level = max(alert_level, 3)
        elif river_status in ['氾濫注意']:
            alerts['river'] = '警戒'
            alert_level = max(alert_level, 2)
        elif river_status in ['水防団待機']:
            alerts['river'] = '注意'
            alert_level = max(alert_level, 1)
        else:
            alerts['river'] = '正常'
        
        # ダム水位チェック
        dam_level = data.get('dam', {}).get('water_level')
        
        if dam_level is not None:
            # ダム水位による判定
            if dam_level >= thresholds['dam_danger']:  # 設計最高水位
                alerts['dam'] = '危険'
                alert_level = max(alert_level, 3)
            elif dam_level >= thresholds['dam_warning']:  # 洪水時最高水位
                alerts['dam'] = '警戒'
                alert_level = max(alert_level, 2)
        
        # 雨量チェック
        hourly_rain = data.get('rainfall', {}).get('hourly')
        cumulative_rain = data.get('rainfall', {}).get('cumulative')
        
        # null値の場合は雨量チェックをスキップ
        if hourly_rain is not None and cumulative_rain is not None:
            if hourly_rain >= 50 or cumulative_rain >= 200:
                alerts['rainfall'] = '危険'
                alert_level = max(alert_level, 3)
            elif hourly_rain >= 30 or cumulative_rain >= 100:
                alerts['rainfall'] = '警戒'
                alert_level = max(alert_level, 2)
            elif hourly_rain >= 10 or cumulative_rain >= 50:
                alerts['rainfall'] = '注意'
                alert_level = max(alert_level, 1)
        
        # 総合アラートレベル設定
        if alert_level >= 3:
            alerts['overall'] = '危険'
        elif alert_level >= 2:
            alerts['overall'] = '警戒'
        elif alert_level >= 1:
            alerts['overall'] = '注意'
        else:
            alerts['overall'] = '正常'
        
        return alerts
    
    def create_weather_forecast_display(self, data: Dict[str, Any], show_weekly: bool = True) -> None:
        """天気予報情報を表示する"""
        st.markdown("## 天気予報（宇部市）")
        
        weather_data = data.get('weather', {})
        
        if not weather_data or not weather_data.get('today', {}).get('weather_text'):
            st.info("天気予報データが利用できません")
            return
        
        # 更新時刻の表示
        if weather_data.get('update_time'):
            try:
                update_time = datetime.fromisoformat(weather_data['update_time'])
                st.caption(f"予報更新時刻 : {update_time.strftime('%Y-%m-%d %H:%M')} JST")
            except:
                pass
        
        # 今日・明日の天気予報を横並びで表示
        col1, col2 = st.columns(2)
        
        # 今日の天気
        with col1:
            st.markdown("### 今日")
            today = weather_data.get('today', {})
            
            # 天気
            weather_text = today.get('weather_text', 'データなし')
            # スペースを削除して2行分確保
            weather_text_cleaned = weather_text.replace('　', '').replace(' ', '')
            st.markdown(f"**天気:**<br>{weather_text_cleaned}", unsafe_allow_html=True)
            # 2行分の高さを確保するための空白行
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 気温
            temp_max = today.get('temp_max')
            temp_min = today.get('temp_min')
            if temp_max is not None and temp_min is not None:
                st.markdown(f"**気温:** {temp_max}°C / {temp_min}°C")
            elif temp_max is not None:
                st.markdown(f"**最高気温:** {temp_max}°C")
            elif temp_min is not None:
                st.markdown(f"**最低気温:** {temp_min}°C")
            
            # 時間別降水確率をグラフで表示
            precip_prob = today.get('precipitation_probability', [])
            precip_times = today.get('precipitation_times', [])
            if precip_prob and precip_times:
                st.markdown(f"**降水確率:**")
                # Plotlyでグラフ作成
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=precip_times,
                    y=precip_prob,
                    mode='lines+markers+text',
                    text=[f'{p}%' if p is not None else '--' for p in precip_prob],
                    textposition='top center',
                    textfont=dict(size=12, color='black'),
                    line=dict(color='#4488ff', width=3),
                    marker=dict(
                        size=12,
                        color='white',
                        line=dict(width=2, color='#4488ff')
                    )
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=30, b=30),
                    xaxis_title="",
                    yaxis_title="降水確率 (%)",
                    yaxis=dict(range=[0, 100], fixedrange=True),
                    xaxis=dict(fixedrange=True),
                    showlegend=False,
                    autosize=True,
                    font=dict(size=9)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="today_weather_chart")
        
        # 明日の天気
        with col2:
            st.markdown("### 明日")
            tomorrow = weather_data.get('tomorrow', {})
            
            # 天気
            weather_text = tomorrow.get('weather_text', 'データなし')
            # スペースを削除して2行分確保
            weather_text_cleaned = weather_text.replace('　', '').replace(' ', '')
            st.markdown(f"**天気:**<br>{weather_text_cleaned}", unsafe_allow_html=True)
            # 2行分の高さを確保するための空白行
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 気温
            temp_max = tomorrow.get('temp_max')
            temp_min = tomorrow.get('temp_min')
            if temp_max is not None and temp_min is not None:
                st.markdown(f"**気温:** {temp_max}°C / {temp_min}°C")
            elif temp_max is not None:
                st.markdown(f"**最高気温:** {temp_max}°C")
            elif temp_min is not None:
                st.markdown(f"**最低気温:** {temp_min}°C")
            
            # 時間別降水確率をグラフで表示
            precip_prob = tomorrow.get('precipitation_probability', [])
            precip_times = tomorrow.get('precipitation_times', [])
            if precip_prob and precip_times:
                st.markdown(f"**降水確率:**")
                # Plotlyでグラフ作成
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=precip_times,
                    y=precip_prob,
                    mode='lines+markers+text',
                    text=[f'{p}%' if p is not None else '--' for p in precip_prob],
                    textposition='top center',
                    textfont=dict(size=12, color='black'),
                    line=dict(color='#4488ff', width=3),
                    marker=dict(
                        size=12,
                        color='white',
                        line=dict(width=2, color='#4488ff')
                    )
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=30, b=30),
                    xaxis_title="",
                    yaxis_title="降水確率 (%)",
                    yaxis=dict(range=[0, 100], fixedrange=True),
                    xaxis=dict(fixedrange=True),
                    showlegend=False,
                    autosize=True,
                    font=dict(size=9)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="tomorrow_weather_chart")
        
        
        # 警戒メッセージ
        today_precip = weather_data.get('today', {}).get('precipitation_probability', [])
        tomorrow_precip = weather_data.get('tomorrow', {}).get('precipitation_probability', [])
        
        # 2日間の最大降水確率を取得
        max_today = max([p for p in today_precip if p is not None], default=0)
        max_tomorrow = max([p for p in tomorrow_precip if p is not None], default=0)
        
        if max_today >= 70 or max_tomorrow >= 70:
            st.warning("■ 降水確率が高くなっています。水位の変化にご注意ください。")
        elif max_today >= 50 or max_tomorrow >= 50:
            st.info("● 降水の可能性があります。河川・ダムの状況を定期的にご確認ください。")
        
        st.markdown("---")
        
        # 週間予報の表示（条件付き）
        if show_weekly:
            self.create_weekly_forecast_display(data)
    
    def get_weather_icon(self, weather_code: str, weather_text: str = "") -> str:
        """天気コードまたは天気テキストから適切な絵文字を返す"""
        if not weather_code and not weather_text:
            return "❓"
        
        # 天気コードベースの判定
        if weather_code:
            code = str(weather_code)
            # 晴れ系
            if code.startswith('1'):
                if code in ['100']:
                    return "☀️"  # 晴れ
                elif code in ['101', '110', '111']:
                    return "🌤️"  # 晴れ時々くもり
                elif code in ['102', '112', '113']:
                    return "🌦️"  # 晴れ一時雨
                else:
                    return "☀️"
            # くもり系
            elif code.startswith('2'):
                if code in ['200']:
                    return "☁️"  # くもり
                elif code in ['201', '210', '211']:
                    return "⛅"  # くもり時々晴れ
                elif code in ['202', '212', '213']:
                    return "🌦️"  # くもり一時雨
                elif code in ['203']:
                    return "🌧️"  # くもり時々雨
                elif code in ['204']:
                    return "🌨️"  # くもり一時雪
                else:
                    return "☁️"
            # 雨系
            elif code.startswith('3'):
                if code in ['300', '313']:
                    return "🌧️"  # 雨
                elif code in ['301']:
                    return "🌦️"  # 雨時々晴れ
                elif code in ['302']:
                    return "🌧️"  # 雨時々くもり
                elif code in ['303', '314']:
                    return "🌨️"  # 雨時々雪、雨のち雪
                elif code in ['308']:
                    return "⛈️"  # 大雨
                elif code in ['311']:
                    return "🌦️"  # 雨のち晴れ
                else:
                    return "🌧️"
            # 雪系
            elif code.startswith('4'):
                if code in ['400', '413']:
                    return "❄️"  # 雪
                elif code in ['401', '411']:
                    return "🌨️"  # 雪時々晴れ、雪のち晴れ
                elif code in ['402']:
                    return "🌨️"  # 雪時々くもり
                elif code in ['403', '414']:
                    return "🌨️"  # 雪時々雨、雪のち雨
                elif code in ['406']:
                    return "❄️"  # 大雪
                else:
                    return "❄️"
        
        # 天気テキストベースの判定（フォールバック）
        if weather_text:
            text = weather_text.lower()
            if "晴" in text:
                if "雨" in text:
                    return "🌦️"
                elif "くもり" in text or "曇" in text:
                    return "🌤️"
                else:
                    return "☀️"
            elif "くもり" in text or "曇" in text:
                if "雨" in text:
                    return "🌧️"
                elif "晴" in text:
                    return "⛅"
                else:
                    return "☁️"
            elif "雨" in text:
                if "大雨" in text or "雷" in text:
                    return "⛈️"
                else:
                    return "🌧️"
            elif "雪" in text:
                return "❄️"
        
        return "❓"
    
    def create_weekly_forecast_display(self, data: Dict[str, Any]) -> None:
        """週間予報情報を表示する"""
        weather_data = data.get('weather', {})
        weekly_forecast = weather_data.get('weekly_forecast', [])
        
        if not weekly_forecast:
            return
        
        st.markdown("## 週間天気予報（山口県）")
        
        # レスポンシブ用のCSS
        st.markdown("""
        <style>
            /* デフォルト（デスクトップ）: 6列表示 */
            .weekly-forecast-container {
                display: grid;
                grid-template-columns: repeat(6, 1fr);
                gap: 10px;
            }
            
            /* タブレット: 4列表示 */
            @media (max-width: 768px) {
                .weekly-forecast-container {
                    grid-template-columns: repeat(4, 1fr);
                }
            }
            
            /* スマートフォン: 2列表示 */
            @media (max-width: 480px) {
                .weekly-forecast-container {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
            
            .weather-day-item {
                text-align: center;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
            }
            
            .weather-date {
                font-weight: bold;
                margin-bottom: 5px;
                font-size: 18px;
            }
            
            .weather-label {
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            .weather-icon {
                font-size: 24px;
                margin: 10px 0 5px 0;
            }
            
            .weather-text {
                font-size: 10px;
                color: #666;
                margin-bottom: 10px;
            }
            
            .weather-precip {
                margin-bottom: 5px;
                font-size: 18px;
            }
            
            .weather-temp {
                font-size: 18px;
                color: #333;
                margin-bottom: 0;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # 週間予報を表形式で表示
        if len(weekly_forecast) >= 7:
            # HTMLコンテナで週間予報を表示
            html_content = '<div class="weekly-forecast-container">'
            
            # 曜日の日本語マッピング
            weekday_jp = {
                'Mon': '月', 'Tue': '火', 'Wed': '水', 'Thu': '木', 
                'Fri': '金', 'Sat': '土', 'Sun': '日'
            }
            
            for day_data in weekly_forecast[1:7]:
                html_content += '<div class="weather-day-item">'
                
                # 日付と曜日
                try:
                    date_obj = datetime.strptime(day_data['date'], '%Y-%m-%d')
                    month_day = date_obj.strftime('%m/%d')
                    day_of_week = day_data.get('day_of_week', date_obj.strftime('%a'))
                    
                    # 今日・明日・明後日のラベル
                    jst = ZoneInfo('Asia/Tokyo')
                    today = datetime.now(jst).date()
                    target_date = date_obj.date()
                    
                    if target_date == today:
                        day_label = "今日"
                    elif target_date == today + timedelta(days=1):
                        day_label = "明日"
                    elif target_date == today + timedelta(days=2):
                        day_label = "明後日"
                    else:
                        # 英語の曜日を日本語に変換
                        day_label = weekday_jp.get(day_of_week, day_of_week)
                    
                    html_content += f'<div class="weather-date">{month_day}</div>'
                    html_content += f'<div class="weather-label">{day_label}</div>'
                except:
                    html_content += f'<div class="weather-date">{day_data.get("date", "")}</div>'
                    html_content += '<div class="weather-label">--</div>'
                
                # 天気アイコン
                weather_code = day_data.get('weather_code', '')
                weather_text = day_data.get('weather_text', 'データなし')
                weather_icon = self.get_weather_icon(weather_code, weather_text)
                
                html_content += f'<div class="weather-icon">{weather_icon}</div>'
                
                # 短縮版のテキスト
                if len(weather_text) > 6:
                    weather_short = weather_text[:6] + "..."
                else:
                    weather_short = weather_text
                html_content += f'<div class="weather-text">{weather_short}</div>'
                
                # 降水確率
                precip_prob = day_data.get('precipitation_probability')
                if precip_prob is not None:
                    if precip_prob >= 70:
                        precip_text = f'雨 <strong>{precip_prob}%</strong>'
                    elif precip_prob >= 50:
                        precip_text = f'雨 <strong>{precip_prob}%</strong>'
                    elif precip_prob >= 30:
                        precip_text = f'曇 {precip_prob}%'
                    else:
                        precip_text = f'晴 {precip_prob}%'
                else:
                    precip_text = '--'
                
                html_content += f'<div class="weather-precip">{precip_text}</div>'
                
                # 気温情報（最高・最低気温）
                temp_max = day_data.get('temp_max')
                temp_min = day_data.get('temp_min')
                
                if temp_max is not None and temp_min is not None:
                    temp_text = f'{temp_max}°/{temp_min}°'
                elif temp_max is not None:
                    temp_text = f'{temp_max}°/--'
                elif temp_min is not None:
                    temp_text = f'--/{temp_min}°'
                else:
                    temp_text = '--/--'
                
                html_content += f'<div class="weather-temp">{temp_text}</div>'
                html_content += '</div>'
            
            html_content += '</div>'
            st.markdown(html_content, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def create_data_analysis_display(self, history_data: List[Dict[str, Any]], enable_graph_interaction: bool, display_hours: int = 24, demo_mode: bool = False) -> None:
        """データ分析セクションを表示する"""
        # データ分析セクション
        st.markdown("## データ分析")
        
        # タブによる表示切り替え
        tab1, tab2 = st.tabs(["グラフ", "データテーブル"])
        
        with tab1:
            # Plotlyの設定（小画面対応を強化）
            plotly_config = {
                'scrollZoom': enable_graph_interaction,
                'doubleClick': 'reset' if enable_graph_interaction else False,
                'displayModeBar': True,
                'displaylogo': False,
                'responsive': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'] if enable_graph_interaction else ['pan2d', 'zoom2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
            }
            
            # 2列レイアウトでグラフを表示
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("河川水位・全放流量")
                fig1 = self.create_river_water_level_graph(history_data, enable_graph_interaction, display_hours, demo_mode)
                st.plotly_chart(fig1, use_container_width=True, config=plotly_config, key="river_water_level_chart")
            
            with col2:
                st.subheader("ダム放流量・時間雨量")
                # 最新の降水強度データを取得
                latest_precipitation_data = None
                try:
                    latest_data = self.load_latest_data()
                    if latest_data and 'precipitation_intensity' in latest_data:
                        latest_precipitation_data = latest_data['precipitation_intensity']
                except:
                    pass
                
                fig2 = self.create_dam_discharge_rainfall_graph(history_data, enable_graph_interaction, latest_precipitation_data, display_hours, demo_mode)
                st.plotly_chart(fig2, use_container_width=True, config=plotly_config, key="dam_discharge_rainfall_chart")
            
            # 2行目
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("ダム貯水位・時間雨量")
                # 最新の降水強度データを取得（ダム放流量と同じものを使用）
                fig3 = self.create_dam_water_level_graph(history_data, enable_graph_interaction, latest_precipitation_data, display_hours, demo_mode)
                st.plotly_chart(fig3, use_container_width=True, config=plotly_config, key="dam_water_level_chart")
            
            with col4:
                st.subheader("ダム流入出量・累加雨量")
                fig4 = self.create_dam_flow_graph(history_data, enable_graph_interaction, display_hours, demo_mode)
                st.plotly_chart(fig4, use_container_width=True, config=plotly_config, key="dam_flow_chart")
            
            # 3行目
            col5, col6 = st.columns(2)
            
            with col5:
                # 降水強度グラフの表示
                # 最新のAPIデータから取得（降水強度・時間雨量グラフ用に再取得）
                latest_api_precipitation_data = None
                try:
                    # 最新データを再度取得（キャッシュから）
                    latest_data_for_api = self.load_latest_data()
                    if latest_data_for_api and 'precipitation_intensity' in latest_data_for_api:
                        latest_api_precipitation_data = latest_data_for_api['precipitation_intensity']
                except:
                    pass
                
                # APIデータがない場合は、履歴から観測値のみ取得
                if not latest_api_precipitation_data and history_data:
                        # 履歴データから観測値を収集
                        all_observations = []
                        update_time = None
                        # 表示期間に基づいてデータをフィルタリング（デモモード時はスキップ）
                        if demo_mode:
                            filtered_history_data = history_data
                        else:
                            time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode=False)
                            if time_min and time_max:
                                filtered_history_data = self.filter_data_by_time_range(history_data, time_min, time_max)
                            else:
                                filtered_history_data = history_data
                        
                        for item in filtered_history_data:
                            precip_data = item.get('precipitation_intensity', {})
                            if precip_data.get('observation'):
                                all_observations.extend(precip_data.get('observation', []))
                                if not update_time and precip_data.get('update_time'):
                                    update_time = precip_data.get('update_time')
                        
                        if all_observations:
                            latest_api_precipitation_data = {
                                'observation': all_observations,
                                'forecast': [],  # 予測値は常に最新APIから取得
                                'update_time': update_time
                            }
                
                # 予測値を最新データから追加（観測値がある場合のみ）
                if latest_api_precipitation_data:
                    try:
                        if 'latest_data_for_api' not in locals():
                            latest_data_for_api = self.load_latest_data()
                        
                        if latest_data_for_api and 'precipitation_intensity' in latest_data_for_api:
                            api_forecast = latest_data_for_api['precipitation_intensity'].get('forecast', [])
                            if api_forecast:
                                latest_api_precipitation_data['forecast'] = api_forecast
                    except:
                        pass
                
                if latest_api_precipitation_data and (
                    latest_api_precipitation_data.get('observation') or 
                    latest_api_precipitation_data.get('forecast')
                ):
                    st.subheader("降水強度・時間雨量")
                    
                    fig5 = self.create_precipitation_intensity_graph(latest_api_precipitation_data, enable_graph_interaction, history_data, display_hours, demo_mode)
                    st.plotly_chart(fig5, use_container_width=True, config=plotly_config, key="precipitation_intensity_chart")
            
            with col6:
                # 空白のカラム（将来の拡張用）
                pass
        
        with tab2:
            st.subheader("データテーブル")
            df_table = self.create_data_table(history_data)
            if not df_table.empty:
                st.dataframe(df_table, use_container_width=True)
                
                # CSVダウンロード
                csv = df_table.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="CSVダウンロード",
                    data=csv,
                    file_name=f"kotogawa_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("表示するデータがありません")
    
    def create_metrics_display(self, data: Dict[str, Any]) -> None:
        """現在の状況表示を作成"""
        if not data:
            st.warning("表示するデータがありません")
            return
        
        # 観測時刻の取得（日本時間で表示）
        observation_time = data.get('data_time')
        if observation_time:
            try:
                # ISOフォーマットから日時を解析
                dt = datetime.fromisoformat(observation_time.replace('Z', '+00:00'))
                # タイムゾーンがない場合は日本時間として扱う
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                else:
                    # UTCから日本時間に変換
                    dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                obs_time_str = dt.strftime('%Y/%m/%d %H:%M')
            except:
                obs_time_str = observation_time
        else:
            obs_time_str = "不明"
        
        # 3つのセクションに分けて表示
        st.markdown("## 現在の観測状況")
        
        # 河川情報と降雨情報を横並びで表示
        river_rain_col1, river_rain_col2 = st.columns(2)
        
        # 河川情報（左側）
        with river_rain_col1:
            st.markdown("### 河川情報")
            st.caption(f"更新時刻 : {obs_time_str}")
            river_subcol1, river_subcol2 = st.columns(2)
            
            with river_subcol1:
                river_level = data.get('river', {}).get('water_level')
                river_status = data.get('river', {}).get('status', '正常')
                if river_level is not None:
                    delta_color = "normal"
                    level_change = data.get('river', {}).get('level_change')
                    if level_change and level_change > 0:
                        delta_color = "inverse"
                    elif level_change and level_change < 0:
                        delta_color = "normal"
                    
                    st.metric(
                        label="水位 (m)",
                        value=f"{river_level:.2f}",
                        delta=f"{level_change:.2f}" if level_change is not None else None,
                        delta_color=delta_color
                    )
                    
                    # ステータス表示
                    if river_status != '正常':
                        if river_status in ['氾濫危険', '避難判断']:
                            st.error(f"危険 {river_status}")
                        elif river_status in ['氾濫注意', '水防団待機']:
                            st.warning(f"注意 {river_status}")
                    else:
                        st.success(f"{river_status}")
                else:
                    st.metric(label="水位 (m)", value="--")
            
            with river_subcol2:
                st.metric(
                    label="観測地点",
                    value="持世寺"
                )
        
        # 降雨情報（右側）
        with river_rain_col2:
            st.markdown("### 降雨情報")
            st.caption(f"更新時刻 : {obs_time_str}")
            rain_subcol1, rain_subcol2 = st.columns(2)
            
            with rain_subcol1:
                hourly_rain = data.get('rainfall', {}).get('hourly')
                if hourly_rain is not None:
                    rain_color = "normal"
                    if hourly_rain > 20:
                        rain_color = "inverse"
                    st.metric(
                        label="60分雨量 (mm)",
                        value=f"{hourly_rain}",
                        delta=data.get('rainfall', {}).get('change'),
                        delta_color=rain_color
                    )
                    if hourly_rain > 30:
                        st.error("雨 大雨注意")
                    elif hourly_rain > 10:
                        st.warning("雨 雨量多め")
                else:
                    st.metric(label="60分雨量 (mm)", value="--")
            
            with rain_subcol2:
                cumulative_rain = data.get('rainfall', {}).get('cumulative')
                if cumulative_rain is not None:
                    st.metric(
                        label="累加雨量 (mm)",
                        value=f"{cumulative_rain}"
                    )
                else:
                    st.metric(label="累加雨量 (mm)", value="--")
        
        # ダム情報（小画面対応：列数を動的調整）
        st.markdown("### ダム情報")
        st.caption(f"更新時刻 : {obs_time_str}")
        dam_col1, dam_col2, dam_col3, dam_col4, dam_col5, dam_col6 = st.columns([1, 1, 1, 1, 1, 0.2])
        
        with dam_col1:
            dam_level = data.get('dam', {}).get('water_level')
            if dam_level is not None:
                st.metric(
                    label="貯水位 (m)",
                    value=f"{dam_level:.2f}",
                    delta=data.get('dam', {}).get('storage_change')
                )
            else:
                st.metric(label="貯水位 (m)", value="--")
        
        with dam_col2:
            storage_rate = data.get('dam', {}).get('storage_rate')
            if storage_rate is not None:
                st.metric(
                    label="貯水率 (%)",
                    value=f"{storage_rate:.1f}"
                )
            else:
                st.metric(label="貯水率 (%)", value="--")
        
        with dam_col3:
            inflow = data.get('dam', {}).get('inflow')
            if inflow is not None:
                st.metric(
                    label="流入量 (m³/s)",
                    value=f"{inflow:.2f}"
                )
            else:
                st.metric(label="流入量 (m³/s)", value="--")
        
        with dam_col4:
            outflow = data.get('dam', {}).get('outflow')
            if outflow is not None:
                st.metric(
                    label="全放流量 (m³/s)",
                    value=f"{outflow:.2f}"
                )
            else:
                st.metric(label="全放流量 (m³/s)", value="--")
        
        with dam_col5:
            st.metric(
                label="ダム名",
                value="厚東川ダム"
            )
        
        with dam_col6:
            # 空のカラム
            pass
    
    def get_common_time_range(self, history_data: List[Dict[str, Any]], display_hours: int = 24, demo_mode: bool = False, demo_datetime: Optional[datetime] = None) -> tuple:
        """履歴データから共通の時間範囲を取得（将来予測値を考慮）"""
        if not history_data:
            return None, None
        
        if demo_mode:
            # デモモード: サンプルデータの日時に基づいて時間範囲を計算
            # 最新のタイムスタンプを取得
            latest_timestamp = None
            for data in history_data:
                if data.get('timestamp'):
                    try:
                        # fromisoformatは+09:00などのタイムゾーン情報を正しく処理できる
                        ts = datetime.fromisoformat(data['timestamp'])
                        # タイムゾーンがない場合のみJSTを設定
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                        else:
                            # 既にタイムゾーンがある場合は、JSTに変換
                            ts = ts.astimezone(ZoneInfo('Asia/Tokyo'))
                        if latest_timestamp is None or ts > latest_timestamp:
                            latest_timestamp = ts
                    except (ValueError, AttributeError):
                        continue
            
            if latest_timestamp is None:
                return None, None
            
            # デモモード用の時間範囲: 最新データ+3時間を終了時刻として、そこから表示期間分遡る
            time_max = latest_timestamp + timedelta(hours=3)
            time_min = time_max - timedelta(hours=display_hours)
            
        else:
            # 通常モード: 現在時刻（日本時間）基準
            now_jst = datetime.now(ZoneInfo('Asia/Tokyo'))
            
            # 表示期間に基づいた開始時刻を計算
            start_time = now_jst - timedelta(hours=display_hours)
            
            # 終了時刻は現在時刻+2時間（予測値表示のため）
            time_min = start_time
            time_max = now_jst + timedelta(hours=2)
        
        return time_min, time_max
    
    def filter_data_by_time_range(self, history_data: List[Dict[str, Any]], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """指定された時間範囲でデータをフィルタリング"""
        filtered_data = []
        
        # start_timeとend_timeがタイムゾーンを持っているか確認
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
        
        for item in history_data:
            data_time = item.get('data_time') or item.get('timestamp', '')
            if not data_time:
                continue
                
            try:
                # fromisoformatは+09:00などのタイムゾーン情報を正しく処理できる
                dt = datetime.fromisoformat(data_time)
                
                # タイムゾーンがない場合のみJSTを設定
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                else:
                    # 既にタイムゾーンがある場合は、JSTに変換
                    dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                
                # 指定された時間範囲内のデータのみ追加
                if start_time <= dt <= end_time:
                    filtered_data.append(item)
                    
            except Exception as e:
                # タイムスタンプ解析エラーの場合はスキップ（デバッグ用にエラーを出力）
                print(f"Timestamp parsing error: {e} for data_time: {data_time}")
                continue
        
        return filtered_data
    
    def create_river_water_level_graph(self, history_data: List[Dict[str, Any]], enable_interaction: bool = False, display_hours: int = 24, demo_mode: bool = False) -> go.Figure:
        """河川水位グラフを作成（河川水位 + ダム全放流量の二軸表示）"""
        # 現在時刻を取得
        now_jst = datetime.now(ZoneInfo('Asia/Tokyo'))
        
        # 表示期間に基づいてデータをフィルタリング（デモモード時はスキップ）
        if demo_mode:
            filtered_data = history_data
        else:
            time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode=False)
            if time_min and time_max:
                filtered_data = self.filter_data_by_time_range(history_data, time_min, time_max)
            else:
                filtered_data = history_data
        
        if not filtered_data:
            fig = go.Figure()
            fig.add_annotation(
                text="表示するデータがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # データをDataFrameに変換
        df_data = []
        for item in filtered_data:
            # 観測時刻（data_time）を使用、なければtimestampを使用
            data_time = item.get('data_time') or item.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
                # タイムゾーンがない場合はJSTとして扱う
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                else:
                    dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
            except:
                continue
                
            row = {'timestamp': dt}
            
            # 河川水位
            river_level = item.get('river', {}).get('water_level')
            if river_level is not None:
                row['river_level'] = river_level
            
            # ダム全放流量
            outflow = item.get('dam', {}).get('outflow')
            if outflow is not None:
                row['outflow'] = outflow
            
            df_data.append(row)
        
        if not df_data:
            fig = go.Figure()
            fig.add_annotation(
                text="有効なデータがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df = pd.DataFrame(df_data)
        
        # 二軸グラフを作成
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 河川水位（左軸）
        if 'river_level' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['river_level'],
                    mode='lines+markers',
                    name='河川水位（持世寺）',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6, color='white', line=dict(width=2, color='#1f77b4'))
                ),
                secondary_y=False
            )
            
            # AI予測の追加
            if AI_PREDICTION_AVAILABLE:
                try:
                    # 予測器はメイン関数で既に初期化済み
                    if 'predictor' not in st.session_state:
                        # フォールバック（念のため）
                        st.session_state.predictor = AdvancedRiverLevelPredictor()
                        st.session_state.last_prediction_model = "エキスパートルール予測"
                    
                    predictor = st.session_state.predictor
                    
                    # 予測実行（履歴データが十分にある場合）
                    if len(history_data) >= 18:  # 3時間分のデータ
                        # ストリーミングモデルの場合は最新データのみ使用
                        if selected_model == "Riverオンライン学習予測" and hasattr(predictor, 'predict_one'):
                            # ストリーミング予測
                            latest_data = history_data[-1] if history_data else None
                            if latest_data:
                                predictions = predictor.predict_one(latest_data)
                                # デバッグ情報を保存
                                st.session_state['prediction_method_used'] = 'predict_one (streaming)'
                            else:
                                predictions = None
                        else:
                            # 従来の予測
                            predictions = predictor.predict(history_data)
                            # デバッグ情報を保存
                            st.session_state['prediction_method_used'] = 'predict (batch)'
                        
                        # River予測でエラーの場合
                        if predictions is None and selected_model == "Riverオンライン学習予測":
                            error_msg = f"Riverオンライン学習予測でエラーが発生しました。"
                            st.session_state['ai_prediction_error'] = error_msg
                        
                        if predictions:
                            # 予測データをDataFrameに変換
                            pred_df = pd.DataFrame([
                                {
                                    'timestamp': datetime.fromisoformat(p['datetime']),
                                    'predicted_level': p['level']
                                }
                                for p in predictions
                            ])
                            
                            # AI予測線を追加（破線）
                            # モデル名を表示に含める
                            model_name = "エキスパート" if selected_model == "エキスパートルール予測" else "River学習"
                            fig.add_trace(
                                go.Scatter(
                                    x=pred_df['timestamp'],
                                    y=pred_df['predicted_level'],
                                    mode='lines',
                                    name=f'河川水位（AI予測: {model_name}）',
                                    line=dict(color='#1f77b4', width=3, dash='dash'),
                                    showlegend=True
                                ),
                                secondary_y=False
                            )
                            
                            # 予測の最初の点に現在値を追加（連続性のため）
                            if not df.empty and 'river_level' in df.columns:
                                last_actual = df.iloc[-1]
                                fig.add_trace(
                                    go.Scatter(
                                        x=[last_actual['timestamp'], pred_df.iloc[0]['timestamp']],
                                        y=[last_actual['river_level'], pred_df.iloc[0]['predicted_level']],
                                        mode='lines',
                                        line=dict(color='#1f77b4', width=3, dash='dash'),
                                        showlegend=False
                                    ),
                                    secondary_y=False
                                )
                            
                            # 予測情報をセッション状態に保存
                            st.session_state.last_prediction_time = datetime.now()
                            st.session_state.last_predictions = predictions
                            st.session_state.last_prediction_model = selected_model
                            
                            # 予測評価の実行（過去の予測と実測値を比較）
                            if EVALUATION_AVAILABLE:
                                try:
                                    # 評価器の初期化（セッション状態で管理）
                                    if 'evaluator' not in st.session_state:
                                        st.session_state.evaluator = PredictionEvaluator()
                                        # 保存済みデータがあれば読み込み
                                        eval_file = "data/evaluation_results.json"
                                        if Path(eval_file).exists():
                                            st.session_state.evaluator.load_from_file(eval_file)
                                    
                                    # 過去の予測があれば評価
                                    if hasattr(st.session_state, 'past_predictions'):
                                        # 3時間前の予測を確認
                                        current_time = datetime.now()
                                        for past_pred_info in st.session_state.past_predictions:
                                            pred_time = past_pred_info['time']
                                            if (current_time - pred_time).total_seconds() >= 10800:  # 3時間以上経過
                                                # 実測値と比較して評価
                                                model_type = 'expert_rule' if past_pred_info['model'] == "エキスパートルール予測" else 'river_online'
                                                st.session_state.evaluator.evaluate_prediction(
                                                    past_pred_info['predictions'],
                                                    history_data,
                                                    model_type
                                                )
                                                # 評価済みを削除
                                                st.session_state.past_predictions.remove(past_pred_info)
                                    
                                    # 現在の予測を保存
                                    if 'past_predictions' not in st.session_state:
                                        st.session_state.past_predictions = []
                                    st.session_state.past_predictions.append({
                                        'time': datetime.now(),
                                        'predictions': predictions,
                                        'model': selected_model
                                    })
                                    
                                    # 定期的に保存
                                    if len(st.session_state.evaluator.evaluation_results.get('expert_rule', {}).get('history', [])) % 10 == 0:
                                        st.session_state.evaluator.save_to_file("data/evaluation_results.json")
                                        
                                except Exception as e:
                                    print(f"予測評価エラー: {e}")
                                
                except Exception as e:
                    # エラーが発生してもグラフ表示は続行
                    print(f"AI予測エラー: {e}")
                    # セッション状態にエラーを保存
                    st.session_state['ai_prediction_error'] = str(e)
        
        # ダム全放流量（右軸）
        if 'outflow' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['outflow'],
                    mode='lines+markers',
                    name='全放流量（厚東川ダム）',
                    line=dict(color='#d62728', width=3),
                    marker=dict(size=6, color='white', line=dict(width=2, color='#d62728'))
                ),
                secondary_y=True
            )
        
        # 氾濫危険水位ライン（5.5m）を追加
        fig.add_hline(
            y=5.5,
            line_dash="dash",
            line_color="red",
            line_width=2,
            secondary_y=False
        )
        
        # 氾濫危険水位のカスタムアノテーション（フォントサイズ調整）
        fig.add_annotation(
            x=0.02,
            y=5.7,
            text="氾濫危険水位 (5.5m)",
            showarrow=False,
            xref="paper",
            yref="y",
            font=dict(color="red", size=12, weight="bold"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )
        
        # 軸の設定（小画面対応）
        fig.update_yaxes(
            title_text="河川水位 (m)",
            range=[0, 6],
            dtick=1,
            secondary_y=False,
            title_font_size=12,
            tickfont_size=12
        )
        fig.update_yaxes(
            title_text="全放流量 (m³/s)",
            range=[0, 900],
            dtick=150,
            secondary_y=True,
            title_font_size=12,
            tickfont_size=12
        )
        
        # 共通の時間範囲を取得して設定
        time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode)
        xaxis_config = dict(
            title_text="時刻",
            title_font_size=12,
            tickfont_size=12
        )
        if time_min and time_max:
            xaxis_config['range'] = [time_min, time_max]
        
        fig.update_xaxes(**xaxis_config)
        
        fig.update_layout(
            height=465,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.30,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            margin=dict(t=30, l=40, r=40, b=140),
            autosize=True,
            font=dict(size=9)
        )
        
        # デモモード時のY軸範囲設定
        if demo_mode:
            fig.update_yaxes(range=[0, 8], secondary_y=False)  # 左軸（河川水位）：最大8
            fig.update_yaxes(range=[0, 1200], secondary_y=True)  # 右軸（全放流量）：最大1200
        
        # インタラクションが無効の場合は軸を固定
        if not enable_interaction:
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True, secondary_y=False)
            fig.update_yaxes(fixedrange=True, secondary_y=True)
        
        return fig
    
    def create_dam_water_level_graph(self, history_data: List[Dict[str, Any]], enable_interaction: bool = False, latest_precipitation_data: Dict[str, Any] = None, display_hours: int = 24, demo_mode: bool = False) -> go.Figure:
        """ダム水位グラフを作成（ダム水位 + 時間雨量の二軸表示）"""
        # 現在時刻を取得（予測データ処理で使用）
        now_jst = datetime.now(ZoneInfo('Asia/Tokyo'))
        
        # 表示期間に基づいてデータをフィルタリング（デモモード時はスキップ）
        if demo_mode:
            filtered_data = history_data
        else:
            time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode=False)
            if time_min and time_max:
                filtered_data = self.filter_data_by_time_range(history_data, time_min, time_max)
            else:
                filtered_data = history_data
        
        if not filtered_data:
            fig = go.Figure()
            fig.add_annotation(
                text="表示するデータがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # データをDataFrameに変換
        df_data = []
        for item in filtered_data:
            # 観測時刻（data_time）を使用、なければtimestampを使用
            data_time = item.get('data_time') or item.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
                # タイムゾーンがない場合はJSTとして扱う
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                else:
                    dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
            except:
                continue
                
            row = {'timestamp': dt}
            
            # ダム水位
            dam_level = item.get('dam', {}).get('water_level')
            if dam_level is not None:
                row['dam_level'] = dam_level
            
            # 雨量
            rainfall = item.get('rainfall', {}).get('hourly')
            if rainfall is not None:
                row['rainfall'] = rainfall
            
            df_data.append(row)
        
        if not df_data:
            fig = go.Figure()
            fig.add_annotation(
                text="有効なデータがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df = pd.DataFrame(df_data)
        
        # 二軸グラフを作成
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # ダム水位（左軸）
        if 'dam_level' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['dam_level'],
                    mode='lines+markers',
                    name='ダム貯水位（厚東川ダム）',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=6, color='white', line=dict(width=2, color='#ff7f0e'))
                ),
                secondary_y=False
            )
        
        # 時間雨量（右軸）
        if 'rainfall' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['rainfall'],
                    name='時間雨量（厚東川ダム）',
                    marker_color='#87CEEB',
                    opacity=0.7,
                    width=600000
                ),
                secondary_y=True
            )
        
        # 降水強度・時間雨量データを追加
        # 表示期間の計算
        end_time = now_jst
        start_time = end_time - timedelta(hours=display_hours)
        
        # 観測値の処理（APIデータを優先、なければ履歴から取得）
        obs_times = []
        obs_intensities = []
        out_of_range_count = 0
        latest_out_of_range_time = None
        
        # まず最新のAPIデータから観測値を取得
        if latest_precipitation_data and latest_precipitation_data.get('observation'):
            for item in latest_precipitation_data['observation']:
                try:
                    dt = datetime.fromisoformat(item['datetime'])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                    else:
                        dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                    
                    # 表示期間内のデータのみを追加
                    if start_time <= dt <= end_time:
                        obs_times.append(dt)
                        obs_intensities.append(item['intensity'])
                    else:
                        # 範囲外データをカウント
                        out_of_range_count += 1
                        if latest_out_of_range_time is None or dt > latest_out_of_range_time:
                            latest_out_of_range_time = dt
                except (ValueError, KeyError):
                    continue
        
        # APIデータがない場合は履歴データから観測値を取得
        if not obs_times and history_data:
            # 表示期間に基づいてデータをフィルタリング（デモモード時はスキップ）
            if demo_mode:
                filtered_history_data = history_data
            else:
                time_min_history, time_max_history = self.get_common_time_range(history_data, display_hours, demo_mode=False)
                if time_min_history and time_max_history:
                    filtered_history_data = self.filter_data_by_time_range(history_data, time_min_history, time_max_history - timedelta(hours=2))
                else:
                    filtered_history_data = history_data
            
            for item in filtered_history_data:
                precip_data = item.get('precipitation_intensity', {})
                if precip_data.get('observation'):
                    for obs in precip_data['observation']:
                        try:
                            dt = datetime.fromisoformat(obs['datetime'])
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                            else:
                                dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                            
                            # 表示期間内のデータのみを追加
                            if start_time <= dt <= end_time:
                                obs_times.append(dt)
                                obs_intensities.append(obs['intensity'])
                            else:
                                # 範囲外データをカウント
                                out_of_range_count += 1
                                if latest_out_of_range_time is None or dt > latest_out_of_range_time:
                                    latest_out_of_range_time = dt
                        except (ValueError, KeyError):
                            continue
        
        # 範囲外データのログ表示
        if out_of_range_count > 0 and latest_out_of_range_time:
            latest_time_str = latest_out_of_range_time.strftime('%Y-%m-%d %H:%M')
            st.info(f"🔍 表示期間外の降水強度観測値: {out_of_range_count}件 (最新: {latest_time_str})")
        
        # 観測値をプロット
        if obs_times and obs_intensities:
            fig.add_trace(
                go.Bar(
                    x=obs_times,
                    y=obs_intensities,
                    name='降水強度・観測値（厚東川ダム by Yahoo!）',
                    marker_color='#DC143C',
                    opacity=0.8,
                    width=600000,
                    hovertemplate='<b>観測値</b><br>%{x|%H:%M}<br>降水強度: %{y:.1f} mm/h<extra></extra>'
                ),
                secondary_y=True
            )
            
        # 予測値の処理（現在時刻以降のみ、APIデータから取得）
        if latest_precipitation_data and latest_precipitation_data.get('forecast'):
                forecast_times = []
                forecast_intensities = []
                for item in latest_precipitation_data['forecast']:
                    try:
                        dt = datetime.fromisoformat(item['datetime'])
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                        else:
                            dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                        
                        # 現在時刻以降のデータまたは過去30分以内の予測データを使用
                        time_diff = (now_jst - dt).total_seconds() / 60  # 分単位の差
                        if dt >= now_jst or time_diff <= 30:
                            forecast_times.append(dt)
                            forecast_intensities.append(item['intensity'])
                    except (ValueError, KeyError):
                        continue
                
                if forecast_times and forecast_intensities:
                    fig.add_trace(
                        go.Bar(
                            x=forecast_times,
                            y=forecast_intensities,
                            name='降水強度・予測値（厚東川ダム by Yahoo!）',
                            marker_color='#FF1493',
                            opacity=0.6,
                            width=600000,
                            hovertemplate='<b>予測値</b><br>%{x|%H:%M}<br>降水強度: %{y:.1f} mm/h<extra></extra>'
                        ),
                        secondary_y=True
                    )
        
        # 軸の設定（小画面対応）
        fig.update_yaxes(
            title_text="ダム貯水位 (m)",
            range=[20, 45],
            dtick=2.5,
            secondary_y=False,
            title_font_size=12,
            tickfont_size=12
        )
        fig.update_yaxes(
            title_text="時間雨量 (mm/h)",
            range=[0, 50],
            dtick=5,
            secondary_y=True,
            title_font_size=12,
            tickfont_size=12
        )
        
        # 共通の時間範囲を取得して設定
        time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode)
        xaxis_config = dict(
            title_text="時刻",
            title_font_size=12,
            tickfont_size=12
        )
        if time_min and time_max:
            xaxis_config['range'] = [time_min, time_max]
        
        fig.update_xaxes(**xaxis_config)
        
        fig.update_layout(
            height=465,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.30,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            margin=dict(t=30, l=40, r=40, b=140),
            autosize=True,
            font=dict(size=9)
        )
        
        # インタラクションが無効の場合は軸を固定
        if not enable_interaction:
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True, secondary_y=False)
            fig.update_yaxes(fixedrange=True, secondary_y=True)
        
        return fig
    
    def create_dam_discharge_rainfall_graph(self, history_data: List[Dict[str, Any]], enable_interaction: bool = False, latest_precipitation_data: Dict[str, Any] = None, display_hours: int = 24, demo_mode: bool = False) -> go.Figure:
        """ダム放流量グラフを作成（ダム放流量 + 時間雨量の二軸表示）"""
        # 現在時刻を取得（予測データ処理で使用）
        now_jst = datetime.now(ZoneInfo('Asia/Tokyo'))
        
        # 表示期間に基づいてデータをフィルタリング（デモモード時はスキップ）
        if demo_mode:
            filtered_data = history_data
        else:
            time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode=False)
            if time_min and time_max:
                filtered_data = self.filter_data_by_time_range(history_data, time_min, time_max)
            else:
                filtered_data = history_data
        
        if not filtered_data:
            fig = go.Figure()
            fig.add_annotation(
                text="表示するデータがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # データをDataFrameに変換
        df_data = []
        for item in filtered_data:
            # 観測時刻（data_time）を使用、なければtimestampを使用
            data_time = item.get('data_time') or item.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
                # タイムゾーンがない場合はJSTとして扱う
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                else:
                    dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
            except:
                continue
                
            row = {'timestamp': dt}
            
            # ダム放流量
            dam_discharge = item.get('dam', {}).get('outflow')
            if dam_discharge is not None:
                row['dam_discharge'] = dam_discharge
            
            # 雨量
            rainfall = item.get('rainfall', {}).get('hourly')
            if rainfall is not None:
                row['rainfall'] = rainfall
            
            df_data.append(row)
        
        if not df_data:
            fig = go.Figure()
            fig.add_annotation(
                text="有効なデータがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df = pd.DataFrame(df_data)
        
        # 二軸グラフを作成
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # ダム放流量（左軸）
        if 'dam_discharge' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['dam_discharge'],
                    mode='lines+markers',
                    name='全放流量（厚東川ダム）',
                    line=dict(color='#d62728', width=3),
                    marker=dict(size=6, color='white', line=dict(width=2, color='#d62728'))
                ),
                secondary_y=False
            )
        
        # 時間雨量（右軸）
        if 'rainfall' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['rainfall'],
                    name='時間雨量（厚東川ダム）',
                    marker_color='#87CEEB',
                    opacity=0.7,
                    width=600000
                ),
                secondary_y=True
            )
        
        # 降水強度・時間雨量データを追加
        # 表示期間の計算
        end_time = now_jst
        start_time = end_time - timedelta(hours=display_hours)
        
        # 観測値の処理（APIデータを優先、なければ履歴から取得）
        obs_times = []
        obs_intensities = []
        out_of_range_count = 0
        latest_out_of_range_time = None
        
        # まず最新のAPIデータから観測値を取得
        if latest_precipitation_data and latest_precipitation_data.get('observation'):
            for item in latest_precipitation_data['observation']:
                try:
                    dt = datetime.fromisoformat(item['datetime'])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                    else:
                        dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                    
                    # 表示期間内のデータのみを追加
                    if start_time <= dt <= end_time:
                        obs_times.append(dt)
                        obs_intensities.append(item['intensity'])
                    else:
                        # 範囲外データをカウント
                        out_of_range_count += 1
                        if latest_out_of_range_time is None or dt > latest_out_of_range_time:
                            latest_out_of_range_time = dt
                except (ValueError, KeyError):
                    continue
        
        # APIデータがない場合は履歴データから観測値を取得
        if not obs_times and history_data:
            # 表示期間に基づいてデータをフィルタリング（デモモード時はスキップ）
            if demo_mode:
                filtered_history_data = history_data
            else:
                time_min_history, time_max_history = self.get_common_time_range(history_data, display_hours, demo_mode=False)
                if time_min_history and time_max_history:
                    filtered_history_data = self.filter_data_by_time_range(history_data, time_min_history, time_max_history - timedelta(hours=2))
                else:
                    filtered_history_data = history_data
            
            for item in filtered_history_data:
                precip_data = item.get('precipitation_intensity', {})
                if precip_data.get('observation'):
                    for obs in precip_data['observation']:
                        try:
                            dt = datetime.fromisoformat(obs['datetime'])
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                            else:
                                dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                            
                            # 表示期間内のデータのみを追加
                            if start_time <= dt <= end_time:
                                obs_times.append(dt)
                                obs_intensities.append(obs['intensity'])
                            else:
                                # 範囲外データをカウント
                                out_of_range_count += 1
                                if latest_out_of_range_time is None or dt > latest_out_of_range_time:
                                    latest_out_of_range_time = dt
                        except (ValueError, KeyError):
                            continue
        
        # 範囲外データのログ表示
        if out_of_range_count > 0 and latest_out_of_range_time:
            latest_time_str = latest_out_of_range_time.strftime('%Y-%m-%d %H:%M')
            st.info(f"🔍 表示期間外の降水強度観測値: {out_of_range_count}件 (最新: {latest_time_str})")
        
        # 観測値をプロット
        if obs_times and obs_intensities:
            fig.add_trace(
                go.Bar(
                    x=obs_times,
                    y=obs_intensities,
                    name='降水強度・観測値（厚東川ダム by Yahoo!）',
                    marker_color='#DC143C',
                    opacity=0.8,
                    width=600000,
                    hovertemplate='<b>観測値</b><br>%{x|%H:%M}<br>降水強度: %{y:.1f} mm/h<extra></extra>'
                ),
                secondary_y=True
            )
            
        # 予測値の処理（現在時刻以降のみ、APIデータから取得）
        if latest_precipitation_data and latest_precipitation_data.get('forecast'):
                forecast_times = []
                forecast_intensities = []
                for item in latest_precipitation_data['forecast']:
                    try:
                        dt = datetime.fromisoformat(item['datetime'])
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                        else:
                            dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                        
                        # 現在時刻以降のデータまたは過去30分以内の予測データを使用
                        time_diff = (now_jst - dt).total_seconds() / 60  # 分単位の差
                        if dt >= now_jst or time_diff <= 30:
                            forecast_times.append(dt)
                            forecast_intensities.append(item['intensity'])
                    except (ValueError, KeyError):
                        continue
                
                if forecast_times and forecast_intensities:
                    fig.add_trace(
                        go.Bar(
                            x=forecast_times,
                            y=forecast_intensities,
                            name='降水強度・予測値（厚東川ダム by Yahoo!）',
                            marker_color='#FF1493',
                            opacity=0.6,
                            width=600000,
                            hovertemplate='<b>予測値</b><br>%{x|%H:%M}<br>降水強度: %{y:.1f} mm/h<extra></extra>'
                        ),
                        secondary_y=True
                    )
        
        # 軸の設定（小画面対応）
        fig.update_yaxes(
            title_text="ダム放流量 (m³/s)",
            range=[0, 1200],
            dtick=100,
            secondary_y=False,
            title_font_size=12,
            tickfont_size=12
        )
        fig.update_yaxes(
            title_text="時間雨量 (mm/h)",
            range=[0, 60],
            dtick=5,
            secondary_y=True,
            title_font_size=12,
            tickfont_size=12
        )
        
        # 共通の時間範囲を取得して設定
        time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode)
        xaxis_config = dict(
            title_text="時刻",
            title_font_size=12,
            tickfont_size=12
        )
        if time_min and time_max:
            xaxis_config['range'] = [time_min, time_max]
        
        fig.update_xaxes(**xaxis_config)
        
        fig.update_layout(
            height=465,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.30,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            margin=dict(t=30, l=40, r=40, b=140),
            autosize=True,
            font=dict(size=9)
        )
        
        # インタラクションが無効の場合は軸を固定
        if not enable_interaction:
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True, secondary_y=False)
            fig.update_yaxes(fixedrange=True, secondary_y=True)
        
        return fig
    
    def create_dam_flow_graph(self, history_data: List[Dict[str, Any]], enable_interaction: bool = False, display_hours: int = 24, demo_mode: bool = False) -> go.Figure:
        """ダム流入出量グラフを作成（流入量・全放流量 + 累加雨量の二軸表示）"""
        # 現在時刻を取得
        now_jst = datetime.now(ZoneInfo('Asia/Tokyo'))
        
        # 表示期間に基づいてデータをフィルタリング（デモモード時はスキップ）
        if demo_mode:
            filtered_data = history_data
        else:
            time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode=False)
            if time_min and time_max:
                filtered_data = self.filter_data_by_time_range(history_data, time_min, time_max)
            else:
                filtered_data = history_data
        
        if not filtered_data:
            fig = go.Figure()
            fig.add_annotation(
                text="表示するデータがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # データをDataFrameに変換
        df_data = []
        for item in filtered_data:
            # 観測時刻（data_time）を使用、なければtimestampを使用
            data_time = item.get('data_time') or item.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
                # タイムゾーンがない場合はJSTとして扱う
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                else:
                    dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
            except:
                continue
                
            row = {'timestamp': dt}
            
            # ダム流入量
            inflow = item.get('dam', {}).get('inflow')
            if inflow is not None:
                row['inflow'] = inflow
            
            # ダム全放流量
            outflow = item.get('dam', {}).get('outflow')
            if outflow is not None:
                row['outflow'] = outflow
            
            # 累加雨量
            cumulative_rainfall = item.get('rainfall', {}).get('cumulative')
            if cumulative_rainfall is not None:
                row['cumulative_rainfall'] = cumulative_rainfall
            
            df_data.append(row)
        
        if not df_data:
            fig = go.Figure()
            fig.add_annotation(
                text="有効なデータがありません",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df = pd.DataFrame(df_data)
        
        # 二軸グラフを作成
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 累加雨量（右軸）- 塗りつぶし背景として最初に追加（マーカーなし）
        if 'cumulative_rainfall' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cumulative_rainfall'],
                    mode='lines',
                    name='累加雨量（厚東川ダム）',
                    line=dict(color='#87CEEB', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(135, 206, 235, 0.3)'
                ),
                secondary_y=True
            )
        
        # ダム流入量（左軸）- 線グラフを累加雨量の上に表示
        if 'inflow' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['inflow'],
                    mode='lines+markers',
                    name='流入量（厚東川ダム）',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(size=6, color='white', line=dict(width=2, color='#2ca02c'))
                ),
                secondary_y=False
            )
        
        # ダム全放流量（左軸）- 線グラフを累加雨量の上に表示
        if 'outflow' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['outflow'],
                    mode='lines+markers',
                    name='全放流量（厚東川ダム）',
                    line=dict(color='#d62728', width=3),
                    marker=dict(size=6, color='white', line=dict(width=2, color='#d62728'))
                ),
                secondary_y=False
            )
        
        # 軸の設定（小画面対応）
        fig.update_yaxes(
            title_text="流量 (m³/s)",
            range=[0, 900],
            dtick=100,
            secondary_y=False,
            title_font_size=12,
            tickfont_size=12
        )
        fig.update_yaxes(
            title_text="累加雨量 (mm)",
            range=[0, 180],
            dtick=20,
            secondary_y=True,
            title_font_size=12,
            tickfont_size=12
        )
        
        # 共通の時間範囲を取得して設定
        time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode)
        xaxis_config = dict(
            title_text="時刻",
            title_font_size=12,
            tickfont_size=12
        )
        if time_min and time_max:
            xaxis_config['range'] = [time_min, time_max]
        
        fig.update_xaxes(**xaxis_config)
        
        fig.update_layout(
            height=465,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.30,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            margin=dict(t=30, l=40, r=40, b=140),
            autosize=True,
            font=dict(size=9)
        )
        
        # デモモード時のY軸範囲設定
        if demo_mode:
            fig.update_yaxes(range=[0, 1200], secondary_y=False)  # 左軸（流入出量）：最大1200
            fig.update_yaxes(range=[0, 300], dtick=25, secondary_y=True)  # 右軸（累加雨量）：最大300、間隔25mm
        
        # インタラクションが無効の場合は軸を固定
        if not enable_interaction:
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True, secondary_y=False)
            fig.update_yaxes(fixedrange=True, secondary_y=True)
        
        return fig
    
    def create_precipitation_intensity_graph(self, precipitation_data: Dict[str, Any], enable_interaction: bool = True, history_data: List[Dict[str, Any]] = None, display_hours: int = 24, demo_mode: bool = False) -> go.Figure:
        """降水強度グラフを作成"""
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 現在時刻を取得
        now_jst = datetime.now(ZoneInfo('Asia/Tokyo'))
        
        # 表示期間の計算
        end_time = now_jst
        start_time = end_time - timedelta(hours=display_hours)
        
        # 観測データの処理（時間範囲フィルタリングあり）
        obs_times = []
        obs_intensities = []
        out_of_range_count = 0
        latest_out_of_range_time = None
        
        if precipitation_data.get('observation'):
            for item in precipitation_data['observation']:
                try:
                    dt = datetime.fromisoformat(item['datetime'])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                    else:
                        dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                    
                    # 表示期間内のデータのみを追加
                    if start_time <= dt <= end_time:
                        obs_times.append(dt)
                        obs_intensities.append(item['intensity'])
                    else:
                        # 範囲外データをカウント
                        out_of_range_count += 1
                        if latest_out_of_range_time is None or dt > latest_out_of_range_time:
                            latest_out_of_range_time = dt
                except (ValueError, KeyError):
                    continue
        
        # 範囲外データのログ表示
        if out_of_range_count > 0 and latest_out_of_range_time:
            latest_time_str = latest_out_of_range_time.strftime('%Y-%m-%d %H:%M')
            st.info(f"🔍 表示期間外の降水強度観測値: {out_of_range_count}件 (最新: {latest_time_str})")
        
        # 予測データの処理（現在時刻以降のみ、時間範囲フィルタリングなし）
        forecast_times = []
        forecast_intensities = []
        
        if precipitation_data.get('forecast'):
            # デバッグ情報：予測データの時刻範囲を表示
            forecast_debug_times = []
            for item in precipitation_data['forecast']:
                try:
                    dt = datetime.fromisoformat(item['datetime'])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                    else:
                        dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                    forecast_debug_times.append(dt)
                    
                    # 現在時刻以降のデータまたは過去30分以内の予測データを使用
                    time_diff = (now_jst - dt).total_seconds() / 60  # 分単位の差
                    if dt >= now_jst or time_diff <= 30:
                        forecast_times.append(dt)
                        forecast_intensities.append(item['intensity'])
                except (ValueError, KeyError):
                    continue
            
        
        # 観測データのプロット（棒グラフ、左軸）
        if obs_times and obs_intensities:
            fig.add_trace(go.Bar(
                x=obs_times,
                y=obs_intensities,
                name='降水強度・観測値（厚東川ダム by Yahoo!）',
                marker=dict(color='#DC143C'),
                hovertemplate='<b>観測値</b><br>%{x|%H:%M}<br>降水強度: %{y:.1f} mm/h<extra></extra>',
                width=600000
            ), secondary_y=False)
        
        # 予測データのプロット（棒グラフ、左軸）
        if forecast_times and forecast_intensities:
            fig.add_trace(go.Bar(
                x=forecast_times,
                y=forecast_intensities,
                name='降水強度・予測値（厚東川ダム by Yahoo!）',
                marker=dict(color='#FF1493', opacity=0.7),
                hovertemplate='<b>予測値</b><br>%{x|%H:%M}<br>降水強度: %{y:.1f} mm/h<extra></extra>',
                width=600000
            ), secondary_y=False)
        
        # 時間雨量データの追加（右軸）
        if history_data:
            rainfall_times = []
            rainfall_values = []
            
            # 表示期間に基づいてデータをフィルタリング（デモモード時はスキップ）
            if demo_mode:
                filtered_history_data = history_data
            else:
                time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode=False)
                if time_min and time_max:
                    filtered_history_data = self.filter_data_by_time_range(history_data, time_min, time_max)
                else:
                    filtered_history_data = history_data
            
            for item in filtered_history_data:
                # 観測時刻（data_time）を使用、なければtimestampを使用
                data_time = item.get('data_time') or item.get('timestamp', '')
                try:
                    dt = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                    else:
                        dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                    
                    rainfall = item.get('rainfall', {}).get('hourly')
                    if rainfall is not None:
                        rainfall_times.append(dt)
                        rainfall_values.append(rainfall)
                except:
                    continue
            
            if rainfall_times and rainfall_values:
                fig.add_trace(go.Bar(
                    x=rainfall_times,
                    y=rainfall_values,
                    name='時間雨量（厚東川ダム）',
                    marker=dict(color='#87CEEB', opacity=0.7),
                    hovertemplate='<b>時間雨量</b><br>%{x|%H:%M}<br>雨量: %{y:.1f} mm/h<extra></extra>',
                    width=600000
                ), secondary_y=True)
        
        # レイアウト設定
        fig.update_layout(
            height=465,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.30,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            margin=dict(t=30, l=40, r=40, b=140),
            autosize=True,
            font=dict(size=9)
        )
        
        # 軸設定 - 履歴データから共通の時間範囲を取得（河川水位グラフと同じ範囲）
        time_min, time_max = None, None
        if history_data:
            time_min, time_max = self.get_common_time_range(history_data, display_hours, demo_mode)
        
        xaxis_config = dict(
            title_text="時刻",
            title_font_size=12,
            tickfont_size=12
        )
        if time_min and time_max:
            xaxis_config['range'] = [time_min, time_max]
        
        fig.update_xaxes(**xaxis_config)
        
        # 左軸（降水強度）の設定
        fig.update_yaxes(
            title_text="降水強度 (mm/h)",
            range=[0, 50],
            dtick=5,
            secondary_y=False,
            title_font_size=12,
            tickfont_size=12
        )
        
        # 右軸（時間雨量）の設定
        fig.update_yaxes(
            title_text="時間雨量 (mm/h)",
            range=[0, 50],
            dtick=5,
            secondary_y=True,
            title_font_size=12,
            tickfont_size=12
        )
        
        # インタラクションが無効の場合は軸を固定
        if not enable_interaction:
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True, secondary_y=False)
            fig.update_yaxes(fixedrange=True, secondary_y=True)
        
        return fig
    
    def create_data_table(self, history_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """データテーブルを作成"""
        if not history_data:
            return pd.DataFrame()
        
        table_data = []
        for item in history_data[-20:]:  # 最新20件
            # 観測時刻（data_time）を使用、なければtimestampを使用
            data_time = item.get('data_time') or item.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
                # タイムゾーンがない場合はJSTとして扱う
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                else:
                    dt = dt.astimezone(ZoneInfo('Asia/Tokyo'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_time = data_time
            
            table_data.append({
                'ダム貯水位(m)': item.get('dam', {}).get('water_level', '--'),
                'ダム貯水率(%)': item.get('dam', {}).get('storage_rate', '--'),
                'ダム流入量(m³/s)': item.get('dam', {}).get('inflow', '--'),
                'ダム全放流量(m³/s)': item.get('dam', {}).get('outflow', '--'),
                '水位(m)（持世寺）': item.get('river', {}).get('water_level', '--'),
                '観測日時': formatted_time
            })
        
        return pd.DataFrame(table_data).iloc[::-1]  # 新しい順に並び替え
    

def main():
    """メイン関数"""
    monitor = KotogawaMonitor()
    
    # サイドバー設定
    # 更新設定
    with st.sidebar.expander("更新設定", expanded=True):
        
        # 自動更新設定
        refresh_interval = st.selectbox(
            "自動更新間隔",
            options=[
                ("自動更新なし", 0),
                ("10分（推奨）", 10 * 60 * 1000),
                ("30分", 30 * 60 * 1000),
                ("60分", 60 * 60 * 1000)
            ],
            index=1,  # デフォルトは10分
            format_func=lambda x: x[0]
        )
        
        # 手動更新ボタン
        if st.button("手動更新", type="primary", key="sidebar_refresh"):
            monitor.load_history_data.clear()
            st.cache_data.clear()
            st.rerun()
    
    # 表示設定
    with st.sidebar.expander("表示設定", expanded=False):
        # 表示期間設定
        display_hours = st.selectbox(
            "表示期間",
            [6, 12, 24, 48, 72, 96, 120],
            index=2,
            format_func=lambda x: f"{x}時間"
        )
        
        # グラフ操作設定
        enable_graph_interaction = st.checkbox(
            "グラフの編集の有効化",
            value=False,
            help="チェックを入れるとグラフの拡大・縮小・移動が可能になります"
        )
        
        
        # 週間天気表示設定
        show_weekly_weather = st.checkbox(
            "週間天気を表示",
            value=True,
            help="チェックを外すと週間天気予報を非表示にします"
        )
        
        # デモモード設定
        demo_mode = st.checkbox(
            "デモモード",
            value=False,
            help="過去の河川・ダムデータ（2023/6/25-7/2）を表示します"
        )
        
        # デモモード時の日時選択
        demo_datetime = None
        if demo_mode:
            st.markdown("**デモモード日時設定**")
            col_date, col_time = st.columns(2)
            
            with col_date:
                demo_date = st.date_input(
                    "日付",
                    value=datetime(2023, 6, 30).date(),
                    min_value=datetime(2023, 6, 25).date(),
                    max_value=datetime(2023, 7, 1).date(),
                    help="2023/6/25〜7/1の範囲で選択",
                    key="demo_date_input"  # 一意のキーを追加
                )
            
            with col_time:
                demo_time = st.time_input(
                    "時刻",
                    value=datetime(2023, 6, 30, 12, 0).time(),
                    help="表示する時刻を選択",
                    key="demo_time_input"  # 一意のキーを追加
                )
            
            # 日時を結合
            demo_datetime = datetime.combine(demo_date, demo_time)
            demo_datetime = demo_datetime.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
    
    # AI予測設定
    with st.sidebar.expander("AI予測設定", expanded=False):
        # AI予測モデル選択
        if AI_PREDICTION_AVAILABLE:
            st.markdown("**AI予測設定**")
            prediction_model = st.radio(
                "予測モデル",
                ["エキスパートルール予測", "Riverオンライン学習予測"],
                index=0,
                help="予測モデルを選択してください"
            )
            
            # セッション状態に保存
            st.session_state.prediction_model = prediction_model
            
            # モデル解説へのリンク
            st.markdown("[🤖 AI予測モデルの詳細解説を見る](/AI予測モデル解説)")
            
            # 現在の予測精度を表示
            if EVALUATION_AVAILABLE and 'evaluator' in st.session_state:
                evaluator = st.session_state.evaluator
                model_type = 'expert_rule' if prediction_model == "エキスパートルール予測" else 'river_online'
                
                if model_type in evaluator.evaluation_results:
                    latest = evaluator.evaluation_results[model_type].get('latest')
                    if latest and 'overall' in latest:
                        score = latest['overall'].get('score', 0)
                        mae = latest['overall'].get('mae', 0)
                        
                        # スコアに応じて色を変更
                        if score >= 80:
                            color = "green"
                        elif score >= 60:
                            color = "orange"
                        else:
                            color = "red"
                            
                        st.markdown(f"**現在の精度スコア:** <span style='color: {color}'>{score:.1f}/100</span>", unsafe_allow_html=True)
                        st.caption(f"平均絶対誤差: {mae:.3f}m")
            
            # モデル動作状態の表示
            st.markdown("---")
            st.markdown("**📊 モデル動作状態**")
            
            # システム診断情報
            with st.expander("🔍 システム診断情報", expanded=False):
                st.caption("**利用可能なモジュール:**")
                st.caption(f"• エキスパートルール予測: {'✅' if AI_PREDICTION_AVAILABLE else '❌'}")
                st.caption(f"• Riverストリーミング予測: {'✅' if RIVER_STREAMING_AVAILABLE else '❌'}")
                st.caption(f"• Riverオンライン学習: {'✅' if RIVER_LEARNING_AVAILABLE else '❌'}")
                st.caption(f"• River import helper: {'✅' if get_river_predictor is not None else '❌'}")
                st.caption(f"• 予測評価機能: {'✅' if EVALUATION_AVAILABLE else '❌'}")
                
                if not RIVER_LEARNING_AVAILABLE:
                    st.caption("**River関連のトラブルシューティング:**")
                    st.caption("1. river ライブラリがインストールされているか確認")
                    st.caption("2. scripts/river_import_helper.py が存在するか確認")
                    st.caption("3. 依存関係のバージョン互換性を確認")
            
            if 'predictor' in st.session_state:
                predictor = st.session_state.predictor
                
                # 実際に動作しているモデルを確認
                actual_model_name = type(predictor).__name__
                
                # 期待されるモデルと実際のモデルを比較
                expected_model = "RiverStreamingPredictor" if prediction_model == "Riverオンライン学習予測" else "AdvancedRiverLevelPredictor"
                
                if actual_model_name == expected_model:
                    st.success(f"✅ 実行中: {actual_model_name}")
                else:
                    st.warning(f"⚠️ 代替モデル実行中: {actual_model_name}")
                    st.caption(f"期待: {expected_model}")
                
                # モデル固有の情報を表示
                if hasattr(predictor, 'n_samples'):
                    st.info(f"学習サンプル数: {predictor.n_samples}")
                elif hasattr(predictor, 'n_learned'):
                    st.info(f"学習回数: {predictor.n_learned}")
                
                # ストリーミングモデルの場合
                if hasattr(predictor, 'feature_extractor'):
                    st.caption("ストリーミング処理: ✅ 有効")
                    if hasattr(predictor, 'delay_estimator'):
                        st.caption("動的遅延推定: ✅ 有効")
                
                # モデルの詳細情報
                if hasattr(predictor, 'get_model_info'):
                    try:
                        model_info = predictor.get_model_info()
                        if 'model_type' in model_info:
                            st.caption(f"アルゴリズム: {model_info['model_type']}")
                    except:
                        pass
                
                # 予測メソッドの確認
                if hasattr(predictor, 'predict_one'):
                    st.caption("予測方式: ストリーミング（predict_one）")
                elif hasattr(predictor, 'predict'):
                    st.caption("予測方式: バッチ（predict）")
                
                # 実際に使用された予測メソッド
                if 'prediction_method_used' in st.session_state:
                    st.caption(f"最終実行: {st.session_state['prediction_method_used']}")
                if 'ai_prediction_error' in st.session_state:
                    st.error(f"最新エラー: {st.session_state['ai_prediction_error']}")
            else:
                st.warning("モデル未初期化")
            
            # モデル再初期化ボタン
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 モデルを再初期化", type="secondary", use_container_width=True):
                    # セッション状態から予測モデル関連をクリア
                    if 'predictor' in st.session_state:
                        del st.session_state['predictor']
                    if 'last_prediction_model' in st.session_state:
                        del st.session_state['last_prediction_model']
                    if 'ai_prediction_error' in st.session_state:
                        del st.session_state['ai_prediction_error']
                    if 'prediction_method_used' in st.session_state:
                        del st.session_state['prediction_method_used']
                    st.success("モデルを再初期化しました")
                    st.rerun()
            
            with col2:
                # Riverモデルの場合は学習データもクリア可能
                if prediction_model == "Riverオンライン学習予測":
                    if st.button("🗑️ 学習データをクリア", type="secondary", use_container_width=True):
                        try:
                            # モデルファイルを削除
                            import os
                            model_files = [
                                'models/river_streaming_model.pkl',
                                'models/river_online_model.pkl'
                            ]
                            for model_file in model_files:
                                if os.path.exists(model_file):
                                    os.remove(model_file)
                            # セッション状態をクリア
                            if 'predictor' in st.session_state:
                                del st.session_state['predictor']
                            if 'last_prediction_model' in st.session_state:
                                del st.session_state['last_prediction_model']
                            st.success("学習データをクリアしました")
                            st.rerun()
                        except Exception as e:
                            st.error(f"学習データのクリアに失敗: {e}")
                        
            # 予測精度評価ページへのリンク
            st.markdown("[📈 予測精度の詳細を見る](/予測精度評価)")
        else:
            st.info("AI予測機能は利用できません")
    
    # アラート閾値設定
    with st.sidebar.expander("アラート設定", expanded=False):
        river_warning = st.number_input("河川警戒水位 (m)", value=3.8, step=0.1)
        river_danger = st.number_input("河川危険水位 (m)", value=5.0, step=0.1)
        dam_warning = st.number_input("ダム警戒水位 (m)", value=39.2, step=0.1, help="洪水時最高水位")
        dam_danger = st.number_input("ダム危険水位 (m)", value=40.0, step=0.1, help="設計最高水位")
    
    thresholds = {
        'river_warning': river_warning,
        'river_danger': river_danger,
        'dam_warning': dam_warning,
        'dam_danger': dam_danger
    }
    
    # システムヘッダーの表示
    st.markdown('<h1 style="text-align: center; margin-top: 0; margin-bottom: 1rem;">厚東川氾濫監視システムv2.0</h1>', unsafe_allow_html=True)
    
    
    # 自動更新の実行（ヘッダーの後に配置）- デモモード時は無効化
    if refresh_interval[1] > 0 and not demo_mode:
        count = st_autorefresh(
            interval=refresh_interval[1],
            limit=None,
            key="autorefresh"
        )
    
    # データ読み込み
    if demo_mode:
        # デモモードの場合はサンプルデータを読み込む
        with st.spinner('デモデータを読み込み中...'):
            sample_data = monitor.load_sample_csv_data()
            if sample_data and demo_datetime:
                # 選択された日時を基準にデータをフィルタリング
                # demo_datetimeより前のデータのみを抽出（表示期間分）
                filtered_sample_data = []
                for data in sample_data:
                    try:
                        data_time = datetime.fromisoformat(data.get('data_time', data.get('timestamp')).replace('Z', '+00:00'))
                        if data_time.tzinfo is None:
                            data_time = data_time.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                        else:
                            data_time = data_time.astimezone(ZoneInfo('Asia/Tokyo'))
                        
                        # 選択日時から過去display_hours時間分のデータを取得
                        if data_time <= demo_datetime and data_time >= demo_datetime - timedelta(hours=display_hours):
                            filtered_sample_data.append(data)
                    except:
                        continue
                
                # 選択日時に最も近いデータを最新データとする
                if filtered_sample_data:
                    latest_data = max(filtered_sample_data, key=lambda x: datetime.fromisoformat(
                        x.get('data_time', x.get('timestamp')).replace('Z', '+00:00')
                    ).astimezone(ZoneInfo('Asia/Tokyo')))
                    history_data = filtered_sample_data
                else:
                    latest_data = None
                    history_data = []
            elif sample_data:
                latest_data = sample_data[-1]  # 最新のデータポイントを取得
                history_data = sample_data
            else:
                latest_data = None
                history_data = []
        cache_key = f"demo_mode_{demo_datetime.isoformat() if demo_datetime else 'default'}"
    else:
        # 通常モード
        with st.spinner('データを更新中...'):
            latest_data = monitor.load_latest_data()
        
        # キャッシュキー取得
        cache_key = monitor.get_cache_key()
        
        # 履歴データの読み込み
        try:
            with st.spinner("履歴データを読み込み中..."):
                history_data = monitor.load_history_data(120, cache_key)
        except Exception as e:
            st.warning(f"履歴データの読み込みに失敗しました: {e}")
            history_data = []
    
    # アラート状態の取得
    if latest_data:
        alerts = monitor.check_alert_status(latest_data, thresholds)
    else:
        alerts = {'overall': 'データなし', 'river': 'データなし', 'dam': 'データなし', 'rainfall': 'データなし'}
    
    # AI予測モデルの初期化（データ読み込み後、表示前に実行）
    if AI_PREDICTION_AVAILABLE and history_data:
        selected_model = st.session_state.get('prediction_model', 'エキスパートルール予測')
        
        # モデルが変更された場合またはモデルが存在しない場合は初期化
        if ('predictor' not in st.session_state or 
            st.session_state.get('last_prediction_model') != selected_model):
            
            try:
                if selected_model == "Riverオンライン学習予測" and RIVER_LEARNING_AVAILABLE:
                    # River予測モデルを取得
                    if get_river_predictor:
                        predictor_instance = get_river_predictor()
                        if predictor_instance:
                            st.session_state.predictor = predictor_instance
                            st.session_state.last_prediction_model = selected_model
                            # ストリーミングモデルの場合の学習
                            if hasattr(predictor_instance, 'learn_one') and len(history_data) >= 18:
                                for i in range(min(10, len(history_data) - 18)):  # 最初の10個で学習
                                    current_data = history_data[i]
                                    future_data = history_data[i+1:i+19]
                                    predictor_instance.learn_one(current_data, future_data)
                        else:
                            raise Exception("River予測モデルの作成に失敗")
                    else:
                        raise Exception("get_river_predictorが利用できません")
                else:
                    # エキスパートルール予測を使用
                    st.session_state.predictor = AdvancedRiverLevelPredictor()
                    st.session_state.last_prediction_model = selected_model
            except Exception as e:
                # エラー時はフォールバック
                st.session_state.predictor = AdvancedRiverLevelPredictor()
                st.session_state.last_prediction_model = "エキスパートルール予測"
                if 'ai_prediction_error' not in st.session_state:
                    st.session_state.ai_prediction_error = str(e)
    
    # デモモード表示
    if demo_mode:
        if demo_datetime:
            # 選択された日時を表示
            demo_datetime_str = demo_datetime.strftime("%Y年%m月%d日 %H:%M")
            st.info(f"📊 デモデータ表示中（選択日時: {demo_datetime_str}）")
        else:
            # 動的な時間範囲を計算
            time_min, time_max = monitor.get_common_time_range(history_data, display_hours, demo_mode)
            if time_min and time_max:
                # 日本時間で表示
                time_min_jst = time_min.strftime("%Y年%m月%d日")
                time_max_jst = time_max.strftime("%Y年%m月%d日")
                st.info(f"📊 デモデータ表示中（{time_min_jst}〜{time_max_jst}）")
            else:
                st.info("📊 デモデータ表示中")
    
    if latest_data:
        # 状態、更新時間、API取得時間を3列で表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if alerts['overall'] == '正常':
                st.success("🟢 現在の状況: 正常")
            elif alerts['overall'] == '危険':
                st.error("🔴 現在の状況: 危険")
            elif alerts['overall'] == '警戒':
                st.warning("🟠 現在の状況: 警戒")
            elif alerts['overall'] == '注意':
                st.warning("🟡 現在の状況: 注意")
            else:
                st.info("⚪ 現在の状況: 確認中")
        
        with col2:
            # 更新時間
            if latest_data.get('data_time'):
                try:
                    dt = datetime.fromisoformat(latest_data['data_time'].replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                    update_time = dt.strftime('%H:%M')
                    st.success(f"🕐 最終更新: {update_time}")
                except:
                    st.error("🕐 最終更新: 取得失敗")
            else:
                st.warning("🕐 最終更新: データなし")
        
        with col3:
            # API取得時間
            precipitation_data = latest_data.get('precipitation_intensity', {})
            api_update_time = precipitation_data.get('update_time')
            if api_update_time:
                try:
                    dt = datetime.fromisoformat(api_update_time.replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                    api_time = dt.strftime('%H:%M')
                    st.success(f"📡 API取得: {api_time}")
                except:
                    st.error("📡 API取得: 取得失敗")
            else:
                st.warning("📡 API取得: データなし")
    else:
        st.warning("⚠️ データの読み込み中...")
    
    st.markdown("---")
    
    # 現在の状況表示
    if latest_data:
        monitor.create_metrics_display(latest_data)
        
        # 天気予報表示
        monitor.create_weather_forecast_display(latest_data, show_weekly_weather)
    
    # データ分析表示
    monitor.create_data_analysis_display(history_data, enable_graph_interaction, display_hours, demo_mode)
    
    # グラフ作成後にAI予測エラーがある場合は表示
    if 'ai_prediction_error' in st.session_state:
        st.error(f"⚠️ AI予測エラー: {st.session_state['ai_prediction_error']}")
        # エラーをクリア
        del st.session_state['ai_prediction_error']
    
    # システム情報（サイドバー）
    with st.sidebar.expander("システム情報", expanded=True):
        # 観測状況
        with st.expander("■ 観測状況", expanded=True):
            # 観測時刻の表示
            if latest_data and latest_data.get('data_time'):
                try:
                    # data_timeを使用（観測時刻）
                    obs_time = datetime.fromisoformat(latest_data['data_time'].replace('Z', '+00:00'))
                    if obs_time.tzinfo is None:
                        obs_time = obs_time.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
                    
                    # 現在時刻（日本時間）
                    now_jst = datetime.now(ZoneInfo('Asia/Tokyo'))
                    time_diff = now_jst - obs_time
                    minutes_ago = int(time_diff.total_seconds() / 60)
                    
                    if minutes_ago < 60:
                        st.success(f"観測時刻 ： {minutes_ago}分前")
                    elif minutes_ago < 120:
                        st.warning(f"観測時刻 ： {minutes_ago}分前")
                    else:
                        st.error(f"観測時刻 ： {minutes_ago}分前")
                except:
                    st.info("● 観測時刻確認中")
            
            # データ統計
            st.info(f"データ件数 ： {len(history_data)}件")
        
        # 警戒レベル説明
        with st.expander("■ 警戒レベル説明", expanded=False):
            st.write(f"""
            **河川水位基準**
            - 正常: 3.80m未満
            - 水防団待機: 3.80m以上
            - 氾濫注意: 5.00m以上
            - 避難判断: 5.10m以上
            - 氾濫危険: 5.50m以上
            
            **ダム水位基準**
            - 警戒: {dam_warning}m以上（洪水時最高水位）
            - 危険: {dam_danger}m以上（設計最高水位）
            
            **雨量基準**
            - 注意: 10mm/h以上
            - 警戒: 30mm/h以上
            - 危険: 50mm/h以上
            """)
        
        # データソース情報
        with st.expander("■ データソース", expanded=False):
            st.write("""
            **厚東川ダム**
            
            ・ 更新間隔 ： 10分

            **厚東川**
            
            ・ 更新間隔 ： 10分

            データ提供:山口県土木防災情報システム
            """)
    
    # アプリ情報
    st.sidebar.markdown("---")
    st.sidebar.caption("厚東川氾濫監視システム v2.0")
    st.sidebar.caption("※ 本システムは山口県公開データを再加工した参考情報です。防災判断は必ず公式発表をご確認ください。")
    st.sidebar.caption("※ 本システムの利用または利用不能により生じた直接・間接の損害について、一切責任を負いません。")
    st.sidebar.caption("Powered by Streamlit")
    

if __name__ == "__main__":
    main()