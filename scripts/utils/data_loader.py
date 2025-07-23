"""
データ読み込み関連のユーティリティモジュール
"""
import json
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz
    ZoneInfo = lambda x: pytz.timezone(x)


def load_latest_data(data_dir: Path) -> Optional[Dict[str, Any]]:
    """最新データを読み込む（タイムスタンプベースのキャッシュ）"""
    latest_file = data_dir / "latest.json"
    
    if not latest_file.exists():
        st.warning("■ データファイルが見つかりません。データ収集スクリプトを実行してください。")
        return None
    
    try:
        # まずファイル内容を読み込んでタイムスタンプを取得
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # データのタイムスタンプをキャッシュキーとして使用（Streamlit.io対応）
        cache_key = data.get('timestamp', str(latest_file.stat().st_mtime))
        return _load_latest_data_cached(str(latest_file), cache_key)
    except Exception as e:
        st.error(f"× データ読み込みエラー: {e}")
        return None


@st.cache_data(ttl=60)  # 1分間キャッシュ（短縮してStreamlit.ioでの更新頻度を上げる）
def _load_latest_data_cached(file_path: str, cache_key: str) -> Optional[Dict[str, Any]]:
    """タイムスタンプをキーとするキャッシュされたデータ読み込み"""
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


def get_cache_key(data_dir: Path) -> str:
    """キャッシュキー用のデータタイムスタンプを取得"""
    try:
        # latest.jsonからタイムスタンプを取得
        latest_file = data_dir / "latest.json"
        if latest_file.exists():
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # データのタイムスタンプをキャッシュキーとして使用
                return data.get('timestamp', str(latest_file.stat().st_mtime))
        return "no_file"
    except Exception:
        return "error"


@st.cache_data(ttl=300)  # 5分間キャッシュ（短縮）
def load_history_data(history_dir: Path, hours: int = 72, cache_key: str = None) -> List[Dict[str, Any]]:
    """履歴データを読み込む（固定期間で全データを読み込み、表示はグラフ側で制御）"""
    history_data = []
    # JST（日本標準時）で現在時刻を取得
    end_time = datetime.now(ZoneInfo('Asia/Tokyo'))
    start_time = end_time - timedelta(hours=hours)
    
    if not history_dir.exists():
        st.info("■ 履歴データディレクトリがありません。データが蓄積されるまでお待ちください。")
        return history_data
    
    error_count = 0
    processed_files = 0
    # 時間に応じて最大処理ファイル数を動的に調整（10分間隔データを想定）
    max_files = min(hours * 6 + 50, 500)  # 余裕を持って設定
    
    # JST時刻で日付ディレクトリを処理（新しいデータから逆順で処理）
    current_time = end_time
    while current_time >= start_time and processed_files < max_files:
        date_dir = (history_dir / 
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


def load_sample_csv_data() -> List[Dict[str, Any]]:
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
                    'water_level': float(row['water_level']) if pd.notna(row['water_level']) else 0.0,
                    'storage_rate': float(row['storage_rate']) if pd.notna(row['storage_rate']) else 0.0,
                    'inflow': float(row['inflow']) if pd.notna(row['inflow']) else 0.0,
                    'outflow': float(row['outflow']) if pd.notna(row['outflow']) else 0.0,
                    'storage_change': 0.0  # サンプルデータには含まれない
                },
                'river': {
                    'water_level': float(water_row['water_level'].iloc[0]) if not water_row.empty and pd.notna(water_row['water_level'].iloc[0]) else 0.0,
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
            st.error("❌ 有効なサンプルデータが1件も読み込めませんでした")
            st.error(f"エラー数: {error_count}件")
            
        return sample_data
        
    except Exception as e:
        st.error(f"❌ CSVファイル読み込みエラー: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []


def get_common_time_range(history_data: List[Dict[str, Any]], 
                         hours_6: bool, hours_12: bool, hours_24: bool,
                         hours_48: bool, hours_72: bool) -> tuple:
    """
    選択されたオプションに基づいて、共通の時間範囲を返す
    
    Returns:
        tuple: (start_time: datetime, end_time: datetime)
    """
    # 履歴データがある場合は最新データの時刻を基準にする
    if history_data:
        try:
            # 最新データのタイムスタンプを取得
            latest_timestamp = history_data[-1]['timestamp']
            end_time = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
            else:
                end_time = end_time.astimezone(ZoneInfo('Asia/Tokyo'))
        except Exception:
            # エラーの場合は現在時刻を使用
            end_time = datetime.now(ZoneInfo('Asia/Tokyo'))
    else:
        # データがない場合は現在時刻を使用
        end_time = datetime.now(ZoneInfo('Asia/Tokyo'))
    
    # 選択された時間範囲から開始時刻を計算
    if hours_6:
        start_time = end_time - timedelta(hours=6)
    elif hours_12:
        start_time = end_time - timedelta(hours=12)
    elif hours_24:
        start_time = end_time - timedelta(hours=24)
    elif hours_48:
        start_time = end_time - timedelta(hours=48)
    elif hours_72:
        start_time = end_time - timedelta(hours=72)
    else:
        # デフォルトは24時間
        start_time = end_time - timedelta(hours=24)
    
    return start_time, end_time


def filter_data_by_time_range(data: List[Dict[str, Any]], 
                            start_time: datetime, 
                            end_time: datetime) -> List[Dict[str, Any]]:
    """
    データを指定された時間範囲でフィルタリング
    
    Args:
        data: フィルタリング対象のデータリスト
        start_time: 開始時刻（datetime、タイムゾーン付き）
        end_time: 終了時刻（datetime、タイムゾーン付き）
    
    Returns:
        フィルタリングされたデータリスト
    """
    filtered_data = []
    
    for item in data:
        try:
            # タイムスタンプを取得して解析
            timestamp = item.get('timestamp', '')
            if not timestamp:
                continue
                
            # ISO形式のタイムスタンプを解析
            item_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if item_time.tzinfo is None:
                item_time = item_time.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
            else:
                item_time = item_time.astimezone(ZoneInfo('Asia/Tokyo'))
            
            # 時間範囲内かチェック
            if start_time <= item_time <= end_time:
                filtered_data.append(item)
                
        except Exception:
            # エラーの場合はスキップ
            continue
    
    return filtered_data