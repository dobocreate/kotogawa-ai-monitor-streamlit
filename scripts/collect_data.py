#!/usr/bin/env python3
"""
厚東川監視システム データ収集スクリプト
山口県土木防災情報システムからデータを取得し、JSON形式で保存する
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Union
import requests
from bs4 import BeautifulSoup

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8以前の場合
    import pytz
    ZoneInfo = lambda x: pytz.timezone(x)

class KotogawaDataCollector:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.history_dir = self.data_dir / "history"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
        
        # URLs for data sources
        self.dam_url = "https://y-bousai.pref.yamaguchi.lg.jp/citizen/dam/kdm_table.aspx"
        self.river_url = "https://y-bousai.pref.yamaguchi.lg.jp/citizen/water/kwl_table.aspx"
        
        # Request settings
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.timeout = 30
        self.max_retries = 5  # リトライ回数を増加
        self.retry_delay = 3  # リトライ間隔（秒）
        
        # Yahoo! Weather API settings
        self.yahoo_api_url = "https://map.yahooapis.jp/weather/V1/place"
        self.yahoo_app_id = "dj00aiZpPW5YTFVqSXc0S2dCcSZzPWNvbnN1bWVyc2VjcmV0Jng9MDA-"
        self.coordinates = "131.289496,34.079891"  # 経度,緯度
        
    def fetch_page(self, url: str, params: Dict[str, str]) -> Optional[BeautifulSoup]:
        """指定されたURLからHTMLを取得し、BeautifulSoupオブジェクトを返す"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    headers=self.headers, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # レスポンスサイズをチェック
                if len(response.content) < 100:
                    raise requests.RequestException(f"Response too small: {len(response.content)} bytes")
                
                return BeautifulSoup(response.content, 'html.parser')
                
            except requests.RequestException as e:
                last_error = e
                error_msg = f"Attempt {attempt + 1}/{self.max_retries} failed: {type(e).__name__}: {e}"
                print(error_msg)
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)  # 指数バックオフ
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to fetch {url} after {self.max_retries} attempts. Last error: {last_error}")
                    return None
            except Exception as e:
                last_error = e
                print(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
    
    def extract_number(self, text: str) -> Optional[float]:
        """テキストから数値を抽出する"""
        if not text:
            return None
        
        # 数値パターンを検索（負の数も含む）
        match = re.search(r'-?\d+\.?\d*', text.strip())
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None
    
    def collect_dam_data(self) -> Dict[str, Any]:
        """ダムデータと降雨データを収集する"""
        # 日本時間で現在時刻を取得し、10分単位に丸める
        jst = ZoneInfo('Asia/Tokyo')
        current_time = datetime.now(jst)
        # 分を10で割って切り捨て、10を掛けることで10分単位に
        minutes = (current_time.minute // 10) * 10
        # 最新の10分単位時刻のデータを取得
        observation_time = current_time.replace(minute=minutes, second=0, microsecond=0)
        # 山口県システムにはJST時間で送信する
        obsdt = observation_time.strftime('%Y%m%d%H%M')
        
        params = {
            'check': '015',     # 厚東川ダムの観測所コード
            'obsdt': obsdt,     # 10分単位に丸めた観測時刻
            'pop': '1'
        }
        soup = self.fetch_page(self.dam_url, params)
        
        if not soup:
            return {
                'dam': {
                    'water_level': None,
                    'storage_rate': None,
                    'inflow': None,
                    'outflow': None,
                    'storage_change': None,
                    'actual_observation_time': None
                },
                'rainfall': {
                    'hourly': None,
                    'cumulative': None,
                    'change': None
                }
            }
        
        dam_data = {
            'water_level': None,
            'storage_rate': None,
            'inflow': None,
            'outflow': None,
            'storage_change': None,
            'actual_observation_time': None
        }
        
        rainfall_data = {
            'hourly': None,
            'cumulative': None,
            'change': None
        }
        
        try:
            # 貯水位パターンで検索（より具体的に）
            full_text = soup.get_text()
            
            # 貯水位を明示的に検索（ダム水位の正確な値）
            water_level_patterns = [
                r'貯水位[:\s]*(\d+\.\d+)\s*m',
                r'現在[:\s]*(\d+\.\d+)\s*m',
                r'水位[:\s]*(\d+\.\d+)\s*m'
            ]
            
            for pattern in water_level_patterns:
                match = re.search(pattern, full_text)
                if match:
                    level = float(match.group(1))
                    # ダム水位の妥当性チェック（30-40mの範囲）
                    if 30 <= level <= 40:
                        dam_data['water_level'] = level
                        break
            
            # テーブルから正確な日時マッチングで目標データを取得
            tables = soup.find_all('table')
            target_date = observation_time.strftime('%Y/%m/%d')
            target_time = observation_time.strftime('%H:%M')
            
            print(f"Looking for dam data: {target_date} {target_time}")
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 9:  # ダムテーブルの最小列数（日付、時刻、貯水位、貯水率、流入量、全放流量、調整流量、60分雨量、累加雨量）
                        try:
                            date_text = cells[0].get_text().strip()
                            time_text = cells[1].get_text().strip()
                            
                            # 日時が完全一致する行を探す
                            if date_text == target_date and time_text == target_time:
                                print(f"Found matching row: {date_text} {time_text}")
                                
                                # 列位置に基づいてデータを抽出
                                # 列2: 貯水位, 列3: 貯水率, 列4: 流入量, 列5: 全放流量
                                water_level_text = cells[2].get_text().strip()
                                storage_rate_text = cells[3].get_text().strip()
                                inflow_text = cells[4].get_text().strip()
                                outflow_text = cells[5].get_text().strip()
                                
                                # 貯水位
                                try:
                                    level = float(water_level_text)
                                    if 30 <= level <= 40:  # 妥当性チェック
                                        dam_data['water_level'] = level
                                        print(f"Dam water level: {level}m")
                                except ValueError:
                                    print(f"Invalid water level: {water_level_text}")
                                
                                # 貯水率
                                try:
                                    rate = float(storage_rate_text)
                                    if 0 <= rate <= 100:  # 妥当性チェック
                                        dam_data['storage_rate'] = rate
                                        print(f"Storage rate: {rate}%")
                                except ValueError:
                                    print(f"Invalid storage rate: {storage_rate_text}")
                                
                                # 流入量
                                try:
                                    inflow = float(inflow_text)
                                    if 0 <= inflow <= 100:  # 範囲を拡張
                                        dam_data['inflow'] = inflow
                                        print(f"Inflow: {inflow} m³/s")
                                except ValueError:
                                    print(f"Invalid inflow: {inflow_text}")
                                
                                # 全放流量
                                try:
                                    outflow = float(outflow_text)
                                    if 0 <= outflow <= 100:  # 範囲を拡張
                                        dam_data['outflow'] = outflow
                                        print(f"Outflow: {outflow} m³/s")
                                except ValueError:
                                    print(f"Invalid outflow: {outflow_text}")
                                
                                # 降雨データの取得（列７: 60分雨量、列８: 累加雨量）
                                if len(cells) > 7:
                                    # 60分雨量
                                    try:
                                        hourly_text = cells[7].get_text().strip()
                                        hourly = int(hourly_text)
                                        if 0 <= hourly <= 200:  # 範囲を拡張
                                            rainfall_data['hourly'] = hourly
                                            print(f"Hourly rainfall: {hourly}mm")
                                    except ValueError:
                                        print(f"Invalid hourly rainfall: {cells[7].get_text().strip()}")
                                
                                if len(cells) > 8:
                                    # 累加雨量
                                    try:
                                        cumulative_text = cells[8].get_text().strip()
                                        cumulative = int(cumulative_text)
                                        if 0 <= cumulative <= 1000:  # 範囲を拡張
                                            rainfall_data['cumulative'] = cumulative
                                            print(f"Cumulative rainfall: {cumulative}mm")
                                    except ValueError:
                                        print(f"Invalid cumulative rainfall: {cells[8].get_text().strip()}")
                                
                                dam_data['actual_observation_time'] = f"{date_text} {time_text}"
                                break  # 目標行が見つかったら終了
                        except (IndexError, ValueError) as e:
                            continue
                
                if dam_data['water_level'] is not None:
                    break  # データが見つかったらテーブル検索終了
            
            # 目標データが見つからなかった場合、最終行（最新データ）を取得
            if dam_data['water_level'] is None:
                print(f"Target data not found. Looking for the latest available data...")
                
                for table in tables:
                    rows = table.find_all('tr')
                    # 最後から順に有効なデータ行を探す
                    for row in reversed(rows):
                        cells = row.find_all('td')
                        if len(cells) >= 9:
                            try:
                                date_text = cells[0].get_text().strip()
                                time_text = cells[1].get_text().strip()
                                
                                # 日付形式のチェック（YYYY/MM/DD）
                                if re.match(r'\d{4}/\d{2}/\d{2}', date_text) and re.match(r'\d{2}:\d{2}', time_text):
                                    # この観測時刻のデータが既に保存されているかチェック
                                    obs_datetime = datetime.strptime(f"{date_text} {time_text}", "%Y/%m/%d %H:%M")
                                    obs_datetime = obs_datetime.replace(tzinfo=jst)
                                    
                                    # ファイルの存在確認
                                    date_dir = self.history_dir / obs_datetime.strftime("%Y") / obs_datetime.strftime("%m") / obs_datetime.strftime("%d")
                                    history_file = date_dir / f"{obs_datetime.strftime('%H%M')}.json"
                                    
                                    if history_file.exists():
                                        print(f"Data for {date_text} {time_text} already exists. Skipping.")
                                        continue
                                    
                                    print(f"Found latest data: {date_text} {time_text}")
                                    
                                    # データを抽出
                                    water_level_text = cells[2].get_text().strip()
                                    storage_rate_text = cells[3].get_text().strip()
                                    inflow_text = cells[4].get_text().strip()
                                    outflow_text = cells[5].get_text().strip()
                                    
                                    # 貯水位
                                    try:
                                        level = float(water_level_text)
                                        if 30 <= level <= 40:
                                            dam_data['water_level'] = level
                                            print(f"Dam water level: {level}m")
                                    except ValueError:
                                        pass
                                    
                                    # 貯水率
                                    try:
                                        rate = float(storage_rate_text)
                                        if 0 <= rate <= 100:
                                            dam_data['storage_rate'] = rate
                                            print(f"Storage rate: {rate}%")
                                    except ValueError:
                                        pass
                                    
                                    # 流入量
                                    try:
                                        inflow = float(inflow_text)
                                        if 0 <= inflow <= 100:
                                            dam_data['inflow'] = inflow
                                            print(f"Inflow: {inflow} m³/s")
                                    except ValueError:
                                        pass
                                    
                                    # 全放流量
                                    try:
                                        outflow = float(outflow_text)
                                        if 0 <= outflow <= 100:
                                            dam_data['outflow'] = outflow
                                            print(f"Outflow: {outflow} m³/s")
                                    except ValueError:
                                        pass
                                    
                                    # 降雨データの取得（列７: 60分雨量、列８: 累加雨量）
                                    if len(cells) > 7:
                                        # 60分雨量
                                        try:
                                            hourly_text = cells[7].get_text().strip()
                                            hourly = int(hourly_text)
                                            if 0 <= hourly <= 200:  # 範囲を拡張
                                                rainfall_data['hourly'] = hourly
                                                print(f"Hourly rainfall: {hourly}mm")
                                        except ValueError:
                                            print(f"Invalid hourly rainfall: {cells[7].get_text().strip()}")
                                    
                                    if len(cells) > 8:
                                        # 累加雨量
                                        try:
                                            cumulative_text = cells[8].get_text().strip()
                                            cumulative = int(cumulative_text)
                                            if 0 <= cumulative <= 1000:  # 範囲を拡張
                                                rainfall_data['cumulative'] = cumulative
                                                print(f"Cumulative rainfall: {cumulative}mm")
                                        except ValueError:
                                            print(f"Invalid cumulative rainfall: {cells[8].get_text().strip()}")
                                    
                                    if dam_data['water_level'] is not None:
                                        dam_data['actual_observation_time'] = f"{date_text} {time_text}"
                                        break
                                        
                            except (IndexError, ValueError) as e:
                                continue
                    
                    if dam_data['water_level'] is not None:
                        break
            
            # 最終的にデータが取得できなかった場合はnullを保持
            if dam_data['water_level'] is None:
                print("No valid dam data found. Keeping null values.")
                
        except Exception as e:
            print(f"Error extracting dam data: {e}")
        
        # 変化量の計算（現在は0を設定）
        rainfall_data['change'] = 0 if rainfall_data['hourly'] is not None else None
        
        return {
            'dam': dam_data,
            'rainfall': rainfall_data
        }
    
    def collect_river_data(self) -> Dict[str, Any]:
        """河川データを収集する"""
        # 日本時間で現在時刻を取得し、10分単位に丸める
        jst = ZoneInfo('Asia/Tokyo')
        current_time = datetime.now(jst)
        # 分を10で割って切り捨て、10を掛けることで10分単位に
        minutes = (current_time.minute // 10) * 10
        # 最新の10分単位時刻のデータを取得
        observation_time = current_time.replace(minute=minutes, second=0, microsecond=0)
        # 山口県システムにはJST時間で送信する
        obsdt = observation_time.strftime('%Y%m%d%H%M')
        
        params = {
            'check': '05067',  # 厚東川（持世寺）の観測所コード
            'obsdt': obsdt,     # 10分単位に丸めた観測時刻
            'pop': '1'
        }
        soup = self.fetch_page(self.river_url, params)
        
        if not soup:
            return {
                'water_level': None,
                'level_change': None,
                'status': 'データなし',
                'actual_observation_time': None
            }
        
        river_data = {
            'water_level': None,
            'level_change': None,
            'status': None,
            'actual_observation_time': None
        }
        
        # 警戒レベル閾値（サイトから取得した実際の値）
        thresholds = {
            'preparedness': 3.80,    # 水防団待機水位
            'caution': 5.00,        # 氾濫注意水位
            'evacuation': 5.10,     # 避難判断水位
            'danger': 5.50          # 氾濫危険水位
        }
        
        try:
            # テーブルから正確な日時マッチングで目標データを取得
            tables = soup.find_all('table')
            target_date = observation_time.strftime('%Y/%m/%d')
            target_time = observation_time.strftime('%H:%M')
            
            print(f"Looking for river data: {target_date} {target_time}")
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:  # 河川テーブルの最小列数（日付、時刻、水位、変化量など）
                        try:
                            date_text = cells[0].get_text().strip()
                            time_text = cells[1].get_text().strip()
                            
                            # 日時が完全一致する行を探す
                            if date_text == target_date and time_text == target_time:
                                print(f"Found matching river row: {date_text} {time_text}")
                                
                                # 列位置に基づいてデータを抽出
                                # 列2: 水位, 列3: 水位変化（推定）
                                water_level_text = cells[2].get_text().strip()
                                
                                # 水位
                                try:
                                    level = float(water_level_text)
                                    if 0.5 <= level <= 10:  # 合理的な水位範囲
                                        river_data['water_level'] = level
                                        print(f"River water level: {level}m")
                                        
                                        # 水位変化（列3があれば）
                                        if len(cells) > 3:
                                            try:
                                                change_text = cells[3].get_text().strip()
                                                # +0.01 や -0.02 のような形式から数値を抽出
                                                change_match = re.search(r'([+-]?\d+\.\d+)', change_text)
                                                if change_match:
                                                    change = float(change_match.group(1))
                                                    river_data['level_change'] = round(change, 2)
                                                    print(f"Water level change: {change}m")
                                                else:
                                                    river_data['level_change'] = 0.0
                                            except (ValueError, IndexError):
                                                river_data['level_change'] = 0.0
                                        else:
                                            river_data['level_change'] = 0.0
                                        
                                        # 警戒レベルの判定
                                        if level >= thresholds['danger']:
                                            river_data['status'] = '氾濫危険'
                                        elif level >= thresholds['evacuation']:
                                            river_data['status'] = '避難判断'
                                        elif level >= thresholds['caution']:
                                            river_data['status'] = '氾濫注意'
                                        elif level >= thresholds['preparedness']:
                                            river_data['status'] = '水防団待機'
                                        else:
                                            river_data['status'] = '正常'
                                        
                                        river_data['actual_observation_time'] = f"{date_text} {time_text}"
                                        break  # 目標行が見つかったら終了
                                except ValueError:
                                    print(f"Invalid river water level: {water_level_text}")
                        except (IndexError, ValueError) as e:
                            continue
                
                if river_data['water_level'] is not None:
                    break  # データが見つかったらテーブル検索終了
            
            # 目標データが見つからなかった場合、最終行（最新データ）を取得
            if river_data['water_level'] is None:
                print(f"Target river data not found. Looking for the latest available data...")
                
                for table in tables:
                    rows = table.find_all('tr')
                    # 最後から順に有効なデータ行を探す
                    for row in reversed(rows):
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            try:
                                date_text = cells[0].get_text().strip()
                                time_text = cells[1].get_text().strip()
                                
                                # 日付形式のチェック（YYYY/MM/DD）
                                if re.match(r'\d{4}/\d{2}/\d{2}', date_text) and re.match(r'\d{2}:\d{2}', time_text):
                                    # この観測時刻のデータが既に保存されているかチェック
                                    obs_datetime = datetime.strptime(f"{date_text} {time_text}", "%Y/%m/%d %H:%M")
                                    obs_datetime = obs_datetime.replace(tzinfo=jst)
                                    
                                    # ファイルの存在確認
                                    date_dir = self.history_dir / obs_datetime.strftime("%Y") / obs_datetime.strftime("%m") / obs_datetime.strftime("%d")
                                    history_file = date_dir / f"{obs_datetime.strftime('%H%M')}.json"
                                    
                                    if history_file.exists():
                                        print(f"River data for {date_text} {time_text} already exists. Skipping.")
                                        continue
                                    
                                    print(f"Found latest river data: {date_text} {time_text}")
                                    
                                    # データを抽出
                                    water_level_text = cells[2].get_text().strip()
                                    
                                    # 水位
                                    try:
                                        level = float(water_level_text)
                                        if 0.5 <= level <= 10:  # 合理的な水位範囲
                                            river_data['water_level'] = level
                                            print(f"River water level: {level}m")
                                            
                                            # 水位変化（列３があれば）
                                            if len(cells) > 3:
                                                try:
                                                    change_text = cells[3].get_text().strip()
                                                    change_match = re.search(r'([+-]?\d+\.\d+)', change_text)
                                                    if change_match:
                                                        change = float(change_match.group(1))
                                                        river_data['level_change'] = round(change, 2)
                                                    else:
                                                        river_data['level_change'] = 0.0
                                                except (ValueError, IndexError):
                                                    river_data['level_change'] = 0.0
                                            else:
                                                river_data['level_change'] = 0.0
                                            
                                            # 警戒レベルの判定
                                            if level >= thresholds['danger']:
                                                river_data['status'] = '氾濫危険'
                                            elif level >= thresholds['evacuation']:
                                                river_data['status'] = '避難判断'
                                            elif level >= thresholds['caution']:
                                                river_data['status'] = '氾濫注意'
                                            elif level >= thresholds['preparedness']:
                                                river_data['status'] = '水防団待機'
                                            else:
                                                river_data['status'] = '正常'
                                            
                                            river_data['actual_observation_time'] = f"{date_text} {time_text}"
                                            break
                                    except ValueError:
                                        pass
                                        
                            except (IndexError, ValueError) as e:
                                continue
                    
                    if river_data['water_level'] is not None:
                        break
            
            # 最終的にデータが取得できなかった場合はnullを保持
            if river_data['water_level'] is None:
                print("No valid river data found. Keeping null value.")
                # 水位が取得できない場合はステータスもnullに
                river_data['status'] = None
                        
        except Exception as e:
            print(f"Error extracting river data: {e}")
        
        return river_data
    
# 旧collect_rainfall_dataメソッドは削除されました
    # 降雨データはcollect_dam_dataメソッドで統合取得されます
    
    def collect_weather_data(self) -> Dict[str, Any]:
        """気象庁APIから天気予報データを収集する"""
        weather_data = {
            'today': {
                'weather_code': None,
                'weather_text': None,
                'temp_max': None,
                'temp_min': None,
                'precipitation_probability': [],
                'precipitation_times': []
            },
            'tomorrow': {
                'weather_code': None,
                'weather_text': None,
                'temp_max': None,
                'temp_min': None,
                'precipitation_probability': [],
                'precipitation_times': []
            },
            'day_after_tomorrow': {
                'weather_code': None,
                'weather_text': None,
                'temp_max': None,
                'temp_min': None,
                'precipitation_probability': [],
                'precipitation_times': []
            },
            'update_time': None,
            'weekly_forecast': []
        }
        
        try:
            # 山口県の天気予報データを取得
            url = "https://www.jma.go.jp/bosai/forecast/data/forecast/350000.json"
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            forecast_data = response.json()
            
            if not forecast_data or len(forecast_data) == 0:
                print("No weather forecast data available")
                return weather_data
            
            # 最新の予報データを取得
            latest_forecast = forecast_data[0]
            
            # 更新時刻を設定
            if 'reportDatetime' in latest_forecast:
                try:
                    jst = ZoneInfo('Asia/Tokyo')
                    update_time = datetime.fromisoformat(latest_forecast['reportDatetime'].replace('Z', '+00:00'))
                    update_time_jst = update_time.astimezone(jst)
                    weather_data['update_time'] = update_time_jst.isoformat()
                except (ValueError, KeyError):
                    pass
            
            # 宇部市のエリア情報を探す（エリアコード350012）
            target_area = None
            if 'timeSeries' in latest_forecast:
                for series in latest_forecast['timeSeries']:
                    if 'areas' in series:
                        for area in series['areas']:
                            if area.get('area', {}).get('code') == '350012':
                                target_area = area
                                break
                        if target_area:
                            break
            
            # 短期予報から天気情報を取得（今日・明日）
            jst = ZoneInfo('Asia/Tokyo')
            now = datetime.now(jst)
            
            # 西部エリアのデータを取得
            west_area_weather = None
            west_area_pop = None
            pop_time_defines = None
            
            if 'timeSeries' in latest_forecast:
                # 天気情報（timeSeries[0]）
                if len(latest_forecast['timeSeries']) > 0:
                    for area in latest_forecast['timeSeries'][0].get('areas', []):
                        if area.get('area', {}).get('code') == '350010':  # 西部
                            west_area_weather = area
                            break
                
                # 降水確率情報（timeSeries[1]）
                if len(latest_forecast['timeSeries']) > 1:
                    pop_time_defines = latest_forecast['timeSeries'][1].get('timeDefines', [])
                    for area in latest_forecast['timeSeries'][1].get('areas', []):
                        if area.get('area', {}).get('code') == '350010':  # 西部
                            west_area_pop = area
                            break
            
            # 天気情報の設定
            if west_area_weather and 'weathers' in west_area_weather:
                weathers = west_area_weather['weathers']
                weather_codes = west_area_weather.get('weatherCodes', [])
                
                if len(weathers) >= 1:
                    weather_data['today']['weather_text'] = weathers[0]
                    if len(weather_codes) >= 1:
                        weather_data['today']['weather_code'] = weather_codes[0]
                
                if len(weathers) >= 2:
                    weather_data['tomorrow']['weather_text'] = weathers[1]
                    if len(weather_codes) >= 2:
                        weather_data['tomorrow']['weather_code'] = weather_codes[1]
            
            # 降水確率の設定（時間別）
            if west_area_pop and 'pops' in west_area_pop and pop_time_defines:
                pops = west_area_pop['pops']
                
                # 時刻と降水確率をペアにする
                today_pops = []
                today_times = []
                tomorrow_pops = []
                tomorrow_times = []
                
                for i, time_str in enumerate(pop_time_defines):
                    if i < len(pops):
                        try:
                            time_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                            time_jst = time_obj.astimezone(jst)
                            pop_value = int(pops[i]) if pops[i] != '' else None
                            
                            # 日付で振り分け
                            if time_jst.date() == now.date():
                                today_pops.append(pop_value)
                                today_times.append(time_jst.strftime('%H時'))
                            elif time_jst.date() == (now + timedelta(days=1)).date():
                                tomorrow_pops.append(pop_value)
                                tomorrow_times.append(time_jst.strftime('%H時'))
                        except (ValueError, IndexError):
                            continue
                
                weather_data['today']['precipitation_probability'] = today_pops
                weather_data['today']['precipitation_times'] = today_times
                weather_data['tomorrow']['precipitation_probability'] = tomorrow_pops
                weather_data['tomorrow']['precipitation_times'] = tomorrow_times
            
            # 週間予報から明後日の天気情報を取得
            if len(forecast_data) > 1:
                week_forecast = forecast_data[1]
                if 'timeSeries' in week_forecast and len(week_forecast['timeSeries']) > 0:
                    time_defines = week_forecast['timeSeries'][0].get('timeDefines', [])
                    
                    # 明後日の日付を探す
                    day_after_tomorrow = now + timedelta(days=2)
                    day_after_tomorrow_str = day_after_tomorrow.strftime('%Y-%m-%d')
                    
                    for i, time_str in enumerate(time_defines):
                        if day_after_tomorrow_str in time_str:
                            # 山口県全体のデータを探す（週間予報は県単位）
                            for area in week_forecast['timeSeries'][0].get('areas', []):
                                if area.get('area', {}).get('code') in ['350000', '81428']:  # 山口県または下関
                                    if 'weatherCodes' in area and i < len(area['weatherCodes']):
                                        weather_data['day_after_tomorrow']['weather_code'] = area['weatherCodes'][i]
                                    
                                    # 週間予報の降水確率
                                    if 'pops' in area and i < len(area['pops']):
                                        try:
                                            pop = int(area['pops'][i]) if area['pops'][i] != '' else None
                                            weather_data['day_after_tomorrow']['precipitation_probability'] = [pop]
                                            weather_data['day_after_tomorrow']['precipitation_times'] = ['日中']
                                        except ValueError:
                                            pass
                                    
                                    # 天気コードから天気テキストを生成（簡易版）
                                    weather_code_map = {
                                        '100': '晴れ', '101': '晴れ時々くもり', '102': '晴れ一時雨',
                                        '110': '晴れ時々くもり一時雨', '111': '晴れ時々くもり一時雪',
                                        '112': '晴れ一時雨', '113': '晴れ時々雨', '114': '晴れ一時雪',
                                        '200': 'くもり', '201': 'くもり時々晴れ', '202': 'くもり一時雨',
                                        '203': 'くもり時々雨', '204': 'くもり一時雪', '210': 'くもり時々晴れ一時雨',
                                        '211': 'くもり時々晴れ一時雪', '212': 'くもり一時雨か雪', '213': 'くもり一時雨か雷雨',
                                        '300': '雨', '301': '雨時々晴れ', '302': '雨時々くもり',
                                        '303': '雨時々雪', '308': '大雨', '311': '雨のち晴れ',
                                        '313': '雨のちくもり', '314': '雨のち雪',
                                        '400': '雪', '401': '雪時々晴れ', '402': '雪時々くもり',
                                        '403': '雪時々雨', '406': '大雪', '411': '雪のち晴れ',
                                        '413': '雪のちくもり', '414': '雪のち雨'
                                    }
                                    if weather_data['day_after_tomorrow']['weather_code'] in weather_code_map:
                                        weather_data['day_after_tomorrow']['weather_text'] = weather_code_map[weather_data['day_after_tomorrow']['weather_code']]
                                    break
                            break
            
            # 気温データの取得（短期予報から）
            if 'timeSeries' in latest_forecast:
                for series in latest_forecast['timeSeries']:
                    if 'areas' in series:
                        # 下関の気温データを使用（宇部に最も近い観測地点）
                        for area in series['areas']:
                            area_code = area.get('area', {}).get('code')
                            area_name = area.get('area', {}).get('name', '')
                            if area_code == '81428' or '下関' in area_name:
                                # 気温データ
                                if 'temps' in area:
                                    temps = area['temps']
                                    # 通常、[今日の最高, 今日の最低, 明日の最低, 明日の最高] の順
                                    if len(temps) >= 4:
                                        try:
                                            weather_data['today']['temp_max'] = int(temps[0]) if temps[0] != '' else None
                                            weather_data['today']['temp_min'] = int(temps[1]) if temps[1] != '' else None
                                            weather_data['tomorrow']['temp_min'] = int(temps[2]) if temps[2] != '' else None
                                            weather_data['tomorrow']['temp_max'] = int(temps[3]) if temps[3] != '' else None
                                        except (ValueError, IndexError):
                                            pass
                                break
            
            # 週間予報から明後日の気温データを取得
            if len(forecast_data) > 1 and 'timeSeries' in forecast_data[1]:
                for series in forecast_data[1]['timeSeries']:
                    if 'areas' in series:
                        for area in series['areas']:
                            if area.get('area', {}).get('code') in ['350000', '81428']:  # 山口県または下関
                                # 気温データの探索
                                time_defines = series.get('timeDefines', [])
                                temps_max = area.get('tempsMax', [])
                                temps_min = area.get('tempsMin', [])
                                
                                day_after_tomorrow = now + timedelta(days=2)
                                day_after_tomorrow_str = day_after_tomorrow.strftime('%Y-%m-%d')
                                
                                for i, time_str in enumerate(time_defines):
                                    if day_after_tomorrow_str in time_str:
                                        if i < len(temps_max) and temps_max[i] != '':
                                            try:
                                                weather_data['day_after_tomorrow']['temp_max'] = int(temps_max[i])
                                            except ValueError:
                                                pass
                                        if i < len(temps_min) and temps_min[i] != '':
                                            try:
                                                weather_data['day_after_tomorrow']['temp_min'] = int(temps_min[i])
                                            except ValueError:
                                                pass
                                        break
                                break
            
            # 週間予報データの収集（7日間）
            if len(forecast_data) > 1:
                week_forecast = forecast_data[1]
                print(f"DEBUG: forecast_data[1]の構造確認")
                print(f"DEBUG: forecast_data[1]にtimeSeriesがあるか: {'timeSeries' in week_forecast}")
                if 'timeSeries' in week_forecast:
                    print(f"DEBUG: timeSeriesの数: {len(week_forecast['timeSeries'])}")
                    for idx, ts in enumerate(week_forecast['timeSeries']):
                        print(f"DEBUG: timeSeries[{idx}]のキー: {list(ts.keys())}")
                        if 'areas' in ts:
                            print(f"DEBUG: timeSeries[{idx}]のエリア数: {len(ts['areas'])}")
                            for area in ts['areas']:
                                area_code = area.get('area', {}).get('code')
                                area_name = area.get('area', {}).get('name')
                                area_keys = list(area.keys())
                                print(f"DEBUG: timeSeries[{idx}] エリア {area_code}({area_name}) のキー: {area_keys}")
                
                if 'timeSeries' in week_forecast and len(week_forecast['timeSeries']) > 0:
                    # 天気コードと降水確率はtimeSeries[0]から取得
                    ts = week_forecast['timeSeries'][0]
                    time_defines = ts.get('timeDefines', [])
                    
                    # 山口県のデータを取得
                    weather_codes = []
                    pops = []
                    
                    for area in ts.get('areas', []):
                        if area.get('area', {}).get('code') in ['350000', '81428']:
                            weather_codes = area.get('weatherCodes', [])
                            pops = area.get('pops', [])
                            break
                    
                    # 気温データを全timeSeriesから検索
                    temps_max = []
                    temps_min = []
                    print(f"DEBUG: 週間予報のtimeSeries数: {len(week_forecast['timeSeries'])}")
                    
                    for ts_idx, ts_temp in enumerate(week_forecast['timeSeries']):
                        print(f"DEBUG: timeSeries[{ts_idx}]のエリア数: {len(ts_temp.get('areas', []))}")
                        for area in ts_temp.get('areas', []):
                            area_code = area.get('area', {}).get('code')
                            area_name = area.get('area', {}).get('name')
                            has_temp_max = 'tempsMax' in area
                            has_temp_min = 'tempsMin' in area
                            print(f"DEBUG: timeSeries[{ts_idx}] エリア: {area_code}({area_name}), tempsMax={has_temp_max}, tempsMin={has_temp_min}")
                            
                            # 温度データがあるエリアを検索（エリアコードは問わず）
                            if has_temp_max and has_temp_min:
                                temps_max = area.get('tempsMax', [])
                                temps_min = area.get('tempsMin', [])
                                print(f"DEBUG: 温度データ発見 エリア({area_code}): 最高気温数={len(temps_max)}, 最低気温数={len(temps_min)}")
                                print(f"DEBUG: 最高気温データ: {temps_max}")
                                print(f"DEBUG: 最低気温データ: {temps_min}")
                                # 温度データが見つかったらbreak
                                if temps_max and temps_min:
                                    break
                        # 温度データが見つかったら外側のループもbreak
                        if temps_max and temps_min:
                            break
                    
                    if not (temps_max and temps_min):
                        print("DEBUG: 温度データが見つかりませんでした")
                    
                    weekly_data = []
                    for i, time_str in enumerate(time_defines):
                        try:
                            date_obj = datetime.fromisoformat(time_str)
                            date_jst = date_obj.astimezone(jst)
                            
                            day_data = {
                                'date': date_jst.strftime('%Y-%m-%d'),
                                'day_of_week': date_jst.strftime('%a'),
                                'weather_code': weather_codes[i] if i < len(weather_codes) else None,
                                'weather_text': None,
                                'precipitation_probability': None,
                                'temp_max': None,
                                'temp_min': None
                            }
                            
                            # 天気コードから天気テキストを生成
                            if day_data['weather_code']:
                                weather_code_map = {
                                    '100': '晴れ', '101': '晴れ時々くもり', '102': '晴れ一時雨',
                                    '110': '晴れ時々くもり一時雨', '111': '晴れ時々くもり一時雪',
                                    '112': '晴れ一時雨', '113': '晴れ時々雨', '114': '晴れ一時雪',
                                    '200': 'くもり', '201': 'くもり時々晴れ', '202': 'くもり一時雨',
                                    '203': 'くもり時々雨', '204': 'くもり一時雪', '210': 'くもり時々晴れ一時雨',
                                    '211': 'くもり時々晴れ一時雪', '212': 'くもり一時雨か雪', '213': 'くもり一時雨か雷雨',
                                    '300': '雨', '301': '雨時々晴れ', '302': '雨時々くもり',
                                    '303': '雨時々雪', '308': '大雨', '311': '雨のち晴れ',
                                    '313': '雨のちくもり', '314': '雨のち雪',
                                    '400': '雪', '401': '雪時々晴れ', '402': '雪時々くもり',
                                    '403': '雪時々雨', '406': '大雪', '411': '雪のち晴れ',
                                    '413': '雪のちくもり', '414': '雪のち雨'
                                }
                                day_data['weather_text'] = weather_code_map.get(day_data['weather_code'], f"天気コード{day_data['weather_code']}")
                            
                            # 降水確率
                            if i < len(pops) and pops[i] != '':
                                try:
                                    day_data['precipitation_probability'] = int(pops[i])
                                except ValueError:
                                    pass
                            
                            # 最高気温
                            if i < len(temps_max) and temps_max[i] != '':
                                try:
                                    day_data['temp_max'] = int(temps_max[i])
                                except ValueError:
                                    pass
                            
                            # 最低気温
                            if i < len(temps_min) and temps_min[i] != '':
                                try:
                                    day_data['temp_min'] = int(temps_min[i])
                                except ValueError:
                                    pass
                            
                            weekly_data.append(day_data)
                        except (ValueError, IndexError):
                            continue
                    
                    weather_data['weekly_forecast'] = weekly_data
            
            print(f"Weather data collected successfully")
            print(f"Today: {weather_data['today']['weather_text']}, Max: {weather_data['today']['temp_max']}°C, Min: {weather_data['today']['temp_min']}°C")
            print(f"Tomorrow: {weather_data['tomorrow']['weather_text']}, Max: {weather_data['tomorrow']['temp_max']}°C, Min: {weather_data['tomorrow']['temp_min']}°C")
            print(f"Day after tomorrow: {weather_data['day_after_tomorrow']['weather_text']}, Max: {weather_data['day_after_tomorrow']['temp_max']}°C, Min: {weather_data['day_after_tomorrow']['temp_min']}°C")
            print(f"Weekly forecast: {len(weather_data['weekly_forecast'])} days")
            
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
        except json.JSONDecodeError as e:
            print(f"Error parsing weather data JSON: {e}")
        except Exception as e:
            print(f"Unexpected error collecting weather data: {e}")
        
        return weather_data
    
    def collect_precipitation_intensity(self) -> Dict[str, Any]:
        """Yahoo! Weather APIから降水強度データを取得する"""
        precipitation_data = {
            'observation': [],
            'forecast': [],  # 予測値は最新データのみに保存（履歴には保存しない）
            'update_time': None
        }
        
        try:
            # 10分間隔でデータを取得
            params = {
                'coordinates': self.coordinates,
                'appid': self.yahoo_app_id,
                'output': 'json',
                'interval': '10'
            }
            
            print(f"Fetching precipitation intensity from Yahoo API...")
            response = requests.get(self.yahoo_api_url, params=params, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Feature' in data and len(data['Feature']) > 0:
                feature = data['Feature'][0]
                
                # 観測データの取得
                if 'Property' in feature and 'WeatherList' in feature['Property']:
                    weather_list = feature['Property']['WeatherList']
                    
                    if 'Weather' in weather_list:
                        weather_data = weather_list['Weather']
                        
                        for weather_item in weather_data:
                            if 'Type' in weather_item and 'Date' in weather_item and 'Rainfall' in weather_item:
                                data_type = weather_item['Type']
                                date_str = weather_item['Date']
                                rainfall = weather_item['Rainfall']
                                
                                # 日時をパース
                                try:
                                    # YYYYMMDDHHmmSS形式をパース
                                    dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                                    jst = ZoneInfo('Asia/Tokyo')
                                    dt_jst = dt.replace(tzinfo=jst)
                                    
                                    rainfall_data = {
                                        'datetime': dt_jst.isoformat(),
                                        'intensity': float(rainfall) if rainfall != '' else 0.0
                                    }
                                    
                                    if data_type == 'observation':
                                        precipitation_data['observation'].append(rainfall_data)
                                    elif data_type == 'forecast':
                                        precipitation_data['forecast'].append(rainfall_data)
                                        
                                except (ValueError, TypeError) as e:
                                    print(f"Error parsing precipitation data: {e}")
                                    continue
                
                # 更新時刻を設定
                jst = ZoneInfo('Asia/Tokyo')
                precipitation_data['update_time'] = datetime.now(jst).isoformat()
                
                print(f"Precipitation intensity data collected: {len(precipitation_data['observation'])} observations, {len(precipitation_data['forecast'])} forecasts")
            
        except requests.RequestException as e:
            print(f"Error fetching precipitation intensity data: {e}")
        except json.JSONDecodeError as e:
            print(f"Error parsing precipitation intensity JSON: {e}")
        except Exception as e:
            print(f"Unexpected error collecting precipitation intensity: {e}")
        
        return precipitation_data
    
    def save_data(self, data: Dict[str, Any], is_error: bool = False, error_info: Dict[str, Any] = None) -> None:
        """データを保存する"""
        jst = ZoneInfo('Asia/Tokyo')
        current_time = datetime.now(jst)
        
        # 最新データを保存（エラーの場合はlatest.jsonは更新しない）
        if not is_error:
            latest_file = self.data_dir / "latest.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        # 履歴データを保存
        date_dir = self.history_dir / current_time.strftime("%Y") / current_time.strftime("%m") / current_time.strftime("%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名の決定
        if is_error:
            # エラー時：error_HHMM.json（エラー発生時刻）
            history_file = date_dir / f"error_{current_time.strftime('%H%M')}.json"
            save_data = {
                'error_time': current_time.isoformat(),
                'error_info': error_info,
                'partial_data': data
            }
        else:
            # 正常時：HHMM.json（観測時刻）
            if 'data_time' in data:
                # ISO形式の文字列からdatetimeオブジェクトに変換
                observation_time = datetime.fromisoformat(data['data_time'])
                history_file = date_dir / f"{observation_time.strftime('%H%M')}.json"
            else:
                # data_timeがない場合は現在時刻を使用（フォールバック）
                history_file = date_dir / f"{current_time.strftime('%H%M')}.json"
            
            # 履歴データから予測値を除外
            save_data = data.copy()
            if 'precipitation_intensity' in save_data:
                # 予測値を削除（観測値のみ保存）
                save_data['precipitation_intensity'] = {
                    'observation': save_data['precipitation_intensity'].get('observation', []),
                    'update_time': save_data['precipitation_intensity'].get('update_time')
                }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Data saved: {history_file.name}")
    
    def cleanup_old_data(self, days_to_keep: int = 7) -> None:
        """古いデータを削除する"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for year_dir in self.history_dir.iterdir():
            if not year_dir.is_dir():
                continue
                
            try:
                year = int(year_dir.name)
                if year < cutoff_date.year:
                    # Remove entire year directory
                    import shutil
                    shutil.rmtree(year_dir)
                    print(f"Removed old data directory: {year_dir}")
                    continue
                    
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    try:
                        month = int(month_dir.name)
                        if year == cutoff_date.year and month < cutoff_date.month:
                            import shutil
                            shutil.rmtree(month_dir)
                            print(f"Removed old data directory: {month_dir}")
                            continue
                            
                        for day_dir in month_dir.iterdir():
                            if not day_dir.is_dir():
                                continue
                            
                            try:
                                day = int(day_dir.name)
                                dir_date = datetime(year, month, day)
                                if dir_date < cutoff_date:
                                    import shutil
                                    shutil.rmtree(day_dir)
                                    print(f"Removed old data directory: {day_dir}")
                            except (ValueError, OSError) as e:
                                print(f"Error processing day directory {day_dir}: {e}")
                                
                    except (ValueError, OSError) as e:
                        print(f"Error processing month directory {month_dir}: {e}")
                        
            except (ValueError, OSError) as e:
                print(f"Error processing year directory {year_dir}: {e}")
    
    def create_daily_summary(self) -> None:
        """前日の日次サマリーを作成する"""
        try:
            jst = ZoneInfo('Asia/Tokyo')
            current_time = datetime.now(jst)
            yesterday = current_time - timedelta(days=1)
            
            # 前日のディレクトリ
            yesterday_dir = self.history_dir / yesterday.strftime("%Y") / yesterday.strftime("%m") / yesterday.strftime("%d")
            if not yesterday_dir.exists():
                return
            
            # 既に日次サマリーが存在する場合はスキップ
            summary_file = yesterday_dir / "daily_summary.json"
            if summary_file.exists():
                return
            
            print(f"Creating daily summary for {yesterday.strftime('%Y-%m-%d')}...")
            
            # 前日のすべてのデータファイルを読み込む
            daily_data = {}
            error_count = 0
            successful_count = 0
            
            for file_path in sorted(yesterday_dir.glob("*.json")):
                # daily_summary.json自体はスキップ
                if file_path.name == "daily_summary.json":
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    
                    # エラーファイルとデータファイルを区別
                    if file_path.name.startswith("error_"):
                        error_count += 1
                    else:
                        successful_count += 1
                        # 観測時刻をキーとしてデータを保存
                        obs_time = file_path.stem  # ファイル名から拡張子を除いた部分
                        daily_data[obs_time] = file_data
                
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            
            # 日次統計を計算
            statistics = self._calculate_daily_statistics(daily_data)
            
            # サマリーデータの作成
            summary = {
                'date': yesterday.strftime('%Y-%m-%d'),
                'created_at': current_time.isoformat(),
                'total_records': successful_count,
                'error_count': error_count,
                'statistics': statistics,
                'hourly_data': daily_data  # 時刻をキーとした全データ
            }
            
            # サマリーファイルを保存
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"Daily summary created: {summary_file.name}")
            
        except Exception as e:
            print(f"Error creating daily summary: {e}")
    
    def _calculate_daily_statistics(self, daily_data: Dict[str, Any]) -> Dict[str, Any]:
        """日次統計を計算する"""
        stats = {
            'dam': {
                'water_level': {'min': None, 'max': None, 'avg': None},
                'storage_rate': {'min': None, 'max': None, 'avg': None},
                'inflow': {'min': None, 'max': None, 'avg': None, 'total': None},
                'outflow': {'min': None, 'max': None, 'avg': None, 'total': None}
            },
            'river': {
                'water_level': {'min': None, 'max': None, 'avg': None}
            },
            'rainfall': {
                'hourly': {'max': None, 'total': None},
                'cumulative': {'max': None}
            }
        }
        
        # 各データタイプごとに値を収集
        dam_levels = []
        dam_rates = []
        dam_inflows = []
        dam_outflows = []
        river_levels = []
        hourly_rains = []
        cumulative_rains = []
        
        for time_key, data in daily_data.items():
            if 'dam' in data:
                if data['dam'].get('water_level') is not None:
                    dam_levels.append(data['dam']['water_level'])
                if data['dam'].get('storage_rate') is not None:
                    dam_rates.append(data['dam']['storage_rate'])
                if data['dam'].get('inflow') is not None:
                    dam_inflows.append(data['dam']['inflow'])
                if data['dam'].get('outflow') is not None:
                    dam_outflows.append(data['dam']['outflow'])
            
            if 'river' in data and data['river'].get('water_level') is not None:
                river_levels.append(data['river']['water_level'])
            
            if 'rainfall' in data:
                if data['rainfall'].get('hourly') is not None:
                    hourly_rains.append(data['rainfall']['hourly'])
                if data['rainfall'].get('cumulative') is not None:
                    cumulative_rains.append(data['rainfall']['cumulative'])
        
        # 統計値の計算
        if dam_levels:
            stats['dam']['water_level'] = {
                'min': min(dam_levels),
                'max': max(dam_levels),
                'avg': round(sum(dam_levels) / len(dam_levels), 2)
            }
        
        if dam_rates:
            stats['dam']['storage_rate'] = {
                'min': min(dam_rates),
                'max': max(dam_rates),
                'avg': round(sum(dam_rates) / len(dam_rates), 1)
            }
        
        if dam_inflows:
            stats['dam']['inflow'] = {
                'min': min(dam_inflows),
                'max': max(dam_inflows),
                'avg': round(sum(dam_inflows) / len(dam_inflows), 2),
                'total': round(sum(dam_inflows) * 600 / 1000000, 2)  # m³/s * 600秒 / 1000000 = 百万m³
            }
        
        if dam_outflows:
            stats['dam']['outflow'] = {
                'min': min(dam_outflows),
                'max': max(dam_outflows),
                'avg': round(sum(dam_outflows) / len(dam_outflows), 2),
                'total': round(sum(dam_outflows) * 600 / 1000000, 2)  # m³/s * 600秒 / 1000000 = 百万m³
            }
        
        if river_levels:
            stats['river']['water_level'] = {
                'min': min(river_levels),
                'max': max(river_levels),
                'avg': round(sum(river_levels) / len(river_levels), 2)
            }
        
        if hourly_rains:
            stats['rainfall']['hourly'] = {
                'max': max(hourly_rains),
                'total': sum(hourly_rains)
            }
        
        if cumulative_rains:
            stats['rainfall']['cumulative'] = {
                'max': max(cumulative_rains)
            }
        
        return stats
    
    def collect_all_data(self) -> Dict[str, Any]:
        """全てのデータを収集する"""
        print("Starting data collection...")
        
        # エラー情報の収集用
        errors = []
        data_collected = {}
        
        # 観測時刻を計算（10分単位で最新の観測時刻）- 日本時間で統一
        jst = ZoneInfo('Asia/Tokyo')
        current_time = datetime.now(jst)
        minutes = (current_time.minute // 10) * 10
        observation_time = current_time.replace(minute=minutes, second=0, microsecond=0)
        
        # ダムデータと降雨データ収集
        print("Collecting dam and rainfall data...")
        try:
            dam_rainfall_data = self.collect_dam_data()
            data_collected['dam'] = dam_rainfall_data['dam']
            data_collected['rainfall'] = dam_rainfall_data['rainfall']
            
            # ダムデータのチェック
            if all(v is None for k, v in dam_rainfall_data['dam'].items() if k != 'storage_change'):
                errors.append({
                    'step': 'dam_data_collection',
                    'error': 'All dam data values are None',
                    'data': dam_rainfall_data['dam']
                })
            
            # 降雨データのチェック - エラー判定から除外（nullでも正常データとして扱う）
            # 雨量データは取得できない場合があるため、エラーとしない
            if all(v is None for k, v in dam_rainfall_data['rainfall'].items()):
                print("Rainfall data is None - treating as normal condition")
        except Exception as e:
            print(f"Error collecting dam and rainfall data: {e}")
            errors.append({
                'step': 'dam_rainfall_data_collection',
                'error': str(e),
                'error_type': type(e).__name__
            })
            data_collected['dam'] = {
                'water_level': None,
                'storage_rate': None,
                'inflow': None,
                'outflow': None,
                'storage_change': None
            }
            data_collected['rainfall'] = {
                'hourly': None,
                'cumulative': None,
                'change': None
            }
        
        # 河川データ収集
        print("Collecting river data...")
        try:
            river_data = self.collect_river_data()
            data_collected['river'] = river_data
            if river_data['water_level'] is None:
                errors.append({
                    'step': 'river_data_collection',
                    'error': 'River water level is None',
                    'data': river_data
                })
        except Exception as e:
            print(f"Error collecting river data: {e}")
            errors.append({
                'step': 'river_data_collection',
                'error': str(e),
                'error_type': type(e).__name__
            })
            data_collected['river'] = {
                'water_level': None,
                'level_change': None,
                'status': None
            }
        
        # 降雨データはダムデータと同時に取得済み
        
        # 天気予報データ収集
        print("Collecting weather forecast data...")
        try:
            weather_data = self.collect_weather_data()
            data_collected['weather'] = weather_data
        except Exception as e:
            print(f"Error collecting weather data: {e}")
            errors.append({
                'step': 'weather_data_collection',
                'error': str(e),
                'error_type': type(e).__name__
            })
            data_collected['weather'] = {
                'today': {
                    'weather_code': None,
                    'weather_text': None,
                    'temp_max': None,
                    'temp_min': None,
                    'precipitation_probability': [],
                    'precipitation_times': []
                },
                'tomorrow': {
                    'weather_code': None,
                    'weather_text': None,
                    'temp_max': None,
                    'temp_min': None,
                    'precipitation_probability': [],
                    'precipitation_times': []
                },
                'day_after_tomorrow': {
                    'weather_code': None,
                    'weather_text': None,
                    'temp_max': None,
                    'temp_min': None,
                    'precipitation_probability': [],
                    'precipitation_times': []
                },
                'update_time': None,
                'weekly_forecast': []
            }
        
        # 降水強度データ収集（Yahoo! Weather API）
        print("Collecting precipitation intensity data...")
        try:
            precipitation_intensity_data = self.collect_precipitation_intensity()
            data_collected['precipitation_intensity'] = precipitation_intensity_data
        except Exception as e:
            print(f"Error collecting precipitation intensity data: {e}")
            errors.append({
                'step': 'precipitation_intensity_data_collection',
                'error': str(e),
                'error_type': type(e).__name__
            })
            data_collected['precipitation_intensity'] = {
                'observation': [],
                'forecast': [],
                'update_time': None
            }
        
        # データを統合（日本時間で保存）
        timestamp_jst = datetime.now(jst)
        observation_time_jst = observation_time  # 既にJST
        
        # 実際の観測時刻を使用（最新データを取得した場合）
        actual_obs_time = None
        # ダムデータの実際の観測時刻をチェック
        if 'dam' in data_collected and data_collected['dam'].get('actual_observation_time'):
            try:
                actual_obs_time = datetime.strptime(
                    data_collected['dam']['actual_observation_time'], 
                    "%Y/%m/%d %H:%M"
                ).replace(tzinfo=jst)
                observation_time_jst = actual_obs_time
                print(f"Using actual dam observation time: {actual_obs_time}")
            except ValueError:
                pass
        
        # 河川データの実際の観測時刻をチェック（ダムと異なる場合がある）
        if 'river' in data_collected and data_collected['river'].get('actual_observation_time'):
            try:
                river_obs_time = datetime.strptime(
                    data_collected['river']['actual_observation_time'], 
                    "%Y/%m/%d %H:%M"
                ).replace(tzinfo=jst)
                # ダムと河川で異なる時刻の場合、より新しい方を使用
                if actual_obs_time is None or river_obs_time > actual_obs_time:
                    observation_time_jst = river_obs_time
                    print(f"Using actual river observation time: {river_obs_time}")
            except ValueError:
                pass
        
        data = {
            'timestamp': timestamp_jst.isoformat(),
            'data_time': observation_time_jst.isoformat(),  # 観測時刻を追加
            'dam': data_collected.get('dam', {}),
            'river': data_collected.get('river', {}),
            'rainfall': data_collected.get('rainfall', {}),
            'weather': data_collected.get('weather', {}),
            'precipitation_intensity': data_collected.get('precipitation_intensity', {})
        }
        
        # actual_observation_timeは保存データから削除（内部使用のみ）
        if 'actual_observation_time' in data['dam']:
            del data['dam']['actual_observation_time']
        if 'actual_observation_time' in data['river']:
            del data['river']['actual_observation_time']
        
        # エラーがある場合はエラーファイルを保存
        if errors:
            error_info = {
                'errors': errors,
                'total_errors': len(errors),
                'observation_time': observation_time_jst.isoformat()
            }
            self.save_data(data, is_error=True, error_info=error_info)
            print(f"Data collection completed with {len(errors)} errors")
        else:
            # 正常データの保存
            self.save_data(data)
            print("Data collection completed successfully")
        
        # 古いデータのクリーンアップ
        self.cleanup_old_data()
        
        # 日次サマリーの作成（前日分）- 使用されていないため無効化
        # self.create_daily_summary()
        
        return data

def main():
    """メイン関数"""
    collector = KotogawaDataCollector()
    
    try:
        data = collector.collect_all_data()
        print("Collection process completed!")
        print(f"Latest data: {json.dumps(data, ensure_ascii=False, indent=2, default=str)}")
    except Exception as e:
        print(f"Critical error during data collection: {e}")
        # クリティカルエラーの場合もエラーファイルを保存
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            import pytz
            ZoneInfo = lambda x: pytz.timezone(x)
        jst = ZoneInfo('Asia/Tokyo')
        current_time = datetime.now(jst)
        error_data = {
            'timestamp': current_time.isoformat(),
            'data_time': None,
            'dam': {'water_level': None, 'storage_rate': None, 'inflow': None, 'outflow': None, 'storage_change': None},
            'river': {'water_level': None, 'level_change': 0.0, 'status': '不明'},
            'rainfall': {'hourly': 0, 'cumulative': 0, 'change': 0},
            'weather': {'today': {'weather_text': None, 'temp_max': None, 'temp_min': None}, 'tomorrow': {'weather_text': None, 'temp_max': None, 'temp_min': None}, 'update_time': None}
        }
        error_info = {
            'errors': [{
                'step': 'main_execution',
                'error': str(e),
                'error_type': type(e).__name__
            }],
            'total_errors': 1,
            'observation_time': None
        }
        collector.save_data(error_data, is_error=True, error_info=error_info)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())