"""
天気予報表示関連のコンポーネント
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz
    ZoneInfo = lambda x: pytz.timezone(x)


def get_weather_icon(weather_code: str, weather_text: str = "") -> str:
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


def create_weather_forecast_display(weather_data: Optional[Dict[str, Any]]) -> None:
    """天気予報を表示"""
    
    if not weather_data:
        st.info("⛅ 天気予報データは利用できません")
        return
    
    try:
        update_time = weather_data.get('update_time', '')
        if update_time:
            # ISO形式のタイムスタンプを解析してJSTに変換
            update_dt = datetime.fromisoformat(update_time.replace('Z', '+00:00'))
            if update_dt.tzinfo is None:
                update_dt = update_dt.replace(tzinfo=ZoneInfo('Asia/Tokyo'))
            else:
                update_dt = update_dt.astimezone(ZoneInfo('Asia/Tokyo'))
            update_time_str = update_dt.strftime('%H:%M')
        else:
            update_time_str = '不明'
    except:
        update_time_str = '不明'
    
    today = weather_data.get('today', {})
    tomorrow = weather_data.get('tomorrow', {})
    
    # 天気アイコンを取得
    today_icon = get_weather_icon(
        today.get('weather_code', ''),
        today.get('weather_text', '')
    )
    tomorrow_icon = get_weather_icon(
        tomorrow.get('weather_code', ''),
        tomorrow.get('weather_text', '')
    )
    
    # 降水確率
    today_rain_probs = today.get('precipitation_probability', [])
    tomorrow_rain_probs = tomorrow.get('precipitation_probability', [])
    
    # 最高降水確率を取得
    today_max_rain = max(today_rain_probs) if today_rain_probs else 0
    tomorrow_max_rain = max(tomorrow_rain_probs) if tomorrow_rain_probs else 0
    
    # 今日と明日を横並びで表示
    col1, col2 = st.columns(2)
    
    with col1:
        # 今日の天気
        st.markdown(f"### 今日 {today_icon}")
        today_weather = today.get('weather_text', '不明')
        st.write(f"**{today_weather}**")
        
        # 気温表示（データがある場合のみ）
        temp_max = today.get('temp_max')
        temp_min = today.get('temp_min')
        if temp_max is not None and temp_min is not None:
            temp_text = f"🌡️ {temp_max}℃ / {temp_min}℃"
        else:
            temp_text = "🌡️ データなし"
        st.write(temp_text)
        
        # 降水確率表示
        if today_rain_probs:
            rain_times = today.get('precipitation_times', [])
            st.write(f"☔ 最高降水確率: **{today_max_rain}%**")
            
            # 降水確率が高い時間帯を強調表示
            high_rain_times = []
            for i, (prob, time) in enumerate(zip(today_rain_probs, rain_times)):
                if prob >= 50:
                    high_rain_times.append(f"{time}: {prob}%")
            
            if high_rain_times:
                st.caption("降水確率50%以上の時間帯:")
                for time_info in high_rain_times:
                    st.caption(f"　• {time_info}")
        else:
            st.write("☔ 降水確率: データなし")
    
    with col2:
        # 明日の天気
        st.markdown(f"### 明日 {tomorrow_icon}")
        tomorrow_weather = tomorrow.get('weather_text', '不明')
        st.write(f"**{tomorrow_weather}**")
        
        # 気温表示（データがある場合のみ）
        temp_max = tomorrow.get('temp_max')
        temp_min = tomorrow.get('temp_min')
        if temp_max is not None and temp_min is not None:
            temp_text = f"🌡️ {temp_max}℃ / {temp_min}℃"
        else:
            temp_text = "🌡️ データなし"
        st.write(temp_text)
        
        # 降水確率表示
        if tomorrow_rain_probs:
            rain_times = tomorrow.get('precipitation_times', [])
            st.write(f"☔ 最高降水確率: **{tomorrow_max_rain}%**")
            
            # 降水確率が高い時間帯を強調表示
            high_rain_times = []
            for i, (prob, time) in enumerate(zip(tomorrow_rain_probs, rain_times)):
                if prob >= 50:
                    high_rain_times.append(f"{time}: {prob}%")
            
            if high_rain_times:
                st.caption("降水確率50%以上の時間帯:")
                for time_info in high_rain_times:
                    st.caption(f"　• {time_info}")
        else:
            st.write("☔ 降水確率: データなし")
    
    # 更新時刻
    st.caption(f"最終更新: {update_time_str}")


def create_weekly_forecast_display(weather_data: Optional[Dict[str, Any]]) -> None:
    """週間天気予報を表示"""
    
    if not weather_data:
        st.info("⛅ 週間天気予報データは利用できません")
        return
    
    weekly_forecast = weather_data.get('weekly_forecast', [])
    
    if not weekly_forecast:
        st.info("📅 週間天気予報データはまだありません")
        return
    
    # データをDataFrameに変換して表示
    forecast_data = []
    for day in weekly_forecast:
        date_str = day.get('date', '')
        if date_str:
            try:
                # 日付を解析してフォーマット
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%m/%d(%a)')
                weekday_jp = ['月', '火', '水', '木', '金', '土', '日'][date_obj.weekday()]
                formatted_date = date_obj.strftime(f'%m/%d({weekday_jp})')
            except:
                formatted_date = date_str
        else:
            formatted_date = '不明'
        
        # 天気アイコンを取得
        weather_icon = get_weather_icon(
            day.get('weather_code', ''),
            day.get('weather_text', '')
        )
        
        # 気温（データがない場合は'-'を表示）
        temp_max = day.get('temp_max')
        temp_min = day.get('temp_min')
        temp_max_str = f"{temp_max}℃" if temp_max is not None else '-'
        temp_min_str = f"{temp_min}℃" if temp_min is not None else '-'
        
        # 降水確率（最高値を表示）
        rain_probs = day.get('precipitation_probability', [])
        max_rain = max(rain_probs) if rain_probs else 0
        
        forecast_data.append({
            '日付': formatted_date,
            '天気': f"{weather_icon} {day.get('weather_text', '不明')}",
            '最高': temp_max_str,
            '最低': temp_min_str,
            '降水確率': f"{max_rain}%"
        })
    
    # 表形式で表示
    import pandas as pd
    df = pd.DataFrame(forecast_data)
    
    # スタイリング付きで表示
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "日付": st.column_config.TextColumn("日付", width="small"),
            "天気": st.column_config.TextColumn("天気", width="medium"),
            "最高": st.column_config.TextColumn("最高", width="small"),
            "最低": st.column_config.TextColumn("最低", width="small"),
            "降水確率": st.column_config.TextColumn("降水確率", width="small"),
        }
    )
    
    # 降水確率が高い日の警告
    high_rain_days = []
    for day in weekly_forecast:
        date_str = day.get('date', '')
        rain_probs = day.get('precipitation_probability', [])
        max_rain = max(rain_probs) if rain_probs else 0
        
        if max_rain >= 70:
            if date_str:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    weekday_jp = ['月', '火', '水', '木', '金', '土', '日'][date_obj.weekday()]
                    formatted_date = date_obj.strftime(f'%m/%d({weekday_jp})')
                    high_rain_days.append(f"{formatted_date}: {max_rain}%")
                except:
                    high_rain_days.append(f"{date_str}: {max_rain}%")
    
    if high_rain_days:
        st.warning("☔ 降水確率70%以上の日:")
        for day_info in high_rain_days:
            st.caption(f"　• {day_info}")