"""
アラート状態チェック関連のユーティリティモジュール
"""
from typing import Dict, Any


# デフォルトの閾値定義
DEFAULT_THRESHOLDS = {
    'river_danger': 5.5,      # 河川危険水位
    'river_warning': 4.0,     # 河川警戒水位
    'river_caution': 3.0,     # 河川注意水位
    'dam_danger': 153.5,      # ダム設計最高水位
    'dam_warning': 150.0,     # ダム洪水時最高水位
    'rain_hourly_danger': 50,     # 時間雨量危険値
    'rain_hourly_warning': 30,    # 時間雨量警戒値
    'rain_hourly_caution': 10,    # 時間雨量注意値
    'rain_cumulative_danger': 200,    # 累積雨量危険値
    'rain_cumulative_warning': 100,   # 累積雨量警戒値
    'rain_cumulative_caution': 50,    # 累積雨量注意値
}


def check_alert_status(data: Dict[str, Any], thresholds: Dict[str, float] = None) -> Dict[str, str]:
    """
    アラート状態をチェック
    
    Args:
        data: 監視データ
        thresholds: アラート閾値（省略時はデフォルト値を使用）
    
    Returns:
        各項目のアラート状態を含む辞書
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
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
        if dam_level >= thresholds.get('dam_danger', DEFAULT_THRESHOLDS['dam_danger']):  # 設計最高水位
            alerts['dam'] = '危険'
            alert_level = max(alert_level, 3)
        elif dam_level >= thresholds.get('dam_warning', DEFAULT_THRESHOLDS['dam_warning']):  # 洪水時最高水位
            alerts['dam'] = '警戒'
            alert_level = max(alert_level, 2)
    
    # 雨量チェック
    hourly_rain = data.get('rainfall', {}).get('hourly')
    cumulative_rain = data.get('rainfall', {}).get('cumulative')
    
    # null値の場合は雨量チェックをスキップ
    if hourly_rain is not None and cumulative_rain is not None:
        if (hourly_rain >= thresholds.get('rain_hourly_danger', DEFAULT_THRESHOLDS['rain_hourly_danger']) or 
            cumulative_rain >= thresholds.get('rain_cumulative_danger', DEFAULT_THRESHOLDS['rain_cumulative_danger'])):
            alerts['rainfall'] = '危険'
            alert_level = max(alert_level, 3)
        elif (hourly_rain >= thresholds.get('rain_hourly_warning', DEFAULT_THRESHOLDS['rain_hourly_warning']) or 
              cumulative_rain >= thresholds.get('rain_cumulative_warning', DEFAULT_THRESHOLDS['rain_cumulative_warning'])):
            alerts['rainfall'] = '警戒'
            alert_level = max(alert_level, 2)
        elif (hourly_rain >= thresholds.get('rain_hourly_caution', DEFAULT_THRESHOLDS['rain_hourly_caution']) or 
              cumulative_rain >= thresholds.get('rain_cumulative_caution', DEFAULT_THRESHOLDS['rain_cumulative_caution'])):
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