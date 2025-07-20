"""
Riverモジュールのインポートヘルパー
"""

# Riverオンライン学習モジュールのインポート
RIVER_LEARNING_AVAILABLE = False
RIVER_MODEL = None

try:
    from .river_streaming_prediction import RiverStreamingPredictor
    RIVER_MODEL = RiverStreamingPredictor
    RIVER_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"RiverStreamingPredictorのインポートエラー: {e}")
    try:
        from .river_online_prediction import RiverOnlinePredictor
        RIVER_MODEL = RiverOnlinePredictor
        RIVER_LEARNING_AVAILABLE = True
    except ImportError as e2:
        print(f"RiverOnlinePredictorのインポートエラー: {e2}")
        pass

def get_river_predictor():
    """利用可能なRiver予測モデルを返す"""
    if RIVER_MODEL:
        return RIVER_MODEL()
    return None