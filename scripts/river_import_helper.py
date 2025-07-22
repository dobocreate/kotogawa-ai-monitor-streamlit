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
    print("River Streaming (ARF + ADWIN) を使用")
except ImportError as e:
    print(f"RiverStreamingPredictorのインポートエラー: {e}")

def get_river_predictor():
    """利用可能なRiver予測モデルを返す"""
    if RIVER_MODEL:
        return RIVER_MODEL()
    return None