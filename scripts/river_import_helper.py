"""
Riverモジュールのインポートヘルパー
"""

# Riverオンライン学習モジュールのインポート
RIVER_LEARNING_AVAILABLE = False
RIVER_MODEL = None

try:
    # まずv2を試す
    from .river_streaming_prediction_v2 import RiverStreamingPredictor
    RIVER_MODEL = RiverStreamingPredictor
    RIVER_LEARNING_AVAILABLE = True
    print("River Streaming v2 (ARF + ADWIN) を使用")
except ImportError:
    try:
        # v2が利用できない場合はv1にフォールバック
        from .river_streaming_prediction import RiverStreamingPredictor
        RIVER_MODEL = RiverStreamingPredictor
        RIVER_LEARNING_AVAILABLE = True
        print("River Streaming v1 を使用")
    except ImportError as e:
        print(f"RiverStreamingPredictorのインポートエラー: {e}")

def get_river_predictor():
    """利用可能なRiver予測モデルを返す"""
    if RIVER_MODEL:
        return RIVER_MODEL()
    return None