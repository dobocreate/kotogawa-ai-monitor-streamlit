# 学習モデル仕様書 — River ストリーミング水位予測モデル

## 1. 目的

- **リアルタイム河川水位予測** を 10 分間隔で提供し、洪水・放流管理の意思決定を支援する。
- 欠測・不規則間隔・概念ドリフトに強い **オンライン学習** を採用し、サーバ負荷とレイテンシを最小化する。

---

## 2. システム全体構成

```mermaid
flowchart LR
    API[河川・雨量 API(JSON)] -->|生データ| StreamService((Stream Fetcher))
    StreamService -->|JSON Record| Predictor(Python + River)
    Predictor -->|推論結果 JSON| Redis[(Redis Cache)]
    Predictor -->|MAE,Drift| Prometheus
    Prometheus --> Grafana((Grafana Dash))
    Predictor -->|Pickle Snapshot| S3[(Object Storage)]
```

---

## 3. データ仕様

| フィールド         | 型        | 説明         | 単位           |
| ------------- | -------- | ---------- | ------------ |
| `timestamp`   | ISO‑8601 | 取得時刻       | –            |
| `water_level` | float    | 観測水位       | m            |
| `dam_outflow` | float    | ダム放流量      | m³/s         |
| `rainfall`    | float    | 直近 10 分降雨量 | mm           |
| `elapsed_min` | float    | 前回観測からの経過分 | min *(自動生成)* |

> **注意**: 欠測値は `null` で受け取り、後段 `StatImputer` が補完。

---

## 4. 特徴量エンジニアリング

| 手順     | River コンポーネント                  | パラメータ                                                                |
| ------ | ------------------------------ | -------------------------------------------------------------------- |
| ラグ生成   | `feature_extraction.Lagger`    | `{'water_level':(1,2,3), 'dam_outflow':(1,2,3), 'rainfall':(1,2,3)}` |
| 欠測補完   | `preprocessing.StatImputer`    | `strategy='mean'` (数値平均)                                             |
| スケーリング | `preprocessing.StandardScaler` | default                                                              |

---

## 5. モデル設定

| 項目              | 値                         |
| --------------- | ------------------------- |
| ライブラリ           | River **0.21.0**          |
| 学習器             | `forest.ARFRegressor`     |
| n\_estimators   | 15                        |
| max\_depth      | 15                        |
| drift\_detector | `drift.ADWIN(delta=1e-3)` |
| seed            | 42                        |

```python
from river import compose, feature_extraction as fx, preprocessing as pp
from river import forest, drift

pipe = compose.Pipeline(
    fx.Lagger({...}),
    pp.StatImputer(),
    pp.StandardScaler(),
    forest.ARFRegressor(
        n_estimators=15,
        max_depth=15,
        drift_detector=drift.ADWIN(delta=1e-3),
        seed=42,
    )
)
```

---

## 6. オンライン学習フロー

1. **入力取得**: API から最新 JSON をポーリング (毎 10 min)。
2. `pipe.predict_one(x)` → 予測値 `ŷ` 返却。
3. Redis に `key=station:timestamp` 形式で保存。有効期限 24 h。
4. 実測水位が到着次第 `pipe.learn_one(x, y)` で学習更新。
5. `metrics.MAE` を更新し、しきい値 0.15 超で Prometheus アラート。
6. 5 分ごとに `pickle` で `/tmp/model.pkl` → S3 バケットにアップロード。

---

## 7. 入出力インターフェース

### 7.1 REST (FastAPI)

```http
POST /predict
Content-Type: application/json
{
  "timestamp": "2025-07-20T15:20:00+09:00",
  "water_level": 1.98,
  "dam_outflow": 134.0,
  "rainfall": 0.2
}
```

**200 OK**

```json
{
  "predicted_water_level": 2.05,
  "mae_last_100": 0.12,
  "model_version": "river-0.21.0-20250720T1520"
}
```

### 7.2 gRPC (Option)

- Service: `WaterLevelPredictor`
- Method: `Predict(Record) returns (Prediction)`

---

## 8. モデル評価指標

| 指標             | 目的     | 計算方法              |
| -------------- | ------ | ----------------- |
| MAE (10 min 先) | 平均誤差   | `metrics.MAE()`   |
| RMSE           | 大誤差重視  | `metrics.RMSE()`  |
| 観測可用率          | 欠測監視   | 受信レコード数 / 期待レコード数 |
| ドリフト検知回数       | 概念ドリフト | `ADWIN` アラーム回数/月  |

---

## 9. 運用・保守

- **モデル更新**: 月次でバッチ再学習 (過去 1 年データ) → A/B テスト後に本番切替。
- **可観測性**: Prometheus + Grafana, Loglevel=INFO, TraceID=UUID。
- **障害対応**: MAE >0.3 かドリフト連発時にロールバック (`model.pkl` スナップショット適用)。
- **依存ライブラリ**: `river==0.21.0`, `aiohttp`, `fastapi`, `uvicorn`, `redis`, `boto3`, `prometheus-client`。
- **デプロイ**: Docker (python:3.10‑slim), GitHub Actions CI/CD, ECR→ECS。

---

## 10. セキュリティ & ガバナンス

- **データ暗号化**: API ↔ Predictor は TLS1.2。S3 バケットは SSE-S3 暗号化。
- **IAM**: Predictor には最小権限ロール (S3 putObject, CloudWatch putMetricData)。
- **PII**: 取り扱い無し。
- **監査ログ**: `uvicorn` access log, AWS CloudTrail。

---

## 11. 既知の制約 / 今後の課題

| ID   | 内容                  | 影響                   | 改善案                 |
| ---- | ------------------- | -------------------- | ------------------- |
| C‑01 | 0.21 系は Polars 依存あり | Docker イメージ容量 +50 MB | 0.21.2 へアップグレード予定   |
| C‑02 | 深層モデル未サポート          | 非線形複雑性に限界            | River+ONNX 経由で微調整検討 |
| C‑03 | 欠測 2 h 超で精度劣化       | 誤警報リスク               | 監視フラグで予測停止          |

---

## 12. 変更履歴

| 版   | 日付         | 変更者 | 主要変更 |
| --- | ---------- | --- | ---- |
| 0.1 | 2025‑07‑20 | ご主人 | 初版作成 |

---

**作成者**: ご主人 **連絡先**: \*\*\* (追記してください)\
**ライセンス**: Internal Use Only

