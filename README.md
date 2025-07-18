# 🌊 厚東川氾濫監視システム v2.5

山口県宇部市の厚東川ダムおよび厚東川（持世寺）の水位・雨量データをリアルタイムで監視・表示するWebアプリケーションです。Yahoo! Weather APIによる降水強度予測機能を搭載。

公開URL：https://kotogawa-monitor-app-fjprveyn8fzb3mbffkwq6i.streamlit.app/

## 📋 概要

- **データソース**: 山口県土木防災情報システム、Yahoo! Weather API
- **更新頻度**: 10分間隔
- **データ保持**: 過去7日間
- **警戒レベル**: 4段階のアラート機能
- **モバイル対応**: レスポンシブデザイン採用

## 🚀 デプロイ方法

### Streamlit Cloud でのデプロイ

1. **GitHubリポジトリの準備**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Streamlit Cloud でアプリを作成**
   - [Streamlit Cloud](https://streamlit.io/cloud) にアクセス
   - GitHubアカウントでサインイン
   - "New app" をクリック
   - リポジトリを選択: `kotogawa-monitor-streamlit`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - "Deploy!" をクリック

3. **GitHub Actions の有効化**
   - リポジトリの Settings → Actions → General
   - "Allow all actions and reusable workflows" を選択
   - 初回データ収集を手動実行:
     ```bash
     python scripts/collect_data.py
     git add data/
     git commit -m "Initial data collection"
     git push
     ```

## 💻 ローカル実行

### 必要要件

- Python 3.9以上
- pip

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd kotogawa-monitor-streamlit

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 実行

```bash
# データ収集（初回）
python scripts/collect_data.py

# Streamlitアプリの起動
streamlit run streamlit_app.py
```

ブラウザで `http://localhost:8501` にアクセス

## 📊 機能

### データ表示
- **河川水位**: 現在値・変化量・警戒ステータス
- **ダム水位**: 水位・貯水率・流入出量
- **雨量**: 時間雨量・累積雨量
- **降水強度**: 観測値・予測値（Yahoo! Weather API）
- **天気予報**: 今日・明日・週間予報
- **時系列グラフ**: 6〜72時間の選択可能な表示期間
  - 河川水位・全放流量
  - ダム放流量・時間雨量（降水強度含む）
  - ダム貯水位・時間雨量（降水強度含む）
  - ダム流入出量・累加雨量
  - 降水強度・時間雨量（Yahoo! Weather API）

### アラート機能
- **4段階の警戒レベル**:
  - 🟢 正常: 3.80m未満
  - 🟡 水防団待機: 3.80m以上
  - 🟠 氾濫注意: 5.00m以上
  - 🔴 氾濫危険: 5.50m以上

### データ管理
- **自動収集**: GitHub Actions による10分間隔
- **履歴保存**: 7日間のデータ保持
- **CSVエクスポート**: データのダウンロード機能
- **自動クリーンアップ**: 古いデータの削除

## 🔧 設定

### サイドバー機能

- **更新設定**: 自動更新間隔（10/30/60分）、手動更新
- **表示設定**: 表示期間（6〜72時間）、グラフ編集、週間天気
- **アラート設定**: 河川・ダムの警戒水位カスタマイズ
- **システム情報**: 観測状況、警戒レベル説明、データソース

### モバイル対応

- 画面幅768px以下でサイドバー自動非表示
- ハンバーガーメニューで開閉可能

## 📁 プロジェクト構造

```
kotogawa-monitor-streamlit/
├── streamlit_app.py          # メインアプリケーション
├── requirements.txt          # 依存パッケージ
├── README.md                 # このファイル
├── .streamlit/
│   └── config.toml          # Streamlit設定
├── data/
│   ├── latest.json          # 最新データ
│   └── history/             # 履歴データ（YYYY/MM/DD/）
├── scripts/
│   ├── collect_data.py      # データ収集スクリプト
│   ├── process_data.py      # データ処理・分析
│   └── cleanup_data.py      # 古いデータ削除
├── .github/
│   └── workflows/
│       ├── data_collection.yml  # 定期データ収集
│       └── cleanup.yml          # 定期クリーンアップ
└── doc/
    └── 厚東川監視システム開発仕様書.txt
```

## 🔄 GitHub Actions

### データ収集ワークフロー
- **実行間隔**: 10分ごと
- **タイムアウト**: 5分
- **処理**: データ取得 → 保存 → Git push

### クリーンアップワークフロー  
- **実行間隔**: 毎日0時（UTC）
- **処理**: 7日以前のデータ削除 → Git push

## 🐛 トラブルシューティング

### データが表示されない場合

1. **データファイルの確認**:
   ```bash
   ls -la data/
   cat data/latest.json
   ```

2. **手動データ収集**:
   ```bash
   python scripts/collect_data.py
   ```

3. **GitHub Actions の確認**:
   - リポジトリの "Actions" タブで実行状況を確認
   - 失敗している場合はログを確認

### アプリがエラーで起動しない場合

1. **依存関係の確認**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Python バージョンの確認**:
   ```bash
   python --version  # 3.9以上
   ```

3. **ファイルパスの確認**:
   - 絶対パスでの実行を推奨
   ```bash
   cd /path/to/kotogawa-monitor-streamlit
   streamlit run streamlit_app.py
   ```

## 📈 今後の拡張

### v2.5の新機能 ✨
- ダム放流量・時間雨量グラフを新規追加
- 週間天気予報を6日表示に変更（データ不完全な最初の日を除外）
- グラフ凡例の「byYahoo!」を「by Yahoo!」に修正
- グラフ配置の最適化（ダム関連グラフを上部に集約）

### v2.3の新機能
- モバイル対応（サイドバー自動非表示）
- サイドバーUI全面改善（入れ子構造）
- 免責事項の明記
- システムヘッダー上部スペース削除

### Phase 2 実装予定
- データ取得エラー時の通知機能
- 過去データの統計分析機能
- 長期予測機能の追加

### Phase 3 実装予定  
- アラート通知機能の強化
- パフォーマンス最適化
- ユーザビリティテストと改善

## 📄 ライセンス

このプロジェクトはMITライセンスのもとで公開されています。

---

## ⚠️ 免責事項

※ 本システムは山口県公開データを再加工した参考情報です。防災判断は必ず公式発表をご確認ください。  
※ 本システムの利用または利用不能により生じた直接・間接の損害について、一切責任を負いません。
