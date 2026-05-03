# Quick Start

## 環境需求

- Python 3.11+
- Anthropic API Key（`claude-sonnet-4-20250514`）

---

## 1. 安裝依賴

```bash
pip install -r requirements.txt
```

---

## 2. 設定環境變數

在專案根目錄建立 `.env`：

```bash
ANTHROPIC_API_KEY=你的key
```

> 如需即時抓取 Finnhub 新聞（選用）：`FINNHUB_API_KEY=你的key`
> 不設定時自動使用本地 `data/news_database.json`。

---

## 3. 執行單次回測

```bash
cd src
python main.py
```

使用預設 `configs/config.yaml`，結果存至 `results/`。

指定不同設定或輸出目錄：

```bash
python main.py --config ../configs/config.yaml --output-dir ../results/my_run
```

跳過 LLM（快速測試用，不消耗 API 額度）：

```bash
python main.py --no-llm
```

---

## 4. 執行三組對照實驗

```bash
cd src
python run_experiments.py
```

依序執行以下三組實驗，每組獨立儲存至 `results/{name}/`：

| 實驗 | 說明 |
|------|------|
| `no_news` | 純結構化分析（無新聞） |
| `fusion_default` | 新聞 + Dynamic Omega（sens=0.5） |
| `weak_fusion` | 新聞 + 保守融合（sens=0.2） |

完成後自動生成比較圖表：

```
results/comparison_cumulative_returns.png   # 三組累積報酬對照
results/comparison_bl_performance.png       # BL 策略跨實驗績效
results/comparison_summary.csv              # 完整數值摘要
```

> 三組實驗共約 231 次 LLM API 呼叫，預計 20-25 分鐘。

---

## 5. 查看結果

每個實驗資料夾包含：

```
results/{experiment_name}/
├── backtest_results.pkl          # 完整數據（可用 pickle.load 讀取）
├── performance_summary.csv       # 四策略績效摘要
├── cumulative_returns.png        # 累積報酬曲線
├── drawdowns.png                 # 回撤圖
├── rolling_metrics.png           # 滾動 Sharpe / 波動率
├── return_distributions.png      # 報酬分佈
├── weights_black_litterman.png   # BL 月度權重
├── weights_markowitz.png         # Markowitz 月度權重
└── weights_equal_weight.png      # 等權重（固定）
```

用 Python 讀取原始數據：

```python
import pickle

with open("results/fusion_default/backtest_results.pkl", "rb") as f:
    r = pickle.load(f)

# 取得 BL 策略每日持倉淨值
pv = r["black_litterman"]["portfolio_values"]
print(pv.tail())
```

---

## 6. 調整參數

編輯 `configs/config.yaml`：

```yaml
black_litterman:
  tau: 0.15              # 調高 → 對 LLM 觀點更敏感
  risk_aversion: 1.8     # 調高 → 組合更保守

llm:
  use_news: true         # false = 無新聞（no_news 模式）
  fusion:
    enabled: true
    agreement_sensitivity: 0.5   # 調高 → Dynamic Omega 效果更強

backtest:
  start_date: "2025-03-01"
  end_date: "2026-01-31"
  rebalance_frequency: "monthly"
```

---

## 常見問題

**Q: API 呼叫量太多，可以加速嗎？**
A: 先用 `--no-llm` 驗證回測邏輯，確認正確後再開 LLM。

**Q: 新聞數據只到 2026-01，之後怎麼辦？**
A: 設定 `FINNHUB_API_KEY` 後，`data_collection.py` 會自動呼叫 Finnhub API 補充最新新聞。或手動更新 `data/news_database.json`。

**Q: 想新增實驗組合怎麼做？**
A: 在 `src/run_experiments.py` 的 `EXPERIMENTS` list 中新增一組 dict，指定 `name`、`desc`、`overrides`（以點分隔的 YAML 路徑覆蓋值）。

**Q: `backtest_results.pkl` 裡有哪些 key？**
A: `black_litterman`、`markowitz`、`equal_weight`、`spy_benchmark`。每個 key 下有 `portfolio_values`（DataFrame）、`total_return`、`weights_history` 等欄位。
