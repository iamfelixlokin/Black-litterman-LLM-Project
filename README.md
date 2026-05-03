# BL Algo Trading — Black-Litterman + LLM + News Fusion

以 Claude LLM 生成市場觀點，融合技術面與新聞情緒，驅動 Black-Litterman 模型最佳化 Magnificent 7 科技股投資組合。包含三組實驗對照（無新聞 / 完整融合 / 保守融合），全面評估 LLM 觀點品質對 BL 績效的影響。

---

## 回測結果（2025-03-03 → 2026-01-02，11 次月度 Rebalance）

### BL 策略跨實驗比較

| 實驗 | 描述 | 總報酬 | 年化報酬 | Sharpe | Max DD |
|------|------|--------|----------|--------|--------|
| **fusion_default** | 新聞 + Dynamic Omega（sens=0.5） | **22.85%** | **22.48%** | **0.634** | -23.2% |
| no_news | 純結構化分析（無新聞） | 21.73% | 21.22% | 0.561 | -20.1% |
| weak_fusion | 新聞 + 保守融合（sens=0.2） | 20.17% | 19.95% | 0.546 | -23.2% |

### 四策略比較（fusion_default 實驗）

| 策略 | 總報酬 | 年化報酬 | 年化波動率 | Sharpe | Max DD |
|------|--------|----------|------------|--------|--------|
| **Black-Litterman (Fusion)** | **22.85%** | **22.48%** | 29.15% | **0.634** | -23.2% |
| Equal Weight | 22.04% | 20.51% | 29.32% | 0.563 | -18.8% |
| Markowitz | 15.53% | 15.19% | 32.73% | 0.342 | -22.3% |
| SPY Benchmark | 10.74% | 9.91% | 19.23% | 0.307 | -14.7% |

> BL + Fusion 在 Sharpe 上優於所有策略，總報酬為 SPY 的 2.1 倍。

---

## 核心架構

```
價格數據（yfinance）+ 新聞情緒（VADER）
              ↓
      Claude LLM 觀點生成
   （Alpha 趨勢 + 新聞情緒 → 融合觀點）
              ↓
     Dynamic Omega 調整
  （技術面與新聞一致 → 降低 Ω，衝突 → 提高 Ω）
              ↓
      Black-Litterman 模型
  （後驗報酬 = Π + τΣP'[PτΣP' + Ω]⁻¹(Q - PΠ)）
              ↓
    最佳化投資組合（月度 Rebalance）
```

### Dynamic Omega 融合機制

```python
agreement = tech_signal × news_signal  # {-1, 0, +1}
fusion_scalar = base_scalar × exp(−k × agreement)
```

- **一致（ACCELERATING + POSITIVE）**: scalar × 0.61 → 更低的 Omega → LLM 更有影響力
- **衝突（DECELERATING + POSITIVE）**: scalar × 1.65 → 更高的 Omega → LLM 影響降低

---

## 目標資產

| 代號 | 公司 |
|------|------|
| AAPL | Apple |
| MSFT | Microsoft |
| GOOGL | Google |
| AMZN | Amazon |
| NVDA | NVIDIA |
| TSLA | Tesla |
| META | Meta |
| SPY | S&P 500 ETF（Benchmark） |

---

## 專案結構

```
bl_algo_trading/
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── configs/
│   └── config.yaml              # 策略參數（tau, risk_aversion, fusion 設定）
├── src/
│   ├── main.py                  # 主入口（單次回測）
│   ├── run_experiments.py       # 多組實驗執行器
│   ├── black_litterman.py       # BL 模型核心
│   ├── llm_view_generator.py    # Claude LLM 觀點生成 + Dynamic Omega
│   ├── news_fetcher.py          # 新聞數據庫 + VADER 情緒分析
│   ├── data_collection.py       # 價格 / 新聞數據整合
│   ├── backtest_engine.py       # 回測引擎（含 Markowitz / Equal Weight / SPY）
│   ├── performance_metrics.py   # 績效指標計算
│   └── utils.py                 # 工具函數（rebalance 日期、log 設定等）
├── data/
│   ├── prices.csv               # 歷史價格緩存（2015-01-02 → 2026-03-31）
│   └── news_database.json       # 新聞數據庫（2025-02-28 → 2026-01-31，12 個月）
├── results/
│   ├── no_news/                 # 實驗 1：純結構化
│   ├── fusion_default/          # 實驗 2：完整融合（τ=0.15, sens=0.5）
│   ├── weak_fusion/             # 實驗 3：保守融合（τ=0.15, sens=0.2）
│   ├── comparison_cumulative_returns.png
│   ├── comparison_bl_performance.png
│   └── comparison_summary.csv
└── logs/                        # 執行日誌
```

---

## 三組實驗說明

| 實驗名稱 | `use_news` | `fusion.enabled` | `agreement_sensitivity` |
|----------|-----------|-----------------|------------------------|
| `no_news` | False | False | — |
| `fusion_default` | True | True | 0.5（預設） |
| `weak_fusion` | True | True | 0.2（保守） |

各實驗結果獨立儲存於 `results/{experiment_name}/`，含完整圖表與 pickle 數據。

---

## 參數設定（config.yaml）

```yaml
backtest:
  start_date: "2025-03-01"
  end_date: "2026-01-31"
  rebalance_frequency: "monthly"   # BMS：每月第一個交易日

black_litterman:
  tau: 0.15
  risk_aversion: 1.8

llm:
  use_news: true
  fusion:
    enabled: true
    agreement_sensitivity: 0.5
```

---

## 技術棧

| 類別 | 工具 |
|------|------|
| LLM | Anthropic Claude（claude-sonnet-4-20250514） |
| 情緒分析 | VADER（本地，無需 API） |
| 新聞數據 | Finnhub API / 本地 JSON 數據庫 |
| 數學優化 | NumPy, SciPy（Markowitz），pypfopt |
| 數據 | yfinance, pandas |
| 視覺化 | matplotlib |

---

## 免責聲明

本專案僅供研究與學習用途，不構成投資建議。所有回測結果基於歷史數據，不代表未來績效。
