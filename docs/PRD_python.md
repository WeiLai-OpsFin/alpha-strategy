# FIN3083 策略设计文档 - Python 版

## 目标

用 Python 完成一个宏观状态切换因子策略。

核心想法很简单：

- 市场流动性宽松时，偏向成长、估值、质量因子。
- 市场流动性紧张时，偏向质量、低波动、股息因子。
- 每月调仓一次。
- 回测时计入交易成本。

## 数据

项目使用这些 CSV 文件：

| 文件 | 用途 |
|---|---|
| `macro_shibor.csv` | 判断宏观流动性状态 |
| `index_weights.csv` | 构造沪深 300 抽样股票池 |
| `stock_prices.csv` | 计算收益率、波动率、Beta |
| `daily_basic.csv` | 计算价值、股息因子 |
| `fina_indicator.csv` | 计算成长、质量因子 |
| `index_prices.csv` | 作为基准指数 |

## 输出

主要输出文件：

- `performance_summary.csv`
- `daily_nav.csv`
- `selected_stocks.csv`
- `factor_weights.csv`
- `regime_sharpe.csv`
- `ic_stability.csv`

## 分工

| 姓名 | 学号 | 分工 | 贡献比例 |
|---|---|---|---|
| Qin Han | 2230018019 | Project Lead, Backtesting Framework | 25% |
| Lai Wei | 2230001108 | Data Acquisition, Macro Indicators | 25% |
| Haolin Zuo | 2230018100 | Factor Engineering, Signal Generation | 25% |
| Jinyang He | 2230006053 | Performance Analysis, Report Writing | 25% |
