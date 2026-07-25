#!/usr/bin/env python3
"""
AIQuant Pro - AI量化交易研究框架
超越 xauusd.team 的多模型投票 + 自适应风控 + 专业回测系统

用法:
    python main.py backtest            # 运行离线回测
    python main.py live --mt5 -d 30    # MT5真实数据 + AI三阶段决策模拟
    python main.py live -d 30          # 模拟数据AI回测
    python main.py dashboard           # 启动监控面板
    python main.py optimize            # Walk-Forward参数优化
    python main.py status              # 查看系统状态
"""
import argparse
import sys
import os
from pathlib import Path

# 确保项目根目录在sys.path中
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime


def _get_multiplier(config: dict, symbol: str) -> float:
    """统一获取合约乘数（与ExecutionEngine._get_contract_multiplier一致）
    缺失配置时抛异常而非静默使用100（防止非黄金品种1000倍误差）"""
    specs = config.get("mt5", {}).get("symbol_specs", {})
    clean_symbol = symbol.replace(".n", "")
    spec = specs.get(clean_symbol)
    if spec is None or "contract_multiplier" not in spec:
        raise ValueError(
            f"品种 {symbol} 未在 symbol_specs 中配置 contract_multiplier。"
            f"请在 config/settings.yaml 的 mt5.symbol_specs.{clean_symbol} 中添加配置。"
        )
    return spec["contract_multiplier"]


def load_config():
    """加载配置文件（支持环境变量替换）"""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = os.path.expandvars(content)
    return yaml.safe_load(content)


def get_package_dir():
    """获取ai_quant_framework包目录"""
    return Path(__file__).parent


def cmd_backtest(args):
    """运行离线回测"""
    print("\n" + "=" * 60)
    print("  AIQuant Pro - 离线回测模式")
    print("=" * 60)

    config = load_config()

    # 生成模拟XAUUSD K线数据
    print("\n[1/4] 生成模拟行情数据...")
    bars = args.bars or 2000
    df = _generate_sample_data(bars)

    # 添加指标
    from ai_quant_framework.core.data_engine import DataEngine
    de = DataEngine(config)
    df = de.add_indicators(df)
    print(f"  生成 {len(df)} 根K线 (OHLCV+EMA+ATR+ADX)")

    # 运行趋势跟随回测
    print("\n[2/4] 运行趋势跟随策略...")
    from ai_quant_framework.strategies.trend_following import TrendFollowingStrategy
    from ai_quant_framework.backtest.engine import BacktestEngine

    tf_strategy = TrendFollowingStrategy({"ema_period": 55, "atr_mult": 2.0})
    signals = tf_strategy.generate_signal_series(df)

    engine = BacktestEngine(config)
    result_tf = engine.run(df, signals, df["atr"] if "atr" in df.columns else None)
    print(f"  信号数: {len(signals[signals != 0])}, 交易数: {result_tf.total_trades}")

    # 运行均值回归回测
    print("\n[3/4] 运行均值回归策略...")
    from ai_quant_framework.strategies.mean_reversion import MeanReversionStrategy

    mr_strategy = MeanReversionStrategy()
    mr_signals = mr_strategy.generate_signal_series(df)
    result_mr = engine.run(df, mr_signals)
    print(f"  信号数: {len(mr_signals[mr_signals != 0])}, 交易数: {result_mr.total_trades}")

    # Walk-Forward优化
    print("\n[4/4] Walk-Forward参数优化...")
    from ai_quant_framework.backtest.walk_forward import WalkForwardOptimizer

    wfo = WalkForwardOptimizer(config)
    param_grid = {
        "ema_period": [21, 55, 89],
        "atr_mult": [1.5, 2.0, 2.5],
    }
    wf_result = wfo.optimize(df, TrendFollowingStrategy, param_grid, n_windows=3)

    # 蒙特卡洛模拟
    from ai_quant_framework.backtest.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(config)
    mc_result = mc.simulate(result_tf.trades)

    # 绩效报告
    from ai_quant_framework.backtest.performance import PerformanceAnalyzer
    print("\n" + PerformanceAnalyzer.summary(result_tf))

    print("\n--- Walk-Forward 最优参数 ---")
    for k, v in wf_result.optimal_params.items():
        print(f"  {k}: {v}")
    print(f"  稳健性评分: {wf_result.robustness_score}/100")
    print(f"  参数稳定性: {wf_result.param_stability}/100")

    print("\n--- 蒙特卡洛模拟 (2000次) ---")
    print(f"  平均收益: {mc_result.mean_return_pct}%")
    print(f"  最差收益: {mc_result.worst_return_pct}%")
    print(f"  95%VaR: {mc_result.var_95_pct}%")
    print(f"  平均回撤: {mc_result.mean_drawdown_pct}%")
    print(f"  最大回撤: {mc_result.max_drawdown_pct}%")
    print(f"  收敛度: {mc_result.convergence:.0%}")

    print("\n--- 均值回归对比 ---")
    print(f"  收益: {result_mr.total_return_pct*100:.2f}%")
    print(f"  夏普: {result_mr.sharpe_ratio}")
    print(f"  胜率: {result_mr.win_rate}%")
    print(f"  交易数: {result_mr.total_trades}")

    print("\n" + "=" * 60)
    print("  回测完成!")
    print("=" * 60 + "\n")


def build_market_prompt_m15_h1(snapshot_m15: dict, snapshot_h1: dict = None) -> str:
    """将 M15/H1 技术数据构建为 AI 分析师期待的结构化 prompt"""
    from ai_quant_framework.config.prompts import AI1_ANALYST_M15_H1

    def _fmt(snap, label):
        if snap is None:
            return f"## {label}\n暂无数据\n"
        price = snap.get("price", snap.get("close", 0))
        atr = snap.get("atr", 0)
        ema_slope = snap.get("ema_slope", 0)
        ema = snap.get("ema", price)
        adx = snap.get("adx", 0)
        slope_desc = "向上" if ema_slope > 0.0002 else ("向下" if ema_slope < -0.0002 else "走平")
        price_vs = f"EMA上方(+${price - ema:.1f})" if price > ema else f"EMA下方(-${ema - price:.1f})"
        return (
            f"## XAUUSD {label}\n\n"
            f"现价: {price:.1f} | 开盘: {snap.get('open', price):.1f}"
            f" | 最高: {snap.get('high', price):.1f} | 最低: {snap.get('low', price):.1f}\n"
            f"EMA斜率: {slope_desc} | 价格位置: {price_vs}\n"
            f"ATR(14): {atr:.1f} | ADX: {adx:.1f}\n"
        )

    market = _fmt(snapshot_h1, "H1 (小时线)") + "\n" + _fmt(snapshot_m15, "M15")
    return AI1_ANALYST_M15_H1 + "\n\n" + market


def cmd_live(args):
    """实时 AI 决策回测 — 三阶段流水线：分析 → 审核 → 执行"""
    import asyncio
    import yaml
    from datetime import datetime
    from ai_quant_framework.ai.ensemble import AIEnsemble
    from ai_quant_framework.core.data_engine import DataEngine, TIMEFRAME_MINUTES

    # 加载配置（使用统一的load_config以支持环境变量替换）
    config = load_config()

    from ai_quant_framework.config.prompts import AI2_RISK_M15_H1

    engine = DataEngine(config)
    symbol = config["mt5"].get("symbol", "XAUUSD")

    # 验证周期参数
    tf_primary = f"M{args.interval}"
    supported_tfs = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
    if tf_primary not in supported_tfs:
        print(f"  不支持的周期: {tf_primary}，支持的周期: {sorted(supported_tfs)}")
        return

    # 1. 加载数据 — MT5真实数据 或 CSV/模拟数据
    df_h1 = None
    tf_secondary = "H1" if args.interval <= 15 else "H4"

    if args.mt5:
        # === MT5 真实数据模式 ===
        print(f"\n  正在连接MT5模拟账户 {config['mt5']['account']}...")
        if not engine.connect_mt5():
            print("  MT5连接失败! 请确保:")
            print("  1. MT5终端已启动并已登录模拟账户")
            print("  2. terminal_path 配置正确")
            return

        bars_primary = min(args.days * 24 * 60 // args.interval, config["data"]["max_bars"])
        bars_secondary = min(args.days * 24 * 60 // TIMEFRAME_MINUTES.get(tf_secondary, 60),
                             config["data"]["max_bars"])

        print(f"  正在从MT5拉取真实历史数据...")
        print(f"    品种: {symbol}")
        print(f"    主周期: {tf_primary} ({bars_primary}根)")
        print(f"    辅助周期: {tf_secondary} ({bars_secondary}根)")

        df = engine.fetch_ohlcv(symbol, tf_primary, bars=bars_primary)
        try:
            df_h1 = engine.fetch_ohlcv(symbol, tf_secondary, bars=bars_secondary)
        except Exception as e:
            print(f"  辅助周期 {tf_secondary} 拉取失败: {e}，仅使用单周期分析")

        df = engine.add_indicators(df)
        if df_h1 is not None and len(df_h1) > 0:
            df_h1 = engine.add_indicators(df_h1)

        engine.disconnect_mt5()
        print(f"  MT5数据获取完成，连接已安全断开")
        data_source = f"MT5真实数据 ({config['mt5']['server']})"
    else:
        # === CSV / 模拟数据模式 ===
        if args.csv:
            df = engine.load_csv(args.csv)
        else:
            sample_path = f"data/XAUUSD_M{args.interval}.csv"
            if os.path.exists(sample_path):
                df = engine.load_csv(sample_path)
            else:
                df = DataEngine.generate_sample_data(days=args.days, interval_min=args.interval)
        data_source = "CSV文件" if args.csv else "模拟生成"

    print(f"\n{'='*60}")
    print(f"   AI 三阶段决策回测")
    print(f"{'='*60}")
    print(f"   数据源: {data_source}")
    print(f"   数据: {len(df)} 根K线 ({args.interval}分钟)")
    print(f"   双周期: M{args.interval} + {tf_secondary}" if df_h1 is not None else "")
    print(f"   时间: {df['time'].iloc[0]} ~ {df['time'].iloc[-1]}")
    print(f"   价格: {df['close'].iloc[0]:.1f} ~ {df['close'].iloc[-1]:.1f}")

    # 2. 风控参数
    risk_cfg = config.get("risk", {})
    max_risk_pct = risk_cfg.get("max_risk_per_trade_pct", 2.0)
    max_drawdown = risk_cfg.get("max_drawdown_pct", 20.0)
    daily_loss_limit = risk_cfg.get("daily_loss_limit_pct", 5.0)

    # 3. 回测状态
    capital = 10000.0
    peak_capital = capital
    trades = []
    position = None
    min_bars = 50

    equity_curve = [capital]
    daily_pnl = 0.0
    current_day = None

    # 预构建H1时间索引(加速查找: O(N log M) 替代 O(N×M))
    h1_time_index = None
    h1_times_sorted = None
    if df_h1 is not None and len(df_h1) > 0:
        h1_times_sorted = df_h1["time"].values  # numpy array for searchsorted
        h1_time_index = df_h1.index.values       # 对应的行索引

    # 创建AI集成引擎（复用客户端）
    ensemble = AIEnsemble(config)

    async def _tick(bar_idx):
        nonlocal capital, peak_capital, position, trades, daily_pnl, current_day, equity_curve

        row = df.iloc[bar_idx]
        snapshot = DataEngine.bar_to_snapshot(row, symbol, f"M{args.interval}")
        bar_time = row["time"]

        # 查找对应的H1快照（使用二分查找，O(log M)替代O(M)全扫描）
        snapshot_h1 = None
        if h1_times_sorted is not None and len(h1_times_sorted) > 0:
            # 强制类型统一为 pd.Timestamp（防止 CSV 字符串与 Timestamp 比较崩溃）
            bar_time_ts = pd.Timestamp(bar_time)
            # searchsorted: 扔回第一个 > bar_time 的位置，减1得到最后一个 <= bar_time 的
            idx = np.searchsorted(h1_times_sorted, bar_time_ts, side="right") - 1
            if idx >= 0:
                h1_row = df_h1.iloc[idx]
                snapshot_h1 = DataEngine.bar_to_snapshot(h1_row, symbol, tf_secondary)

        # 日终结算
        if isinstance(bar_time, pd.Timestamp):
            bar_day = bar_time.date()
        else:
            bar_day = bar_time
        if current_day is None:
            current_day = bar_day
        
        # 日切换重置（必须在日内亏损检查之前）
        if bar_day != current_day:
            daily_pnl = 0.0
            current_day = bar_day
        
        # 日内亏损检查（每周期都检查，而非仅日切）
        if daily_pnl < -daily_loss_limit * 0.01 * capital:
            print(f"  [!!!] {current_day} 日亏损超限 ({daily_pnl:.1f})，熔断清仓")
            if position:
                multiplier = _get_multiplier(config, symbol)
                point_size = config["mt5"].get("point_size", 0.01)
                spread_points = config["mt5"].get("spread_points", 30)
                half_spread = spread_points * point_size / 2
                # BUY平仓用bid（扣半个点差），SELL平仓用ask（加半个点差）
                if position["side"] == "BUY":
                    exit_price = snapshot["close"] - half_spread
                else:
                    exit_price = snapshot["close"] + half_spread
                pnl = (exit_price - position["entry"]) * position["size"] * multiplier * (1 if position["side"] == "BUY" else -1)
                capital += pnl
                trades.append({**position, "exit_bar": bar_idx, "exit_price": exit_price,
                               "pnl": pnl, "reason": "日亏损熔断"})
                position = None
                equity_curve.append(capital)

        # --- 持仓检查 ---
        if position:
            price = snapshot["close"]
            entry = position["entry"]
            side = position["side"]
            exit_reason = None

            # 使用 high/low 判断 bar 内触发（与回测引擎一致）
            bar_low = snapshot.get("low", price)
            bar_high = snapshot.get("high", price)
            half_spread = config["backtest"].get("spread_points", 3) * 0.01 / 2

            if side == "BUY":
                if bar_low <= position["sl"]:
                    exit_reason = "止损"
                    price = position["sl"] - half_spread
                elif bar_high >= position["tp"]:
                    exit_reason = "止盈"
                    price = position["tp"] - half_spread
            else:  # SELL
                if bar_high >= position["sl"]:
                    exit_reason = "止损"
                    price = position["sl"] + half_spread
                elif bar_low <= position["tp"]:
                    exit_reason = "止盈"
                    price = position["tp"] + half_spread

            # 追踪未实现盈亏(用于回撤计算)
            multiplier = _get_multiplier(config, symbol)
            unrealized = (price - entry) * position["size"] * multiplier * (1 if side == "BUY" else -1)
            current_equity = capital + unrealized
            equity_curve.append(current_equity)
            peak_capital = max(peak_capital, current_equity)

            # 最大回撤熔断检查
            current_dd = (peak_capital - current_equity) / peak_capital
            if current_dd >= max_drawdown * 0.01:
                pnl = unrealized
                capital += pnl
                daily_pnl += pnl
                trades.append({**position, "exit_bar": bar_idx, "exit_price": price,
                               "pnl": pnl, "reason": f"最大回撤熔断({current_dd:.1%})", "exit_time": str(bar_time)})
                print(f"  [!!!] 最大回撤熔断 {current_dd:.1%}，强制平仓 PnL={pnl:+.1f} 权益={capital:.0f}")
                position = None
                equity_curve.append(capital)
                return

            if exit_reason:
                pnl = (price - entry) * position["size"] * multiplier * (1 if side == "BUY" else -1)
                capital += pnl
                daily_pnl += pnl
                peak_capital = max(peak_capital, capital)
                trades.append({**position, "exit_bar": bar_idx, "exit_price": price,
                               "pnl": pnl, "reason": exit_reason, "exit_time": str(bar_time)})
                print(f"  [{exit_reason}] {side} @{price:.1f} PnL={pnl:+.1f} 权益={capital:.0f}")
                position = None
                equity_curve.append(capital)  # 平仓后的权益
            # (未平仓时已在前面记录unrealized)

            return  # 持仓时不发新单

        equity_curve.append(capital)

        # --- AI 决策 (双周期prompt) ---
        prompt = build_market_prompt_m15_h1(snapshot, snapshot_h1)

        try:
            signal = await ensemble.analyze(prompt)
        except Exception as e:
            print(f"  [bar {bar_idx}] AI分析失败: {e}")
            return

        if not signal.is_trade_signal:
            print(f"  [{str(bar_time)[:19]}] {signal.direction} {signal.confidence:.0f}% | "
                  f"观望 (同意={signal.model_agreement:.0%}, {signal.reasoning[:40]})")
            return

        # 阶段2: 风控审核（fail-closed：故障时拒绝交易而非自动通过）
        try:
            review = await ensemble.risk_review(AI2_RISK_M15_H1, signal)
        except Exception as e:
            print(f"  [bar {bar_idx}] 风控审核失败: {e}, 拒绝交易(fail-closed)")
            review = {"approved": False, "approved_count": 0, "total_voters": 0, "reasons": [f"审核系统故障: {e}"]}

        if not review.get("approved", True):
            print(f"  [{str(bar_time)[:19]}] {signal.direction} {signal.confidence:.0f}% | "
                  f"风控REJECT ({review.get('approved_count',0)}/{review.get('total_voters',1)})")
            return

        # 应用AI-2风控审核的调整建议（如扩大止损、调整止盈）
        # 使用统一正则提取，避免重复import
        import re
        point_size_adj = 0.01
        point_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*个?点')

        for reason in review.get("reasons", []):
            if not isinstance(reason, str):
                continue
            # 止损调整：扩大/放宽/放大 → 远离入场价
            sl_expand_kw = ("放大", "扩大", "放宽", "加宽", "延伸", "拉大", "拓宽", "放远", "推远", "移动")
            sl_tighten_kw = ("收紧", "收窄", "缩小", "压缩", "减窄", "拉近", "推进", "靠近")
            tp_expand_kw = ("上调", "提高", "放宽", "扩大", "延伸", "拉大", "调高", "推高", "远移", "上移")
            tp_tighten_kw = ("下调", "降低", "收紧", "收窄", "压缩", "调低", "下移", "保守", "近移")

            if "止损" in reason and any(kw in reason for kw in sl_expand_kw):
                match = point_pattern.search(reason)
                if match:
                    adj_points = float(match.group(1))
                    if signal.direction == "BUY":
                        signal.stop_loss -= adj_points * point_size_adj
                    else:
                        signal.stop_loss += adj_points * point_size_adj
                    print(f"  [风控调整] 止损扩大 {adj_points} 点 → SL={signal.stop_loss:.2f}")
                else:
                    print(f"  [风控调整] ⚠ 识别到止损扩大请求，但未提取到点数: {reason[:80]}")
            # 止损调整：收紧/收窄 → 靠近入场价
            elif "止损" in reason and any(kw in reason for kw in sl_tighten_kw):
                match = point_pattern.search(reason)
                if match:
                    adj_points = float(match.group(1))
                    if signal.direction == "BUY":
                        signal.stop_loss += adj_points * point_size_adj
                    else:
                        signal.stop_loss -= adj_points * point_size_adj
                    print(f"  [风控调整] 止损收紧 {adj_points} 点 → SL={signal.stop_loss:.2f}")
                else:
                    print(f"  [风控调整] ⚠ 识别到止损收紧请求，但未提取到点数: {reason[:80]}")
            # 止盈调整：上调/提高/放宽 → 远离入场价（追求更多利润）
            if "止盈" in reason and any(kw in reason for kw in tp_expand_kw):
                match = point_pattern.search(reason)
                if match:
                    adj_points = float(match.group(1))
                    if signal.direction == "BUY":
                        signal.take_profit += adj_points * point_size_adj
                    else:
                        signal.take_profit -= adj_points * point_size_adj
                    print(f"  [风控调整] 止盈上调 {adj_points} 点 → TP={signal.take_profit:.2f}")
                else:
                    print(f"  [风控调整] ⚠ 识别到止盈上调请求，但未提取到点数: {reason[:80]}")
            # 止盈调整：下调/降低/收紧 → 靠近入场价
            elif "止盈" in reason and any(kw in reason for kw in tp_tighten_kw):
                match = point_pattern.search(reason)
                if match:
                    adj_points = float(match.group(1))
                    if signal.direction == "BUY":
                        signal.take_profit -= adj_points * point_size_adj
                    else:
                        signal.take_profit += adj_points * point_size_adj
                    print(f"  [风控调整] 止盈下调 {adj_points} 点 → TP={signal.take_profit:.2f}")
                else:
                    print(f"  [风控调整] ⚠ 识别到止盈下调请求，但未提取到点数: {reason[:80]}")

        # 阶段3: 执行
        # 模拟模式使用 next_open 入场（避免未来信息优势），与回测引擎一致
        if bar_idx + 1 < len(df):
            price = df.iloc[bar_idx + 1]["open"]
        else:
            price = snapshot["close"]
        
        # 交易成本模型
        spread_points = config["backtest"].get("spread_points", 3)
        slippage_points = config["backtest"].get("slippage_points", 2)
        commission_pct = config["backtest"].get("commission_pct", 0.05)
        
        point_size = 0.01
        half_spread = spread_points * point_size / 2
        
        # 获取合约乘数（必须在仓位计算之前）
        multiplier = _get_multiplier(config, symbol)
        
        # 入场价：优先使用AI建议入场价（如果有效且与市价差距合理）
        atr_value = snapshot.get("atr", 0) or 0.5
        use_signal_entry = False
        if signal.entry_price > 0:
            price_diff = abs(signal.entry_price - price)
            # 如果AI建议入场价与市价差距在1/4 ATR以内，使用信号入场价
            if price_diff <= atr_value * 0.25:
                entry_price = signal.entry_price
                use_signal_entry = True
        
        # 默认使用市场价入场（含点差）
        if not use_signal_entry:
            entry_price = price + (half_spread if signal.direction == "BUY" else -half_spread)
        
        # 应用滑点
        slippage = np.random.uniform(0, slippage_points) * point_size
        if signal.direction == "BUY":
            entry_price += slippage
        else:
            entry_price -= slippage
        
        risk_amount = capital * max_risk_pct * 0.01
        sl_dist = abs(entry_price - signal.stop_loss)
        # 安全约束: 最小止损距离10点 (避免仓位过大)
        MIN_SL_DIST = 10.0 * point_size
        sl_dist = max(sl_dist, MIN_SL_DIST)
        
        # 正确的仓位计算公式：size = risk_amount / (sl_dist * multiplier)
        # 因为 PnL = (price - entry) * size * multiplier
        # 所以 risk_amount = sl_dist * size * multiplier → size = risk_amount / (sl_dist * multiplier)
        size = risk_amount / (sl_dist * multiplier) if sl_dist > 0 else 0.01
        
        # 最大仓位: 不超过账户15%名义价值（包含乘数因子）
        MAX_NOTIONAL = capital * 0.15
        max_size = MAX_NOTIONAL / (entry_price * multiplier)
        size = max(0.01, min(round(size, 2), round(max_size, 2)))
        
        # 手续费 (必须在size计算之后)
        commission = entry_price * size * multiplier * commission_pct * 0.01
        capital -= commission

        position = {
            "side": signal.direction,
            "entry": entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "size": size,
            "bar": bar_idx,
            "entry_time": str(bar_time),
            "confidence": signal.confidence,
            "model_agreement": signal.model_agreement,
            "review_approved": review.get("approved_count", 0),
            "review_total": review.get("total_voters", 0),
        }

        rr = abs(signal.take_profit - price) / max(sl_dist, 0.01)
        print(f"  [{str(bar_time)[:19]}] {signal.direction} {signal.confidence:.0f}% | "
              f"entry={price:.1f} sl={signal.stop_loss:.1f} tp={signal.take_profit:.1f} "
              f"rr={rr:.1f} | 风控={review['approved_count']}/{review['total_voters']} | "
              f"仓位={size:.2f}")

    # 4. 运行回测
    decision_indices = range(min_bars, len(df))
    total = len(decision_indices)

    print(f"\n{'='*60}")
    print(f"   开始回测 {total} 个决策点 ({min_bars} 预热)...")
    if args.delay > 0:
        print(f"   决策间隔: {args.delay}s (限流保护)")
    print(f"{'='*60}\n")

    t_start = datetime.now()

    async def _run():
        try:
            for i, bar_idx in enumerate(decision_indices):
                if i % max(1, total // 20) == 0 and i > 0:
                    elapsed = (datetime.now() - t_start).total_seconds()
                    eta = elapsed / i * (total - i) if i > 0 else 0
                    print(f"  ...进度 {i}/{total} ({100*i//total}%) 耗时{elapsed:.0f}s ETA{eta:.0f}s")
                if args.delay > 0 and i > 0:
                    await asyncio.sleep(args.delay)
                await _tick(bar_idx)
        finally:
            try:
                await ensemble.close()
                print("  [AI] 客户端连接已关闭")
            except Exception as e:
                print(f"  [AI] 关闭客户端失败: {e}")

    asyncio.run(_run())

    # 强制平仓（包含半个点差成本，与正常出场一致）
    if position:
        last_price = df["close"].iloc[-1]
        multiplier = _get_multiplier(config, symbol)
        point_size = config["mt5"].get("point_size", 0.01)
        spread_points = config["mt5"].get("spread_points", 30)
        half_spread = spread_points * point_size / 2
        # BUY平仓用bid（扣半个点差），SELL平仓用ask（加半个点差）
        if position["side"] == "BUY":
            exit_price = last_price - half_spread
        else:
            exit_price = last_price + half_spread
        pnl = (exit_price - position["entry"]) * position["size"] * multiplier * (1 if position["side"] == "BUY" else -1)
        capital += pnl
        trades.append({**position, "exit_bar": len(df)-1, "exit_price": last_price,
                       "pnl": pnl, "reason": "回测结束平仓"})
        position = None

    # 5. 输出报告
    elapsed = (datetime.now() - t_start).total_seconds()
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    win_pnls = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
    loss_pnls = [t["pnl"] for t in trades if t.get("pnl", 0) <= 0]

    print(f"\n{'='*60}")
    print(f"   回测报告")
    print(f"{'='*60}")
    print(f"   数据源: {data_source}")
    print(f"   总耗时: {elapsed:.0f}s ({total} 决策点)")
    print(f"   交易次数: {len(trades)}")
    print(f"   胜率: {100*wins/len(trades):.1f}%" if trades else "   胜率: N/A")
    print(f"   初始资本: ${10000:.0f}")
    print(f"   最终权益: ${capital:.0f}")
    print(f"   总收益: ${capital-10000:+.0f} ({(capital/10000-1)*100:+.1f}%)")
    print(f"   最大回撤: {100*(1-min(equity_curve)/peak_capital):.1f}%" if peak_capital > 0 else "")
    if win_pnls:
        print(f"   盈利单: {len(win_pnls)} 笔, 平均+{sum(win_pnls)/len(win_pnls):.1f}, 最大+{max(win_pnls):.1f}")
    if loss_pnls:
        print(f"   亏损单: {len(loss_pnls)} 笔, 平均{sum(loss_pnls)/len(loss_pnls):.1f}, 最大{min(loss_pnls):.1f}")

    # 按退出原因统计
    reasons = {}
    for t in trades:
        r = t.get("reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    if reasons:
        print(f"   平仓原因: {reasons}")
    print(f"{'='*60}")


def cmd_dashboard(args):
    """启动Web监控面板"""
    print("启动 AIQuant Pro Dashboard...")
    config = load_config()

    from ai_quant_framework.dashboard.app import run_dashboard

    # 尝试连接MT5
    execution_engine = None
    data_engine = None
    try:
        from ai_quant_framework.core.data_engine import DataEngine
        from ai_quant_framework.core.execution_engine import ExecutionEngine

        data_engine = DataEngine(config)
        if data_engine.connect_mt5():
            execution_engine = ExecutionEngine(config, data_engine)
            execution_engine.initialize()
            print("MT5已连接")
        else:
            print("MT5未连接，以监控模式运行")
    except Exception as e:
        print(f"MT5初始化跳过: {e}")

    run_dashboard(config, execution_engine, data_engine)


def cmd_status(args):
    """查看系统状态"""
    print("\n=== AIQuant Pro 系统状态 ===\n")

    # 检查配置文件
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    if config_path.exists():
        config = load_config()
        print(f"[OK] 配置文件: {config_path}")
        print(f"  品种: {config['mt5']['symbol']}")
        print(f"  AI模型数: {len(config['ai']['models'])}")
        print(f"  风控方法: {config['risk']['position']['method']}")
        print(f"  回测初始资金: ${config['backtest']['initial_capital']}")
    else:
        print("[FAIL] 配置文件缺失")
        return

    # 检查依赖
    deps = ["numpy", "pandas", "yaml", "openai", "fastapi", "uvicorn"]
    for dep in deps:
        try:
            __import__(dep)
            print(f"[OK] 依赖: {dep}")
        except ImportError:
            print(f"[MISS] 依赖: {dep} (pip install {dep})")

    # 检查MT5
    try:
        import MetaTrader5 as mt5
        print("[OK] MetaTrader5 Python API")
    except ImportError:
        print("[INFO] MetaTrader5 未安装 (仅影响实时交易)")

    # 检查API Keys
    for m in config["ai"]["models"]:
        key = os.getenv(m["api_key_env"])
        status = "已设置" if key else "未设置"
        print(f"[{'OK' if key else 'WARN'}] {m['name']}: API Key {status} ({m['api_key_env']})")

    print(f"\n  项目根目录: {Path(__file__).parent}")
    print("=" * 50 + "\n")


def _generate_sample_data(bars: int = 2000) -> pd.DataFrame:
    """生成模拟XAUUSD行情数据"""
    np.random.seed(42)

    dates = pd.date_range(start="2025-01-01", periods=bars, freq="5min")
    close = 2650.0
    prices = []

    volatility = 0.0003
    trend = 0.00002

    for i in range(bars):
        ret = np.random.normal(trend, volatility)
        close *= (1 + ret)
        close = max(2500, min(2800, close))

        spread = close * np.random.uniform(0.0002, 0.001)
        high = close + abs(np.random.normal(0, close * 0.0005))
        low = close - abs(np.random.normal(0, close * 0.0005))
        open_price = close + np.random.normal(0, close * 0.0002)

        prices.append({
            "time": dates[i],
            "open": round(open_price, 2),
            "high": round(max(open_price, high, close), 2),
            "low": round(min(open_price, low, close), 2),
            "close": round(close, 2),
            "volume": int(np.random.randint(100, 5000)),
            "spread": int(spread * 100),
        })

    return pd.DataFrame(prices).set_index("time")


def cmd_optimize(args):
    """Walk-Forward参数优化"""
    print("=" * 60)
    print("  AIQuant Pro - 参数优化模式")
    print("=" * 60)

    config = load_config()
    df = _generate_sample_data(args.bars or 3000)
    config["backtest"]["walk_forward"]["train_pct"] = 0.7
    config["backtest"]["walk_forward"]["test_pct"] = 0.3

    from ai_quant_framework.core.data_engine import DataEngine
    de = DataEngine(config)
    df = de.add_indicators(df)

    from ai_quant_framework.strategies.trend_following import TrendFollowingStrategy
    from ai_quant_framework.backtest.walk_forward import WalkForwardOptimizer

    param_grid = {
        "ema_period": [13, 21, 34, 55, 89, 144],
        "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
        "min_rr": [1.0, 1.5, 2.0],
    }

    wfo = WalkForwardOptimizer(config)
    result = wfo.optimize(df, TrendFollowingStrategy, param_grid, n_windows=5)

    print("\n--- 各窗口结果 ---")
    for w in result.windows:
        print(f"  窗口{w['window']}: 训练Sharpe={w['train_sharpe']:.2f} "
              f"测试Sharpe={w['test_sharpe']:.2f} "
              f"测试收益={w['test_return_pct']*100:.1f}%")

    print(f"\n--- 最优参数 ---")
    for k, v in result.optimal_params.items():
        print(f"  {k}: {v}")
    print(f"\n稳健性评分: {result.robustness_score}/100")
    print(f"参数稳定性: {result.param_stability}/100")
    print("=" * 60)


# --- CLI入口 ---
def main():
    parser = argparse.ArgumentParser(
        description="AIQuant Pro - AI量化交易研究框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py backtest              运行回测
  python main.py backtest -b 5000      生成5000根K线回测
  python main.py live --mt5 -d 30      MT5真实数据AI模拟(30天M15+H1)
  python main.py live --mt5 -d 7 -i 5  MT5真实数据短线回测(7天M5+H1)
  python main.py live -d 30            模拟数据回测
  python main.py live --csv data/xxx.csv  使用CSV文件回测
  python main.py optimize              参数优化
  python main.py dashboard             启动Web面板
  python main.py status                查看系统状态
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    p_bt = subparsers.add_parser("backtest", help="运行离线回测")
    p_bt.add_argument("-b", "--bars", type=int, default=2000, help="K线数量")

    p_live = subparsers.add_parser("live", help="启动实时AI决策回测")
    p_live.add_argument("-d", "--days", type=int, default=30, help="数据天数 (MT5模式拉取历史数据, CSV模式生成模拟数据)")
    p_live.add_argument("-i", "--interval", type=int, default=15, help="K线间隔(分钟, 默认15=M15)")
    p_live.add_argument("--csv", type=str, default=None, help="使用指定CSV文件")
    p_live.add_argument("--mt5", action="store_true", default=False,
                        help="从MT5真实账户拉取历史数据 (双周期M15+H1分析)")
    p_live.add_argument("--delay", type=float, default=0.0,
                        help="每决策点间隔延迟(秒, 默认0, 建议1.0以上避免限流)")

    p_dash = subparsers.add_parser("dashboard", help="启动Web监控面板")

    p_opt = subparsers.add_parser("optimize", help="Walk-Forward参数优化")
    p_opt.add_argument("-b", "--bars", type=int, default=3000, help="K线数量")

    p_stat = subparsers.add_parser("status", help="查看系统状态")

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "live":
        cmd_live(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
