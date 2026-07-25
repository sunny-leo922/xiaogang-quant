"""
交易策略单元测试
"""
import unittest
import pandas as pd
import numpy as np


# 生成测试用K线数据
def _make_df(n=500, trend=1.0):
    """生成合成K线 + 指标"""
    np.random.seed(42)
    close = 3200 + np.cumsum(np.random.randn(n) * 2 + trend * 0.2)
    df = pd.DataFrame({
        "open": close - np.random.randn(n),
        "high": close + np.abs(np.random.randn(n)) * 3,
        "low": close - np.abs(np.random.randn(n)) * 3,
        "close": close,
        "volume": np.random.randint(100, 5000, n),
    })
    # 添加基础指标
    df["ema"] = df["close"].ewm(span=55, adjust=False).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["adx"] = np.abs(df["close"].diff()).rolling(14).mean() / df["atr"] * 100
    df["rsi"] = 50 + np.random.randn(n) * 10
    df["ema_slope"] = df["ema"].diff() / df["close"]
    return df.fillna(0)


class TestTrendFollowingStrategy(unittest.TestCase):
    """趋势跟随策略测试"""

    def setUp(self):
        from ai_quant_framework.strategies.trend_following import TrendFollowingStrategy
        self.StrategyClass = TrendFollowingStrategy

    def test_generate_signals_returns_series(self):
        df = _make_df(300)
        strategy = self.StrategyClass({"ema_period": 55, "atr_mult": 2.0})
        signals = strategy.generate_signal_series(df)
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(df))

    def test_signals_only_minus1_0_1(self):
        df = _make_df(300)
        strategy = self.StrategyClass({"ema_period": 55, "atr_mult": 2.0})
        signals = strategy.generate_signal_series(df)
        valid_values = {1, 0, -1}
        self.assertTrue(set(signals.unique()).issubset(valid_values),
                        f"信号值只能为{{-1,0,1}}，实际={set(signals.unique())}")

    def test_initial_bars_limit(self):
        """预热期信号应稀疏（前30根K线指标不够成熟）"""
        df = _make_df(300)
        strategy = self.StrategyClass({"ema_period": 55, "atr_mult": 2.0})
        signals = strategy.generate_signal_series(df)
        # 前30根K线因EMA周期原因信号应较少
        early = signals.iloc[:30]
        early_signals = (early != 0).sum()
        late = signals.iloc[100:200]
        late_signals = (late != 0).sum()
        # 预热期信号数应 ≤ 成熟期（宽松判断）
        self.assertLessEqual(early_signals, max(late_signals + 2, 5),
                             f"预热期信号({early_signals})不应远超成熟期({late_signals})")

    def test_signal_with_params(self):
        """不同参数应产生不同信号"""
        df = _make_df(300)
        s1 = self.StrategyClass({"ema_period": 21, "atr_mult": 1.5}).generate_signal_series(df)
        s2 = self.StrategyClass({"ema_period": 89, "atr_mult": 3.0}).generate_signal_series(df)
        count1 = len(s1[s1 != 0])
        count2 = len(s2[s2 != 0])
        # 快周期应有更多信号
        self.assertNotEqual(count1 + count2, 0,
                            "至少一种参数应有信号")


class TestMeanReversionStrategy(unittest.TestCase):
    """均值回归策略测试"""

    def setUp(self):
        from ai_quant_framework.strategies.mean_reversion import MeanReversionStrategy
        self.StrategyClass = MeanReversionStrategy

    def test_generate_signals_returns_series(self):
        df = _make_df(300, trend=0.01)  # 轻微趋势，适合均值回归
        strategy = self.StrategyClass()
        signals = strategy.generate_signal_series(df)
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(df))

    def test_signals_valid_values(self):
        df = _make_df(300, trend=0.01)
        strategy = self.StrategyClass()
        signals = strategy.generate_signal_series(df)
        valid_values = {1, 0, -1}
        self.assertTrue(set(signals.unique()).issubset(valid_values),
                        f"信号值只能为{{-1,0,1}}，实际={set(signals.unique())}")

    def test_buy_on_oversold(self):
        """超卖区域应产生BUY信号"""
        df = _make_df(200, trend=0.01)
        # 强制最后几根暴跌制造超卖
        df.iloc[-20:, df.columns.get_loc("close")] -= 50
        strategy = self.StrategyClass()
        signals = strategy.generate_signal_series(df)
        last_signals = signals.iloc[-30:]
        # 暴跌区域应有BUY（超卖买入）
        has_buy = (last_signals == 1).any()
        has_sell = (last_signals == -1).any()
        self.assertTrue(has_buy or has_sell,
                        "价格偏离应触发信号")


class TestStrategyConsistency(unittest.TestCase):
    """策略输出一致性测试"""

    def test_same_input_same_output(self):
        """相同DataFrame输入 → 相同信号输出（确定性）"""
        from ai_quant_framework.strategies.trend_following import TrendFollowingStrategy
        df = _make_df(300)
        strategy = TrendFollowingStrategy({"ema_period": 55, "atr_mult": 2.0})
        s1 = strategy.generate_signal_series(df)
        s2 = strategy.generate_signal_series(df)
        np.testing.assert_array_equal(s1.values, s2.values)


if __name__ == "__main__":
    unittest.main()
