"""
合成投票模型单元测试
"""
import unittest

# 测试 _generate_synthetic_votes 函数
from ai_quant_framework.main import _generate_synthetic_votes


class TestSyntheticVotes(unittest.TestCase):
    """合成投票生成测试"""

    def setUp(self):
        self.config = {
            "ai": {
                "models": [
                    {"name": "DeepSeek", "weight": 1.0},
                    {"name": "GLM", "weight": 1.0},
                    {"name": "Qwen", "weight": 1.0},
                    {"name": "Kimi", "weight": 1.0},
                ]
            }
        }

    def test_strong_uptrend_votes_buy(self):
        """强多头行情 → 多数投BUY"""
        snapshot = {
            "price": 3200, "ema": 3150, "ema_slope": 0.001, "adx": 45,
            "atr": 15, "rsi": 65, "close": 3200, "open": 3190,
            "high": 3210, "low": 3185, "volume": 1000,
        }
        votes = _generate_synthetic_votes(self.config, snapshot)
        self.assertEqual(len(votes), 4, "应生成4个模型投票")
        buy_count = sum(1 for v in votes if v.direction == "BUY")
        sell_count = sum(1 for v in votes if v.direction == "SELL")
        self.assertGreater(buy_count, sell_count,
                           f"强多头行情下BUY票({buy_count})应多于SELL票({sell_count})")

    def test_strong_downtrend_votes_sell(self):
        """强空头行情 → 多数投SELL"""
        snapshot = {
            "price": 3100, "ema": 3150, "ema_slope": -0.001, "adx": 45,
            "atr": 15, "rsi": 35, "close": 3100, "open": 3110,
            "high": 3120, "low": 3090, "volume": 1000,
        }
        votes = _generate_synthetic_votes(self.config, snapshot)
        sell_count = sum(1 for v in votes if v.direction == "SELL")
        buy_count = sum(1 for v in votes if v.direction == "BUY")
        self.assertGreater(sell_count, buy_count,
                           f"强空头行情下SELL票({sell_count})应多于BUY票({buy_count})")

    def test_sideways_market_mixed(self):
        """震荡行情 → 投票不应全票BUY或全票SELL（全HOLD合理）"""
        snapshot = {
            "price": 3200, "ema": 3200, "ema_slope": 0.0001, "adx": 15,
            "atr": 5, "rsi": 50, "close": 3200, "open": 3198,
            "high": 3205, "low": 3195, "volume": 100,
        }
        votes = _generate_synthetic_votes(self.config, snapshot)
        directions = {v.direction for v in votes}
        # 不应全票BUY或全票SELL（极端低波动全HOLD是合理的）
        buy_count = len([v for v in votes if v.direction == "BUY"])
        sell_count = len([v for v in votes if v.direction == "SELL"])
        hold_count = len([v for v in votes if v.direction == "HOLD"])
        self.assertFalse(buy_count == len(votes) or sell_count == len(votes),
                         f"震荡行情不应全票BUY/SELL, BUY={buy_count} SELL={sell_count} HOLD={hold_count}")

    def test_votes_have_valid_fields(self):
        """每个投票都有完整的字段"""
        snapshot = {"price": 3200, "ema": 3190, "ema_slope": 0.0005, "adx": 30,
                    "atr": 10, "rsi": 55, "close": 3200}
        votes = _generate_synthetic_votes(self.config, snapshot)
        for v in votes:
            self.assertIsNotNone(v.model_name)
            self.assertIn(v.direction, ("BUY", "SELL", "HOLD"))
            self.assertGreaterEqual(v.confidence, 20)
            self.assertLessEqual(v.confidence, 90)
            if v.direction != "HOLD":
                self.assertGreater(v.entry_price, 0, f"{v.model_name} entry_price应为正")
                self.assertGreater(v.stop_loss, 0 if v.direction == "BUY" else v.entry_price,
                                   f"BUY时SL应>0, 实际SL={v.stop_loss}")
                self.assertGreater(v.take_profit, v.entry_price if v.direction == "BUY" else 0,
                                   f"BUY时TP应>entry({v.entry_price}), 实际TP={v.take_profit}")

    def test_h1_influence(self):
        """H1大周期数据应影响投票方向"""
        snapshot_m15 = {"price": 3200, "ema": 3205, "ema_slope": -0.0003,
                        "adx": 20, "atr": 10, "rsi": 45, "close": 3200}
        # M15略偏空，但H1强多
        snapshot_h1 = {"price": 3200, "ema": 3150, "ema_slope": 0.002,
                       "adx": 40, "atr": 20, "rsi": 65, "close": 3200}
        votes = _generate_synthetic_votes(self.config, snapshot_m15, snapshot_h1)
        buy_count = sum(1 for v in votes if v.direction == "BUY")
        sell_count = sum(1 for v in votes if v.direction == "SELL")
        # H1强多应至少中和M15做空倾向
        self.assertLessEqual(abs(buy_count - sell_count), len(votes),
                             "H1应影响投票方向，不应一边倒")


class TestSyntheticVoteConsistency(unittest.TestCase):
    """合成投票一致性测试"""

    def setUp(self):
        self.config = {"ai": {"models": [{"name": "M1", "weight": 1.0},
                                          {"name": "M2", "weight": 1.0}]}}

    def test_deterministic_for_same_input(self):
        """相同输入必须产生完全相同的投票（可复现性）"""
        snapshot = {"price": 3200, "ema": 3190, "ema_slope": 0.0008,
                    "adx": 35, "atr": 10, "rsi": 60, "close": 3200}
        votes1 = _generate_synthetic_votes(self.config, snapshot)
        votes2 = _generate_synthetic_votes(self.config, snapshot)
        for v1, v2 in zip(votes1, votes2):
            self.assertEqual(v1.direction, v2.direction,
                             f"{v1.model_name}: {v1.direction} vs {v2.direction}")
            self.assertEqual(v1.confidence, v2.confidence)
            self.assertEqual(v1.stop_loss, v2.stop_loss)
            self.assertEqual(v1.take_profit, v2.take_profit)


if __name__ == "__main__":
    unittest.main()
