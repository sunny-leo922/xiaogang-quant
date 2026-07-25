"""
信号融合引擎单元测试
"""
import unittest
from ai_quant_framework.core.signal_engine import SignalEngine, ModelVote, Signal


class TestModelVote(unittest.TestCase):
    """ModelVote 数据类测试"""

    def test_create_buy_vote(self):
        v = ModelVote("DeepSeek", "BUY", 75.0, 3200.5, 3180.0, 3260.0, "趋势看涨")
        self.assertEqual(v.model_name, "DeepSeek")
        self.assertEqual(v.direction, "BUY")
        self.assertEqual(v.confidence, 75.0)

    def test_create_hold_vote(self):
        v = ModelVote("GLM", "HOLD", 40.0, 0, 0, 0, "震荡观望")
        self.assertEqual(v.direction, "HOLD")


class TestSignalFusion(unittest.TestCase):
    """信号融合逻辑测试"""

    def setUp(self):
        config = {
            "ai": {
                "models": [
                    {"name": "DeepSeek", "weight": 1.0},
                    {"name": "GLM", "weight": 1.0},
                    {"name": "Qwen", "weight": 1.0},
                ],
                "ensemble": {
                    "min_models_agree": 2,
                    "min_confidence": 60,
                    "hold_on_conflict": False,
                    "conflict_threshold": 0.3,
                }
            },
            "mt5": {"symbol": "XAUUSD.n", "mode": "demo"}
        }
        self.engine = SignalEngine(config)

    def test_unanimous_buy(self):
        """全票买 → BUY"""
        votes = [
            ModelVote("DeepSeek", "BUY", 80, 3200, 3180, 3260, "看涨"),
            ModelVote("GLM", "BUY", 70, 3205, 3185, 3270, "看涨"),
            ModelVote("Qwen", "BUY", 65, 3198, 3178, 3255, "看涨"),
        ]
        signal = self.engine.fuse_votes(votes)
        self.assertEqual(signal.direction, "BUY")
        self.assertTrue(signal.is_trade_signal)
        self.assertGreater(signal.confidence, 60)

    def test_unanimous_sell(self):
        """全票卖 → SELL"""
        votes = [
            ModelVote("DeepSeek", "SELL", 80, 3200, 3220, 3140, "看跌"),
            ModelVote("GLM", "SELL", 70, 3195, 3215, 3135, "看跌"),
        ]
        signal = self.engine.fuse_votes(votes)
        self.assertEqual(signal.direction, "SELL")
        self.assertTrue(signal.is_trade_signal)

    def test_all_hold(self):
        """全票观望 → HOLD"""
        votes = [
            ModelVote("DeepSeek", "HOLD", 50, 0, 0, 0, "观望"),
            ModelVote("GLM", "HOLD", 40, 0, 0, 0, "观望"),
        ]
        signal = self.engine.fuse_votes(votes)
        self.assertEqual(signal.direction, "HOLD")
        self.assertFalse(signal.is_trade_signal)

    def test_split_h2_buy_win(self):
        """2:1 BUY胜出"""
        votes = [
            ModelVote("DeepSeek", "BUY", 75, 3200, 3180, 3250, "看涨"),
            ModelVote("GLM", "BUY", 65, 3195, 3175, 3245, "看涨"),
            ModelVote("Qwen", "SELL", 60, 3205, 3230, 3165, "高估"),
        ]
        signal = self.engine.fuse_votes(votes)
        self.assertEqual(signal.direction, "BUY")
        self.assertGreater(signal.model_agreement, 0.5)

    def test_empty_votes(self):
        """空投票列表 → HOLD"""
        signal = self.engine.fuse_votes([])
        self.assertEqual(signal.direction, "HOLD")
        self.assertFalse(signal.is_trade_signal)

    def test_weighted_fusion(self):
        """加权投票测试：高权重模型影响更大（需足够同意模型数）"""
        votes = [
            ModelVote("DeepSeek", "SELL", 50, 3200, 3220, 3150, "看跌"),
            ModelVote("GLM", "BUY", 90, 3195, 3170, 3255, "强烈看涨"),
            ModelVote("Qwen", "BUY", 45, 3200, 3185, 3245, "轻微看涨"),
        ]
        weights = {"DeepSeek": 0.5, "GLM": 2.0, "Qwen": 0.8}
        # GLM(weight=2.0, conf=90) + Qwen(weight=0.8, conf=45) = 180 + 36 = 216
        # DeepSeek(weight=0.5, conf=50) = 25
        # BUY wins and 2 models agree (>= min_models_agree=2)
        signal = self.engine.fuse_votes(votes, model_weights=weights)
        self.assertEqual(signal.direction, "BUY")

    def test_model_agreement_calc(self):
        """模型同意率计算"""
        votes = [
            ModelVote("A", "BUY", 70, 3200, 3180, 3260, ""),
            ModelVote("B", "BUY", 65, 3195, 3175, 3250, ""),
            ModelVote("C", "SELL", 50, 3205, 3230, 3165, ""),
            ModelVote("D", "HOLD", 30, 0, 0, 0, ""),
        ]
        signal = self.engine.fuse_votes(votes)
        # 4 models, 2 agree on BUY → 50%
        self.assertAlmostEqual(signal.model_agreement, 0.5, places=1)


class TestSignalProperties(unittest.TestCase):
    """Signal 对象属性测试"""

    def test_is_trade_signal_buy(self):
        s = Signal("BUY", 75, 3200, 3180, 3260, "看涨")
        self.assertTrue(s.is_trade_signal)

    def test_is_trade_signal_hold(self):
        s = Signal("HOLD", 30, 0, 0, 0, "观望")
        self.assertFalse(s.is_trade_signal)


if __name__ == "__main__":
    unittest.main()
