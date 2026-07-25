"""
执行引擎单元测试
"""
import unittest
from unittest.mock import MagicMock, patch
from ai_quant_framework.core.execution_engine import ExecutionEngine


class TestExecutionEngineConfig(unittest.TestCase):
    """执行引擎配置测试"""

    def setUp(self):
        self.config = {
            "mt5": {
                "symbol": "XAUUSD.n",
                "mode": "demo",
                "deviation_points": 10,
                "deviation_volatile": 30,
                "point_size": 0.01,
            },
            "risk": {
                "max_positions": 3,
                "max_daily_loss_pct": 0.02,
            },
        }

    def test_deviation_normal_reads_config(self):
        """应从配置读取普通行情偏差"""
        engine = ExecutionEngine(self.config, data_engine=MagicMock())
        self.assertEqual(engine.deviation_normal, 10)

    def test_deviation_volatile_reads_config(self):
        """应从配置读取高波动偏差"""
        engine = ExecutionEngine(self.config, data_engine=MagicMock())
        self.assertEqual(engine.deviation_volatile, 30)

    def test_deviation_default_when_missing(self):
        """配置缺失时使用默认值"""
        config_no_dev = {"mt5": {"symbol": "XAUUSD.n", "mode": "demo"}}
        engine = ExecutionEngine(config_no_dev, data_engine=MagicMock())
        self.assertEqual(engine.deviation_normal, 10)  # 默认10
        self.assertEqual(engine.deviation_volatile, 30)  # 默认30

    def test_get_deviation_normal(self):
        engine = ExecutionEngine(self.config, data_engine=MagicMock())
        self.assertEqual(engine._get_deviation(is_volatile=False), 10)

    def test_get_deviation_volatile(self):
        engine = ExecutionEngine(self.config, data_engine=MagicMock())
        self.assertEqual(engine._get_deviation(is_volatile=True), 30)

    def test_config_stored(self):
        engine = ExecutionEngine(self.config, data_engine=MagicMock())
        self.assertEqual(engine.config["mt5"]["symbol"], "XAUUSD.n")


if __name__ == "__main__":
    unittest.main()
