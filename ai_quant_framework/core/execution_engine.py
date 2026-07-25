"""
MT5 交易执行引擎
负责：开仓/平仓/止损/止盈订单管理
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


@dataclass
class OrderResult:
    """订单执行结果"""
    success: bool
    ticket: int = 0
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    error: str = ""


class ExecutionEngine:
    """MT5订单执行引擎"""

    def __init__(self, config: dict, data_engine):
        self.config = config
        self.data_engine = data_engine
        self.config = config
        self.symbol = config["mt5"]["symbol"]
        self.magic_number = 20240722  # 魔数，标识EA订单
        self._active = False
        # 从配置读取偏差容忍（可覆盖硬编码默认值）
        mt5_cfg = config.get("mt5", {})
        self.deviation_normal = mt5_cfg.get("deviation_points", 10)
        self.deviation_volatile = mt5_cfg.get("deviation_volatile", 30)

    def _get_deviation(self, is_volatile: bool = False) -> int:
        """返回当前行情下的偏差容忍点数"""
        return self.deviation_volatile if is_volatile else self.deviation_normal

    def initialize(self) -> bool:
        """初始化"""
        if not MT5_AVAILABLE:
            print("[Execution] MetaTrader5 不可用")
            return False
        self._active = True
        return True

    def market_order(self,
                     direction: str,
                     volume: float,
                     sl_price: float = 0.0,
                     tp_price: float = 0.0,
                     comment: str = "AIQuant",
                     is_volatile: bool = False) -> OrderResult:
        """
        市价单

        Args:
            direction: 'BUY' 或 'SELL'
            volume: 手数
            sl_price: 止损价
            tp_price: 止盈价
            comment: 订单备注
        """
        if not self._active:
            return OrderResult(False, error="执行引擎未激活")

        # 获取当前价格
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(False, error="无法获取报价")

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = sl_price if sl_price > 0 else 0
            tp = tp_price if tp_price > 0 else 0
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = sl_price if sl_price > 0 else 0
            tp = tp_price if tp_price > 0 else 0

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self._get_deviation(is_volatile),
            "magic": self.magic_number,            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                False, ticket=result.order,
                error=f"{result.retcode}: {result.comment}"
            )

        return OrderResult(
            True, ticket=result.order, volume=result.volume,
            price=result.price, sl=sl_price, tp=tp_price,
        )

    def close_position(self, ticket: int, volume: float = None) -> OrderResult:
        """平仓指定订单"""
        if not self._active:
            return OrderResult(False, error="未激活")

        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return OrderResult(False, error=f"找不到持仓 {ticket}")

        p = pos[0]
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(False, error="无法获取当前价格")
        
        price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume or p.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self._get_deviation(is_volatile),
            "magic": self.magic_number,            "comment": "AIQuant Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, error=f"{result.retcode}: {result.comment}")

        return OrderResult(True, ticket=ticket, volume=result.volume, price=result.price)

    def modify_sl_tp(self, ticket: int,
                     new_sl: float = None,
                     new_tp: float = None) -> OrderResult:
        """修改已有持仓的止损止盈"""
        if not self._active:
            return OrderResult(False, error="未激活")

        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return OrderResult(False, error=f"找不到持仓 {ticket}")

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.symbol,
        }
        if new_sl is not None:
            request["sl"] = new_sl
        if new_tp is not None:
            request["tp"] = new_tp

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, error=f"{result.retcode}: {result.comment}")

        return OrderResult(True, ticket=ticket)

    def get_open_positions(self) -> list[dict]:
        """获取所有AIQuant的持仓"""
        if not self._active:
            return []

        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []

        current_price = mt5.symbol_info_tick(self.symbol)
        if current_price is None:
            return []
        bid = current_price.bid
        ask = current_price.ask

        multiplier = self._get_contract_multiplier(self.symbol)

        result = []
        for p in positions:
            if p.magic != self.magic_number:
                continue

            if p.type == mt5.ORDER_TYPE_BUY:
                pnl = (bid - p.price_open) * p.volume * multiplier
                current_p = bid
            else:
                pnl = (p.price_open - ask) * p.volume * multiplier
                current_p = ask

            pnl_pct = pnl / (p.price_open * p.volume * multiplier) * 100 if p.price_open > 0 else 0

            result.append({
                "ticket": p.ticket,
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "open_price": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "current_price": current_p,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "open_time": p.time,
            })

        return result

    def _get_contract_multiplier(self, symbol: str) -> float:
        """获取合约乘数，缺失时抛异常（防止非黄金品种1000倍误差）"""
        specs = self.config.get("mt5", {}).get("symbol_specs", {})
        clean_symbol = symbol.replace(".n", "")
        spec = specs.get(clean_symbol)
        if spec is None or "contract_multiplier" not in spec:
            raise ValueError(
                f"品种 {symbol} 未在 symbol_specs 中配置 contract_multiplier。"
            )
        return spec["contract_multiplier"]

    def get_account_info(self) -> dict:
        """获取账户信息"""
        if not self._active:
            return {"balance": 0, "equity": 0, "margin_free": 0, "margin_level": 0}

        info = mt5.account_info()
        if info is None:
            return {"balance": 0, "equity": 0, "margin_free": 0, "margin_level": 0}

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
        }

    def calculate_volume(self,
                         risk_amount: float,
                         stop_points: float,
                         point_size: float = 0.01) -> float:
        """
        根据风险金额计算手数

        Args:
            risk_amount: 愿意承担的风险金额
            stop_points: 止损点数
            point_size: 每点价值
        """
        if stop_points <= 0:
            return 0.01
        multiplier = self._get_contract_multiplier(self.symbol)
        volume = risk_amount / (stop_points * point_size * multiplier)
        volume = max(0.01, round(volume, 2))
        return volume
