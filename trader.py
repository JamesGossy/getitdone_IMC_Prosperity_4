import json
import math
from typing import Dict, List
from datamodel import (
    Order,
    OrderDepth,
    Symbol,
    TradingState,
)
from bt_logger import logger

# ── Position limits ──────────────────────────────────────────────────────────
LIMITS: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 50,
    "INTARIAN_PEPPER_ROOT": 50,
}

# ── ASH_COATED_OSMIUM: hard fair value (like Rainforest Resin) ───────────────
ACO_FAIR = 10_000

# ── INTARIAN_PEPPER_ROOT: linear trend parameters ────────────────────────────
# Price = day_start_price + 0.001 * timestamp
IPR_SLOPE = 0.001


class Trader:
    def __init__(self):
        self._ipr_day_start: float | None = None
        self._ipr_last_ts: int = -1

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Restore persisted state
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                self._ipr_day_start = saved.get("ipr_day_start", None)
                self._ipr_last_ts = saved.get("ipr_last_ts", -1)
            except:
                pass

        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            limit = LIMITS.get(product, 50)

            if product == "ASH_COATED_OSMIUM":
                orders = self._trade_aco(product, order_depth, position, limit)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders = self._trade_ipr(product, order_depth, position, limit, state.timestamp)
            else:
                orders = []

            if orders:
                result[product] = orders

        trader_data_str = json.dumps({
            "ipr_day_start": self._ipr_day_start,
            "ipr_last_ts": self._ipr_last_ts,
        })
        logger.flush(state, result, 0, trader_data_str)

        return result, 0, trader_data_str

    # ── ASH COATED OSMIUM ────────────────────────────────────────────────────
    def _trade_aco(self, product: str, od: OrderDepth, position: int, limit: int) -> List[Order]:
        """
        Market making around the fixed fair value of 10,000.
        - Take any ask below fair / bid above fair (mispriced orders)
        - Passive quotes: improve best bid by +1, undercut best ask by -1
        - Inventory skew to stay balanced
        """
        orders: List[Order] = []
        buy_cap = limit - position
        sell_cap = limit + position

        # ── 1. Take mispriced orders ─────────────────────────────────────────
        for ask in sorted(od.sell_orders.keys()):
            if ask >= ACO_FAIR:
                break
            take = min(-od.sell_orders[ask], buy_cap)
            if take > 0:
                orders.append(Order(product, ask, take))
                buy_cap -= take

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid <= ACO_FAIR:
                break
            give = min(od.buy_orders[bid], sell_cap)
            if give > 0:
                orders.append(Order(product, bid, -give))
                sell_cap -= give

        # ── 2. Passive quotes (overbid / undercut), skewed by inventory ──────
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else ACO_FAIR - 8
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else ACO_FAIR + 8

        # Inventory skew: if long, lower bid; if short, raise ask
        inv_skew = int(position / limit * 3)
        make_bid = min(best_bid + 1, ACO_FAIR - 1) - max(0, inv_skew)
        make_ask = max(best_ask - 1, ACO_FAIR + 1) - min(0, inv_skew)

        logger.print(f"ACO fair={ACO_FAIR} pos={position} make_bid={make_bid} make_ask={make_ask}")

        if buy_cap > 0 and make_bid < ACO_FAIR:
            orders.append(Order(product, make_bid, buy_cap))
        if sell_cap > 0 and make_ask > ACO_FAIR:
            orders.append(Order(product, make_ask, -sell_cap))

        return orders

    # ── INTARIAN PEPPER ROOT ─────────────────────────────────────────────────
    def _trade_ipr(
        self, product: str, od: OrderDepth, position: int, limit: int, timestamp: int
    ) -> List[Order]:
        """
        Two-pronged strategy:
        1. PRIMARY: Go max long ASAP and hold — trend is +0.001/tick = +1000/day
        2. SECONDARY: Market make around the known fair value (trend line)

        Fair value = day_start_price + 0.001 * timestamp
        day_start_price is calibrated from wall mid on the first tick of each day.
        """
        if not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        # Calibrate day_start on first tick, or when timestamp resets (new day)
        if self._ipr_day_start is None or timestamp < self._ipr_last_ts:
            self._ipr_day_start = mid - IPR_SLOPE * timestamp
            logger.print(f"IPR day_start calibrated: {self._ipr_day_start:.4f} at ts={timestamp}")

        self._ipr_last_ts = timestamp

        fair = self._ipr_day_start + IPR_SLOPE * timestamp
        logger.print(f"IPR fair={fair:.4f} mid={mid:.2f} pos={position} ts={timestamp}")

        orders: List[Order] = []
        buy_cap = limit - position
        sell_cap = limit + position

        # ── 1. PRIMARY: Buy aggressively — trend is always up ─────────────────
        # Take any ask at or below fair + small tolerance
        for ask in sorted(od.sell_orders.keys()):
            if ask > fair + 2:
                break
            take = min(-od.sell_orders[ask], buy_cap)
            if take > 0:
                orders.append(Order(product, ask, take))
                buy_cap -= take

        # ── 2. Sell anything wildly above fair (mean reversion of noise) ─────
        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid < fair + 5:
                break
            give = min(od.buy_orders[bid], sell_cap)
            if give > 0:
                orders.append(Order(product, bid, -give))
                sell_cap -= give

        # ── 3. Passive quotes — bias heavily toward buying (trend is up) ──────
        if buy_cap > 0:
            passive_bid = min(math.floor(fair) - 1, best_bid + 1)
            orders.append(Order(product, passive_bid, buy_cap))

        # Only sell passively well above fair to avoid missing the trend
        if sell_cap > 0:
            passive_ask = max(math.ceil(fair) + 5, best_ask - 1)
            orders.append(Order(product, passive_ask, -sell_cap))

        return orders