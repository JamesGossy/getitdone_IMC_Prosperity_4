import json
import math
from typing import Any, Dict, List

from datamodel import (
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


# ── Logger (Visualizer Compatible) ──────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        output = [
            self.compress_state(state, state.traderData),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ]
        print(json.dumps(output, cls=ProsperityEncoder, separators=(",", ":")))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [depth.buy_orders, depth.sell_orders] for s, depth in state.order_depths.items()},
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        if hasattr(observations, "conversionObservations"):
            for product, observation in observations.conversionObservations.items():
                conversion_observations[product] = [
                    observation.bidPrice,
                    observation.askPrice,
                    observation.transportFees,
                    observation.exportTariff,
                    observation.importTariff,
                    observation.sugarPrice,
                    observation.sunlightIndex,
                ]
        return [getattr(observations, "plainValueObservations", {}), conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed


logger = Logger()


# ── Parameters & Limits ──────────────────────────────────────────────────────
LIMITS: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

# ASH_COATED_OSMIUM: hard fair value (like Rainforest Resin)
ACO_FAIR = 10_000

# INTARIAN_PEPPER_ROOT: linear trend — Price ≈ day_start + 0.001 * timestamp
IPR_SLOPE = 0.001


# ── Trader ───────────────────────────────────────────────────────────────────
class Trader:
    def __init__(self):
        self._ipr_day_start: float | None = None
        self._ipr_last_ts: int = -1

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # 1. Restore persisted state
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                self._ipr_day_start = saved.get("ipr_day_start", None)
                self._ipr_last_ts = saved.get("ipr_last_ts", -1)
            except Exception:
                pass

        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        # 2. Process each product
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

        # 3. Persist state
        trader_data_str = json.dumps({
            "ipr_day_start": self._ipr_day_start,
            "ipr_last_ts": self._ipr_last_ts,
        })

        # 4. Flush visualizer logs
        logger.flush(state, result, conversions, trader_data_str)

        return result, conversions, trader_data_str

    # ── ASH COATED OSMIUM ────────────────────────────────────────────────────
    def _trade_aco(
        self, product: str, od: OrderDepth, position: int, limit: int
    ) -> List[Order]:
        """
        Market-make around the fixed fair value of 10,000.
          1. Take any mispriced orders (asks below fair, bids above fair).
          2. Post passive quotes one tick inside best bid/ask, skewed by inventory.
        """
        orders: List[Order] = []
        buy_cap = limit - position
        sell_cap = limit + position

        # Take mispriced asks
        for ask in sorted(od.sell_orders.keys()):
            if ask >= ACO_FAIR:
                break
            take = min(-od.sell_orders[ask], buy_cap)
            if take > 0:
                orders.append(Order(product, ask, take))
                buy_cap -= take

        # Take mispriced bids
        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid <= ACO_FAIR:
                break
            give = min(od.buy_orders[bid], sell_cap)
            if give > 0:
                orders.append(Order(product, bid, -give))
                sell_cap -= give

        # Passive quotes with inventory skew
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else ACO_FAIR - 5
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else ACO_FAIR + 5
        inv_skew = int(position / limit * 2)

        m_bid = min(best_bid + 1, ACO_FAIR - 1) - max(0, inv_skew)
        m_ask = max(best_ask - 1, ACO_FAIR + 1) - min(0, inv_skew)

        logger.print(f"ACO fair={ACO_FAIR} pos={position} m_bid={m_bid} m_ask={m_ask}")

        if buy_cap > 0 and m_bid < ACO_FAIR:
            orders.append(Order(product, m_bid, buy_cap))
        if sell_cap > 0 and m_ask > ACO_FAIR:
            orders.append(Order(product, m_ask, -sell_cap))

        return orders

    # ── INTARIAN PEPPER ROOT ─────────────────────────────────────────────────
    def _trade_ipr(
        self, product: str, od: OrderDepth, position: int, limit: int, timestamp: int
    ) -> List[Order]:
        """
        Trend-following strategy on a +0.001/tick rising price.
          1. PRIMARY   — aggressively buy asks at or below fair + 2 (capture trend).
          2. SECONDARY — sell bids wildly above fair + 5 (mean-reversion noise).
          3. PASSIVE   — resting bid just below fair; resting ask well above fair.

        Fair value = day_start_price + IPR_SLOPE * timestamp.
        day_start_price is calibrated from mid-price on the first tick of each day
        (detected when timestamp resets below the previous tick's timestamp).
        """
        if not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        # Calibrate day_start on first tick or new day (timestamp reset)
        if self._ipr_day_start is None or timestamp < self._ipr_last_ts:
            self._ipr_day_start = mid - IPR_SLOPE * timestamp
            logger.print(f"IPR day_start calibrated: {self._ipr_day_start:.4f} at ts={timestamp}")

        self._ipr_last_ts = timestamp
        fair = self._ipr_day_start + IPR_SLOPE * timestamp

        logger.print(f"IPR fair={fair:.4f} mid={mid:.2f} pos={position} ts={timestamp}")

        orders: List[Order] = []
        buy_cap = limit - position
        sell_cap = limit + position

        # 1. Aggressively buy asks at or below fair + 2
        for ask in sorted(od.sell_orders.keys()):
            if ask > fair + 4:
                break
            take = min(-od.sell_orders[ask], buy_cap)
            if take > 0:
                orders.append(Order(product, ask, take))
                buy_cap -= take

        # 2. Sell bids that are wildly above fair (noise mean-reversion)
        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid < fair + 7:  
                break
            give = min(od.buy_orders[bid], sell_cap)
            if give > 0:
                orders.append(Order(product, bid, -give))
                sell_cap -= give

        # 3. Passive quotes — biased heavily toward buying
        if buy_cap > 0:
            passive_bid = min(math.floor(fair) - 1, best_bid + 1)
            orders.append(Order(product, passive_bid, buy_cap))

        # Only sell passively well above fair to avoid missing the uptrend
        if sell_cap > 0:
            passive_ask = max(math.ceil(fair) + 5, best_ask - 1)
            orders.append(Order(product, passive_ask, -sell_cap))

        return orders