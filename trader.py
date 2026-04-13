import json
import math
from typing import Any, Dict, List
from datamodel import (
    Listing, 
    Observation, 
    Order, 
    OrderDepth, 
    ProsperityEncoder, 
    Symbol, 
    Trade, 
    TradingState
)

# ── Position limits ──────────────────────────────────────────────────────────
LIMITS: Dict[str, int] = {
    "EMERALDS": 20,
    "TOMATOES": 20,
}

# ── EMERALDS: hard fair value ────────────────────────────────────────────────
EMERALD_FAIR = 10_000

# ── TOMATOES: market-making parameters ───────────────────────────────────────
TOM_BASE_SPREAD = 3     
TOM_EMA_ALPHA   = 0.15  
TOM_INV_SKEW    = 0.4   
TOM_AGGR_THRESH = 8     

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Visualizer expects: [state, orders, conversions, trader_data, logs]
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
        if hasattr(observations, 'conversionObservations'):
            for product, observation in observations.conversionObservations.items():
                conversion_observations[product] = [
                    observation.bidPrice, observation.askPrice, observation.transportFees,
                    observation.exportTariff, observation.importTariff, observation.sugarPrice,
                    observation.sunlightIndex,
                ]
        return [getattr(observations, 'plainValueObservations', {}), conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

logger = Logger()

class Trader:
    def __init__(self):
        self._ema: Dict[str, float] = {}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Restore EMA state from traderData
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                self._ema = saved.get("ema", {})
            except:
                self._ema = {}

        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            limit = LIMITS.get(product, 20)

            if product == "EMERALDS":
                orders = self._trade_emeralds(product, order_depth, position, limit)
            elif product == "TOMATOES":
                orders = self._trade_tomatoes(product, order_depth, position, limit)
            else:
                orders = []

            if orders:
                result[product] = orders

        # Persist state and flush logs
        trader_data_str = json.dumps({"ema": self._ema})
        logger.flush(state, result, 0, trader_data_str)
        
        return result, 0, trader_data_str

    def _trade_emeralds(self, product: str, od: OrderDepth, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_capacity = limit - position
        sell_capacity = limit + position

        # Market Taking
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= EMERALD_FAIR: break
            take = min(-od.sell_orders[ask_price], buy_capacity)
            if take > 0:
                orders.append(Order(product, ask_price, take))
                buy_capacity -= take

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= EMERALD_FAIR: break
            give = min(od.buy_orders[bid_price], sell_capacity)
            if give > 0:
                orders.append(Order(product, bid_price, -give))
                sell_capacity -= give

        # Passive Quoting with inventory skew
        logger.print(f"FAIR:EMERALDS:{EMERALD_FAIR}")
        skew = int(position / limit * 2)
        passive_bid = EMERALD_FAIR - 1 - max(0, skew)
        passive_ask = EMERALD_FAIR + 1 - min(0, skew)

        if buy_capacity > 0:
            orders.append(Order(product, passive_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(product, passive_ask, -sell_capacity))

        return orders

    def _trade_tomatoes(self, product: str, od: OrderDepth, position: int, limit: int) -> List[Order]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is None or best_ask is None:
            return []

        mid = (best_bid + best_ask) / 2.0
        self._ema[product] = TOM_EMA_ALPHA * mid + (1 - TOM_EMA_ALPHA) * self._ema.get(product, mid)
        fair = self._ema[product]
        logger.print(f"FAIR:TOMATOES:{fair:.4f}")

        orders: List[Order] = []
        buy_capacity = limit - position
        sell_capacity = limit + position

        # Aggressive taking
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price > fair - TOM_AGGR_THRESH: break
            take = min(-od.sell_orders[ask_price], buy_capacity)
            if take > 0:
                orders.append(Order(product, ask_price, take))
                buy_capacity -= take

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price < fair + TOM_AGGR_THRESH: break
            give = min(od.buy_orders[bid_price], sell_capacity)
            if give > 0:
                orders.append(Order(product, bid_price, -give))
                sell_capacity -= give

        # Passive quotes
        skew = position * TOM_INV_SKEW
        our_bid = min(math.floor(fair - TOM_BASE_SPREAD - skew), best_bid)
        our_ask = max(math.ceil(fair + TOM_BASE_SPREAD - skew), best_ask)

        if buy_capacity > 0:
            orders.append(Order(product, our_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(product, our_ask, -sell_capacity))

        return orders