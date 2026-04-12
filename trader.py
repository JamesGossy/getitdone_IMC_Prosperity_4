"""
IMC Prosperity 4 - Trading Algorithm
=====================================
Products: EMERALDS, TOMATOES

Strategy Summary
----------------
EMERALDS:
  - True fair value is almost always 10000 (confirmed across both data days).
  - The book quotes bid ~9992 / ask ~10008, giving us a huge edge.
  - We aggressively BUY anything below 10000 and SELL anything above 10000.
  - Edge: ~8 per trade on average vs a near-zero-risk position.

TOMATOES:
  - Noisy mean-reverting product (~std 20, mean ~5000-5007 per day).
  - We use a short rolling mid-price EMA as fair value estimate.
  - Market-make around that fair value with tight spreads,
    and lean the quotes based on current inventory to stay flat.
"""

from datamodel import (
    OrderDepth,
    TradingState,
    Order,
    ConversionObservation,
)
from typing import Dict, List
import json
import math

# ── Position limits ──────────────────────────────────────────────────────────
LIMITS: Dict[str, int] = {
    "EMERALDS": 20,
    "TOMATOES": 20,
}

# ── EMERALDS: hard fair value (confirmed from data) ──────────────────────────
EMERALD_FAIR = 10_000

# ── TOMATOES: market-making parameters ───────────────────────────────────────
TOM_BASE_SPREAD = 3        # half-spread around fair value (3 each side)
TOM_EMA_ALPHA   = 0.15     # smoothing factor for rolling mid-price EMA
TOM_INV_SKEW    = 0.4      # price skew per unit of inventory (cents)
TOM_AGGR_THRESH = 8        # cross fair by this much → take liquidity aggressively


class Trader:
    def __init__(self):
        # Persistent state stored as JSON in traderData string
        self._ema: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        # Restore state
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                self._ema = saved.get("ema", {})
            except Exception:
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

        # Persist state
        trader_data = json.dumps({"ema": self._ema})
        return result, 0, trader_data

    # ── EMERALDS ──────────────────────────────────────────────────────────────
    def _trade_emeralds(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        limit: int,
    ) -> List[Order]:
        """
        Fair value = 10000 (hard-coded, validated across both training days).
        - Lift every ask below 10000 (we buy cheap).
        - Hit every bid above 10000 (we sell dear).
        - Post passive quotes at 9999 / 10001 to earn the spread.
        """
        orders: List[Order] = []
        buy_capacity  = limit - position   # how many more we can buy
        sell_capacity = limit + position   # how many more we can sell

        # ── Take asks below fair value ──
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= EMERALD_FAIR:
                break
            vol = -od.sell_orders[ask_price]   # sell_orders volumes are negative
            take = min(vol, buy_capacity)
            if take > 0:
                orders.append(Order(product, ask_price, take))
                buy_capacity -= take

        # ── Hit bids above fair value ──
        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= EMERALD_FAIR:
                break
            vol = od.buy_orders[bid_price]
            give = min(vol, sell_capacity)
            if give > 0:
                orders.append(Order(product, bid_price, -give))
                sell_capacity -= give

        # ── Passive quotes at fair ± 1 ──
        # Skew based on inventory to mean-revert position
        skew = int(position / limit * 2)  # -2 to +2

        passive_bid = EMERALD_FAIR - 1 - max(0,  skew)
        passive_ask = EMERALD_FAIR + 1 - min(0,  skew)

        if buy_capacity > 0:
            orders.append(Order(product, passive_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(product, passive_ask, -sell_capacity))

        return orders

    # ── TOMATOES ──────────────────────────────────────────────────────────────
    def _trade_tomatoes(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        limit: int,
    ) -> List[Order]:
        """
        Mean-reverting market-making strategy.
        1. Compute current mid-price.
        2. Update EMA as our fair value estimate.
        3. Skew quotes based on inventory (lean to stay flat).
        4. Aggressively take price if it crosses our fair value by ≥ THRESH.
        """
        # ── Compute mid price ──
        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is None or best_ask is None:
            return []

        mid = (best_bid + best_ask) / 2.0

        # ── Update EMA ──
        alpha = TOM_EMA_ALPHA
        if product not in self._ema:
            self._ema[product] = mid
        else:
            self._ema[product] = alpha * mid + (1 - alpha) * self._ema[product]

        fair = self._ema[product]

        orders: List[Order] = []
        buy_capacity  = limit - position
        sell_capacity = limit + position

        # ── Aggressive taking: price far from fair ──
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price > fair - TOM_AGGR_THRESH:
                break
            vol  = -od.sell_orders[ask_price]
            take = min(vol, buy_capacity)
            if take > 0:
                orders.append(Order(product, ask_price, take))
                buy_capacity -= take

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price < fair + TOM_AGGR_THRESH:
                break
            vol  = od.buy_orders[bid_price]
            give = min(vol, sell_capacity)
            if give > 0:
                orders.append(Order(product, bid_price, -give))
                sell_capacity -= give

        # ── Inventory skew ──
        skew = position * TOM_INV_SKEW   # positive position → lower our bid/ask

        # ── Passive quotes ──
        our_bid = math.floor(fair - TOM_BASE_SPREAD - skew)
        our_ask = math.ceil (fair + TOM_BASE_SPREAD - skew)

        # Don't post inside the current book (we'd just cross ourselves)
        our_bid = min(our_bid, best_bid)
        our_ask = max(our_ask, best_ask)

        if buy_capacity > 0:
            orders.append(Order(product, our_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(product, our_ask, -sell_capacity))

        return orders