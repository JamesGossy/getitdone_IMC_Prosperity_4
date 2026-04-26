"""
HYDROGEL_PACK trader for IMC Prosperity 4.

Strategy summary (see HYDROGEL_PACK_PLAN.md for derivation):
  - HYDROGEL_PACK is dominated by two bots: Mark 14 (passive MM at mid +/- 8)
    and Mark 38 (anti-informed taker that crosses Mark 14's spread).
  - We market-make 1 tick inside Mark 14's wall (so quotes at FV +/- 7),
    skew quotes by inventory, and adjust FV by short-term order-book
    imbalance and long-EMA fade.
  - In the rare narrow-spread regime (~3.3% of timestamps), some other
    agent is quoting inside Mark 14's wall. We use level-2 of the book
    as the wall in that case, and step back if the spread room collapses.
  - Take any resting order priced through fair value.

Compatible with: nabayansaha/imc-prosperity-4-backtester (CLI: prosperity4btest).
HYDROGEL_PACK is NOT in the backtester's default LIMITS dict, so you must
pass `--limit HYDROGEL_PACK:200` on the command line.
"""

import json
import math
from typing import Any, List

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)

# =============================================================================
# CONFIG  --  edit these for parameter sweeps; everything else is structural
# =============================================================================
SYMBOL = "HYDROGEL_PACK"
POSITION_LIMIT = 200

# Fair-value model
ALPHA_IMB     = 2.0     # weight on (bid_v1 - ask_v1) / (bid_v1 + ask_v1)
ALPHA_EMA     = 0.07    # weight on (wall_mid - slow_ema) reversion
EMA_HALFLIFE  = 1000    # ticks; EWM halflife for the slow anchor

# Quoting
EDGE_X        = 7       # half-spread of our quotes around FV (1 tick inside Mark 14's wall)
K_INV         = 0.05    # inventory skew per unit position; shifts both bid & ask by -K_INV*pos

# Taking
TAKE_THRESHOLD = 0      # take an ask if (FV - ask) > TAKE_THRESHOLD; symmetric for bids

# Regime detection
NARROW_SPREAD_THRESHOLD = 10  # inner_spread <= this => narrow regime, use L2 as wall

# Make-side risk: if room inside Mark 14's wall is too tight (someone undercut us
# from inside), skip making this tick rather than fight for thin edge.
STEP_BACK_IF_NO_ROOM = True

# Optional: cap individual quote size (None = use full remaining capacity).
MAX_QUOTE_SIZE = None


# =============================================================================
# Logger -- visualizer-compatible Logger from sample.py (do not modify)
# =============================================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]],
              conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return out

    def compress_observations(self, observations: Observation) -> list[Any]:
        co = {}
        for product, obs in observations.conversionObservations.items():
            co[product] = [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex,
            ]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = value[:mid] + ("..." if mid < len(value) else "")
            if len(json.dumps(cand)) <= max_length:
                out = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# =============================================================================
# Trader
# =============================================================================
class Trader:

    def run(self, state: TradingState):
        # ----- restore persistent state -----
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}
        ema = td.get("ema", None)

        result: dict[Symbol, list[Order]] = {}

        # If hydrogel isn't in the book this tick, do nothing
        if SYMBOL not in state.order_depths:
            return result, 0, json.dumps(td)

        depth: OrderDepth = state.order_depths[SYMBOL]
        position = state.position.get(SYMBOL, 0)

        # Sort book: bids high->low, asks low->high
        bids = sorted(depth.buy_orders.items(),  key=lambda kv: -kv[0])
        asks = sorted(depth.sell_orders.items(), key=lambda kv:  kv[0])

        if not bids or not asks:
            # One-sided book; can't form fair value robustly. Skip this tick.
            return {SYMBOL: []}, 0, json.dumps(td)

        bb1, bv1 = bids[0]
        ba1, av1 = asks[0]
        bb2 = bids[1][0] if len(bids) >= 2 else bb1
        ba2 = asks[1][0] if len(asks) >= 2 else ba1
        bv1_abs = abs(bv1)
        av1_abs = abs(av1)
        inner_spread = ba1 - bb1

        # ----- 1. wall_mid (regime-aware) -----
        # In wide regime (>96% of ticks) Mark 14 is at L1, so L1-mid is wall_mid.
        # In narrow regime someone tighter is at L1; Mark 14's wall is at L2.
        if inner_spread <= NARROW_SPREAD_THRESHOLD:
            wall_mid = (bb2 + ba2) / 2.0
            inner_bid_wall = bb2
            inner_ask_wall = ba2
        else:
            wall_mid = (bb1 + ba1) / 2.0
            inner_bid_wall = bb1
            inner_ask_wall = ba1

        # ----- 2. EMA update (persisted across ticks) -----
        if ema is None:
            ema = wall_mid
        else:
            alpha = 1.0 - 0.5 ** (1.0 / EMA_HALFLIFE)
            ema = (1.0 - alpha) * ema + alpha * wall_mid
        td["ema"] = ema

        # ----- 3. order-book imbalance (short-horizon predictor) -----
        v1_total = bv1_abs + av1_abs
        imbalance = (bv1_abs - av1_abs) / v1_total if v1_total > 0 else 0.0

        # ----- 4. fair value -----
        fv = wall_mid + ALPHA_IMB * imbalance - ALPHA_EMA * (wall_mid - ema)

        orders: List[Order] = []

        # ----- 5. TAKE: scan asks for prices through fair (BUY them) -----
        # Track separate "what would my position become if all my buys filled" for
        # buy-side capacity. Sells use a separate counter (orders on opposite sides
        # don't both consume the same headroom -- only one side can fill on a given
        # resting order).
        cur_buy_pos  = position    # position assuming all queued buys fill
        cur_sell_pos = position    # position assuming all queued sells fill

        for ask_price, ask_vol in asks:
            ask_vol_abs = abs(ask_vol)
            should_take = False
            if (fv - ask_price) > TAKE_THRESHOLD:
                should_take = True
            elif ask_price <= fv and position < 0:
                # at-or-better-than-fair flatten when short
                should_take = True
            if not should_take:
                break  # asks are sorted ascending; once one fails, all higher do too

            buyable = min(ask_vol_abs, POSITION_LIMIT - cur_buy_pos)
            if buyable > 0:
                orders.append(Order(SYMBOL, ask_price, buyable))
                cur_buy_pos += buyable

        # ----- 5b. TAKE: scan bids for prices through fair (SELL into them) -----
        for bid_price, bid_vol in bids:
            bid_vol_abs = abs(bid_vol)
            should_take = False
            if (bid_price - fv) > TAKE_THRESHOLD:
                should_take = True
            elif bid_price >= fv and position > 0:
                should_take = True
            if not should_take:
                break

            sellable = min(bid_vol_abs, POSITION_LIMIT + cur_sell_pos)
            if sellable > 0:
                orders.append(Order(SYMBOL, bid_price, -sellable))
                cur_sell_pos -= sellable

        # ----- 6. MAKE: post resting quotes inside Mark 14's wall -----
        bought_so_far = sum(o.quantity for o in orders if o.quantity > 0)
        sold_so_far   = sum(-o.quantity for o in orders if o.quantity < 0)
        max_buy  = POSITION_LIMIT - position - bought_so_far
        max_sell = POSITION_LIMIT + position - sold_so_far

        # Skew BOTH quotes by -K_INV*pos (long => both shift down => bid less aggressive,
        # ask more aggressive; net biases us to sell).
        bid_offset = EDGE_X + K_INV * position    # bigger when long => deeper bid
        ask_offset = EDGE_X - K_INV * position    # smaller when long => tighter ask

        bid_target = math.floor(fv - bid_offset)
        ask_target = math.ceil(fv + ask_offset)

        # ----- room check: must be strictly inside Mark 14's wall, not crossing -----
        # spread_room is the number of price-tick slots strictly INSIDE the walls.
        # walls at e.g. 9991 / 10007 => slots 9992..10006 => 15 slots => spread_room=15.
        spread_room = inner_ask_wall - inner_bid_wall - 1
        no_room_to_make = (spread_room <= 0)

        post_bid = True
        post_ask = True

        # If quote target falls AT or OUTSIDE Mark 14's wall on its own side,
        # skip that side. (Long => bid drifts to/below wall => stop bidding => good.)
        if bid_target <= inner_bid_wall:
            post_bid = False
        if ask_target >= inner_ask_wall:
            post_ask = False

        # If targets crossed each other (very high |skew|), clamp.
        if post_bid and post_ask and bid_target >= ask_target:
            mid_anchor = (inner_bid_wall + inner_ask_wall) / 2.0
            if position > 0:
                # prefer selling: pin ask, drop bid
                ask_target = max(ask_target, math.ceil(mid_anchor) + 0)
                bid_target = ask_target - 1
            else:
                bid_target = min(bid_target, math.floor(mid_anchor) - 0)
                ask_target = bid_target + 1
            # Re-validate inside walls
            if bid_target <= inner_bid_wall: post_bid = False
            if ask_target >= inner_ask_wall: post_ask = False

        if STEP_BACK_IF_NO_ROOM and no_room_to_make:
            post_bid = False
            post_ask = False

        if post_bid and max_buy > 0:
            qty = max_buy if MAX_QUOTE_SIZE is None else min(max_buy, MAX_QUOTE_SIZE)
            if qty > 0:
                orders.append(Order(SYMBOL, int(bid_target), int(qty)))

        if post_ask and max_sell > 0:
            qty = max_sell if MAX_QUOTE_SIZE is None else min(max_sell, MAX_QUOTE_SIZE)
            if qty > 0:
                orders.append(Order(SYMBOL, int(ask_target), -int(qty)))

        result[SYMBOL] = orders

        # ----- 7. lightweight diagnostics for the visualizer -----
        logger.print(
            f"ts={state.timestamp} pos={position} "
            f"wall_mid={wall_mid:.2f} fv={fv:.2f} ema={ema:.2f} imb={imbalance:+.2f} "
            f"sp_inner={inner_spread} room={spread_room} "
            f"bid={bid_target if post_bid else '-'} ask={ask_target if post_ask else '-'} "
            f"max_buy={max_buy} max_sell={max_sell}"
        )

        trader_data = json.dumps(td)
        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data