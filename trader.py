import json
import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional

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

# ── Normal distribution helpers ──────────────────────────────────────────────
_N = NormalDist()

def _ncdf(x: float) -> float:
    return _N.cdf(x)

def _npdf(x: float) -> float:
    return _N.pdf(x)


# ── Black-Scholes utilities ──────────────────────────────────────────────────

def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return S * _npdf(d1) * math.sqrt(T)


def implied_vol(
    price: float, S: float, K: float, T: float,
    max_iter: int = 50, tol: float = 1e-5
) -> Optional[float]:
    """Bisection implied vol — robust, no Newton instability."""
    if T <= 1e-8:
        return None
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + tol:
        return None
    lo, hi = 1e-4, 5.0
    # Quick sanity check
    if bs_call(S, K, T, hi) < price:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        val = bs_call(S, K, T, mid)
        if abs(val - price) < tol:
            return mid
        if val < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


# ── Wall Mid ─────────────────────────────────────────────────────────────────

def wall_mid(od: OrderDepth) -> Optional[float]:
    """Average of the bid wall (highest buy volume) and ask wall (highest sell volume)."""
    if not od.buy_orders or not od.sell_orders:
        return None
    bid_wall = max(od.buy_orders, key=lambda p: od.buy_orders[p])
    # sell_orders have negative quantities; most negative = largest size
    ask_wall = min(od.sell_orders, key=lambda p: od.sell_orders[p])
    return (bid_wall + ask_wall) / 2.0


# ── Parameters ───────────────────────────────────────────────────────────────

# Position limits
HGP_LIMIT   = 200
VEV_LIMIT   = 200
OPT_LIMIT   = 300

# Hydrogel Pack
HGP_FAIR = 10_000

# VEV underlying mean-reversion
VEV_MR_THR    = 15      # ticks from EMA to trigger aggressive trade
VEV_MR_WINDOW = 10      # EMA window (fast)

# Options IV scalping
TTE_DAYS_START   = 5    # days to expiry at start of Round 3 (round 3 = day 5 out of 7)
THR_OPEN         = 0.5  # open position when |deviation| > THR_OPEN
THR_CLOSE        = 0.0  # close position when deviation crosses THR_CLOSE
IV_SCALPING_THR  = 0.7  # switch_mean must exceed this to activate scalping
IV_SCALPING_WIN  = 100  # EMA window for abs deviation (switch signal)
THEO_NORM_WIN    = 20   # EMA window for theo_diff mean
LOW_VEGA_ADJ     = 0.5  # extra threshold for near-zero-vega options

# All option symbols and their strikes
OPTION_SYMBOLS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
STRIKES = {sym: int(sym.split("_")[1]) for sym in OPTION_SYMBOLS}

UNDERLYING = "VELVETFRUIT_EXTRACT"
HGP        = "HYDROGEL_PACK"

# Timestamps per day (1 day = 1_000_000 timestamp units based on 0–999,900)
TS_PER_DAY = 1_000_000


# ── EMA helper ───────────────────────────────────────────────────────────────

def ema_update(old: float, value: float, window: int) -> float:
    alpha = 2.0 / (window + 1)
    return alpha * value + (1.0 - alpha) * old


# ── Trader ───────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # 1. Restore persisted state
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}

        ts = state.timestamp

        # Detect day boundary (timestamp reset → new day)
        last_ts = td.get("last_ts", -1)
        days_elapsed = td.get("days_elapsed", 0)
        if ts < last_ts:
            days_elapsed += 1
        td["last_ts"] = ts
        td["days_elapsed"] = days_elapsed

        # TTE in years: starts at TTE_DAYS_START days, decreases each day + within-day
        tte = (TTE_DAYS_START - days_elapsed - ts / TS_PER_DAY) / 365.0

        result: Dict[Symbol, List[Order]] = {}

        # 2. VEV underlying (needed for options, computed first)
        vev_mid: Optional[float] = None
        if UNDERLYING in state.order_depths:
            od_vev = state.order_depths[UNDERLYING]
            vev_mid = wall_mid(od_vev)
            if vev_mid is None and od_vev.buy_orders and od_vev.sell_orders:
                vev_mid = (max(od_vev.buy_orders) + min(od_vev.sell_orders)) / 2.0
            orders_vev = self._trade_vev(od_vev, state.position.get(UNDERLYING, 0), vev_mid, td)
            if orders_vev:
                result[UNDERLYING] = orders_vev

        # 3. Hydrogel Pack
        if HGP in state.order_depths:
            od_hgp = state.order_depths[HGP]
            orders_hgp = self._trade_hgp(od_hgp, state.position.get(HGP, 0))
            if orders_hgp:
                result[HGP] = orders_hgp

        # 4. Options IV scalping
        if vev_mid is not None and tte > 1e-5:
            for sym in OPTION_SYMBOLS:
                if sym not in state.order_depths:
                    continue
                od_opt = state.order_depths[sym]
                pos = state.position.get(sym, 0)
                orders_opt = self._trade_option(sym, od_opt, pos, vev_mid, tte, td)
                if orders_opt:
                    result[sym] = orders_opt

        # 5. Persist state
        new_td = json.dumps(td)

        logger.flush(state, result, 0, new_td)
        return result, 0, new_td

    # ── HYDROGEL PACK ────────────────────────────────────────────────────────

    def _trade_hgp(self, od: OrderDepth, position: int) -> List[Order]:
        """Market-make around fixed fair value of 10,000."""
        orders: List[Order] = []
        buy_cap  = HGP_LIMIT - position
        sell_cap = HGP_LIMIT + position
        fair = HGP_FAIR

        # Take mispriced asks (below fair)
        for ask in sorted(od.sell_orders):
            if ask > fair:
                break
            if ask == fair and position >= 0:
                break
            take = min(-od.sell_orders[ask], buy_cap)
            if take > 0:
                orders.append(Order(HGP, ask, take))
                buy_cap -= take

        # Take mispriced bids (above fair)
        for bid in sorted(od.buy_orders, reverse=True):
            if bid < fair:
                break
            if bid == fair and position <= 0:
                break
            give = min(od.buy_orders[bid], sell_cap)
            if give > 0:
                orders.append(Order(HGP, bid, -give))
                sell_cap -= give

        # Passive quotes with inventory skew
        best_bid = max(od.buy_orders) if od.buy_orders else fair - 5
        best_ask = min(od.sell_orders) if od.sell_orders else fair + 5
        inv_skew = int(position / HGP_LIMIT * 3)

        m_bid = min(best_bid + 1, fair - 1) - max(0, inv_skew)
        m_ask = max(best_ask - 1, fair + 1) - min(0, inv_skew)

        if buy_cap > 0 and m_bid < fair:
            orders.append(Order(HGP, m_bid, buy_cap))
        if sell_cap > 0 and m_ask > fair:
            orders.append(Order(HGP, m_ask, -sell_cap))

        return orders

    # ── VELVETFRUIT EXTRACT ──────────────────────────────────────────────────

    def _trade_vev(
        self, od: OrderDepth, position: int,
        wm: Optional[float], td: Dict
    ) -> List[Order]:
        """Market-make + EMA mean-reversion on VEV underlying."""
        if not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = wm if wm is not None else (best_bid + best_ask) / 2.0

        # EMA update
        ema = td.get("ema_vev", mid)
        ema = ema_update(ema, mid, VEV_MR_WINDOW)
        td["ema_vev"] = ema

        orders: List[Order] = []
        buy_cap  = VEV_LIMIT - position
        sell_cap = VEV_LIMIT + position
        deviation = mid - ema

        # Aggressive mean-reversion trades
        if deviation > VEV_MR_THR and sell_cap > 0:
            # Price is high relative to EMA — sell aggressively at best bid
            give = min(od.buy_orders[best_bid], sell_cap)
            if give > 0:
                orders.append(Order(UNDERLYING, best_bid, -give))
                sell_cap -= give

        elif deviation < -VEV_MR_THR and buy_cap > 0:
            # Price is low relative to EMA — buy aggressively at best ask
            take = min(-od.sell_orders[best_ask], buy_cap)
            if take > 0:
                orders.append(Order(UNDERLYING, best_ask, take))
                buy_cap -= take

        # Take any asks clearly below mid, bids clearly above mid
        for ask in sorted(od.sell_orders):
            if ask >= mid - 2:
                break
            take = min(-od.sell_orders[ask], buy_cap)
            if take > 0:
                orders.append(Order(UNDERLYING, ask, take))
                buy_cap -= take

        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= mid + 2:
                break
            give = min(od.buy_orders[bid], sell_cap)
            if give > 0:
                orders.append(Order(UNDERLYING, bid, -give))
                sell_cap -= give

        # Passive quotes with inventory skew
        inv_skew = int(position / VEV_LIMIT * 3)
        m_bid = min(best_bid + 1, int(mid) - 1) - max(0, inv_skew)
        m_ask = max(best_ask - 1, int(mid) + 1) - min(0, inv_skew)

        if buy_cap > 0 and m_bid < mid:
            orders.append(Order(UNDERLYING, m_bid, buy_cap))
        if sell_cap > 0 and m_ask > mid:
            orders.append(Order(UNDERLYING, m_ask, -sell_cap))

        return orders

    # ── OPTIONS IV SCALPING ──────────────────────────────────────────────────

    def _trade_option(
        self, sym: str, od: OrderDepth, position: int,
        S: float, tte: float, td: Dict
    ) -> List[Order]:
        """IV scalping: trade deviations of option price from BS theoretical."""
        if not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        wm = wall_mid(od)
        if wm is None:
            wm = (best_bid + best_ask) / 2.0

        K = STRIKES[sym]
        buy_cap  = OPT_LIMIT - position
        sell_cap = OPT_LIMIT + position

        # Compute IV from mid price
        mid_price = (best_bid + best_ask) / 2.0
        iv = implied_vol(mid_price, S, K, tte)
        if iv is None:
            # Can't compute IV (deep OTM, near-zero extrinsic, etc.) — try to close only
            orders = []
            if position > 0:
                orders.append(Order(sym, best_bid, -position))
            elif position < 0:
                orders.append(Order(sym, best_ask, -position))
            return orders

        # Update IV EMA (smooth estimate of "fair" IV)
        iv_ema_key = f"{sym}_iv_ema"
        iv_ema = td.get(iv_ema_key, iv)
        iv_ema = ema_update(iv_ema, iv, THEO_NORM_WIN)
        td[iv_ema_key] = iv_ema

        # BS theoretical price using smoothed IV
        theo = bs_call(S, K, tte, iv_ema)

        # theo_diff = how much market price deviates from BS fair
        theo_diff = wm - theo

        # EMA of theo_diff (tracks slow drift of deviation)
        diff_ema_key = f"{sym}_diff_ema"
        diff_ema = td.get(diff_ema_key, 0.0)
        diff_ema = ema_update(diff_ema, theo_diff, THEO_NORM_WIN)
        td[diff_ema_key] = diff_ema

        # EMA of abs deviation from mean_diff (measures oscillation amplitude)
        abs_dev_key = f"{sym}_abs_dev_ema"
        abs_dev_ema = td.get(abs_dev_key, 0.0)
        abs_dev = abs(theo_diff - diff_ema)
        abs_dev_ema = ema_update(abs_dev_ema, abs_dev, IV_SCALPING_WIN)
        td[abs_dev_key] = abs_dev_ema

        # Compute vega to detect low-vega options (add extra buffer)
        vega = bs_vega(S, K, tte, iv_ema)
        low_vega_adj = LOW_VEGA_ADJ if vega <= 1.0 else 0.0

        # Centered deviation from mean
        deviation = theo_diff - diff_ema

        orders: List[Order] = []

        if abs_dev_ema >= IV_SCALPING_THR:
            # Enough oscillation — actively scalp
            thr_o = THR_OPEN + low_vega_adj

            if deviation > thr_o and sell_cap > 0:
                # Option is overpriced vs smile — sell at best bid
                orders.append(Order(sym, best_bid, -sell_cap))
            elif deviation < -thr_o and buy_cap > 0:
                # Option is underpriced vs smile — buy at best ask
                orders.append(Order(sym, best_ask, buy_cap))

            # Close trades when deviation returns toward mean
            if deviation > THR_CLOSE and position > 0:
                orders.append(Order(sym, best_bid, -position))
            elif deviation < -THR_CLOSE and position < 0:
                orders.append(Order(sym, best_ask, -position))

        else:
            # Not enough volatility — close any open positions
            if position > 0:
                orders.append(Order(sym, best_bid, -position))
            elif position < 0:
                orders.append(Order(sym, best_ask, -position))

        return orders
