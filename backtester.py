"""
IMC Prosperity 4 — Custom Backtester + Visualiser
===================================================
Mirrors the logic of the prosperity4bt repo exactly:
  - Same CSV parsing (prices + trades)
  - Same order matching (order depth first, then market trades)
  - Same position limit enforcement
  - Same PnL accounting
  - Same risk metrics (Sharpe, Sortino, Calmar, max drawdown)

Plus a built-in matplotlib visualiser with:
  - PnL over time per product + total
  - Position over time per product
  - Mid-price over time with your trades overlaid
  - Summary stats table

Usage
-----
  python backtester.py <trader_file.py> [round] [--data TUTORIAL_ROUND_1] [--merge-pnl] [--no-vis]

Examples
--------
  python backtester.py trader.py 1 --data TUTORIAL_ROUND_1
  python backtester.py trader.py 1 --data TUTORIAL_ROUND_1 --merge-pnl
  python backtester.py trader.py 1 --data TUTORIAL_ROUND_1 --no-vis
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

# ── Try to use the installed datamodel, fall back to a bundled minimal one ──
try:
    from prosperity4bt import datamodel as _dm
    sys.modules.setdefault("datamodel", _dm)
    from datamodel import Order, OrderDepth, Trade, TradingState, Listing, Observation
except ImportError:
    # Minimal datamodel bundled so backtester works standalone
    from dataclasses import dataclass as _dc
    from typing import Dict, List

    @_dc
    class Order:
        symbol: str
        price: int
        quantity: int

    class OrderDepth:
        def __init__(self):
            self.buy_orders: Dict[int, int] = {}
            self.sell_orders: Dict[int, int] = {}

    @_dc
    class Trade:
        symbol: str
        price: int
        quantity: int
        buyer: str = ""
        seller: str = ""
        timestamp: int = 0

    @_dc
    class Listing:
        symbol: str
        product: str
        denomination: int

    class Observation:
        def __init__(self, plain=None, conversion=None):
            self.plainValueObservations = plain or {}
            self.conversionObservations = conversion or {}

    class TradingState:
        def __init__(self, traderData, timestamp, listings, order_depths,
                     own_trades, market_trades, position, observations):
            self.traderData = traderData
            self.timestamp = timestamp
            self.listings = listings
            self.order_depths = order_depths
            self.own_trades = own_trades
            self.market_trades = market_trades
            self.position = position
            self.observations = observations


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_LIMIT = 50
KNOWN_LIMITS: dict[str, int] = {"EMERALDS": 80, "TOMATOES": 80}


def get_limit(symbol: str, overrides: Optional[dict[str, int]] = None) -> int:
    if overrides and symbol in overrides:
        return overrides[symbol]
    return KNOWN_LIMITS.get(symbol, DEFAULT_LIMIT)


@dataclass
class PriceRow:
    day: int
    timestamp: int
    product: str
    bid_prices: list[int]
    bid_volumes: list[int]
    ask_prices: list[int]
    ask_volumes: list[int]
    mid_price: float


@dataclass
class DayData:
    round_num: int
    day_num: int
    products: list[str]
    prices: dict[int, dict[str, PriceRow]]          # ts -> product -> row
    trades: dict[int, dict[str, list[Trade]]]        # ts -> product -> trades
    profit_loss: dict[str, float]                    # running PnL per product


def _col_ints(cols: list[str], indices: list[int]) -> list[int]:
    out = []
    for i in indices:
        v = cols[i].strip()
        if not v:
            break
        out.append(int(v))
    return out


def load_day(data_dir: Path, round_num: int, day_num: int) -> Optional[DayData]:
    prices_file = data_dir / f"round{round_num}" / f"prices_round_{round_num}_day_{day_num}.csv"
    trades_file = data_dir / f"round{round_num}" / f"trades_round_{round_num}_day_{day_num}.csv"

    if not prices_file.exists():
        return None

    prices_list: list[PriceRow] = []
    for line in prices_file.read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split(";")
        if len(cols) < 16:
            continue
        prices_list.append(PriceRow(
            day=int(cols[0]),
            timestamp=int(cols[1]),
            product=cols[2].strip(),
            bid_prices=_col_ints(cols, [3, 5, 7]),
            bid_volumes=_col_ints(cols, [4, 6, 8]),
            ask_prices=_col_ints(cols, [9, 11, 13]),
            ask_volumes=_col_ints(cols, [10, 12, 14]),
            mid_price=float(cols[15]),
        ))

    trades_list: list[Trade] = []
    if trades_file.exists():
        for line in trades_file.read_text(encoding="utf-8").splitlines()[1:]:
            cols = line.split(";")
            if len(cols) < 7:
                continue
            trades_list.append(Trade(
                symbol=cols[3].strip(),
                price=int(float(cols[5])),
                quantity=int(cols[6]),
                buyer=cols[1].strip(),
                seller=cols[2].strip(),
                timestamp=int(cols[0]),
            ))

    prices_by_ts: dict[int, dict[str, PriceRow]] = defaultdict(dict)
    for row in prices_list:
        prices_by_ts[row.timestamp][row.product] = row

    trades_by_ts: dict[int, dict[str, list[Trade]]] = defaultdict(lambda: defaultdict(list))
    for t in trades_list:
        trades_by_ts[t.timestamp][t.symbol].append(t)

    products = sorted({r.product for r in prices_list})

    return DayData(
        round_num=round_num,
        day_num=day_num,
        products=products,
        prices=dict(prices_by_ts),
        trades=dict(trades_by_ts),
        profit_loss={p: 0.0 for p in products},
    )


# ═══════════════════════════════════════════════════════════════════════════
# ORDER MATCHING  (mirrors prosperity4bt/runner.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketTrade:
    trade: Trade
    buy_qty: int
    sell_qty: int


def _match_buy(state: TradingState, data: DayData, order: Order,
               mkt: list[MarketTrade], limits_override) -> list[Trade]:
    trades = []
    od = state.order_depths[order.symbol]

    for price in sorted(p for p in od.sell_orders if p <= order.price):
        lim = get_limit(order.symbol, limits_override)
        pos = state.position.get(order.symbol, 0)
        vol = min(order.quantity, abs(od.sell_orders[price]), max(0, lim - pos))
        if vol <= 0:
            continue
        trades.append(Trade(order.symbol, price, vol, "SUBMISSION", "", state.timestamp))
        state.position[order.symbol] = pos + vol
        data.profit_loss[order.symbol] -= price * vol
        od.sell_orders[price] += vol
        if od.sell_orders[price] == 0:
            del od.sell_orders[price]
        order.quantity -= vol
        if order.quantity == 0:
            return trades

    for mt in mkt:
        if mt.sell_qty == 0 or mt.trade.price > order.price:
            continue
        lim = get_limit(order.symbol, limits_override)
        pos = state.position.get(order.symbol, 0)
        vol = min(order.quantity, mt.sell_qty, max(0, lim - pos))
        if vol <= 0:
            continue
        trades.append(Trade(order.symbol, order.price, vol, "SUBMISSION", mt.trade.seller, state.timestamp))
        state.position[order.symbol] = pos + vol
        data.profit_loss[order.symbol] -= order.price * vol
        mt.sell_qty -= vol
        order.quantity -= vol
        if order.quantity == 0:
            return trades

    return trades


def _match_sell(state: TradingState, data: DayData, order: Order,
                mkt: list[MarketTrade], limits_override) -> list[Trade]:
    trades = []
    od = state.order_depths[order.symbol]

    for price in sorted((p for p in od.buy_orders if p >= order.price), reverse=True):
        lim = get_limit(order.symbol, limits_override)
        pos = state.position.get(order.symbol, 0)
        vol = min(abs(order.quantity), od.buy_orders[price], max(0, pos + lim))
        if vol <= 0:
            continue
        trades.append(Trade(order.symbol, price, vol, "", "SUBMISSION", state.timestamp))
        state.position[order.symbol] = pos - vol
        data.profit_loss[order.symbol] += price * vol
        od.buy_orders[price] -= vol
        if od.buy_orders[price] == 0:
            del od.buy_orders[price]
        order.quantity += vol
        if order.quantity == 0:
            return trades

    for mt in mkt:
        if mt.buy_qty == 0 or mt.trade.price < order.price:
            continue
        lim = get_limit(order.symbol, limits_override)
        pos = state.position.get(order.symbol, 0)
        vol = min(abs(order.quantity), mt.buy_qty, max(0, pos + lim))
        if vol <= 0:
            continue
        trades.append(Trade(order.symbol, order.price, vol, mt.trade.buyer, "SUBMISSION", state.timestamp))
        state.position[order.symbol] = pos - vol
        data.profit_loss[order.symbol] += order.price * vol
        mt.buy_qty -= vol
        order.quantity += vol
        if order.quantity == 0:
            return trades

    return trades


def _enforce_limits(state, data: DayData, orders: dict, limits_override):
    for product in list(orders.keys()):
        pos = state.position.get(product, 0)
        lim = get_limit(product, limits_override)
        total_long  = sum(o.quantity for o in orders[product] if o.quantity > 0)
        total_short = sum(abs(o.quantity) for o in orders[product] if o.quantity < 0)
        if pos + total_long > lim or pos - total_short < -lim:
            print(f"  [LIMIT] Orders for {product} cancelled (would exceed ±{lim})")
            del orders[product]


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST RESULT  (records everything needed for visualisation)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TimestepRecord:
    timestamp: int
    product: str
    mid_price: float
    pnl: float           # mark-to-market PnL at this timestamp
    position: int


@dataclass
class TradeRecord:
    timestamp: int
    product: str
    price: int
    quantity: int        # positive = bought, negative = sold


@dataclass
class DayResult:
    round_num: int
    day_num: int
    records: list[TimestepRecord] = field(default_factory=list)
    my_trades: list[TradeRecord]  = field(default_factory=list)

    def final_pnl(self) -> dict[str, float]:
        """PnL per product at the last timestamp."""
        last_ts = max(r.timestamp for r in self.records)
        return {
            r.product: r.pnl
            for r in self.records
            if r.timestamp == last_ts
        }


# ═══════════════════════════════════════════════════════════════════════════
# CORE RUN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_day(trader, data: DayData, limits_override=None, verbose=False) -> DayResult:
    result = DayResult(round_num=data.round_num, day_num=data.day_num)

    trader_data = ""
    state = TradingState(
        traderData=trader_data,
        timestamp=0,
        listings={},
        order_depths={},
        own_trades={},
        market_trades={},
        position={p: 0 for p in data.products},
        observations=Observation({}, {}),
    )

    timestamps = sorted(data.prices.keys())
    total = len(timestamps)

    for i, ts in enumerate(timestamps):
        # Progress bar
        pct = int(50 * i / total)
        bar = "█" * pct + "░" * (50 - pct)
        print(f"\r  [{bar}] {i}/{total}", end="", flush=True)

        state.timestamp = ts
        state.traderData = trader_data
        state.own_trades = {}
        state.market_trades = {}

        # Build order depths
        for product in data.products:
            od = OrderDepth()
            row = data.prices[ts].get(product)
            if row:
                for p, v in zip(row.bid_prices, row.bid_volumes):
                    od.buy_orders[p] = v
                for p, v in zip(row.ask_prices, row.ask_volumes):
                    od.sell_orders[p] = -v
            state.order_depths[product] = od
            state.listings[product] = Listing(product, product, 1)

        # Call trader
        buf = StringIO()
        buf.close = lambda: None
        with redirect_stdout(buf):
            try:
                orders, conversions, trader_data = trader.run(state)
            except Exception as e:
                print(f"\n  [ERROR] Trader raised at ts={ts}: {e}")
                orders, conversions, trader_data = {}, 0, trader_data

        if verbose and buf.getvalue().strip():
            print(f"\n  [LOG ts={ts}] {buf.getvalue().strip()}")

        # Enforce limits
        _enforce_limits(state, data, orders, limits_override)

        # Match orders
        market_trades_by_product = {
            prod: [MarketTrade(t, t.quantity, t.quantity) for t in trades]
            for prod, trades in data.trades.get(ts, {}).items()
        }

        for product in data.products:
            for order in orders.get(product, []):
                mkt = market_trades_by_product.get(product, [])
                if order.quantity > 0:
                    filled = _match_buy(state, data, order, mkt, limits_override)
                elif order.quantity < 0:
                    filled = _match_sell(state, data, order, mkt, limits_override)
                else:
                    filled = []

                for t in filled:
                    qty = t.quantity if t.buyer == "SUBMISSION" else -t.quantity
                    result.my_trades.append(TradeRecord(ts, product, t.price, qty))

        # Record mark-to-market state
        for product in data.products:
            row = data.prices[ts].get(product)
            if not row:
                continue
            pos = state.position.get(product, 0)
            mtm_pnl = data.profit_loss[product] + pos * row.mid_price
            result.records.append(TimestepRecord(
                timestamp=ts,
                product=product,
                mid_price=row.mid_price,
                pnl=mtm_pnl,
                position=pos,
            ))

    print(f"\r  [{'█'*50}] {total}/{total}", flush=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# RISK METRICS  (same formulas as prosperity4bt/metrics.py)
# ═══════════════════════════════════════════════════════════════════════════

TRADING_DAYS_PER_YEAR = 252


def risk_metrics(day_results: list[DayResult]) -> dict:
    # Stitch equity curves across days
    all_levels: list[float] = []
    day_pnls: list[float] = []
    offset = 0.0

    for dr in day_results:
        by_ts: dict[int, float] = defaultdict(float)
        for r in dr.records:
            by_ts[r.timestamp] += r.pnl
        levels = [v for _, v in sorted(by_ts.items())]
        if not levels:
            continue
        shifted = [offset + x for x in levels]
        all_levels.extend(shifted)
        offset = shifted[-1]
        day_pnls.append(levels[-1])

    if not all_levels:
        return {}

    final_pnl = all_levels[-1]

    # Max drawdown
    hwm = all_levels[0]
    max_dd_abs = 0.0
    max_dd_pct = float("nan")
    for e in all_levels:
        hwm = max(hwm, e)
        dd = hwm - e
        max_dd_abs = max(max_dd_abs, dd)
        if hwm > 0:
            p = dd / hwm
            max_dd_pct = p if math.isnan(max_dd_pct) else max(max_dd_pct, p)

    # Sharpe / Sortino on daily PnL
    if len(day_pnls) >= 2:
        m = mean(day_pnls)
        s = stdev(day_pnls)
        sharpe = (m / s) if s > 0 else float("nan")
        down_sq = sum(min(0.0, r) ** 2 for r in day_pnls)
        d = math.sqrt(down_sq / len(day_pnls))
        sortino = (m / d) if d > 0 else (float("inf") if m > 0 else float("nan"))
    else:
        sharpe = sortino = float("nan")

    ann_sharpe = sharpe * math.sqrt(TRADING_DAYS_PER_YEAR) if not math.isnan(sharpe) else float("nan")
    calmar = (final_pnl / max_dd_abs) if max_dd_abs > 0 else float("nan")

    return {
        "final_pnl": final_pnl,
        "sharpe_ratio": sharpe,
        "annualized_sharpe": ann_sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_abs": max_dd_abs,
        "max_drawdown_pct": max_dd_pct,
        "calmar_ratio": calmar,
        "_stitched_levels": all_levels,
        "_day_results": day_results,
    }


def _fmt(x: float, int_style=False) -> str:
    if math.isnan(x):  return "n/a"
    if math.isinf(x):  return "inf" if x > 0 else "-inf"
    return f"{x:,.0f}" if int_style else f"{x:,.4f}"


def print_metrics(m: dict) -> None:
    print(f"  final_pnl:          {_fmt(m['final_pnl'], True)}")
    print(f"  sharpe_ratio:       {_fmt(m['sharpe_ratio'])}")
    print(f"  annualized_sharpe:  {_fmt(m['annualized_sharpe'])}")
    print(f"  sortino_ratio:      {_fmt(m['sortino_ratio'])}")
    print(f"  max_drawdown_abs:   {_fmt(m['max_drawdown_abs'], True)}")
    print(f"  max_drawdown_pct:   {_fmt(m['max_drawdown_pct'])}")
    print(f"  calmar_ratio:       {_fmt(m['calmar_ratio'])}")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALISER
# ═══════════════════════════════════════════════════════════════════════════

def visualise(day_results: list[DayResult], metrics: dict, merge_pnl: bool) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("matplotlib not installed — skipping visualisation (pip install matplotlib)")
        return

    # ── Colour palette ──────────────────────────────────────────────────
    BG       = "#0d1117"
    PANEL    = "#161b22"
    BORDER   = "#30363d"
    TEXT     = "#e6edf3"
    MUTED    = "#8b949e"
    GREEN    = "#3fb950"
    RED      = "#f85149"
    BLUE     = "#58a6ff"
    ORANGE   = "#d29922"
    PURPLE   = "#bc8cff"
    PRODUCT_COLORS = [BLUE, ORANGE, PURPLE, GREEN, RED, "#ff7b72", "#79c0ff"]

    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  BORDER,
    })

    # ── Gather all products ──────────────────────────────────────────────
    products = sorted({r.product for dr in day_results for r in dr.records})
    n_products = len(products)
    pcols = {p: PRODUCT_COLORS[i % len(PRODUCT_COLORS)] for i, p in enumerate(products)}

    # ── Build merged time series ─────────────────────────────────────────
    # timestamps are stitched (offset across days so they increase monotonically)
    stitched: dict[str, dict] = {p: {"ts": [], "mid": [], "pnl": [], "pos": []} for p in products}
    all_ts_total: dict[int, float] = {}  # stitched ts -> total PnL
    my_trades_merged: list[TradeRecord] = []

    ts_offset = 0
    pnl_offset: dict[str, float] = {p: 0.0 for p in products}

    for di, dr in enumerate(day_results):
        by_ts: dict[int, dict[str, TimestepRecord]] = defaultdict(dict)
        for r in dr.records:
            by_ts[r.timestamp][r.product] = r

        last_ts_in_day = max(by_ts.keys()) if by_ts else 0

        for ts in sorted(by_ts.keys()):
            sts = ts + ts_offset
            total_pnl = 0.0
            for p in products:
                rec = by_ts[ts].get(p)
                if rec:
                    adj_pnl = rec.pnl + pnl_offset[p]
                    stitched[p]["ts"].append(sts)
                    stitched[p]["mid"].append(rec.mid_price)
                    stitched[p]["pnl"].append(adj_pnl)
                    stitched[p]["pos"].append(rec.position)
                    total_pnl += adj_pnl
            all_ts_total[sts] = total_pnl

        for t in dr.my_trades:
            my_trades_merged.append(TradeRecord(t.timestamp + ts_offset, t.product, t.price, t.quantity))

        if merge_pnl and di + 1 < len(day_results):
            for p in products:
                last_recs = [r for r in dr.records if r.timestamp == last_ts_in_day and r.product == p]
                if last_recs:
                    pnl_offset[p] += last_recs[0].pnl

        ts_offset += last_ts_in_day + 100

    # ── Layout: 3 rows × 2 cols ──────────────────────────────────────────
    n_rows = 2 + n_products   # PnL total, positions, one mid-price per product
    fig = plt.figure(figsize=(18, 4 + 3 * n_rows), facecolor=BG)
    fig.suptitle("IMC Prosperity 4 — Backtest Results", fontsize=16,
                 color=TEXT, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.45,
                           left=0.07, right=0.97, top=0.97, bottom=0.04)

    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    def style_ax(ax, title):
        ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=6)
        ax.grid(True)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    comma_fmt = FuncFormatter(lambda x, _: f"{x:,.0f}")

    # ── Row 0: Total PnL ─────────────────────────────────────────────────
    ax = axes[0]
    style_ax(ax, "Total PnL (mark-to-market)")
    ts_sorted = sorted(all_ts_total)
    vals = [all_ts_total[t] for t in ts_sorted]
    ax.plot(ts_sorted, vals, color=GREEN, linewidth=1.5, label="Total")
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.fill_between(ts_sorted, 0, vals,
                    where=[v >= 0 for v in vals], alpha=0.15, color=GREEN)
    ax.fill_between(ts_sorted, 0, vals,
                    where=[v < 0 for v in vals],  alpha=0.15, color=RED)
    ax.yaxis.set_major_formatter(comma_fmt)

    # Per-product PnL on same axis
    for p in products:
        d = stitched[p]
        ax.plot(d["ts"], d["pnl"], linewidth=0.8, color=pcols[p],
                alpha=0.6, linestyle="--", label=p)
    ax.legend(fontsize=7, ncol=len(products)+1)

    # ── Row 1: Positions ─────────────────────────────────────────────────
    ax = axes[1]
    style_ax(ax, "Positions")
    for p in products:
        d = stitched[p]
        ax.step(d["ts"], d["pos"], where="post", color=pcols[p], linewidth=1.2, label=p)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.legend(fontsize=7)

    # Day boundary lines on all axes
    if len(day_results) > 1:
        boundary_ts = []
        _off = 0
        for di, dr in enumerate(day_results[:-1]):
            by_ts2 = {r.timestamp for r in dr.records}
            last = max(by_ts2) if by_ts2 else 0
            boundary_ts.append(_off + last + 50)
            _off += last + 100
        for bts in boundary_ts:
            for ax in axes:
                ax.axvline(bts, color=MUTED, linewidth=0.6, linestyle=":", alpha=0.6)

    # ── Rows 2+: Mid-price per product with trades overlaid ───────────────
    for pi, p in enumerate(products):
        ax = axes[2 + pi]
        style_ax(ax, f"{p} — Mid Price + Your Trades")
        d = stitched[p]
        ax.plot(d["ts"], d["mid"], color=pcols[p], linewidth=1.0, label="mid price")

        buys  = [t for t in my_trades_merged if t.product == p and t.quantity > 0]
        sells = [t for t in my_trades_merged if t.product == p and t.quantity < 0]

        if buys:
            ax.scatter([t.timestamp for t in buys],  [t.price for t in buys],
                       marker="^", color=GREEN, s=40, zorder=5, label="buy")
        if sells:
            ax.scatter([t.timestamp for t in sells], [t.price for t in sells],
                       marker="v", color=RED,   s=40, zorder=5, label="sell")
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(comma_fmt)

    # ── Stats table (text box) ───────────────────────────────────────────
    m = metrics
    stats_lines = [
        f"Final PnL:  {_fmt(m['final_pnl'], True):>12}",
        f"Sharpe:     {_fmt(m['sharpe_ratio']):>12}",
        f"Ann.Sharpe: {_fmt(m['annualized_sharpe']):>12}",
        f"Sortino:    {_fmt(m['sortino_ratio']):>12}",
        f"MaxDD(abs): {_fmt(m['max_drawdown_abs'], True):>12}",
        f"MaxDD(%):   {_fmt(m['max_drawdown_pct']):>12}",
        f"Calmar:     {_fmt(m['calmar_ratio']):>12}",
    ]
    fig.text(0.985, 0.995, "\n".join(stats_lines),
             ha="right", va="top", fontsize=8, color=TEXT,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=BORDER))

    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("\nChart saved to backtest_results.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def load_trader(path: Path):
    sys.path.insert(0, str(path.parent))
    # Inject datamodel so trader's `from datamodel import ...` resolves
    try:
        from prosperity4bt import datamodel as _dm
        sys.modules["datamodel"] = _dm
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location("trader_module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trader()


def main():
    parser = argparse.ArgumentParser(
        description="IMC Prosperity 4 Backtester + Visualiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("algorithm", type=Path, help="Path to trader .py file")
    parser.add_argument("round", type=int, nargs="?", default=1,
                        help="Round number to backtest (default: 1)")
    parser.add_argument("--data", type=Path, default=Path("TUTORIAL_ROUND_1"),
                        help="Path to data directory (default: TUTORIAL_ROUND_1)")
    parser.add_argument("--days", nargs="+", type=int,
                        help="Specific day numbers to test, e.g. --days -2 -1 0")
    parser.add_argument("--merge-pnl", action="store_true",
                        help="Carry PnL forward across days (cumulative)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip the matplotlib visualiser")
    parser.add_argument("--verbose", action="store_true",
                        help="Print trader stdout logs")
    parser.add_argument("--limit", action="append", default=[],
                        metavar="PRODUCT:N",
                        help="Override position limit, e.g. --limit EMERALDS:20")
    args = parser.parse_args()

    # Parse limit overrides
    limits_override = None
    if args.limit:
        limits_override = {}
        for item in args.limit:
            sym, num = item.split(":", 1)
            limits_override[sym.strip()] = int(num.strip())

    # Discover days
    round_num = args.round
    if args.days:
        day_nums = args.days
    else:
        day_nums = []
        for d in range(-5, 10):
            f = args.data / f"round{round_num}" / f"prices_round_{round_num}_day_{d}.csv"
            if f.exists():
                day_nums.append(d)

    if not day_nums:
        print(f"No data found in {args.data}/round{round_num}/")
        print("Expected files like: prices_round_1_day_-1.csv")
        sys.exit(1)

    print(f"Found days: {day_nums}")
    print(f"Loading trader from {args.algorithm}...")
    trader = load_trader(args.algorithm)

    day_results: list[DayResult] = []

    for day_num in day_nums:
        data = load_day(args.data, round_num, day_num)
        if data is None:
            print(f"  Skipping day {day_num} — file not found")
            continue

        print(f"\nBacktesting round {round_num} day {day_num}:")
        result = run_day(trader, data, limits_override, args.verbose)
        day_results.append(result)

        # Day summary
        pnl = result.final_pnl()
        total = sum(pnl.values())
        for p, v in sorted(pnl.items()):
            print(f"  {p}: {v:,.0f}")
        print(f"  Total profit: {total:,.0f}")

    if not day_results:
        print("No days ran successfully.")
        sys.exit(1)

    if len(day_results) > 1:
        print("\nProfit summary:")
        grand = 0.0
        for dr in day_results:
            p = sum(dr.final_pnl().values())
            print(f"  Round {dr.round_num} day {dr.day_num}: {p:,.0f}")
            grand += p
        print(f"  Total profit: {grand:,.0f}")

    m = risk_metrics(day_results)
    print("\nRisk metrics (full trading period):")
    print_metrics(m)

    if not args.no_vis:
        visualise(day_results, m, args.merge_pnl)


if __name__ == "__main__":
    main()