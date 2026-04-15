import json
from typing import Any
from datamodel import (
    Observation,
    Order,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


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