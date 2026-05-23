from AlgorithmImports import *

class AdaptiveMomentumElite(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2026, 3, 1)
        self.SetCash(10000)

        self.UniverseSettings.Resolution = Resolution.Daily

        # ---------------------------------------------------
        # MARKET
        # ---------------------------------------------------

        self.spy = self.AddEquity(
            "SPY",
            Resolution.Daily
        ).Symbol

        self.SetBenchmark("SPY")

        self.spy_ema200 = self.EMA(
            self.spy,
            200,
            Resolution.Daily
        )

        self.spy_roc20 = self.ROC(
            self.spy,
            20,
            Resolution.Daily
        )

        self.spy_atr = self.ATR(
            self.spy,
            14,
            MovingAverageType.Simple,
            Resolution.Daily
        )

        # ---------------------------------------------------
        # UNIVERSE
        # ---------------------------------------------------

        self.AddUniverse(self.CoarseSelection)

        self.data = {}

        self.last_rebalance = None

        self.performance_multiplier = 1.0

        self.max_positions_neutral = 5
        self.max_positions_bull = 7

        self.rank_weights_neutral = [
            0.36,
            0.26,
            0.18,
            0.12,
            0.08
        ]

        self.rank_weights_bull = [
            0.42,
            0.24,
            0.14,
            0.08,
            0.05,
            0.04,
            0.03
        ]

        self.SetWarmUp(220)

    def CoarseSelection(self, coarse):

        filtered = [
            x for x in coarse
            if x.HasFundamentalData
            and x.Price > 20
            and x.DollarVolume > 100000000
        ]

        top = sorted(
            filtered,
            key=lambda x: x.DollarVolume,
            reverse=True
        )[:100]

        return [x.Symbol for x in top]

    def OnSecuritiesChanged(self, changes):

        for sec in changes.AddedSecurities:

            symbol = sec.Symbol

            if symbol == self.spy:
                continue

            if symbol in self.data:
                continue

            self.data[symbol] = {

                "roc20": self.ROC(
                    symbol,
                    20,
                    Resolution.Daily
                ),

                "roc90": self.ROC(
                    symbol,
                    90,
                    Resolution.Daily
                ),

                "ema100": self.EMA(
                    symbol,
                    100,
                    Resolution.Daily
                ),

                "atr": self.ATR(
                    symbol,
                    14,
                    MovingAverageType.Simple,
                    Resolution.Daily
                ),

                "highest": 0.0,

                "stop": None,

                "last_roc20": 0.0
            }

        for sec in changes.RemovedSecurities:

            symbol = sec.Symbol

            if not self.Portfolio[symbol].Invested:
                self.data.pop(symbol, None)

    def OnData(self, data):

        if self.IsWarmingUp:
            return

        self.ManageRisk()

        # ---------------------------------------------------
        # PERFORMANCE MULTIPLIER
        # ---------------------------------------------------

        for symbol in self.Portfolio.Keys:

            if not self.Portfolio[symbol].Invested:
                continue

            pnl = self.Portfolio[
                symbol
            ].UnrealizedProfit

            if pnl > 0:
                self.performance_multiplier *= 1.03
            else:
                self.performance_multiplier *= 0.97

        self.performance_multiplier = max(
            0.7,
            min(
                self.performance_multiplier,
                1.8
            )
        )

        # ---------------------------------------------------
        # MARKET REGIME
        # ---------------------------------------------------

        breadth_ready = 0
        breadth_above = 0

        for symbol, sd in self.data.items():

            if not (
                sd["roc20"].IsReady
                and sd["ema100"].IsReady
            ):
                continue

            price = self.Securities[
                symbol
            ].Price

            if price <= 0:
                continue

            breadth_ready += 1

            if (
                price >
                sd["ema100"].Current.Value
            ):
                breadth_above += 1

        breadth = (
            breadth_above / breadth_ready
            if breadth_ready > 0
            else 0
        )

        bull_mode = (
            self.Securities[self.spy].Price >
            self.spy_ema200.Current.Value
            and breadth > 0.55
            and self.spy_roc20.Current.Value > 0
        )

        rebalance_days = 3 if bull_mode else 7

        if self.last_rebalance:

            if (
                self.Time - self.last_rebalance
            ).days < rebalance_days:

                return

        self.last_rebalance = self.Time

        # ---------------------------------------------------
        # CANDIDATES
        # ---------------------------------------------------

        candidates = []

        spy_roc20 = (
            self.spy_roc20.Current.Value
        )

        for symbol, sd in self.data.items():

            if not (
                sd["roc20"].IsReady
                and sd["roc90"].IsReady
                and sd["ema100"].IsReady
                and sd["atr"].IsReady
            ):
                continue

            price = self.Securities[
                symbol
            ].Price

            if price <= 0:
                continue

            if (
                price <
                sd["ema100"].Current.Value
            ):
                continue

            roc20 = (
                sd["roc20"].Current.Value
            )

            roc90 = (
                sd["roc90"].Current.Value
            )

            if bull_mode:

                if roc20 < 0.03:
                    continue

                if roc90 < 0.06:
                    continue

            else:

                if roc20 < 0.05:
                    continue

                if roc90 < 0.10:
                    continue

            rs = roc20 - spy_roc20

            accel = max(
                0,
                roc20 - sd["last_roc20"]
            )

            score = (
                0.45 * roc20
                + 0.25 * roc90
                + 0.20 * rs
                + 0.10 * accel
            )

            candidates.append(
                (
                    symbol,
                    score
                )
            )

        if not candidates:
            return

        ranked = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True
        )

        max_positions = (
            self.max_positions_bull
            if bull_mode
            else self.max_positions_neutral
        )

        selected = [
            x[0]
            for x in ranked[:max_positions]
        ]

        # ---------------------------------------------------
        # LEADER PERSISTENCE
        # ---------------------------------------------------

        top_15 = [
            x[0]
            for x in ranked[:15]
        ]

        for symbol in list(self.Portfolio.Keys):

            if not self.Portfolio[symbol].Invested:
                continue

            if symbol not in top_15:

                # FIX: Ensure the symbol exists in self.data before accessing
                if symbol not in self.data:
                    continue

                price = self.Securities[
                    symbol
                ].Price

                ema100 = self.data[
                    symbol
                ]["ema100"].Current.Value

                if price < ema100:
                    self.Liquidate(symbol)

        # ---------------------------------------------------
        # POSITIONING
        # ---------------------------------------------------

        rank_weights = (
            self.rank_weights_bull
            if bull_mode
            else self.rank_weights_neutral
        )

        for i, symbol in enumerate(selected):

            sd = self.data[symbol]

            weight = (
                rank_weights[i]
                * self.performance_multiplier
            )

            # PYRAMID LEADERS
            if (
                bull_mode
                and i < 2
                and self.Securities[symbol].Price >
                sd["highest"]
            ):
                weight *= 1.15

            weight = min(weight, 0.45)

            self.SetHoldings(
                symbol,
                weight
            )

            price = self.Securities[
                symbol
            ].Price

            atr = sd["atr"].Current.Value

            if (
                sd["stop"] is None
            ):

                sd["highest"] = price

                stop_mult = (
                    3.25
                    if bull_mode
                    else 2.5
                )

                sd["stop"] = (
                    price
                    - stop_mult * atr
                )

            sd["last_roc20"] = roc20

    def ManageRisk(self):

        bull_mode = (
            self.Securities[self.spy].Price >
            self.spy_ema200.Current.Value
        )

        for symbol, sd in self.data.items():

            if not self.Portfolio[symbol].Invested:
                continue

            if not sd["atr"].IsReady:
                continue

            price = self.Securities[
                symbol
            ].Price

            atr = (
                sd["atr"]
                .Current.Value
            )

            if price > sd["highest"]:

                sd["highest"] = price

                stop_mult = (
                    3.25
                    if bull_mode
                    else 2.5
                )

                new_stop = (
                    price
                    - stop_mult * atr
                )

                if sd["stop"] is None:

                    sd["stop"] = new_stop

                else:

                    sd["stop"] = max(
                        sd["stop"],
                        new_stop
                    )

            if (
                sd["stop"] is not None
                and price < sd["stop"]
            ):

                self.Liquidate(symbol)

                sd["highest"] = 0
                sd["stop"] = None
