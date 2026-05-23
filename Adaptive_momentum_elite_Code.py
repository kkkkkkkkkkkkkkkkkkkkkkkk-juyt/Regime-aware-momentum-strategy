from AlgorithmImports import *

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False


class AdaptiveMomentumElite(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2026, 5, 1)
        self.SetCash(10000)

        self.UniverseSettings.Resolution = Resolution.Daily

        # =====================================================
        # MODEL SETTINGS
        # =====================================================

        self.lookback_days = 756
        self.horizon_days = 20

        self.max_training_symbols = 60
        self.min_training_rows = 1200

        self.feature_columns = [
            "roc20",
            "roc60",
            "roc90",
            "roc120",
            "ema50_dist",
            "ema100_dist",
            "rs20",
            "atr_pct",
            "rel_volume",
            "accel",
            "drawdown60",
            "volatility_expansion",
            "regime"
        ]

        self.model = None
        self.model_ready = False
        self.model_last_trained = None

        # =====================================================
        # MARKET
        # =====================================================

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

        # =====================================================
        # UNIVERSE
        # =====================================================

        self.AddUniverse(self.CoarseSelection)

        self.data = {}

        # FIXED: STORE COARSE DOLLAR VOLUME
        self.coarse_dollar_volume = {}

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

        self.SetWarmUp(260)

        # =====================================================
        # QUARTERLY MODEL TRAINING
        # =====================================================

        self.Schedule.On(
            self.DateRules.MonthStart(self.spy, 3),
            self.TimeRules.AfterMarketOpen(self.spy, 30),
            self.TrainPredictionModel
        )

    # =====================================================
    # COARSE SELECTION
    # =====================================================

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

        # FIXED: STORE DOLLAR VOLUME
        self.coarse_dollar_volume = {
            x.Symbol: float(x.DollarVolume)
            for x in top
        }

        return [x.Symbol for x in top]

    # =====================================================
    # SECURITIES CHANGED
    # =====================================================

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

                "roc60": self.ROC(
                    symbol,
                    60,
                    Resolution.Daily
                ),

                "roc90": self.ROC(
                    symbol,
                    90,
                    Resolution.Daily
                ),

                "roc120": self.ROC(
                    symbol,
                    120,
                    Resolution.Daily
                ),

                "ema50": self.EMA(
                    symbol,
                    50,
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

                "close_window": RollingWindow[float](130),

                "volume_window": RollingWindow[float](30),

                "roc20_window": RollingWindow[float](25),

                "atr_window": RollingWindow[float](30),

                "highest": 0.0,

                "stop": None,

                "last_roc20": 0.0,

                "last_volume": 0.0
            }

        for sec in changes.RemovedSecurities:

            symbol = sec.Symbol

            if not self.Portfolio[symbol].Invested:
                self.data.pop(symbol, None)

    # =====================================================
    # WARMUP FINISHED
    # =====================================================

    def OnWarmupFinished(self):

        self.TrainPredictionModel()

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def OnData(self, data):

        if self.IsWarmingUp:
            return

        # =====================================================
        # UPDATE WINDOWS
        # =====================================================

        for symbol, bar in data.Bars.items():

            if symbol not in self.data:
                continue

            sd = self.data[symbol]

            sd["close_window"].Add(float(bar.Close))

            sd["volume_window"].Add(float(bar.Volume))

            if sd["roc20"].IsReady:

                sd["roc20_window"].Add(
                    float(sd["roc20"].Current.Value)
                )

            if sd["atr"].IsReady:

                sd["atr_window"].Add(
                    float(sd["atr"].Current.Value)
                )

            sd["last_volume"] = float(bar.Volume)

        # =====================================================
        # RISK MANAGEMENT
        # =====================================================

        self.ManageRisk()

        # =====================================================
        # PERFORMANCE MULTIPLIER
        # =====================================================

        for symbol in self.Portfolio.Keys:

            if not self.Portfolio[symbol].Invested:
                continue

            pnl = self.Portfolio[symbol].UnrealizedProfit

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

        # =====================================================
        # MARKET REGIME
        # =====================================================

        breadth_ready = 0
        breadth_above = 0

        for symbol, sd in self.data.items():

            if not (
                sd["roc20"].IsReady
                and sd["ema100"].IsReady
            ):
                continue

            price = self.Securities[symbol].Price

            if price <= 0:
                continue

            breadth_ready += 1

            if price > sd["ema100"].Current.Value:
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

        # =====================================================
        # CANDIDATES
        # =====================================================

        candidates = []

        prob_cache = {}

        spy_roc20 = (
            self.spy_roc20.Current.Value
        )

        for symbol, sd in self.data.items():

            if not (
                sd["roc20"].IsReady
                and sd["roc60"].IsReady
                and sd["roc90"].IsReady
                and sd["roc120"].IsReady
                and sd["ema50"].IsReady
                and sd["ema100"].IsReady
                and sd["atr"].IsReady
            ):
                continue

            price = self.Securities[symbol].Price

            if price <= 0:
                continue

            if (
                price <
                sd["ema100"].Current.Value
            ):
                continue

            roc20 = sd["roc20"].Current.Value
            roc90 = sd["roc90"].Current.Value

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

            accel = 0.0

            if sd["roc20_window"].Count > 20:

                accel = (
                    float(sd["roc20_window"][0])
                    - float(sd["roc20_window"][20])
                )

            # =====================================================
            # MOMENTUM DOMINANT SCORE
            # =====================================================

            base_score = (
                0.50 * roc20
                + 0.25 * roc90
                + 0.15 * rs
                + 0.10 * max(0, accel)
            )

            # =====================================================
            # ML PROBABILITY
            # =====================================================

            ml_prob = self.GetPredictionProbability(
                symbol,
                sd
            )

            prob_cache[symbol] = ml_prob

            # =====================================================
            # SMALL ML TILT ONLY
            # =====================================================

            adjusted_score = (
                base_score
                * (
                    1.0
                    + 0.15 * (ml_prob - 0.5)
                )
            )

            candidates.append(
                (
                    symbol,
                    adjusted_score
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

        # =====================================================
        # LEADER PERSISTENCE
        # =====================================================

        top_15 = [
            x[0]
            for x in ranked[:15]
        ]

        for symbol in list(self.Portfolio.Keys):

            if not self.Portfolio[symbol].Invested:
                continue

            if symbol not in top_15:

                if symbol not in self.data:
                    continue

                price = self.Securities[symbol].Price

                ema100 = self.data[
                    symbol
                ]["ema100"].Current.Value

                if price < ema100:

                    self.Liquidate(symbol)

                    self.data[symbol]["highest"] = 0

                    self.data[symbol]["stop"] = None

        # =====================================================
        # POSITIONING
        # =====================================================

        rank_weights = (
            self.rank_weights_bull
            if bull_mode
            else self.rank_weights_neutral
        )

        for i, symbol in enumerate(selected):

            sd = self.data[symbol]

            ml_prob = prob_cache.get(
                symbol,
                0.5
            )

            weight = (
                rank_weights[i]
                * self.performance_multiplier
            )

            # =====================================================
            # MILD ML POSITION TILT
            # =====================================================

            confidence_multiplier = (
                1.0
                + ((ml_prob - 0.5) * 0.20)
            )

            weight *= confidence_multiplier

            # =====================================================
            # PYRAMID LEADERS
            # =====================================================

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

            price = self.Securities[symbol].Price

            atr = sd["atr"].Current.Value

            if sd["stop"] is None:

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

            sd["last_roc20"] = (
                sd["roc20"].Current.Value
            )

    # =====================================================
    # RISK MANAGEMENT
    # =====================================================

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

            price = self.Securities[symbol].Price

            atr = sd["atr"].Current.Value

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

            # =====================================================
            # ATR EXIT ONLY
            # =====================================================

            if (
                sd["stop"] is not None
                and price < sd["stop"]
            ):

                self.Liquidate(symbol)

                sd["highest"] = 0

                sd["stop"] = None

    # =====================================================
    # HELPERS
    # =====================================================

    def _safe_mean(self, window, length):

        if window.Count < length:
            return None

        total = 0.0

        for i in range(length):

            total += float(window[i])

        return total / length

    # =====================================================
    # LIVE FEATURES
    # =====================================================

    def _build_live_feature_vector(
        self,
        symbol,
        sd
    ):

        price = self.Securities[symbol].Price

        if price <= 0:
            return None

        if not (
            sd["roc20"].IsReady
            and sd["roc60"].IsReady
            and sd["roc90"].IsReady
            and sd["roc120"].IsReady
            and sd["ema50"].IsReady
            and sd["ema100"].IsReady
            and sd["atr"].IsReady
            and self.spy_ema200.IsReady
            and self.spy_roc20.IsReady
        ):
            return None

        roc20 = float(sd["roc20"].Current.Value)

        roc60 = float(sd["roc60"].Current.Value)

        roc90 = float(sd["roc90"].Current.Value)

        roc120 = float(sd["roc120"].Current.Value)

        ema50 = float(sd["ema50"].Current.Value)

        ema100 = float(sd["ema100"].Current.Value)

        ema50_dist = (
            price / ema50 - 1.0
        )

        ema100_dist = (
            price / ema100 - 1.0
        )

        rs20 = (
            roc20
            - float(self.spy_roc20.Current.Value)
        )

        atr_pct = (
            float(sd["atr"].Current.Value)
            / price
        )

        rel_volume = 1.0

        avg_vol = self._safe_mean(
            sd["volume_window"],
            20
        )

        if (
            avg_vol is not None
            and avg_vol > 0
            and sd["last_volume"] > 0
        ):

            rel_volume = (
                float(sd["last_volume"])
                / float(avg_vol)
            )

        accel = 0.0

        if sd["roc20_window"].Count > 20:

            accel = (
                float(sd["roc20_window"][0])
                - float(sd["roc20_window"][20])
            )

        drawdown60 = 0.0

        if sd["close_window"].Count >= 60:

            recent_high = max(
                float(sd["close_window"][i])
                for i in range(60)
            )

            if recent_high > 0:

                drawdown60 = (
                    price / recent_high - 1.0
                )

        volatility_expansion = 1.0

        avg_atr = self._safe_mean(
            sd["atr_window"],
            20
        )

        if (
            avg_atr is not None
            and avg_atr > 0
        ):

            volatility_expansion = (
                float(sd["atr"].Current.Value)
                / float(avg_atr)
            )

        regime = (
            1.0
            if self.Securities[self.spy].Price >
            self.spy_ema200.Current.Value
            else 0.0
        )

        features = np.array([
            roc20,
            roc60,
            roc90,
            roc120,
            ema50_dist,
            ema100_dist,
            rs20,
            atr_pct,
            rel_volume,
            accel,
            drawdown60,
            volatility_expansion,
            regime
        ])

        return features

    # =====================================================
    # ML PROBABILITY
    # =====================================================

    def GetPredictionProbability(
        self,
        symbol,
        sd
    ):

        if (
            not self.model_ready
            or self.model is None
        ):
            return 0.5

        features = self._build_live_feature_vector(
            symbol,
            sd
        )

        if features is None:
            return 0.5

        try:

            X = np.array([features])

            prob = float(
                self.model.predict_proba(X)[0, 1]
            )

            if (
                np.isnan(prob)
                or np.isinf(prob)
            ):
                return 0.5

            return max(
                0.0,
                min(1.0, prob)
            )

        except:

            return 0.5

    # =====================================================
    # TRAINING FRAME
    # =====================================================

    def _prepare_training_frame(
        self,
        symbol_df,
        spy_df
    ):

        if (
            symbol_df.empty
            or spy_df.empty
        ):
            return pd.DataFrame()

        df = symbol_df[
            ["close", "high", "low", "volume"]
        ].copy()

        spy = spy_df[
            ["close"]
        ].copy()

        spy = spy.rename(
            columns={
                "close": "spy_close"
            }
        )

        joined = df.join(
            spy,
            how="inner"
        )

        if joined.empty:
            return pd.DataFrame()

        joined["roc20"] = joined["close"].pct_change(20)

        joined["roc60"] = joined["close"].pct_change(60)

        joined["roc90"] = joined["close"].pct_change(90)

        joined["roc120"] = joined["close"].pct_change(120)

        joined["ema50"] = (
            joined["close"]
            .ewm(span=50, adjust=False)
            .mean()
        )

        joined["ema100"] = (
            joined["close"]
            .ewm(span=100, adjust=False)
            .mean()
        )

        joined["ema50_dist"] = (
            joined["close"]
            / joined["ema50"]
            - 1.0
        )

        joined["ema100_dist"] = (
            joined["close"]
            / joined["ema100"]
            - 1.0
        )

        prev_close = joined["close"].shift(1)

        tr1 = (
            joined["high"]
            - joined["low"]
        )

        tr2 = (
            joined["high"]
            - prev_close
        ).abs()

        tr3 = (
            joined["low"]
            - prev_close
        ).abs()

        joined["atr"] = (
            pd.concat(
                [tr1, tr2, tr3],
                axis=1
            )
            .max(axis=1)
            .rolling(14)
            .mean()
        )

        joined["atr_pct"] = (
            joined["atr"]
            / joined["close"]
        )

        joined["atr_avg20"] = (
            joined["atr"]
            .rolling(20)
            .mean()
        )

        joined["volatility_expansion"] = (
            joined["atr"]
            / joined["atr_avg20"]
        )

        joined["rel_volume"] = (
            joined["volume"]
            / joined["volume"]
            .rolling(20)
            .mean()
        )

        joined["accel"] = (
            joined["roc20"]
            - joined["roc20"].shift(20)
        )

        joined["drawdown60"] = (
            joined["close"]
            / joined["close"]
            .rolling(60)
            .max()
            - 1.0
        )

        joined["spy_ema200"] = (
            joined["spy_close"]
            .ewm(span=200, adjust=False)
            .mean()
        )

        joined["regime"] = (
            joined["spy_close"]
            > joined["spy_ema200"]
        ).astype(float)

        joined["rs20"] = (
            joined["roc20"]
            - joined["spy_close"]
            .pct_change(20)
        )

        joined["future_stock_ret"] = (
            joined["close"]
            .shift(-self.horizon_days)
            / joined["close"]
            - 1.0
        )

        joined["future_spy_ret"] = (
            joined["spy_close"]
            .shift(-self.horizon_days)
            / joined["spy_close"]
            - 1.0
        )

        joined["label"] = (
            joined["future_stock_ret"]
            > joined["future_spy_ret"]
        ).astype(int)

        joined = joined.replace(
            [np.inf, -np.inf],
            np.nan
        )

        joined = joined.dropna()

        if joined.empty:
            return pd.DataFrame()

        cols = (
            self.feature_columns
            + ["label"]
        )

        return joined[cols]

    # =====================================================
    # MODEL TRAINING
    # =====================================================

    def TrainPredictionModel(self):

        if self.IsWarmingUp:
            return

        try:

            symbols = [
                s for s in self.data.keys()
                if s != self.spy
                and self.Securities[s].Price > 0
            ]

            if len(symbols) == 0:
                return

            # FIXED: USE STORED COARSE DOLLAR VOLUME
            symbols = sorted(
                symbols,
                key=lambda s: self.coarse_dollar_volume.get(s, 0),
                reverse=True
            )[:self.max_training_symbols]

            history_symbols = (
                symbols + [self.spy]
            )

            lookback = (
                self.lookback_days
                + self.horizon_days
                + 130
            )

            history = self.History(
                history_symbols,
                lookback,
                Resolution.Daily
            )

            if (
                history is None
                or len(history) == 0
            ):
                return

            frames = []

            for symbol in symbols:

                try:

                    sym_hist = history.loc[
                        symbol
                    ].copy()

                    spy_hist = history.loc[
                        self.spy
                    ].copy()

                    frame = self._prepare_training_frame(
                        sym_hist,
                        spy_hist
                    )

                    if not frame.empty:

                        frames.append(frame)

                except:
                    continue

            if len(frames) == 0:
                return

            train_df = pd.concat(
                frames,
                ignore_index=True
            )

            train_df = train_df.dropna()

            if (
                len(train_df)
                < self.min_training_rows
            ):
                return

            train_df = train_df.sample(
                frac=1,
                random_state=42
            )

            X = train_df[
                self.feature_columns
            ].values

            y = train_df["label"].values

            if len(np.unique(y)) < 2:
                return

            split_idx = int(
                len(train_df) * 0.8
            )

            X_train = X[:split_idx]

            y_train = y[:split_idx]

            if XGBOOST_AVAILABLE:

                self.model = xgb.XGBClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    min_child_weight=8,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=42
                )

            else:

                self.model = GradientBoostingClassifier(
                    random_state=42
                )

            self.model.fit(
                X_train,
                y_train
            )

            self.model_ready = True

            self.model_last_trained = self.Time

            self.Debug(
                f"MODEL TRAINED: {self.Time}"
            )

        except Exception as e:

            self.Debug(
                f"MODEL TRAINING ERROR: {str(e)}"
            )

            self.model_ready = False
