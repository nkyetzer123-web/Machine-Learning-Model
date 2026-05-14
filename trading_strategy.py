"""
trading_strategy.py  —  Stage 2
=================================
Hybrid ML Trading Signal Generator + Automated Paper Portfolio
--------------------------------------------------------------
Runs automatically via GitHub Actions every weekday at 7am UK time.

Modes:
    python3 trading_strategy.py                  # manage portfolio + generate signals
    python3 trading_strategy.py --backtest       # 5-year backtest
    python3 trading_strategy.py --signals-only   # signals only, no portfolio changes

Files produced:
    signals.json           — today's ML signals
    portfolio_state.json   — open/closed positions + running P&L
    daily_log.json         — append-only log of every decision made

Requirements:
    pip3 install yfinance scikit-learn pandas numpy
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# UNIVERSE
# ─────────────────────────────────────────────────────────────

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM",  "V",    "JNJ",  "WMT",   "XOM",  "LLY",  "AVGO",
    "MA",   "PG",   "UNH",  "HD",    "MRK",  "CVX",  "ABBV",
    "COST", "PEP",  "ADBE", "CRM",   "AMD",  "TXN",  "NFLX",
    "AMGN", "QCOM", "GS",   "MS",    "CAT",  "DE",   "BA",
]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
ETFS   = ["SPY", "QQQ", "VTI", "VUSA.L", "ISF.L", "IWDA.AS"]

ASSET_TYPE = {}
for t in STOCKS: ASSET_TYPE[t] = "stock"
for t in CRYPTO:  ASSET_TYPE[t] = "crypto"
for t in ETFS:    ASSET_TYPE[t] = "etf"

ALL_TICKERS = STOCKS + CRYPTO + ETFS
SPY_TICKER  = "SPY"

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────

FORWARD            = 20
STOP_PCT           = 0.03
TP_PCT             = 0.06
BUY_THRESH         = 0.65
WATCH_THRESH       = 0.55
SHORT_WATCH_THRESH = 0.40
SHORT_THRESH       = 0.30
MAX_POS            = 5
START_CAP          = 1000.0
RISK_PER           = 0.01
BACKTEST_YRS       = 5
TRAIN_MONTHS       = 24
TEST_MONTHS        = 3
PURGE_DAYS         = FORWARD
MIN_SHARES_STOCK   = 1
MIN_UNITS_CRYPTO   = 0.00001

# File paths (all in same directory as script)
PORTFOLIO_FILE = "portfolio_state.json"
SIGNALS_FILE   = "signals.json"
LOG_FILE       = "daily_log.json"
BACKTEST_FILE  = "backtest_results.json"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def calc_units(ticker, risk_amt, stop_dist):
    if stop_dist <= 0: return 0
    raw   = risk_amt / stop_dist
    atype = ASSET_TYPE.get(ticker, "stock")
    if atype == "crypto":
        return max(MIN_UNITS_CRYPTO, round(raw, 8))
    return max(MIN_SHARES_STOCK, int(raw))

def get_ticker_data(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw

def classify_signal(prob, regime_bull, atype, vol_ok, vol_r):
    if regime_bull:
        if prob >= BUY_THRESH and vol_ok and vol_r > 0.8: return "BUY"
        if prob >= WATCH_THRESH: return "WATCH"
        return "SKIP"
    else:
        if atype != "crypto":
            if prob < SHORT_THRESH:       return "SHORT"
            if prob < SHORT_WATCH_THRESH: return "SHORT WATCH"
        return "SKIP"


# ─────────────────────────────────────────────────────────────
# PORTFOLIO STATE  (read / write / helpers)
# ─────────────────────────────────────────────────────────────

def load_portfolio():
    """Load portfolio_state.json or create a fresh one."""
    if Path(PORTFOLIO_FILE).exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "start_capital":  START_CAP,
        "cash":           START_CAP,
        "open_positions": {},
        "closed_trades":  [],
        "created":        today_str(),
        "last_run":       None,
    }

def save_portfolio(state):
    state["last_run"] = now_str()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"    💾  Saved {PORTFOLIO_FILE}")

def portfolio_value(state, current_prices):
    """Total portfolio value = cash + mark-to-market of open positions."""
    val = state["cash"]
    for ticker, pos in state["open_positions"].items():
        price = current_prices.get(ticker)
        if price is None: continue
        if pos["direction"] == "long":
            val += (price - pos["entry_price"]) * pos["units"] + pos["entry_price"] * pos["units"]
        else:
            val += pos["entry_price"] * pos["units"] + (pos["entry_price"] - price) * pos["units"]
    return val


# ─────────────────────────────────────────────────────────────
# DAILY LOG  (append-only)
# ─────────────────────────────────────────────────────────────

def append_log(entry):
    """Append one day's log entry to daily_log.json."""
    log = []
    if Path(LOG_FILE).exists():
        with open(LOG_FILE) as f:
            try:
                log = json.load(f)
            except Exception:
                log = []
    log.insert(0, entry)   # newest first
    log = log[:365]        # keep last 365 entries
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"    📋  Updated {LOG_FILE}")


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def compute_features(df, spy):
    c     = df["Close"].squeeze()
    v     = df["Volume"].squeeze()
    spy_c = spy["Close"].squeeze().reindex(c.index).ffill()
    f     = pd.DataFrame(index=c.index)
    f["ret_5d"]         = c.pct_change(5)
    f["ret_20d"]        = c.pct_change(20)
    f["ret_60d"]        = c.pct_change(60)
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    f["above_sma50"]    = (c > sma50).astype(int)
    f["above_sma200"]   = (c > sma200).astype(int)
    f["golden_cross"]   = (sma50 > sma200).astype(int)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"]         = 100 - (100 / (1 + gain / (loss + 1e-9)))
    f["vol_ratio"]      = v / (v.rolling(20).mean() + 1e-9)
    f["volatility_20d"] = c.pct_change().rolling(20).std()
    f["rs_vs_spy_20d"]  = c.pct_change(20) - spy_c.pct_change(20)
    f["regime_bull"]    = (spy_c > spy_c.rolling(200).mean()).astype(int)
    return f

def compute_target(df, spy, forward=FORWARD):
    c     = df["Close"].squeeze()
    spy_c = spy["Close"].squeeze().reindex(c.index).ffill()
    return (c.pct_change(forward).shift(-forward) > spy_c.pct_change(forward).shift(-forward)).astype(int).rename("target")


# ─────────────────────────────────────────────────────────────
# WALK-FORWARD WITH PURGING  (backtest only)
# ─────────────────────────────────────────────────────────────

def walk_forward_predict(X, y):
    results    = []
    dates      = X.index
    cursor     = pd.Timestamp(dates[0]) + pd.DateOffset(months=TRAIN_MONTHS)
    end        = pd.Timestamp(dates[-1])
    test_delta = pd.DateOffset(months=TEST_MONTHS)
    while cursor < end:
        test_end    = min(cursor + test_delta, end)
        train_mask  = dates < cursor
        test_mask   = (dates >= cursor) & (dates <= test_end)
        bound_idx   = np.searchsorted(dates, cursor)
        purge_mask  = np.zeros(len(dates), dtype=bool)
        purge_mask[max(0, bound_idx - PURGE_DAYS):bound_idx] = True
        clean_train = train_mask & ~purge_mask
        if clean_train.sum() < 60 or test_mask.sum() == 0:
            cursor += test_delta; continue
        m = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
        m.fit(X[clean_train], y[clean_train])
        for dt, p in zip(dates[test_mask], m.predict_proba(X[test_mask])[:, 1]):
            results.append((dt, float(p)))
        cursor += test_delta
    return results


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL  (for live signals + portfolio)
# ─────────────────────────────────────────────────────────────

def train_model(X, y):
    split = int(len(X) * 0.80)
    m = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20,
                               class_weight="balanced", random_state=42, n_jobs=-1)
    m.fit(X.iloc[:split], y.iloc[:split])
    acc = accuracy_score(y.iloc[split:], m.predict(X.iloc[split:]))
    return m, float(acc)


# ─────────────────────────────────────────────────────────────
# MAIN RUN  (signals + portfolio management)
# ─────────────────────────────────────────────────────────────

def run(tickers, signals_only=False):
    end   = datetime.today()
    start = end - timedelta(days=365 * 3 + 60)

    print(f"\n[1/5] Loading portfolio state …")
    state = load_portfolio()
    print(f"      Cash: £{state['cash']:.2f}  |  Open positions: {len(state['open_positions'])}")

    print(f"[2/5] Downloading SPY …")
    spy = get_ticker_data(SPY_TICKER, start, end)
    if spy.empty:
        print("ERROR: Cannot download SPY."); return

    regime_bull = bool(spy["Close"].iloc[-1] > spy["Close"].rolling(200).mean().iloc[-1])
    regime_str  = "Bullish" if regime_bull else "Bearish"
    print(f"      Market regime: {regime_str}")

    print(f"[3/5] Processing {len(tickers)} assets …")
    signals      = []
    all_accs     = []
    current_prices = {}
    asset_models   = {}   # ticker -> (model, feats_latest)

    for ticker in tickers:
        try:
            df = get_ticker_data(ticker, start, end)
            if len(df) < 220: continue
            feats  = compute_features(df, spy)
            target = compute_target(df, spy)
            combo  = feats.join(target).dropna()
            if len(combo) < 100: continue

            X = combo.drop(columns=["target"])
            y = combo["target"]
            model, acc = train_model(X, y)
            prob  = float(model.predict_proba(X.iloc[[-1]])[0][1])
            close = float(df["Close"].iloc[-1])
            rsi   = float(feats["rsi_14"].iloc[-1])
            vol_r = float(feats["vol_ratio"].iloc[-1])
            vol20 = float(feats["volatility_20d"].iloc[-1])
            atype = ASSET_TYPE.get(ticker, "stock")
            vol_ok = vol20 < 0.08 if atype == "crypto" else vol20 < 0.04

            all_accs.append(acc)
            current_prices[ticker] = close
            asset_models[ticker]   = (model, prob, vol_ok, vol_r, close, rsi, vol20, atype, acc)

            sig = classify_signal(prob, regime_bull, atype, vol_ok, vol_r)
            risk_amt  = state["cash"] * RISK_PER
            stop_dist = close * STOP_PCT
            units     = calc_units(ticker, risk_amt, stop_dist)
            units_str = f"{units:.8f}" if atype == "crypto" else str(int(units))

            signals.append({
                "ticker": ticker, "asset_type": atype, "signal": sig,
                "probability": round(prob, 4), "close": round(close, 4),
                "rsi": round(rsi, 1), "vol_ratio": round(vol_r, 2),
                "volatility": round(vol20, 4), "model_acc": round(acc, 4),
                "example_units": units_str,
            })
            print(f"    {ticker:12s} {sig:12s}  prob={prob:.2f}")

        except Exception as e:
            print(f"    WARNING: skipped {ticker} — {e}")

    # Save signals.json
    order = {"BUY": 0, "SHORT": 1, "WATCH": 2, "SHORT WATCH": 3, "SKIP": 4}
    signals.sort(key=lambda x: (order.get(x["signal"], 5),
                                -x["probability"] if x["signal"] in ("BUY","WATCH") else x["probability"]))
    signals_out = {
        "generated": now_str(), "regime": regime_str,
        "model_accuracy": round(float(np.mean(all_accs)) if all_accs else 0, 4),
        "tickers_run": len(tickers), "signals": signals,
    }
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals_out, f, indent=2)
    print(f"\n    Signals written to {SIGNALS_FILE}")

    if signals_only:
        print("    --signals-only flag set, skipping portfolio management.")
        return

    # ── Portfolio management ──
    print(f"\n[4/5] Managing portfolio …")
    log_actions  = []
    closed_today = []
    opened_today = []

    # Close positions
    for ticker in list(state["open_positions"].keys()):
        pos   = state["open_positions"][ticker]
        price = current_prices.get(ticker)
        if price is None:
            log_actions.append({"action": "HOLD", "ticker": ticker, "reason": "No price data today"})
            continue

        direction = pos["direction"]
        days_held = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days

        if direction == "long":
            hit_stop = price <= pos["stop"]
            hit_tp   = price >= pos["tp"]
            pnl      = (price - pos["entry_price"]) * pos["units"]
        else:
            hit_stop = price >= pos["stop"]
            hit_tp   = price <= pos["tp"]
            pnl      = (pos["entry_price"] - price) * pos["units"]

        ret = pnl / (pos["entry_price"] * pos["units"] + 1e-9)

        if hit_stop or hit_tp or days_held >= FORWARD:
            reason = "Stop-loss" if hit_stop else ("Take-profit" if hit_tp else f"Hold period ({days_held}d)")
            state["cash"] += pnl
            trade_record = {
                "ticker": ticker, "asset_type": ASSET_TYPE.get(ticker, "stock"),
                "direction": direction, "entry_date": pos["entry_date"],
                "exit_date": today_str(), "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(price, 4), "units": pos["units"],
                "pnl": round(pnl, 2), "return": round(ret, 4),
                "exit_reason": reason, "ml_score": pos.get("ml_score"),
            }
            state["closed_trades"].append(trade_record)
            del state["open_positions"][ticker]
            closed_today.append(trade_record)
            emoji = "✅" if pnl > 0 else "❌"
            print(f"    {emoji} CLOSED {direction.upper()} {ticker:10s}  P&L: £{pnl:+.2f}  ({reason})")
            log_actions.append({
                "action": "CLOSE", "ticker": ticker, "direction": direction,
                "pnl": round(pnl, 2), "return": round(ret, 4),
                "exit_reason": reason, "exit_price": round(price, 4),
            })
        else:
            unrealised = round(pnl, 2)
            print(f"    ↗  HOLD   {direction.upper()} {ticker:10s}  unrealised: £{unrealised:+.2f}  (day {days_held})")
            log_actions.append({
                "action": "HOLD", "ticker": ticker, "direction": direction,
                "unrealised_pnl": unrealised, "days_held": days_held,
            })

    # Open new positions
    slots = MAX_POS - len(state["open_positions"])
    if slots > 0:
        candidates = []
        for sig in signals:
            ticker = sig["ticker"]
            if ticker in state["open_positions"]: continue
            if sig["signal"] == "BUY":
                candidates.append((ticker, sig["probability"], current_prices.get(ticker, 0), "long", sig["probability"]))
            elif sig["signal"] == "SHORT":
                candidates.append((ticker, sig["probability"], current_prices.get(ticker, 0), "short", sig["probability"]))

        candidates.sort(key=lambda x: -x[1] if x[3]=="long" else x[1])

        for ticker, prob, price, direction, raw_prob in candidates[:slots]:
            if price <= 0: continue
            atype     = ASSET_TYPE.get(ticker, "stock")
            risk_amt  = state["cash"] * RISK_PER
            stop_dist = price * STOP_PCT
            units     = calc_units(ticker, risk_amt, stop_dist)
            cost      = units * price

            if cost > state["cash"] * 0.40:
                units = round((state["cash"] * 0.40) / price, 8) if atype=="crypto" else int((state["cash"]*0.40)/price)
                cost  = units * price

            if units <= 0 or cost > state["cash"]: continue

            stop = price * (1 + STOP_PCT) if direction == "short" else price * (1 - STOP_PCT)
            tp   = price * (1 - TP_PCT)   if direction == "short" else price * (1 + TP_PCT)

            state["open_positions"][ticker] = {
                "direction":   direction,
                "entry_price": round(price, 4),
                "units":       units,
                "stop":        round(stop, 4),
                "tp":          round(tp, 4),
                "entry_date":  today_str(),
                "ml_score":    round(raw_prob, 4),
                "asset_type":  atype,
            }
            opened_today.append({"ticker": ticker, "direction": direction, "price": round(price, 4),
                                  "units": units, "ml_score": round(raw_prob, 4)})
            units_str = f"{units:.8f}" if atype=="crypto" else str(int(units))
            emoji = "🟢" if direction == "long" else "🔴"
            print(f"    {emoji} OPENED {direction.upper()} {ticker:10s}  @ £{price:.2f}  units={units_str}  stop=£{stop:.2f}  tp=£{tp:.2f}")
            log_actions.append({
                "action": "OPEN", "ticker": ticker, "direction": direction,
                "entry_price": round(price, 4), "units": units,
                "stop": round(stop, 4), "tp": round(tp, 4), "ml_score": round(raw_prob, 4),
            })
    else:
        print(f"    ℹ️  Portfolio full ({MAX_POS} positions). No new trades.")

    # Portfolio summary
    port_val = portfolio_value(state, current_prices)
    pnl_total = port_val - state["start_capital"]
    closed_all = state["closed_trades"]
    wins = [t for t in closed_all if t["pnl"] > 0]
    win_rate = len(wins) / len(closed_all) if closed_all else 0.0

    print(f"\n[5/5] Saving state …")
    print(f"      Portfolio value : £{port_val:.2f}  ({pnl_total:+.2f} total P&L)")
    print(f"      Cash            : £{state['cash']:.2f}")
    print(f"      Open positions  : {len(state['open_positions'])}")
    print(f"      Closed trades   : {len(closed_all)}  (win rate {win_rate*100:.1f}%)")

    save_portfolio(state)

    # Build daily log entry
    log_entry = {
        "date":            today_str(),
        "timestamp":       now_str(),
        "regime":          regime_str,
        "portfolio_value": round(port_val, 2),
        "cash":            round(state["cash"], 2),
        "total_pnl":       round(pnl_total, 2),
        "open_positions":  len(state["open_positions"]),
        "win_rate":        round(win_rate, 4),
        "closed_today":    closed_today,
        "opened_today":    opened_today,
        "actions":         log_actions,
        "signals_summary": {
            "buys":  sum(1 for s in signals if s["signal"] == "BUY"),
            "shorts": sum(1 for s in signals if s["signal"] == "SHORT"),
            "watches": sum(1 for s in signals if s["signal"] == "WATCH"),
        },
    }
    append_log(log_entry)

    print(f"\n✅  Done — import portfolio_state.json and daily_log.json into your dashboard")


# ─────────────────────────────────────────────────────────────
# BACKTEST  (unchanged from Stage 1)
# ─────────────────────────────────────────────────────────────

def run_backtest(tickers, output_path=BACKTEST_FILE):
    end   = datetime.today()
    start = end - timedelta(days=365 * BACKTEST_YRS + 60)

    print(f"[1/4] Downloading SPY ({BACKTEST_YRS}-year window) …")
    spy = get_ticker_data(SPY_TICKER, start, end)
    if spy.empty:
        print("ERROR: Cannot download SPY."); return

    print(f"[2/4] Downloading and processing {len(tickers)} assets …")
    asset_probs = {}
    asset_close = {}

    for ticker in tickers:
        try:
            df = get_ticker_data(ticker, start, end)
            if len(df) < 300: continue
            feats  = compute_features(df, spy)
            target = compute_target(df, spy)
            combo  = feats.join(target).dropna()
            if len(combo) < 150: continue
            X = combo.drop(columns=["target"])
            y = combo["target"]
            wf = walk_forward_predict(X, y)
            if not wf: continue
            dates, probs = zip(*wf)
            asset_probs[ticker] = pd.Series(probs, index=pd.DatetimeIndex(dates))
            asset_close[ticker] = df["Close"].squeeze()
            print(f"    {ticker:12s}  {len(wf)} prediction dates")
        except Exception as e:
            print(f"    WARNING: skipped {ticker} — {e}")

    if not asset_probs:
        print("ERROR: No assets processed."); return

    all_dates = sorted(set().union(*[set(s.index) for s in asset_probs.values()]))
    all_dates = [d for d in all_dates if d >= start + timedelta(days=365)]

    spy_close  = spy["Close"].squeeze()
    spy_sma200 = spy_close.rolling(200).mean()
    regime_s   = (spy_close > spy_sma200).reindex(pd.DatetimeIndex(all_dates)).ffill().fillna(False)

    print(f"[3/4] Simulating portfolio ({len(all_dates)} trading days) …")
    portfolio    = START_CAP
    equity_curve = [{"date": str(all_dates[0])[:10], "value": round(portfolio, 2)}]
    open_pos     = {}
    all_trades   = []
    by_asset     = {t: {"trades": 0, "wins": 0, "pnl": 0.0} for t in asset_probs}

    for today in all_dates:
        today_str_bt = str(today)[:10]
        regime_bull  = bool(regime_s.get(today, False))
        to_close     = []

        for ticker, pos in open_pos.items():
            try: price = float(asset_close[ticker].asof(today))
            except: continue
            if np.isnan(price): continue
            days_held = (today - pos["entry_date"]).days
            direction = pos["direction"]
            if direction == "long":
                hit_stop = price <= pos["stop"]; hit_tp = price >= pos["tp"]
                pnl = (price - pos["entry_price"]) * pos["units"]
            else:
                hit_stop = price >= pos["stop"]; hit_tp = price <= pos["tp"]
                pnl = (pos["entry_price"] - price) * pos["units"]
            ret = pnl / (pos["entry_price"] * pos["units"] + 1e-9)
            if hit_stop or hit_tp or days_held >= FORWARD:
                exit_reason = "Stop-loss" if hit_stop else ("Take-profit" if hit_tp else "Hold period")
                portfolio += pnl
                atype = ASSET_TYPE.get(ticker, "stock")
                units_disp = f"{pos['units']:.8f}" if atype=="crypto" else str(int(pos["units"]))
                all_trades.append({"ticker": ticker, "asset_type": atype, "direction": direction,
                    "entry_date": str(pos["entry_date"])[:10], "exit_date": today_str_bt,
                    "entry_price": round(pos["entry_price"],4), "exit_price": round(price,4),
                    "units": units_disp, "pnl": round(pnl,2), "return": round(ret,4), "exit_reason": exit_reason})
                if ticker in by_asset:
                    by_asset[ticker]["trades"] += 1; by_asset[ticker]["pnl"] += pnl
                    if pnl > 0: by_asset[ticker]["wins"] += 1
                to_close.append(ticker)

        for t in to_close: del open_pos[t]

        if len(open_pos) < MAX_POS:
            candidates = []
            for ticker, prob_s in asset_probs.items():
                if ticker in open_pos: continue
                try: prob = float(prob_s.asof(today))
                except: continue
                if np.isnan(prob): continue
                try: price = float(asset_close[ticker].asof(today))
                except: continue
                if np.isnan(price) or price <= 0: continue
                atype = ASSET_TYPE.get(ticker, "stock")
                sig = classify_signal(prob, regime_bull, atype, True, 1.0)
                if sig == "BUY":   candidates.append((ticker, prob, price, "long"))
                elif sig == "SHORT": candidates.append((ticker, prob, price, "short"))

            candidates.sort(key=lambda x: -x[1] if x[3]=="long" else x[1])
            for ticker, prob, price, direction in candidates[:MAX_POS - len(open_pos)]:
                risk_amt  = portfolio * RISK_PER
                stop_dist = price * STOP_PCT
                units     = calc_units(ticker, risk_amt, stop_dist)
                cost      = units * price
                if cost > portfolio * 0.40:
                    atype = ASSET_TYPE.get(ticker, "stock")
                    units = round((portfolio*0.40)/price,8) if atype=="crypto" else int((portfolio*0.40)/price)
                    cost  = units * price
                if units <= 0 or cost <= 0: continue
                stop = price*(1+STOP_PCT) if direction=="short" else price*(1-STOP_PCT)
                tp   = price*(1-TP_PCT)   if direction=="short" else price*(1+TP_PCT)
                open_pos[ticker] = {"entry_price":price,"units":units,"direction":direction,
                                    "stop":stop,"tp":tp,"entry_date":today}

        equity_curve.append({"date": today_str_bt, "value": round(portfolio, 2)})

    # Force close remaining
    last_date = all_dates[-1]
    for ticker, pos in open_pos.items():
        try: price = float(asset_close[ticker].asof(last_date))
        except: continue
        if np.isnan(price): continue
        direction = pos["direction"]
        pnl = (price-pos["entry_price"])*pos["units"] if direction=="long" else (pos["entry_price"]-price)*pos["units"]
        ret = pnl/(pos["entry_price"]*pos["units"]+1e-9)
        portfolio += pnl
        atype = ASSET_TYPE.get(ticker,"stock")
        all_trades.append({"ticker":ticker,"asset_type":atype,"direction":direction,
            "entry_date":str(pos["entry_date"])[:10],"exit_date":str(last_date)[:10],
            "entry_price":round(pos["entry_price"],4),"exit_price":round(price,4),
            "units":f"{pos['units']:.8f}" if atype=="crypto" else str(int(pos["units"])),
            "pnl":round(pnl,2),"return":round(ret,4),"exit_reason":"End of backtest"})
        if ticker in by_asset:
            by_asset[ticker]["trades"]+=1; by_asset[ticker]["pnl"]+=pnl
            if pnl>0: by_asset[ticker]["wins"]+=1

    total_return = (portfolio - START_CAP) / START_CAP
    wins = [t for t in all_trades if t["pnl"]>0]
    win_rate = len(wins)/len(all_trades) if all_trades else 0.0
    avg_pnl  = float(np.mean([t["pnl"] for t in all_trades])) if all_trades else 0.0
    long_t   = [t for t in all_trades if t["direction"]=="long"]
    short_t  = [t for t in all_trades if t["direction"]=="short"]
    values   = [e["value"] for e in equity_curve]
    peak=values[0]; max_dd=0.0
    for v in values:
        if v>peak: peak=v
        dd=(v-peak)/peak
        if dd<max_dd: max_dd=dd
    daily_rets = [(values[i]-values[i-1])/values[i-1] for i in range(1,len(values))]
    sharpe = (np.mean(daily_rets)/(np.std(daily_rets)+1e-9))*np.sqrt(252) if daily_rets else 0.0
    by_asset_out = {t:{"trades":d["trades"],"win_rate":round(d["wins"]/d["trades"],4) if d["trades"] else 0,
        "pnl":round(d["pnl"],2),"return":round(d["pnl"]/START_CAP,4),"asset_type":ASSET_TYPE.get(t,"stock")}
        for t,d in by_asset.items() if d["trades"]>0}

    result = {
        "generated": now_str(),
        "period": f"{BACKTEST_YRS}-year backtest ({str(all_dates[0])[:10]} → {str(all_dates[-1])[:10]})",
        "start_cap": START_CAP,
        "stats": {
            "total_return":round(total_return,4),"final_value":round(portfolio,2),
            "win_rate":round(win_rate,4),"total_trades":len(all_trades),
            "long_trades":len(long_t),"short_trades":len(short_t),
            "long_win_rate":round(len([t for t in long_t if t["pnl"]>0])/len(long_t),4) if long_t else 0,
            "short_win_rate":round(len([t for t in short_t if t["pnl"]>0])/len(short_t),4) if short_t else 0,
            "max_drawdown":round(max_dd,4),"sharpe":round(float(sharpe),4),
            "avg_trade_pnl":round(avg_pnl,2),
            "best_trade":round(max((t["pnl"] for t in all_trades),default=0),2),
            "worst_trade":round(min((t["pnl"] for t in all_trades),default=0),2),
        },
        "equity": equity_curve, "trades": all_trades, "by_asset": by_asset_out,
    }

    print(f"[4/4] Writing {output_path} …")
    with open(output_path,"w") as f: json.dump(result,f,indent=2)
    print(f"\n✅  Backtest complete — {len(all_trades)} trades · {total_return*100:+.1f}% return · Sharpe {sharpe:.2f}")
    print(f"    Import '{output_path}' into Dashboard → ML Strategy → Backtest")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Trading — Stage 2")
    parser.add_argument("--backtest",     action="store_true", help="Run 5-year backtest")
    parser.add_argument("--signals-only", action="store_true", help="Generate signals only, no portfolio changes")
    parser.add_argument("--tickers",      nargs="+", default=None)
    parser.add_argument("--output",       type=str,  default=None)
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers] if args.tickers else ALL_TICKERS

    if args.backtest:
        out = args.output or BACKTEST_FILE
        print(f"═══ ML TRADING — BACKTEST MODE ({BACKTEST_YRS}yr walk-forward + purging)")
        print(f"    Universe: {len(tickers)} assets  |  Start capital: £{START_CAP:,.0f}")
        print(f"    Longs: BUY >{BUY_THRESH*100:.0f}%  |  Shorts: SHORT <{SHORT_THRESH*100:.0f}%\n")
        run_backtest(tickers, output_path=out)
    else:
        print(f"═══ ML TRADING — {'SIGNALS ONLY' if args.signals_only else 'PORTFOLIO MODE'}")
        print(f"    Universe: {len(tickers)} assets  |  Run at: {now_str()}\n")
        run(tickers, signals_only=args.signals_only)
