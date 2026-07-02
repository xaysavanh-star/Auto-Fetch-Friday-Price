#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
session_effect_test.py
=======================================================================
ຄຳຖາມ: XAUUSD ມີ "session/ເວລາ effect" ຈິງບໍ່ - ຄື return ແຕກຕ່າງກັນຢ່າງມີ
ນັຍສຳຄັນຕາມ (1) ຊົ່ວໂມງຂອງມື້ (London/NY/Asian session) ຫຼື (2) ວັນຂອງອາທິດ?

ນີ້ຄື signal ປະເພດໃໝ່ໂດຍສິ້ນເຊີງ (ບໍ່ອີງລວດລາຍລາຄາ/indicator ຄື 8 ການທົດລອງ
ກ່ອນໜ້າ) - ອີງ**ເວລາ** ລ້ວນໆ (timestamp), ຊຶ່ງເປັນ predictor ທີ່ບໍ່ມີ look-
ahead ໂດຍທຳມະຊາດ (ຮູ້ໄດ້ລ່ວງໜ້າ 100%).

ໂຕທົດສອບ 1 (hour-of-day): ຄິດໄລ່ mean return ຂອງແຕ່ລະຊົ່ວໂມງ (0-23, ຕາມ
timestamp ໃນ CSV) ຫຼັງຫັກ cost (ຖືກ close-to-close 1 ຊົ່ວໂມງ), ຫາຊົ່ວໂມງທີ່
mean ຫ່າງຈາກສູນຢ່າງມີນັຍສຳຄັນ (HAC t).

ໂຕທົດສອບ 2 (day-of-week): ຄືກັນແຕ່ຄິດໄລ່ຕໍ່ວັນ (ຖືກ close-to-close 1 ວັນ).

Split IS/OOS ດ້ວຍ walk-forward (5 fold expanding) - IS ຫາຊົ່ວໂມງ/ວັນທີ່ໃຫ້
mean ດີທີ່ສຸດ, OOS ພິສູດຄືນ.

ໝາຍເຫດ: ບໍ່ມີ synthetic sanity-check ໃນ script ນີ້ (ບໍ່ຈຳເປັນ - ການຄິດໄລ່
mean-by-group ແມ່ນ arithmetic ກົງໄປກົງມາ, ບໍ່ມີ logic ສະລັບສັບຊ້ອນຄື pivot/
divergence ທີ່ຕ້ອງການ sanity-check).
=======================================================================
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(2025)

CONFIG = {
    "data_path": "XAUUSD_M15.csv",
    "point_size": 0.01,

    "n_folds": 6,
    "min_abs_t": 3.0,
    "min_days_per_group": 300,   # ຂັ້ນຕ່ຳຈຳນວນວັນ/ຊົ່ວໂມງ-instance ຕໍ່ group ຈຶ່ງເຊື່ອຖືໄດ້
}


def load_ohlcv(path):
    df = None
    for sep in [",", "\t", ";"]:
        try:
            tmp = pd.read_csv(path, sep=sep)
            if tmp.shape[1] >= 4:
                df = tmp; break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"ບໍ່ສາມາດອ່ານໄຟລ໌: {path}")
    df.columns = [str(c).strip().lower().strip("<>") for c in df.columns]
    rn = {}
    for c in df.columns:
        if c in ("open", "o"): rn[c] = "open"
        elif c in ("high", "h"): rn[c] = "high"
        elif c in ("low", "l"): rn[c] = "low"
        elif c in ("close", "c"): rn[c] = "close"
        elif "date" in c: rn[c] = "date"
        elif "time" in c and "tickvol" not in c: rn[c] = "time"
        elif c == "spread": rn[c] = "spread"
    df = df.rename(columns=rn)
    if "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    else:
        dt = pd.to_datetime(df["date"], errors="coerce")
    df = df.assign(dt=dt).dropna(subset=["dt"]).sort_values("dt").set_index("dt")
    cols = ["open", "high", "low", "close"]
    if "spread" in df.columns:
        cols.append("spread")
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[cols].dropna()


def newey_west_t(x, lag):
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    mu = x.mean(); e = x - mu
    s = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * (e[k:] @ e[:-k]) / n
    s = max(s, 1e-18)
    return mu, mu / np.sqrt(s / n)


def bps(x):
    return np.nan if x is None or (isinstance(x, float) and not np.isfinite(x)) else x * 1e4


# =====================================================================
# ໂຕທົດສອບ 1: hour-of-day
# =====================================================================
def build_hourly_returns(df):
    """ ret ຂອງແທ່ງ M15 ແຕ່ລະໂຕ (close-to-close, ບໍ່ຫັກ cost) - ໃຊ້ພຽງ scan ຫາ
    ຊົ່ວໂມງທີ່ໜ້າສົນໃຈເບື້ອງຕົ້ນ. cost-adjusted trade ຈິງຢູ່ build_hour_trades()."""
    close = df["close"]
    ret = close.pct_change()
    df = df.copy()
    df["ret"] = ret
    df["hour"] = df.index.hour
    return df


def build_all_hour_trades(df, cfg):
    """Precompute ຄັ້ງດຽວ (vectorized): trade 1/ວັນ/ຊົ່ວໂມງ ທັງໝົດ, ຄືນ DataFrame
    columns=[date, hour, entry_time, net] - ໃຊ້ filter ຊ້ຳໆພາຍຫຼັງ ໄວກວ່າ recompute."""
    if "spread" in df.columns:
        cost = df["spread"] * cfg["point_size"] / df["close"]
    else:
        cost = pd.Series(cfg.get("cost_roundtrip", 0.00025), index=df.index)
    d = pd.DataFrame({
        "date": df.index.normalize(), "hour": df.index.hour,
        "open": df["open"].values, "close": df["close"].values, "cost": cost.values,
    }, index=df.index)
    d["entry_time"] = d.index
    agg = d.groupby(["date", "hour"]).agg(
        entry_time=("entry_time", "first"), entry_price=("open", "first"),
        exit_price=("close", "last"), cost=("cost", "first"))
    agg["net"] = agg["exit_price"] / agg["entry_price"] - 1.0 - agg["cost"]
    return agg.reset_index()


def scan_hour_trade_is(all_trades, is_start, is_end, cfg, verbose=True):
    if verbose:
        print("\n=== IN-SAMPLE scan: hour-trade (1/ວັນ, ຫັກ cost ຈິງ) mean (bps) | HAC t ===")
    sub_all = all_trades[(all_trades["entry_time"] >= is_start) & (all_trades["entry_time"] < is_end)]
    rows = []
    for h in range(24):
        net = sub_all.loc[sub_all["hour"] == h, "net"].values
        if len(net) < 60:
            continue
        mu, t = newey_west_t(net, 1)
        if verbose:
            print(f"  hour={h:>2}: n={len(net):>5}  mean={bps(mu):>6.2f}bps  t={t:>5.2f}")
        rows.append(dict(hour=h, mu=mu, t=t, n=len(net)))
    if not rows:
        return None
    best = max(rows, key=lambda r: r["t"] if np.isfinite(r["t"]) else -999)
    return best if best["mu"] > 0 else None


def evaluate_hour_trade_oos(all_trades, hour, oos_start, oos_end, cfg):
    sub = all_trades[(all_trades["entry_time"] >= oos_start) & (all_trades["entry_time"] < oos_end)]
    net = sub.loc[sub["hour"] == hour, "net"].values
    mu, t = newey_west_t(net, 1)
    lo, hi = block_boot_ci(net, 10, 2000)
    sig = np.isfinite(t) and abs(t) >= cfg["min_abs_t"] and mu > 0 and np.isfinite(lo) and lo > 0
    enough = len(net) >= 60
    winrate = (net > 0).mean() * 100 if len(net) else np.nan
    return dict(hour=hour, n=len(net), mu=mu, t=t, lo=lo, hi=hi, winrate=winrate,
                sig=sig, enough=enough, passed=(mu > 0 and sig and enough))


def block_boot_ci(x, block, n_boot, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    if n < block + 1:
        return np.nan, np.nan
    nb = int(np.ceil(n / block)); smax = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        st = RNG.integers(0, smax + 1, size=nb)
        means[b] = np.concatenate([x[s:s + block] for s in st])[:n].mean()
    return np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def run_walk_forward_hour_trade(df, cfg):
    n = len(df)
    n_folds = cfg["n_folds"]
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    print(f"\n{'='*70}\nWALK-FORWARD [hour-trade, ຫັກ cost ຈິງ]: {n_folds - 1} OOS folds\n{'='*70}")
    all_trades = build_all_hour_trades(df, cfg)
    reports = []
    for k in range(1, n_folds):
        is_start, is_end = df.index[0], df.index[edges[k]]
        oos_start = df.index[edges[k]]
        oos_end = df.index[edges[k + 1] - 1] + pd.Timedelta(minutes=15)
        best = scan_hour_trade_is(all_trades, is_start, is_end, cfg, verbose=False)
        if best is None:
            print(f"\n--- Fold {k}: IS ບໍ່ພົບຊົ່ວໂມງທີ່ mean>0 -> FAIL")
            reports.append(dict(passed=False))
            continue
        m = evaluate_hour_trade_oos(all_trades, best["hour"], oos_start, oos_end, cfg)
        print(f"\n--- Fold {k}: IS best hour={best['hour']} (t={best['t']:.2f}) | "
              f"OOS n={m['n']} win={m['winrate']:.1f}% mean={bps(m['mu']):.2f}bps t={m['t']:.2f} "
              f"CI=[{bps(m['lo']):.2f},{bps(m['hi']):.2f}] -> {'PASS' if m['passed'] else 'FAIL'}")
        reports.append(m)
    n_pass = sum(1 for m in reports if m.get("passed"))
    print(f"\nSUMMARY [hour-trade cost-adjusted]: {n_pass}/{len(reports)} fold PASS")
    return n_pass, len(reports)


def scan_hour_is(df_is, cfg, verbose=True):
    if verbose:
        print("\n=== IN-SAMPLE scan: mean return ຕໍ່ຊົ່ວໂມງ (bps) | HAC t ===")
    rows = []
    for h in range(24):
        sub = df_is.loc[df_is["hour"] == h, "ret"].dropna().values
        if len(sub) < cfg["min_days_per_group"]:
            continue
        mu, t = newey_west_t(sub, 4)   # lag=4 (1 ຊົ່ວໂມງ = 4 ແທ່ງ M15, ອະນຸໂລມ autocorrelation ໃນຊົ່ວໂມງ)
        if verbose:
            print(f"  hour={h:>2}: n={len(sub):>6}  mean={bps(mu):>6.2f}bps  t={t:>5.2f}")
        rows.append(dict(hour=h, mu=mu, t=t, n=len(sub)))
    if not rows:
        return None
    best = max(rows, key=lambda r: r["t"] if np.isfinite(r["t"]) else -999)
    return best if best["mu"] > 0 else None


def evaluate_hour_oos(df_oos, hour, cfg):
    sub = df_oos.loc[df_oos["hour"] == hour, "ret"].dropna().values
    mu, t = newey_west_t(sub, 4)
    sig = np.isfinite(t) and abs(t) >= cfg["min_abs_t"] and mu > 0
    enough = len(sub) >= cfg["min_days_per_group"]
    return dict(hour=hour, n=len(sub), mu=mu, t=t, sig=sig, enough=enough, passed=(mu > 0 and sig and enough))


# =====================================================================
# ໂຕທົດສອບ 2: day-of-week
# =====================================================================
def build_daily_returns(df):
    """ຄິດໄລ່ close-to-close ຂອງແຕ່ລະວັນ (calendar day), ໝາຍ day-of-week."""
    daily_close = df["close"].resample("1D").last().dropna()
    ret = daily_close.pct_change().dropna()
    out = pd.DataFrame({"ret": ret})
    out["dow"] = out.index.dayofweek   # 0=ຈັນ ... 6=ອາທິດ
    return out


def scan_dow_is(df_is, cfg, verbose=True):
    if verbose:
        print("\n=== IN-SAMPLE scan: mean return ຕໍ່ວັນຂອງອາທິດ (bps) | HAC t ===")
    names = ["ຈັນ", "ອັງຄານ", "ພຸດ", "ພະຫັດ", "ສຸກ", "ເສົາ", "ອາທິດ"]
    rows = []
    for d in range(7):
        sub = df_is.loc[df_is["dow"] == d, "ret"].dropna().values
        if len(sub) < 30:
            continue
        mu, t = newey_west_t(sub, 1)
        if verbose:
            print(f"  {names[d]:<8}: n={len(sub):>5}  mean={bps(mu):>6.2f}bps  t={t:>5.2f}")
        rows.append(dict(dow=d, mu=mu, t=t, n=len(sub)))
    if not rows:
        return None
    best = max(rows, key=lambda r: r["t"] if np.isfinite(r["t"]) else -999)
    return best if best["mu"] > 0 else None


def evaluate_dow_oos(df_oos, dow, cfg):
    sub = df_oos.loc[df_oos["dow"] == dow, "ret"].dropna().values
    mu, t = newey_west_t(sub, 1)
    sig = np.isfinite(t) and abs(t) >= cfg["min_abs_t"] and mu > 0
    enough = len(sub) >= 30
    return dict(dow=dow, n=len(sub), mu=mu, t=t, sig=sig, enough=enough, passed=(mu > 0 and sig and enough))


# =====================================================================
# WALK-FORWARD
# =====================================================================
def run_walk_forward_hour(df, cfg):
    n = len(df)
    n_folds = cfg["n_folds"]
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    print(f"\n{'='*70}\nWALK-FORWARD [hour-of-day]: {n_folds - 1} OOS folds\n{'='*70}")
    reports = []
    for k in range(1, n_folds):
        df_is = df.iloc[:edges[k]]
        df_oos = df.iloc[edges[k]:edges[k + 1]]
        best = scan_hour_is(df_is, cfg, verbose=False)
        if best is None:
            print(f"\n--- Fold {k}: IS ບໍ່ພົບຊົ່ວໂມງທີ່ mean>0 -> FAIL")
            reports.append(dict(passed=False))
            continue
        m = evaluate_hour_oos(df_oos, best["hour"], cfg)
        print(f"\n--- Fold {k}: IS best hour={best['hour']} (t={best['t']:.2f}) | "
              f"OOS n={m['n']} mean={bps(m['mu']):.2f}bps t={m['t']:.2f} "
              f"-> {'PASS' if m['passed'] else 'FAIL'}")
        reports.append(m)
    n_pass = sum(1 for m in reports if m.get("passed"))
    print(f"\nSUMMARY [hour-of-day]: {n_pass}/{len(reports)} fold PASS")
    return n_pass, len(reports)


def run_walk_forward_dow(daily_df, cfg):
    n = len(daily_df)
    n_folds = cfg["n_folds"]
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    print(f"\n{'='*70}\nWALK-FORWARD [day-of-week]: {n_folds - 1} OOS folds\n{'='*70}")
    reports = []
    for k in range(1, n_folds):
        df_is = daily_df.iloc[:edges[k]]
        df_oos = daily_df.iloc[edges[k]:edges[k + 1]]
        best = scan_dow_is(df_is, cfg, verbose=False)
        if best is None:
            print(f"\n--- Fold {k}: IS ບໍ່ພົບວັນທີ່ mean>0 -> FAIL")
            reports.append(dict(passed=False))
            continue
        m = evaluate_dow_oos(df_oos, best["dow"], cfg)
        names = ["ຈັນ", "ອັງຄານ", "ພຸດ", "ພະຫັດ", "ສຸກ", "ເສົາ", "ອາທິດ"]
        print(f"\n--- Fold {k}: IS best day={names[best['dow']]} (t={best['t']:.2f}) | "
              f"OOS n={m['n']} mean={bps(m['mu']):.2f}bps t={m['t']:.2f} "
              f"-> {'PASS' if m['passed'] else 'FAIL'}")
        reports.append(m)
    n_pass = sum(1 for m in reports if m.get("passed"))
    print(f"\nSUMMARY [day-of-week]: {n_pass}/{len(reports)} fold PASS")
    return n_pass, len(reports)


def main():
    cfg = CONFIG
    print(f">>> REAL data: {cfg['data_path']}")
    df_raw = load_ohlcv(cfg["data_path"])
    print(f"Bars: {len(df_raw):,} | {df_raw.index[0]} -> {df_raw.index[-1]}")

    df_hour = build_hourly_returns(df_raw)
    print("\n--- ຮູບແບບເຕັມ (ບໍ່ split, ອະທິບາຍພາບລວມ) ---")
    scan_hour_is(df_hour, cfg, verbose=True)
    p1, t1 = run_walk_forward_hour(df_hour, cfg)

    df_daily = build_daily_returns(df_raw)
    scan_dow_is(df_daily, cfg, verbose=True)
    p2, t2 = run_walk_forward_dow(df_daily, cfg)

    p3, t3 = run_walk_forward_hour_trade(df_raw, cfg)

    print(f"\n{'#'*70}")
    print(f"ສະຫຼຸບ: hour-of-day(raw) {p1}/{t1} | day-of-week {p2}/{t2} | hour-trade(ຫັກ cost ຈິງ) {p3}/{t3}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
