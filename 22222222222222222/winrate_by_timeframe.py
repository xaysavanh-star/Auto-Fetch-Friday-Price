#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
winrate_by_timeframe.py
=======================================================================
ຄຳຖາມ: "ອັດຕາຊະນະ" (win rate = % ຂອງ trade ທີ່ໄດ້ກຳໄລ ຫຼັງຫັກ cost ຈິງ)
ຂອງ spike-fade ແລະ momentum ໃນ timeframe ໃດຫຼາຍທີ່ສຸດ?

ວິທີການ: resample XAUUSD_M15.csv (ຂໍ້ມູນຈິງ) ຂຶ້ນເປັນ M15 / M30 / H1 / H4 / D1
ແລ້ວທົດສອບ z-score spike signal ດຽວກັນ (ຄືກັນກັບ spike_fade_zone_filter.py /
spike_momentum_test.py) ໃນແຕ່ລະ TF, z_thr ຄົງທີ່ (2.5) ເພື່ອບໍ່ໃຫ້ grid ໃຫຍ່
ເກີນໄປ, horizon [3,5,10] ແທ່ງ (ຄວາມໝາຍເວລາຈິງແຕກຕ່າງກັນຕາມ TF - ນັ້ນແມ່ນຈຸດປະສົງ).

Split IS/OOS 70/30 ຄັ້ງດຽວ (ບໍ່ແມ່ນ full walk-forward ຄື script ອື່ນ - ນີ້ເປັນ
scan ຫາທິດທາງ, ບໍ່ແມ່ນການພິສູດສຸດທ້າຍ). ລາຍງານ win rate + mean bps + HAC t
ຂອງ OOS ເທົ່ານັ້ນ, ບໍ່ cherry-pick (ສະແດງທຸກ combo).

ໝາຍເຫດ: win rate ສູງ ບໍ່ໄດ້ແປວ່າກຳໄລ (ຖ້າ average loss ໃຫຍ່ກວ່າ average win
ຫຼາຍ) - ລາຍງານ mean bps ຄຽງໄວ້ນຳເພື່ອໃຫ້ຄົບຖ້ວນ.
=======================================================================
"""

import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "XAUUSD_M15.csv"
POINT_SIZE = 0.01
COST_ROUNDTRIP = 0.00025   # fallback ຖ້າບໍ່ມີ spread column
VOL_WINDOW = 100
Z_THR = 2.5
HORIZONS = [3, 5, 10]
DIRECTIONS = ["fade", "momentum"]
TIMEFRAMES = [("M15", "15min"), ("M30", "30min"), ("H1", "1h"), ("H4", "4h"), ("D1", "1D")]
IS_FRACTION = 0.70


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


def resample_tf(df, rule):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "spread" in df.columns:
        agg["spread"] = "mean"
    out = df.resample(rule).agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def prepare(df):
    close = df["close"]
    ret = np.log(close).diff()
    sigma = ret.rolling(VOL_WINDOW).std().shift(1)
    df = df.copy()
    df["z"] = ret / sigma
    if "spread" in df.columns:
        df["cost"] = (df["spread"] * POINT_SIZE / df["close"]).clip(lower=0)
    else:
        df["cost"] = COST_ROUNDTRIP
    return df


def build_trades(df, z_thr, horizon, direction):
    close = df["close"].values
    open_ = df["open"].values
    z = df["z"].values
    cost = df["cost"].values
    n = len(close)

    up_val = -1.0 if direction == "fade" else 1.0
    dn_val = 1.0 if direction == "fade" else -1.0
    raw_side = np.zeros(n)
    raw_side[z >= z_thr] = up_val
    raw_side[z <= -z_thr] = dn_val

    side = np.zeros(n)
    last_trigger = -(horizon + 1)
    for i in range(n):
        if raw_side[i] != 0 and (i - last_trigger) > horizon:
            side[i] = raw_side[i]
            last_trigger = i

    pos = np.arange(n)
    entry_i = pos + 1
    exit_i = pos + horizon
    ok = exit_i < n
    fwd = np.full(n, np.nan)
    fwd[ok] = close[exit_i[ok]] / open_[entry_i[ok]] - 1.0

    mask = (side != 0) & np.isfinite(fwd) & np.isfinite(cost)
    idx = np.where(mask)[0]
    net = side[idx] * fwd[idx] - cost[idx]
    return net


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
    return np.nan if not np.isfinite(x) else x * 1e4


def main():
    print(f">>> ໂຫຼດ {DATA_PATH} ...")
    df_m15 = load_ohlcv(DATA_PATH)
    print(f"M15 bars: {len(df_m15):,} | {df_m15.index[0]} -> {df_m15.index[-1]}\n")

    rows = []
    for tf_name, rule in TIMEFRAMES:
        df_tf = resample_tf(df_m15, rule) if rule != "15min" else df_m15
        df_tf = prepare(df_tf)
        split = int(len(df_tf) * IS_FRACTION)
        df_oos = df_tf.iloc[split:]
        n_bars = len(df_tf)

        for direction in DIRECTIONS:
            for h in HORIZONS:
                net = build_trades(df_oos, Z_THR, h, direction)
                if len(net) < 20:
                    rows.append(dict(tf=tf_name, direction=direction, horizon=h,
                                      n=len(net), winrate=np.nan, mean_bps=np.nan, t=np.nan))
                    continue
                winrate = (net > 0).mean() * 100
                mu, t = newey_west_t(net, h)
                rows.append(dict(tf=tf_name, direction=direction, horizon=h,
                                  n=len(net), winrate=winrate, mean_bps=bps(mu), t=t))

        print(f"{tf_name}: {n_bars:,} ແທ່ງທັງໝົດ (OOS ~{len(df_oos):,} ແທ່ງ)")

    res = pd.DataFrame(rows)
    res.to_csv("winrate_by_timeframe.csv", index=False)

    print(f"\n{'='*78}")
    print(f"{'TF':<5}{'ທິດທາງ':<12}{'horizon':<9}{'n_trade':<9}{'win rate':<11}{'mean(bps)':<12}{'HAC t':<8}")
    print("=" * 78)
    for _, r in res.iterrows():
        wr = f"{r['winrate']:.1f}%" if np.isfinite(r["winrate"]) else "n/a"
        mb = f"{r['mean_bps']:.2f}" if np.isfinite(r["mean_bps"]) else "n/a"
        tt = f"{r['t']:.2f}" if np.isfinite(r["t"]) else "n/a"
        print(f"{r['tf']:<5}{r['direction']:<12}{int(r['horizon']):<9}{int(r['n']):<9}{wr:<11}{mb:<12}{tt:<8}")
    print("=" * 78)

    best = res.loc[res["winrate"].idxmax()]
    print(f"\nອັດຕາຊະນະສູງສຸດ: {best['tf']} / {best['direction']} / horizon={int(best['horizon'])} "
          f"= {best['winrate']:.1f}% (n={int(best['n'])}, mean={best['mean_bps']:.2f}bps, t={best['t']:.2f})")
    print("ບັນທຶກຕາຕະລາງເຕັມໃສ່ winrate_by_timeframe.csv ແລ້ວ")


if __name__ == "__main__":
    main()
