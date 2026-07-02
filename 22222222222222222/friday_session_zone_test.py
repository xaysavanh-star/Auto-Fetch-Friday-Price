#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
friday_session_zone_test.py
=======================================================================
ຄຳຖາມ: Friday effect (ຢືນຢັນແລ້ວວ່າເລີ່ມສ້າງຕົວແຕ່ວັນພະຫັດ, ບໍ່ແມ່ນ weekend
gap - ເບິ່ງ friday_effect_test.py ຂໍ້ 7) — drift ນີ້ເກີດຂຶ້ນຊ່ວງໃດຂອງ
ວັນພະຫັດ-ວັນສຸກແທ້ໆ? ຄວນຈຳກັດ trade ໃສ່ໂຊນເວລາໃດໜຶ່ງ ແທນຖືທັງວັນສຸກບໍ່?

ວິທີການ: ຄິດໄລ່ mean return ຂອງແທ່ງ M15 ແຕ່ລະໂຕ (ບໍ່ຫັກ cost) ຈັດກຸ່ມຕາມ
(ວັນ, ຊົ່ວໂມງ) ຈາກວັນພະຫັດ 00:00 ຮອດວັນສຸກ 23:45, ແລ້ວສະສົມ (cumulative)
ຕາມລຳດັບເວລາ - ຈະເຫັນວ່າ drift ສ້າງຕົວແບບໃດ (ຄ່ອຍໆ ຫຼື ພຸ່ງຢູ່ຊົ່ວໂມງໃດ
ຊົ່ວໂມງໜຶ່ງ). ຈາກນັ້ນທົດສອບ window ທີ່ແຄບກວ່າ (ຫັກ cost ຈິງ) ທຽບກັບ
baseline ເຕັມວັນສຸກ ວ່າ efficiency (t-stat ຕໍ່ຫົວໜ່ວຍເວລາຖື) ດີກວ່າບໍ່.

ໝາຍເຫດ: script standalone.
=======================================================================
"""

import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(2025)

CONFIG = {
    "data_path": "XAUUSD_M15.csv",
    "point_size": 0.01,
    "cost_roundtrip": 0.00025,
    "n_boot": 2000,
    # candidate window ທີ່ຈະທົດສອບ: (entry_dow, entry_hour, exit_dow, exit_hour)
    # dow: 0=ຈັນ 1=ອັງຄານ 2=ພຸດ 3=ພະຫັດ 4=ສຸກ
    "candidate_windows": [
        ("baseline: Fri 00:00 -> Fri 23:45", 4, 0, 4, 23),
        ("Thu 12:00 -> Fri 23:45",             3, 12, 4, 23),
        ("Thu 00:00 -> Fri 23:45",             3, 0, 4, 23),
        ("Thu 12:00 -> Fri 12:00",             3, 12, 4, 12),
        ("Fri 00:00 -> Fri 12:00",             4, 0, 4, 12),
        ("Fri 12:00 -> Fri 23:45",             4, 12, 4, 23),
    ],
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


def newey_west_t(x, lag=1):
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


def bps(x):
    return np.nan if x is None or (isinstance(x, float) and not np.isfinite(x)) else x * 1e4


def print_cumulative_curve(df):
    print(f"\n{'='*70}\nCumulative drift ໃນ Thu-Fri (mean return ສະສົມ, ບໍ່ຫັກ cost)\n{'='*70}")
    d = df.copy()
    d["ret"] = np.log(d["close"]).diff()
    d["dow"] = d.index.dayofweek
    d["hour"] = d.index.hour
    sub = d[d["dow"].isin([3, 4])]
    hourly = sub.groupby(["dow", "hour"])["ret"].mean()
    hourly = hourly.sort_index()
    cum = 0.0
    names = {3: "ພະຫັດ", 4: "ສຸກ"}
    for (dow, hour), mu in hourly.items():
        cum += mu
        if hour % 2 == 0:
            bar = "#" * max(0, int(bps(cum) / 2))
            print(f"  {names[dow]} {hour:>2}:00  cum={bps(cum):>7.2f}bps  {bar}")


def build_window_trades(df, cfg, entry_dow, entry_hour, exit_dow, exit_hour):
    """1 trade/ອາທິດ: enter open ຂອງແທ່ງທຳອິດທີ່ dow==entry_dow ແລະ hour==entry_hour,
    exit close ຂອງແທ່ງສຸດທ້າຍທີ່ dow==exit_dow ແລະ hour==exit_hour, ຫັກ cost ຈິງ."""
    if "spread" in df.columns:
        cost = df["spread"] * cfg["point_size"] / df["close"]
    else:
        cost = pd.Series(cfg["cost_roundtrip"], index=df.index)
    d = pd.DataFrame({
        "dow": df.index.dayofweek, "hour": df.index.hour,
        "open": df["open"].values, "close": df["close"].values, "cost": cost.values,
    }, index=df.index)
    # ໝາຍ "week id" ດ້ວຍ (ISO year, ISO week) ເພື່ອຈັບຄູ່ entry/exit ອາທິດດຽວກັນ
    iso = df.index.isocalendar()
    d["week_id"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)

    entry_rows = d[(d["dow"] == entry_dow) & (d["hour"] == entry_hour)]
    entry_first = entry_rows.groupby("week_id").first()

    exit_rows = d[(d["dow"] == exit_dow) & (d["hour"] == exit_hour)]
    exit_last = exit_rows.groupby("week_id").last()

    joined = entry_first.join(exit_last, lsuffix="_en", rsuffix="_ex", how="inner")
    net = joined["close_ex"] / joined["open_en"] - 1.0 - joined["cost_en"]
    return net.values


def main():
    cfg = CONFIG
    print(f">>> REAL data: {cfg['data_path']}")
    df = load_ohlcv(cfg["data_path"])
    print(f"Bars: {len(df):,} | {df.index[0]} -> {df.index[-1]}")

    print_cumulative_curve(df)

    print(f"\n{'='*70}\nປຽບທຽບ window ຕ່າງໆ (ຫັກ cost ຈິງ)\n{'='*70}")
    print(f"{'Window':<32}{'n':<6}{'ຊົ່ວໂມງຖື':<10}{'mean(bps)':<11}{'HAC t':<8}{'CI':<20}")
    print("-" * 90)
    for name, edow, ehour, xdow, xhour in cfg["candidate_windows"]:
        net = build_window_trades(df, cfg, edow, ehour, xdow, xhour)
        mu, t = newey_west_t(net, 1)
        lo, hi = block_boot_ci(net, 10, cfg["n_boot"])
        hold_hours = (xdow - edow) * 24 + (xhour - ehour)
        ci_txt = f"[{bps(lo):.1f},{bps(hi):.1f}]"
        print(f"{name:<32}{len(net):<6}{hold_hours:<10}{bps(mu):<11.2f}{t:<8.2f}{ci_txt:<20}")

    print("\nໝາຍເຫດ: efficiency ດີກວ່າ = t-stat ໃກ້ຄຽງ ຫຼື ດີກວ່າ baseline ແຕ່ຖືເວລາສັ້ນກວ່າ")
    print("(ຫຼຸດຄວາມສ່ຽງຈາກ event ບໍ່ຄາດຄິດໃນຊ່ວງທີ່ຖືເພີ່ມ)")


if __name__ == "__main__":
    main()
