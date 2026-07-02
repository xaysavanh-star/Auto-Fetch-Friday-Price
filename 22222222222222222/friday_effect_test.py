#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
friday_effect_test.py
=======================================================================
ຂຸດເລິກ "Friday effect" (ພົບຈາກ session_effect_test.py: mean=13.43bps,
t=4.09 ໃນ IS scan ເຕັມ, ແຕ່ບໍ່ຜ່ານ walk-forward OOS ຍ້ອນຕົວຢ່າງນ້ອຍ
~1182 ວັນສຸກ/23ປີ ÷ 5 fold = ~236/fold, ອາດພະລັງທາງສະຖິຕິບໍ່ພໍ).

ຂະຫຍາຍການທົດສອບອອກເປັນ 5 ມູມ:
   1. Friday (ຄືກັນກັບ trade ຈິງ: ເຂົ້າ open ວັນສຸກ, ອອກ close ວັນສຸກ, ຫັກ
      cost ຈິງ) vs mean = 0 (Newey-West + block bootstrap CI)
   2. Friday vs ວັນອື່ນທັງໝົດ (permutation test ຄວາມແຕກຕ່າງ mean - ຄືວິທີ
      near-vs-far ທີ່ໃຊ້ໃນ spike_fade_zone_filter.py)
   3. Walk-forward ດ້ວຍ fold ໜ້ອຍລົງ (3 fold ແທນ 5) ເພື່ອໃຫ້ແຕ່ລະ OOS
      window ມີ n ຫຼາຍພໍ (ພະລັງທາງສະຖິຕິສູງຂຶ້ນ)
   4. ແຍກ NFP-Friday (ວັນສຸກທຳອິດຂອງເດືອນ, ວັນປະກາດ Non-Farm Payroll ຈິງ
      ໃນ US) ອອກຈາກ Friday ທຳມະດາ - ເບິ່ງວ່າ effect ມາຈາກ NFP ຫຼືບໍ່
   5. ຄວາມຄົງທີ່ຂ້າມຍຸກ (sub-period stability): ແບ່ງ 23 ປີ ອອກເປັນ 4 ຊ່ວງ
      ເບິ່ງວ່າ Friday mean ບວກຄົງທີ່ທຸກຊ່ວງ ຫຼືມາຈາກຊ່ວງໃດຊ່ວງໜຶ່ງ

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
    "date_start": None,   # None = ໃຊ້ທັງໝົດ (2003-2026)
    "date_end": None,
    "point_size": 0.01,
    "cost_roundtrip": 0.00025,
    "n_perm": 2000,
    "n_boot": 2000,
    "min_abs_t": 3.0,
    "max_perm_p": 0.01,
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


def perm_diff_pvalue(a, b, n_perm):
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); n1 = len(a)
    cnt = 0
    for _ in range(n_perm):
        RNG.shuffle(pool)
        if (pool[:n1].mean() - pool[n1:].mean()) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def bps(x):
    return np.nan if x is None or (isinstance(x, float) and not np.isfinite(x)) else x * 1e4


def build_daily_trades(df, cfg):
    """1 trade/ວັນ: enter open ຂອງແທ່ງທຳອິດຂອງວັນ, exit close ຂອງແທ່ງສຸດທ້າຍ,
    ຫັກ cost ຈິງ (spread ຂອງແທ່ງເຂົ້າ). ຄືນ DataFrame index=date, columns=[net,dow,is_nfp]."""
    if "spread" in df.columns:
        cost = df["spread"] * cfg["point_size"] / df["close"]
    else:
        cost = pd.Series(cfg["cost_roundtrip"], index=df.index)
    d = pd.DataFrame({
        "date": df.index.normalize(), "open": df["open"].values,
        "close": df["close"].values, "cost": cost.values,
    }, index=df.index)
    agg = d.groupby("date").agg(entry_price=("open", "first"), exit_price=("close", "last"),
                                 cost=("cost", "first"))
    agg["net"] = agg["exit_price"] / agg["entry_price"] - 1.0 - agg["cost"]
    agg = agg.reset_index()
    agg["dow"] = agg["date"].dt.dayofweek
    agg["month"] = agg["date"].dt.to_period("M")
    first_fri_per_month = agg.loc[agg["dow"] == 4].groupby("month")["date"].min()
    agg["first_fri_of_month"] = agg["month"].map(first_fri_per_month)
    agg["is_nfp_friday"] = (agg["dow"] == 4) & (agg["date"] == agg["first_fri_of_month"])
    return agg


def analyze_year_by_year(trades, cfg):
    from math import comb
    print(f"\n{'='*70}\n6) ຄວາມສອດຄ່ອງແຕ່ລະປີ (ຈຸດຂໍ້ມູນອິດສະຫຼະ, sign test)\n{'='*70}")
    t = trades.copy()
    t["year"] = t["date"].dt.year
    fri = t[t["dow"] == 4]
    yearly = fri.groupby("year")["net"].agg(["mean", "count"])
    n_pos, n_total = 0, 0
    for year, row in yearly.iterrows():
        if row["count"] < 20:
            continue
        n_total += 1
        if row["mean"] > 0:
            n_pos += 1
        print(f"  {year}: n={int(row['count']):>3}  mean={bps(row['mean']):>7.2f}bps  [{'+' if row['mean'] > 0 else '-'}]")
    pval = sum(comb(n_total, i) * (0.5 ** n_total) for i in range(n_pos, n_total + 1)) if n_total else np.nan
    print(f"\n  {n_pos}/{n_total} ປີ ເປັນບວກ — sign test p={pval:.4f} (H0: ບັງເອີນ 50/50)")


def analyze_session_decomposition(trades, cfg):
    print(f"\n{'='*70}\n7) ຫາວ່າ effect ຢູ່ຊ່ວງໃດຂອງວັນສຸກ (decompose)\n{'='*70}")
    t = trades.sort_values("date").reset_index(drop=True)
    t["prev_dow"] = t["dow"].shift(1)
    t["prev_close"] = t["exit_price"].shift(1)
    t["next_dow"] = t["dow"].shift(-1)
    t["next_open"] = t["entry_price"].shift(-1)

    sub_a = t[(t["dow"] == 4) & (t["prev_dow"] == 3)]
    ret_a = (sub_a["exit_price"] / sub_a["prev_close"] - 1.0).values
    mu_a, t_a = newey_west_t(ret_a, 1)
    print(f"  (a) Thu close -> Fri close       : n={len(ret_a):>4}  mean={bps(mu_a):>7.2f}bps  t={t_a:>5.2f}")

    fri = trades[trades["dow"] == 4]
    mu_b, t_b = newey_west_t(fri["net"].values, 1)
    print(f"  (b) Fri open  -> Fri close (cost): n={len(fri):>4}  mean={bps(mu_b):>7.2f}bps  t={t_b:>5.2f}")

    sub_c = t[(t["dow"] == 4) & (t["next_dow"] == 0)]
    ret_c = (sub_c["next_open"] / sub_c["exit_price"] - 1.0).values
    mu_c, t_c = newey_west_t(ret_c, 1)
    print(f"  (c) Fri close -> Mon open (gap)  : n={len(ret_c):>4}  mean={bps(mu_c):>7.2f}bps  t={t_c:>5.2f}")

    ret_d = (sub_c["next_open"] / sub_c["entry_price"] - 1.0).values
    mu_d, t_d = newey_west_t(ret_d, 1)
    print(f"  (d) Fri open  -> Mon open (ລວມ)  : n={len(ret_d):>4}  mean={bps(mu_d):>7.2f}bps  t={t_d:>5.2f}")

    strongest = max([("(a) Thu->Fri close", t_a), ("(b) Fri session", t_b),
                      ("(c) weekend gap", t_c), ("(d) Fri open->Mon open", t_d)],
                     key=lambda p: p[1] if np.isfinite(p[1]) else -999)
    print(f"\n  -> ສ່ວນທີ່ແຮງທີ່ສຸດ: {strongest[0]} (t={strongest[1]:.2f})")


def main():
    cfg = CONFIG
    print(f">>> REAL data: {cfg['data_path']}")
    df = load_ohlcv(cfg["data_path"])
    if cfg.get("date_start"):
        df = df[df.index >= cfg["date_start"]]
    if cfg.get("date_end"):
        df = df[df.index <= cfg["date_end"]]
    print(f">>> ຊ່ວງເວລາທີ່ໃຊ້: {df.index[0]} -> {df.index[-1]} ({len(df):,} ແທ່ງ)")
    trades = build_daily_trades(df, cfg)
    print(f"ຈຳນວນວັນທັງໝົດ: {len(trades):,} | {trades['date'].min().date()} -> {trades['date'].max().date()}")

    fri = trades.loc[trades["dow"] == 4, "net"].values
    other = trades.loc[trades["dow"] != 4, "net"].values
    print(f"ວັນສຸກ: n={len(fri)}  |  ວັນອື່ນ: n={len(other)}")

    # ===== 1. Friday vs 0 (ຫັກ cost ຈິງແລ້ວ) =====
    print(f"\n{'='*70}\n1) Friday trade vs 0 (ຫັກ cost ຈິງ, ທັງ 23 ປີ)\n{'='*70}")
    mu, t = newey_west_t(fri, 1)
    lo, hi = block_boot_ci(fri, 10, cfg["n_boot"])
    winrate = (fri > 0).mean() * 100
    print(f"  n={len(fri)}  win rate={winrate:.1f}%  mean={bps(mu):.2f}bps  HAC t={t:.2f}")
    print(f"  bootstrap 95% CI: [{bps(lo):.2f}, {bps(hi):.2f}] bps")
    sig = np.isfinite(t) and abs(t) >= cfg["min_abs_t"] and mu > 0 and np.isfinite(lo) and lo > 0
    print(f"  [{'PASS' if sig else 'FAIL'}] significant ຫຼັງຫັກ cost (HAC t>={cfg['min_abs_t']}, CI>0)")

    # ===== 2. Friday vs ວັນອື່ນ (permutation) =====
    print(f"\n{'='*70}\n2) Friday vs ວັນອື່ນທັງໝົດ (permutation test)\n{'='*70}")
    diff, pdiff = perm_diff_pvalue(fri, other, cfg["n_perm"])
    print(f"  diff = {bps(diff):.2f}bps  permutation p = {pdiff:.4f}")
    print(f"  [{'PASS' if pdiff <= cfg['max_perm_p'] and diff > 0 else 'FAIL'}] Friday ດີກວ່າວັນອື່ນຢ່າງມີນັຍສຳຄັນ")

    # ===== 3. Walk-forward 3 fold (ໃຫຍ່ຂຶ້ນ ເພື່ອພະລັງສະຖິຕິ) =====
    print(f"\n{'='*70}\n3) Walk-forward (3 fold, ຂະຫຍາຍຂະໜາດ OOS)\n{'='*70}")
    n = len(trades)
    edges = np.linspace(0, n, 4).astype(int)
    fold_res = []
    for k in range(1, 3):
        is_t = trades.iloc[:edges[k]]
        oos_t = trades.iloc[edges[k]:edges[k + 1]]
        is_fri = is_t.loc[is_t["dow"] == 4, "net"].values
        mu_is, t_is = newey_west_t(is_fri, 1)
        oos_fri = oos_t.loc[oos_t["dow"] == 4, "net"].values
        mu_oos, t_oos = newey_west_t(oos_fri, 1)
        lo_oos, hi_oos = block_boot_ci(oos_fri, 10, cfg["n_boot"])
        passed = (mu_is > 0) and np.isfinite(t_oos) and t_oos >= cfg["min_abs_t"] and mu_oos > 0 and np.isfinite(lo_oos) and lo_oos > 0
        print(f"  Fold {k}: IS n={len(is_fri)} t={t_is:.2f} | OOS n={len(oos_fri)} "
              f"mean={bps(mu_oos):.2f}bps t={t_oos:.2f} CI=[{bps(lo_oos):.2f},{bps(hi_oos):.2f}] "
              f"-> {'PASS' if passed else 'FAIL'}")
        fold_res.append(passed)
    print(f"  SUMMARY: {sum(fold_res)}/{len(fold_res)} fold PASS")

    # ===== 4. NFP-Friday vs Friday ທຳມະດາ =====
    print(f"\n{'='*70}\n4) NFP-Friday (ວັນສຸກທຳອິດຂອງເດືອນ) vs Friday ທຳມະດາ\n{'='*70}")
    nfp = trades.loc[trades["is_nfp_friday"], "net"].values
    normal_fri = trades.loc[(trades["dow"] == 4) & (~trades["is_nfp_friday"]), "net"].values
    mu_nfp, t_nfp = newey_west_t(nfp, 1)
    mu_nf, t_nf = newey_west_t(normal_fri, 1)
    print(f"  NFP-Friday    : n={len(nfp)}  mean={bps(mu_nfp):.2f}bps  t={t_nfp:.2f}")
    print(f"  Friday ທຳມະດາ : n={len(normal_fri)}  mean={bps(mu_nf):.2f}bps  t={t_nf:.2f}")
    diff2, pdiff2 = perm_diff_pvalue(nfp, normal_fri, cfg["n_perm"])
    print(f"  diff (NFP-normal) = {bps(diff2):.2f}bps  p={pdiff2:.4f}")
    print("  " + ("-> effect ອາດມາຈາກ NFP ໂດຍສະເພາະ" if (mu_nfp > mu_nf and pdiff2 < 0.10)
                   else "-> effect ບໍ່ໄດ້ພຸ່ງໃສ່ NFP ໂດຍສະເພາະ, ກະຈາຍທົ່ວວັນສຸກທຸກອາທິດ"))

    # ===== 5. ຄວາມຄົງທີ່ຂ້າມຍຸກ =====
    print(f"\n{'='*70}\n5) ຄວາມຄົງທີ່ຂ້າມຍຸກ (4 ຊ່ວງ, ~5-6 ປີ/ຊ່ວງ)\n{'='*70}")
    edges5 = np.linspace(0, n, 5).astype(int)
    for k in range(4):
        seg = trades.iloc[edges5[k]:edges5[k + 1]]
        seg_fri = seg.loc[seg["dow"] == 4, "net"].values
        mu_s, t_s = newey_west_t(seg_fri, 1)
        d0, d1 = seg["date"].iloc[0].date(), seg["date"].iloc[-1].date()
        print(f"  ຊ່ວງ {k+1} ({d0} -> {d1}): n={len(seg_fri)}  mean={bps(mu_s):.2f}bps  t={t_s:.2f}")

    analyze_year_by_year(trades, cfg)
    analyze_session_decomposition(trades, cfg)


if __name__ == "__main__":
    main()
