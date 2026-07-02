#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barrier_exit_test.py
=======================================================================
ຄຳຖາມ: ຖ້າປ່ຽນ exit ຈາກ "ຄົງທີ່ N ແທ່ງ" (fixed-bar, ໃຊ້ໃນ script ກ່ອນໜ້າທັງໝົດ)
ໄປເປັນ "ATR-based Stop-Loss / Take-Profit" (ໃກ້ຄຽງວິທີເທຣດຈິງກວ່າ) — z-score
spike signal ຈະມີ edge ຂຶ້ນມາບໍ່ (ບໍ່ວ່າຈະ fade ຫຼື momentum)?

ນີ້ຄືການແກ້ຈຸດອ່ອນໂຄງສ້າງທີ່ພົບຈາກ 5 ການທົດລອງກ່ອນໜ້າ: exit ແບບ fixed-bar
ອາດບັງ edge ທີ່ມີແທ້ (ບໍ່ໃຫ້ winner ແລ່ນ, ບໍ່ຕັດ loser ໄວ).

Exit logic (bar-by-bar, ບໍ່ໃຊ້ future data ນອກຈາກ high/low ຂອງແທ່ງທີ່ຜ່ານໄປແລ້ວ):
   Long : SL = entry - sl_mult*ATR,  TP = entry + tp_mult*ATR
   Short: SL = entry + sl_mult*ATR,  TP = entry - tp_mult*ATR
   ໄລຍະ scan ບາຮ໌ຫຼັງ entry ຈົນຮອດ max_hold ແທ່ງ; ຖ້າແທ່ງໃດແຕະທັງ SL ແລະ TP
   ພ້ອມກັນ ໃຫ້ຖືວ່າ SL ຖືກແຕະກ່ອນ (conservative, ກັນ over-optimistic).
   ຖ້າບໍ່ແຕະຫຍັງພາຍໃນ max_hold -> ອອກທີ່ close ຂອງແທ່ງສຸດທ້າຍ.
ATR ແລະ z-score ຄິດໄລ່ causal ຄືກັນກັບ script ອື່ນ. Cost = spread ຈິງ.

ໂຕທົດສອບ: Newey-West HAC t-test + direction-permutation test + block
bootstrap CI + walk-forward OOS (5 fold expanding), ຄືໂຄງສ້າງເກົ່າ.
ທົດສອບທັງ 2 ທິດທາງ (fade, momentum) ດ້ວຍ engine ດຽວກັນ.

ໝາຍເຫດ: script standalone.
=======================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(2025)

CONFIG = {
    "use_synthetic": False,
    "data_path": "XAUUSD_M15.csv",

    "vol_window": 100,
    "atr_period": 14,
    "z_thr": 2.5,               # ຄົງທີ່ (ບໍ່ grid, ອີງຄ່າກາງທີ່ໃຊ້ໃນ script ກ່ອນໜ້າ)
    "sl_mults": [1.0, 1.5, 2.0],
    "tp_mults": [1.5, 2.0, 3.0],
    "max_hold": 40,             # ຂອບເຂດເວລາສູງສຸດ (ແທ່ງ) ກ່ອນບັງຄັບອອກ
    "directions": ["fade", "momentum"],

    "point_size": 0.01,
    "cost_roundtrip": 0.00025,  # fallback ສະເພາະ synthetic

    "use_walk_forward": True,
    "n_folds": 6,
    "n_perm": 2000,
    "n_boot": 2000,
    "block_len": 20,

    "min_abs_t": 3.0,
    "max_perm_p": 0.01,
    "min_trades": 80,
    "n_hypotheses_tried": 2,    # ທົດສອບ 2 ທິດທາງ (fade + momentum)

    # synthetic: 'none' (ບໍ່ມີໂຄງສ້າງ) | 'reversion' (jump ແລ້ວກັບຕົວ - ຄວນເຫັນ fade PASS, momentum FAIL)
    "synth_mode": "reversion",
    "synth_n": 200000,
    "synth_seed": 11,

    "results_csv": "barrier_exit_walk_forward_log.csv",
}


# =====================================================================
# DATA
# =====================================================================
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


def make_synthetic(n, seed, mode):
    """
    mode: 'none' -> random walk ລ້ວນໆ (ຄວນ FAIL ທັງ 2 ທິດ)
          'reversion' -> jump ແລ້ວກັບຕົວ (ຄວນ fade PASS, momentum FAIL)
    """
    rng = np.random.default_rng(seed)
    base_vol = 0.0009
    rets = rng.normal(0, base_vol, n)
    jump = rng.random(n) < 0.012
    jdir = rng.choice([-1.0, 1.0], n)
    jmag = 0.010 * (0.5 + rng.random(n))
    rets = rets + jump * jdir * jmag

    if mode == "reversion":
        for t in np.where(jump)[0]:
            if t + 10 >= n:
                continue
            for k in range(1, 11):
                rets[t + k] -= 1.0 * jdir[t] * jmag[t] / 10.0

    close = 2000.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2015-01-01", periods=n, freq="15min")
    o = np.concatenate([[close[0]], close[:-1]])
    noise = np.abs(rng.normal(0, base_vol, n)) * close
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) + noise,
                         "low": np.minimum(o, close) - noise, "close": close}, index=idx)


# =====================================================================
# FEATURES
# =====================================================================
def compute_atr(df, period):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def prepare(df, cfg):
    close = df["close"]
    ret = np.log(close).diff()
    sigma = ret.rolling(cfg["vol_window"]).std().shift(1)
    df = df.copy()
    df["z"] = ret / sigma
    df["atr"] = compute_atr(df, cfg["atr_period"])
    if "spread" in df.columns:
        df["cost"] = (df["spread"] * cfg["point_size"] / df["close"]).clip(lower=0)
    else:
        df["cost"] = cfg["cost_roundtrip"]
    return df


def simulate_barrier_trades(df, z_thr, direction, sl_mult, tp_mult, max_hold):
    """ຄືນ array ຂອງ net return ຕໍ່ trade, ໃຊ້ ATR SL/TP barrier exit (ບໍ່ແມ່ນ fixed-bar)."""
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    z = df["z"].values
    atr = df["atr"].values
    cost = df["cost"].values
    n = len(close)

    up_val = -1.0 if direction == "fade" else 1.0
    dn_val = 1.0 if direction == "fade" else -1.0
    raw_side = np.zeros(n)
    raw_side[z >= z_thr] = up_val
    raw_side[z <= -z_thr] = dn_val

    cd = max_hold
    side = np.zeros(n)
    last_trigger = -(cd + 1)
    for i in range(n):
        if raw_side[i] != 0 and (i - last_trigger) > cd:
            side[i] = raw_side[i]
            last_trigger = i

    net_out = []
    for i in np.where(side != 0)[0]:
        entry_i = i + 1
        if entry_i >= n or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        s = side[i]
        entry_price = open_[entry_i]
        if s > 0:
            sl_price = entry_price - sl_mult * atr[i]
            tp_price = entry_price + tp_mult * atr[i]
        else:
            sl_price = entry_price + sl_mult * atr[i]
            tp_price = entry_price - tp_mult * atr[i]

        last_j = min(entry_i + max_hold - 1, n - 1)
        exit_price = None
        for j in range(entry_i, last_j + 1):
            if s > 0:
                if low[j] <= sl_price:
                    exit_price = sl_price; break
                if high[j] >= tp_price:
                    exit_price = tp_price; break
            else:
                if high[j] >= sl_price:
                    exit_price = sl_price; break
                if low[j] <= tp_price:
                    exit_price = tp_price; break
        if exit_price is None:
            exit_price = close[last_j]

        fwd = exit_price / entry_price - 1.0
        net_out.append(s * fwd - cost[i])

    return np.array(net_out)


# =====================================================================
# STATS
# =====================================================================
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


def sign_perm_pvalue(net, n_perm):
    """H0: ທິດທາງ trade ບໍ່ມີຄວາມໝາຍ - ສຸ່ມ sign ຂອງແຕ່ລະ trade (±net ດ້ວຍຄວາມນ່າຈະເປັນ
    ເທົ່າກັນ) ແລ້ວປຽບທຽບ mean ຈິງ ກັບ null."""
    net = np.asarray(net, float)
    n = len(net)
    if n < 5:
        return np.nan
    obs = net.mean()
    cnt = 0
    for _ in range(n_perm):
        flips = RNG.choice(np.array([-1.0, 1.0]), size=n)
        if (flips * np.abs(net)).mean() >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def bps(x):
    return np.nan if x is None or (isinstance(x, float) and not np.isfinite(x)) else x * 1e4


# =====================================================================
# IS SELECTION + OOS VALIDATION
# =====================================================================
def scan_is(df_is, cfg, direction, verbose=True):
    best = None
    if verbose:
        print(f"\n=== IN-SAMPLE scan ({direction}): mean (bps) | HAC t ===")
    for sl in cfg["sl_mults"]:
        for tp in cfg["tp_mults"]:
            net = simulate_barrier_trades(df_is, cfg["z_thr"], direction, sl, tp, cfg["max_hold"])
            if len(net) < 40:
                continue
            mu, t = newey_west_t(net, cfg["max_hold"])
            if verbose:
                print(f"  sl={sl} tp={tp}: n={len(net):>5}  mean={bps(mu):>6.2f}bps  t={t:>5.2f}")
            if mu > 0 and np.isfinite(t) and (best is None or t > best["t"]):
                best = dict(sl=sl, tp=tp, t=t, mu=mu, n=len(net))
    return best


def evaluate_oos(df_oos, cfg, direction, sl, tp):
    net = simulate_barrier_trades(df_oos, cfg["z_thr"], direction, sl, tp, cfg["max_hold"])
    mu, t = newey_west_t(net, cfg["max_hold"])
    lo, hi = block_boot_ci(net, cfg["block_len"], cfg["n_boot"])
    pdir = sign_perm_pvalue(net, cfg["n_perm"]) if len(net) else np.nan
    winrate = (net > 0).mean() * 100 if len(net) else np.nan

    sig = np.isfinite(t) and abs(t) >= cfg["min_abs_t"] and mu > 0 and np.isfinite(lo) and lo > 0
    sig_perm = np.isfinite(pdir) and pdir <= cfg["max_perm_p"]
    enough = len(net) >= cfg["min_trades"]

    return dict(sl=sl, tp=tp, n=len(net), mu=mu, t=t, lo=lo, hi=hi, pdir=pdir, winrate=winrate,
                sig=sig, sig_perm=sig_perm, enough=enough,
                passed=(mu > 0 and sig and sig_perm and enough))


def print_oos_report(m, cfg, label="OUT-OF-SAMPLE"):
    print(f"\n=== {label} validation ===")
    print(f"  param: sl_mult={m['sl']}, tp_mult={m['tp']}, max_hold={cfg['max_hold']}")
    print(f"  trades: n={m['n']:>4}  win rate={m['winrate']:>5.1f}%  mean={bps(m['mu']):>6.2f}bps  HAC t={m['t']:>5.2f}")
    print(f"  bootstrap 95% CI : [{bps(m['lo']):.2f}, {bps(m['hi']):.2f}] bps")
    print(f"  sign-perm p      : {m['pdir']:.4f}")

    print("\n=== SCORECARD (OOS) ===")
    print(f"  [{'PASS' if m['mu'] > 0 else 'FAIL'}] mean > 0")
    print(f"  [{'PASS' if m['sig'] else 'FAIL'}] significant (HAC t>= {cfg['min_abs_t']}, CI>0)")
    print(f"  [{'PASS' if m['sig_perm'] else 'FAIL'}] direction ດີກວ່າສຸ່ມ (perm p<= {cfg['max_perm_p']})")
    print(f"  [{'PASS' if m['enough']  else 'FAIL'}] trades >= {cfg['min_trades']}")

    if m["passed"]:
        print("\n>>> VERDICT: ມີ edge ຈິງ (ຜ່ານ OOS) — ຄຸ້ມພິຈາລະນາໃສ່ໃນ EA.")
    else:
        print("\n>>> VERDICT: ບໍ່ພິສູດວ່າຊ່ວຍ.")


def run_walk_forward(df, cfg, direction):
    n = len(df)
    n_folds = cfg["n_folds"]
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    reports = []

    print(f"\n{'='*70}\nWALK-FORWARD [{direction}]: {n_folds - 1} OOS folds (expanding IS window)\n{'='*70}")
    for k in range(1, n_folds):
        df_is = df.iloc[:edges[k]]
        df_oos = df.iloc[edges[k]:edges[k + 1]]
        print(f"\n--- Fold {k}/{n_folds - 1}  IS bars={len(df_is):,}  OOS bars={len(df_oos):,}  "
              f"OOS {df_oos.index[0].date()} -> {df_oos.index[-1].date()} ---")

        best = scan_is(df_is, cfg, direction, verbose=False)
        if best is None:
            print("  IS: ບໍ່ພົບ (sl,tp) ທີ່ mean>0 -> FAIL")
            reports.append(dict(passed=False, n=0, mu=np.nan, sl=None, tp=None))
            continue

        print(f"  IS best: sl={best['sl']} tp={best['tp']} (HAC t={best['t']:.2f})")
        m = evaluate_oos(df_oos, cfg, direction, best["sl"], best["tp"])
        print_oos_report(m, cfg, label=f"Fold {k} OOS")
        m["fold"] = k
        reports.append(m)

    n_pass = sum(1 for m in reports if m.get("passed"))
    total = len(reports)
    means = [float(bps(m["mu"])) for m in reports if np.isfinite(m.get("mu", np.nan))]
    print(f"\n{'='*70}\nWALK-FORWARD SUMMARY [{direction}]: {n_pass}/{total} fold ຜ່ານ scorecard ຄົບທຸກເງື່ອນໄຂ")
    if means:
        print(f"mean return ຂອງແຕ່ລະ fold (bps): {[round(d, 1) for d in means]}")

    n_hyp = max(1, cfg["n_hypotheses_tried"])
    adj_p = cfg["max_perm_p"] / n_hyp
    print(f"ໝາຍເຫດ: Bonferroni-adjusted p threshold ({n_hyp} ທິດທາງທົດສອບ) = {adj_p:.5f}")

    if n_pass >= max(1, int(np.ceil(total * 0.6))):
        print(">>> OVERALL: edge ຄົງທີ່ຢູ່ຫຼາຍ fold — ໜ້າເຊື່ອຖືກວ່າ split ດຽວ")
    else:
        print(">>> OVERALL: edge ບໍ່ຄົງທີ່ຂ້າມ fold — ຢ່າເອົາໃສ່ EA ຈາກຜົນນີ້ຢ່າງດຽວ")
    print("=" * 70)

    rows = []
    for m in reports:
        rows.append({
            "run_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "direction": direction, "fold": m.get("fold"),
            "sl_mult": m.get("sl"), "tp_mult": m.get("tp"),
            "n": m.get("n"), "winrate_pct": m.get("winrate"),
            "mean_bps": bps(m.get("mu")), "t": m.get("t"),
            "perm_p": m.get("pdir"), "passed": m.get("passed", False),
        })
    out_path = cfg["results_csv"]
    header = not os.path.exists(out_path)
    pd.DataFrame(rows).to_csv(out_path, mode="a", header=header, index=False)
    print(f"ບັນທຶກຜົນ {len(rows)} fold ໃສ່ {out_path} ແລ້ວ (append)")
    return n_pass, total


def main():
    cfg = CONFIG
    if cfg["use_synthetic"]:
        print(f">>> SYNTHETIC (mode={cfg['synth_mode']}) — ໃຊ້ validate ໂຕທົດສອບ, ຜົນບໍ່ມີຄວາມໝາຍທາງເງິນ")
        df = make_synthetic(cfg["synth_n"], cfg["synth_seed"], cfg["synth_mode"])
    else:
        print(f">>> REAL data: {cfg['data_path']}")
        df = load_ohlcv(cfg["data_path"])

    df = prepare(df, cfg)
    print(f"Bars: {len(df):,} | {df.index[0]} -> {df.index[-1]}")
    print(f"Cost: {'ອິງ spread ຈິງ (bps ສະເລ່ຍ=' + format(bps(df['cost'].mean()), '.2f') + ')' if 'spread' in df.columns else 'flat ' + format(bps(cfg['cost_roundtrip']), '.2f') + 'bps'}")

    summary = {}
    for direction in cfg["directions"]:
        if cfg["use_walk_forward"]:
            n_pass, total = run_walk_forward(df, cfg, direction)
            summary[direction] = f"{n_pass}/{total}"
        else:
            split = int(len(df) * 0.70)
            df_is, df_oos = df.iloc[:split].copy(), df.iloc[split:].copy()
            best = scan_is(df_is, cfg, direction, verbose=True)
            if best is None:
                print(f"\n>>> [{direction}] IS: ບໍ່ພົບ (sl,tp) ທີ່ mean>0.")
                continue
            m = evaluate_oos(df_oos, cfg, direction, best["sl"], best["tp"])
            print_oos_report(m, cfg, label=direction.upper())

    if summary:
        print(f"\n{'#'*70}\nສະຫຼຸບທຸກທິດທາງ: {summary}\n{'#'*70}")


if __name__ == "__main__":
    main()
