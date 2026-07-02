#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_momentum_test.py
=======================================================================
ຄຳຖາມ: ຫຼັງ return spike (|z| >= z_thr), ລາຄາ "ໄປຕໍ່ໃນທິດເດີມ" (momentum)
ດີກວ່າ "ຍ້ອນກັບ" (fade) ຫຼືບໍ່?

ຕົ້ນເຫດຜົນ: spike_fade_zone_filter.py ພິສູດແລ້ວວ່າ fade-the-spike (ບໍ່ວ່າຈະ
ໃສ່ zone filter ຫຼືບໍ່) ໃຫ້ mean return ເປັນລົບຢ່າງມີນັຍສຳຄັນເກືອບທຸກ
walk-forward fold ໃນ XAUUSD M15 ຂໍ້ມູນຈິງ (23 ປີ) — script ນີ້ທົດສອບທິດ
ກົງກັນຂ້າມ: ຊື້ຕາມທິດ spike ແທນ (momentum / continuation).

Signal: side = sign(z)
   z >= z_thr  -> long  (ຕາມທິດ spike ຂຶ້ນ)
   z <= -z_thr -> short (ຕາມທິດ spike ລົງ)
Entry: open ຂອງແທ່ງຖັດຈາກ signal (ບໍ່ແມ່ນ close ຂອງແທ່ງ signal ເອງ).
Exit: close ຫຼັງ horizon ແທ່ງ.
Cost: spread ຈິງຕໍ່ແທ່ງ (scale ຕາມລາຄາ), ຄືກັນກັບ spike_fade_zone_filter.py.
Cooldown: ຫຼັງ signal ຮັບ 1 ຄັ້ງ, ຂ້າມ `horizon` ແທ່ງ ບໍ່ຮັບ signal ໃໝ່
          (ກັນ trade ຊ້ອນກັນ ບໍ່ independent).

ໂຕທົດສອບ:
   H0: mean(net) <= 0        -> momentum ບໍ່ມີ edge
   H1: mean(net) > 0 ຄົງທີ່  -> momentum ມີ edge ຈິງ
ໃຊ້: Newey-West HAC t-test + direction-permutation test + block bootstrap CI
     + walk-forward OOS (5 fold expanding, ຄືກັນກັບ spike_fade_zone_filter.py).

ໝາຍເຫດ: script standalone, ບໍ່ import spike_fade_zone_filter.py.
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
    "use_log_returns": True,

    "cost_roundtrip": 0.00025,   # fallback ສະເພາະ synthetic (ບໍ່ມີ column spread)
    "point_size": 0.01,          # 1 point = 0.01 (XAUUSD, 2 ຫຼັກທົດສະນິຍົມ)

    "z_thresholds": [2.0, 2.5, 3.0],
    "horizons":     [3, 5, 10],

    "use_walk_forward": True,
    "n_folds": 6,
    "n_perm": 2000,
    "n_boot": 2000,
    "block_len": 20,

    # verdict
    "min_abs_t": 3.0,
    "max_perm_p": 0.01,
    "min_trades": 150,
    "n_hypotheses_tried": 1,     # ປັບຂຶ້ນຖ້າລອງ z/horizon grid ອື່ນນອກ script

    # synthetic: 'none' (ບໍ່ມີ momentum, ຄວນ FAIL) | 'momentum' (ມີ continuation ຈິງ, ຄວນ PASS)
    "synth_mode": "momentum",
    "synth_n": 360000,
    "synth_seed": 11,

    "results_csv": "momentum_walk_forward_log.csv",
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
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
    else:
        dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.assign(dt=dt).dropna(subset=["dt"]).sort_values("dt").set_index("dt")
    cols = ["open", "high", "low", "close"]
    if "spread" in df.columns:
        cols.append("spread")
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[cols].dropna()


def make_synthetic(n, seed, mode):
    """
    ສ້າງ data ປອມ. mode:
      'none'     -> ບໍ່ມີ continuation ຫຼັງ jump (ຄວນໄດ້ NO EDGE)
      'momentum' -> ລາຄາໄປຕໍ່ໃນທິດ jump ຫຼັງ jump (momentum ຄວນຊ່ວຍ)
    """
    rng = np.random.default_rng(seed)
    base_vol = 0.0009
    rets = rng.normal(0, base_vol, n)
    jump = rng.random(n) < 0.012
    jdir = rng.choice([-1.0, 1.0], n)
    jmag = 0.010 * (0.5 + rng.random(n))
    rets = rets + jump * jdir * jmag

    if mode == "momentum":
        cont_strength = 0.8
        for t in np.where(jump)[0]:
            if t + 5 >= n:
                continue
            for k in range(1, 6):
                rets[t + k] += cont_strength * jdir[t] * jmag[t] / 5.0

    close = 2000.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2015-01-01", periods=n, freq="15min")
    o = np.concatenate([[close[0]], close[:-1]])
    noise = np.abs(rng.normal(0, base_vol, n)) * close
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) + noise,
                         "low": np.minimum(o, close) - noise, "close": close}, index=idx)


# =====================================================================
# FEATURES (all causal, no look-ahead)
# =====================================================================
def compute_atr(df, period):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def prepare(df, cfg):
    close = df["close"]
    ret = np.log(close).diff() if cfg["use_log_returns"] else close.pct_change()
    sigma = ret.rolling(cfg["vol_window"]).std().shift(1)     # past-only
    df = df.copy()
    df["ret"] = ret
    df["z"] = ret / sigma
    df["atr"] = compute_atr(df, cfg["atr_period"])
    if "spread" in df.columns:
        df["cost"] = (df["spread"] * cfg["point_size"] / df["close"]).clip(lower=0)
    else:
        df["cost"] = cfg["cost_roundtrip"]
    return df


def build_trades(df, z_thr, horizon):
    """ຄືນ DataFrame ຂອງ trade momentum: side, fwd (raw forward return), cost, net.
    Cooldown ແລະ entry-at-next-open ຄືກັນກັບ spike_fade_zone_filter.py."""
    close = df["close"].values
    open_ = df["open"].values
    z = df["z"].values
    cost = df["cost"].values
    n = len(close)

    raw_side = np.zeros(n)
    raw_side[z >= z_thr] = 1.0    # spike ຂຶ້ນ -> long ຕາມທິດ
    raw_side[z <= -z_thr] = -1.0  # spike ລົງ -> short ຕາມທິດ

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
    s = side[idx]; f = fwd[idx]; c = cost[idx]
    net = s * f - c
    return pd.DataFrame({"pos": idx, "side": s, "fwd": f, "cost": c, "net": net})


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


def perm_direction_pvalue(fwd, cost, obs_mean, n_perm):
    """H0: ທິດທາງ (side) ບໍ່ມີຄວາມໝາຍ - ສຸ່ມທິດແທນ ກໍ່ໄດ້ຜົນຄືກັນ.
    ປຽບທຽບ mean(net) ຈິງ ກັບ null distribution ຈາກທິດສຸ່ມ (±1) ຂອງ trade ດຽວກັນ."""
    fwd = np.asarray(fwd, float); cost = np.asarray(cost, float)
    n = len(fwd)
    if n < 5:
        return np.nan
    cnt = 0
    for _ in range(n_perm):
        rs = RNG.choice(np.array([-1.0, 1.0]), size=n)
        m = (rs * fwd - cost).mean()
        if m >= obs_mean:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


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


# =====================================================================
# IS SELECTION + OOS VALIDATION
# =====================================================================
def scan_is(df_is, cfg, verbose=True):
    """ໃນ IS: ຫາ (z,h) ທີ່ momentum mean return / HAC t ແຮງທີ່ສຸດ."""
    best = None
    if verbose:
        print("\n=== IN-SAMPLE scan: momentum mean (bps) | HAC t ===")
    for z in cfg["z_thresholds"]:
        for h in cfg["horizons"]:
            tr = build_trades(df_is, z, h)
            net = tr["net"].values
            if len(net) < 60:
                continue
            mu, t = newey_west_t(net, h)
            if verbose:
                print(f"  z={z} h={h:>2}: n={len(net):>5}  mean={bps(mu):>6.2f}bps  t={t:>5.2f}")
            if mu > 0 and np.isfinite(t) and (best is None or t > best["t"]):
                best = dict(z=z, h=h, t=t, mu=mu, n=len(net))
    return best


def evaluate_oos(df_oos, cfg, z, h):
    tr = build_trades(df_oos, z, h)
    net = tr["net"].values
    mu, t = newey_west_t(net, h)
    lo, hi = block_boot_ci(net, cfg["block_len"], cfg["n_boot"])
    pdir = perm_direction_pvalue(tr["fwd"].values, tr["cost"].values, mu, cfg["n_perm"]) if len(net) else np.nan

    sig = np.isfinite(t) and abs(t) >= cfg["min_abs_t"] and mu > 0 and np.isfinite(lo) and lo > 0
    sig_perm = np.isfinite(pdir) and pdir <= cfg["max_perm_p"]
    enough = len(net) >= cfg["min_trades"]

    return dict(z=z, h=h, n=len(net), mu=mu, t=t, lo=lo, hi=hi, pdir=pdir,
                sig=sig, sig_perm=sig_perm, enough=enough,
                passed=(mu > 0 and sig and sig_perm and enough))


def print_oos_report(m, cfg, label="OUT-OF-SAMPLE"):
    print(f"\n=== {label} validation ===")
    print(f"  param: z={m['z']}, horizon={m['h']}")
    print(f"  momentum trades : n={m['n']:>4}  mean={bps(m['mu']):>6.2f}bps  HAC t={m['t']:>5.2f}")
    print(f"  bootstrap 95% CI: [{bps(m['lo']):.2f}, {bps(m['hi']):.2f}] bps")
    print(f"  direction-perm p: {m['pdir']:.4f}  (H0: ທິດສຸ່ມກໍ່ໄດ້ຜົນຄືກັນ)")

    print("\n=== SCORECARD (OOS) ===")
    print(f"  [{'PASS' if m['mu'] > 0 else 'FAIL'}] mean > 0")
    print(f"  [{'PASS' if m['sig'] else 'FAIL'}] significant (HAC t>= {cfg['min_abs_t']}, CI>0)")
    print(f"  [{'PASS' if m['sig_perm'] else 'FAIL'}] direction ດີກວ່າສຸ່ມ (perm p<= {cfg['max_perm_p']})")
    print(f"  [{'PASS' if m['enough']  else 'FAIL'}] trades >= {cfg['min_trades']}")

    if m["passed"]:
        print("\n>>> VERDICT: momentum ມີ edge ຈິງ (ຜ່ານ OOS) — ຄຸ້ມພິຈາລະນາໃສ່ໃນ EA.")
    else:
        print("\n>>> VERDICT: momentum ບໍ່ພິສູດວ່າຊ່ວຍ.")


def run_walk_forward(df, cfg):
    n = len(df)
    n_folds = cfg["n_folds"]
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    reports = []

    print(f"\n{'='*70}\nWALK-FORWARD: {n_folds - 1} OOS folds (expanding IS window)\n{'='*70}")
    for k in range(1, n_folds):
        df_is = df.iloc[:edges[k]]
        df_oos = df.iloc[edges[k]:edges[k + 1]]
        print(f"\n--- Fold {k}/{n_folds - 1}  IS bars={len(df_is):,}  OOS bars={len(df_oos):,}  "
              f"OOS {df_oos.index[0].date()} -> {df_oos.index[-1].date()} ---")

        best = scan_is(df_is, cfg, verbose=False)
        if best is None:
            print("  IS: ບໍ່ພົບ (z,h) ທີ່ mean>0 -> FAIL")
            reports.append(dict(passed=False, n=0, mu=np.nan, z=None, h=None))
            continue

        print(f"  IS best: z={best['z']} h={best['h']} (HAC t={best['t']:.2f})")
        m = evaluate_oos(df_oos, cfg, best["z"], best["h"])
        print_oos_report(m, cfg, label=f"Fold {k} OOS")
        m["fold"] = k
        reports.append(m)

    n_pass = sum(1 for m in reports if m.get("passed"))
    total = len(reports)
    means = [float(bps(m["mu"])) for m in reports if np.isfinite(m.get("mu", np.nan))]
    print(f"\n{'='*70}\nWALK-FORWARD SUMMARY: {n_pass}/{total} fold ຜ່ານ scorecard ຄົບທຸກເງື່ອນໄຂ")
    if means:
        print(f"mean return ຂອງແຕ່ລະ fold (bps): {[round(d, 1) for d in means]}")

    n_hyp = max(1, cfg["n_hypotheses_tried"])
    adj_p = cfg["max_perm_p"] / n_hyp
    if n_hyp > 1:
        print(f"\nໝາຍເຫດ: Bonferroni-adjusted p threshold ({n_hyp} hypothesis) = {adj_p:.5f}")

    if n_pass >= max(1, int(np.ceil(total * 0.6))):
        print(">>> OVERALL: edge ຄົງທີ່ຢູ່ຫຼາຍ fold — ໜ້າເຊື່ອຖືກວ່າ split ດຽວ")
    else:
        print(">>> OVERALL: edge ບໍ່ຄົງທີ່ຂ້າມ fold — ຢ່າເອົາໃສ່ EA ຈາກຜົນນີ້ຢ່າງດຽວ")
    print("=" * 70)

    rows = []
    for m in reports:
        rows.append({
            "run_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fold": m.get("fold"),
            "z": m.get("z"), "horizon": m.get("h"),
            "n": m.get("n"), "mean_bps": bps(m.get("mu")), "t": m.get("t"),
            "perm_p": m.get("pdir"), "passed": m.get("passed", False),
        })
    out_path = cfg["results_csv"]
    header = not os.path.exists(out_path)
    pd.DataFrame(rows).to_csv(out_path, mode="a", header=header, index=False)
    print(f"ບັນທຶກຜົນ {len(rows)} fold ໃສ່ {out_path} ແລ້ວ (append)")


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

    if cfg["use_walk_forward"]:
        run_walk_forward(df, cfg)
        return

    split = int(len(df) * 0.70)
    df_is, df_oos = df.iloc[:split].copy(), df.iloc[split:].copy()
    best = scan_is(df_is, cfg, verbose=True)
    if best is None:
        print("\n>>> IS: ບໍ່ພົບ (z,h) ທີ່ mean>0. VERDICT: momentum ບໍ່ຊ່ວຍ.")
        return
    print(f"\nBest IS combo: z={best['z']}, h={best['h']} (HAC t={best['t']:.2f}, mean={bps(best['mu']):.1f}bps)")
    m = evaluate_oos(df_oos, cfg, best["z"], best["h"])
    print_oos_report(m, cfg)


if __name__ == "__main__":
    main()
