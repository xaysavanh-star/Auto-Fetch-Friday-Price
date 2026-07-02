#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rsi_divergence_test.py
=======================================================================
ຄຳຖາມ: "RSI divergence" (ລາຄາ higher-high/lower-low ແຕ່ RSI ບໍ່ໄດ້) ໃຫ້
edge ຈິງ ໃນ XAUUSD M15 ຫຼືບໍ່?

ນິຍາມ (regular divergence, ຮູບແບບຄລາສສິກ - ບໍ່ລວມ hidden divergence):
   Bearish: swing high ຫຼ້າສຸດຂອງລາຄາ ສູງກວ່າ swing high ກ່ອນໜ້າ (higher-high)
            ແຕ່ RSI ຂອງ swing high ຫຼ້າສຸດ ຕ່ຳກວ່າ swing high ກ່ອນໜ້າ (lower-high)
            -> ຄາດວ່າຈະກັບຕົວລົງ -> short
   Bullish: swing low ຫຼ້າສຸດ ຕ່ຳກວ່າ swing low ກ່ອນໜ້າ (lower-low)
            ແຕ່ RSI ຂອງ swing low ຫຼ້າສຸດ ສູງກວ່າ swing low ກ່ອນໜ້າ (higher-low)
            -> ຄາດວ່າຈະກັບຕົວຂຶ້ນ -> long

Swing high/low ນິຍາມແບບ pivot (center window): bar i ເປັນ swing high ຖ້າ
close[i] ສູງສຸດໃນຊ່ວງ [i-pivot_window, i+pivot_window]. ນີ້ "ຮູ້ໄດ້" ຈິງ
ພຽງຫຼັງຜ່ານໄປ pivot_window ແທ່ງ (confirmation lag) — signal ຈຶ່ງຖືກຍົກໄປໃຊ້
ຢູ່ index i+pivot_window (ບໍ່ແມ່ນ i) ເພື່ອບໍ່ໃຫ້ look-ahead.

Entry: open ຂອງແທ່ງຖັດຈາກ confirm bar.  Exit: close ຫຼັງ horizon ແທ່ງ.
Cost: spread ຈິງຕໍ່ແທ່ງ. Cooldown: ຄືກັນກັບ script ອື່ນ (ກັນ signal ຊ້ອນກັນ).

ໂຕທົດສອບ (ຄືໂຄງສ້າງ spike_momentum_test.py):
   H0: mean(net) <= 0        -> RSI divergence ບໍ່ມີ edge
   H1: mean(net) > 0 ຄົງທີ່  -> RSI divergence ມີ edge ຈິງ
ໃຊ້: Newey-West HAC t-test + direction-permutation test + block bootstrap CI
     + walk-forward OOS (5 fold expanding).

ໝາຍເຫດ: script standalone, ບໍ່ import ໄຟລ໌ອື່ນ.
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

    "rsi_period": 14,           # ຄ່າມາດຕະຖານ, ບໍ່ໃສ່ໃນ grid (ກັນ multiple-testing ບານ)
    "point_size": 0.01,
    "cost_roundtrip": 0.00025,  # fallback ສະເພາະ synthetic

    # grids (ນ້ອຍໆ ເພື່ອຄຸມ multiple-testing)
    "pivot_windows": [3, 5, 8],     # ຄວາມກວ້າງ swing confirmation
    "horizons":      [5, 10, 20],   # holding period ຫຼັງ entry
    "min_swing_gap": 5,             # ໄລຍະຫ່າງຂັ້ນຕ່ຳລະຫວ່າງ swing pair ທີ່ປຽບທຽບ

    "use_walk_forward": True,
    "n_folds": 6,
    "n_perm": 2000,
    "n_boot": 2000,
    "block_len": 20,

    # verdict (min_trades ຕ່ຳກວ່າ script ອື່ນ ເພາະ divergence event ໜ້ອຍກວ່າ spike)
    "min_abs_t": 3.0,
    "max_perm_p": 0.01,
    "min_trades": 80,
    "n_hypotheses_tried": 1,

    # synthetic: 'none' (ບໍ່ມີ edge, ຄວນ FAIL) | 'divergence' (ມີ edge ຈິງ, ຄວນ PASS)
    "synth_mode": "divergence",
    "synth_n": 360000,
    "synth_seed": 11,

    "results_csv": "rsi_divergence_walk_forward_log.csv",
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


def make_synthetic(n, seed, mode, rsi_period=14, pivot_window=5, min_gap=5, fade_strength=0.006):
    """
    ສ້າງ data ປອມ. ວິທີການ: ສ້າງ random walk ພື້ນຖານກ່ອນ, ຄິດໄລ່ RSI+pivot ດ້ວຍ
    logic ດຽວກັນກັບ build_trades() ຈິງ, ຫາຈຸດທີ່ຈະເປັນ "divergence" ຕາມນິຍາມນັ້ນ,
    ແລ້ວຄ່ອຍ inject ການກັບຕົວ (reversion) ຢູ່ຈຸດເຫຼົ່ານັ້ນ (mode='divergence').
    ວິທີນີ້ຮັບປະກັນວ່າ signal ທີ່ injected ກົງກັບສິ່ງທີ່ detector ຈິງຈະຈັບໄດ້.
    mode='none' -> pure random walk, ບໍ່ inject ຫຍັງ (ຄວນໄດ້ NO EDGE).
    """
    rng = np.random.default_rng(seed)
    base_vol = 0.0009
    rets = rng.normal(0, base_vol, n)
    base_close = 2000.0 * np.exp(np.cumsum(rets))

    if mode == "divergence":
        s = pd.Series(base_close)
        rsi_v = compute_rsi(s, rsi_period).values
        is_high, is_low = find_pivots(s, pivot_window)
        high_idx = np.where(is_high)[0]
        low_idx = np.where(is_low)[0]

        for j in range(1, len(high_idx)):
            t1, t2 = high_idx[j - 1], high_idx[j]
            if t2 - t1 < min_gap or not (np.isfinite(rsi_v[t1]) and np.isfinite(rsi_v[t2])):
                continue
            if base_close[t2] > base_close[t1] and rsi_v[t2] < rsi_v[t1]:   # bearish divergence
                ct = t2 + pivot_window
                if ct + 15 < n:
                    rets[ct:ct + 15] -= fade_strength / 15.0
        for j in range(1, len(low_idx)):
            t1, t2 = low_idx[j - 1], low_idx[j]
            if t2 - t1 < min_gap or not (np.isfinite(rsi_v[t1]) and np.isfinite(rsi_v[t2])):
                continue
            if base_close[t2] < base_close[t1] and rsi_v[t2] > rsi_v[t1]:   # bullish divergence
                ct = t2 + pivot_window
                if ct + 15 < n:
                    rets[ct:ct + 15] += fade_strength / 15.0

    close = 2000.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2015-01-01", periods=n, freq="15min")
    o = np.concatenate([[close[0]], close[:-1]])
    noise = np.abs(rng.normal(0, base_vol, n)) * close
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) + noise,
                         "low": np.minimum(o, close) - noise, "close": close}, index=idx)


# =====================================================================
# FEATURES
# =====================================================================
def compute_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def find_pivots(series, window):
    """bar i ເປັນ swing high/low ຖ້າ close[i] ສູງສຸດ/ຕ່ຳສຸດ ໃນຊ່ວງ ±window ອ້ອມຮອບ.
    (raw pivot ຢູ່ index i - ຕ້ອງ shift ໄປ i+window ຕອນໃຊ້ຈິງ ເພື່ອບໍ່ໃຫ້ look-ahead)."""
    roll_max = series.rolling(2 * window + 1, center=True).max()
    roll_min = series.rolling(2 * window + 1, center=True).min()
    is_high = (series == roll_max) & roll_max.notna()
    is_low = (series == roll_min) & roll_min.notna()
    return is_high.values, is_low.values


def prepare(df, cfg):
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], cfg["rsi_period"])
    if "spread" in df.columns:
        df["cost"] = (df["spread"] * cfg["point_size"] / df["close"]).clip(lower=0)
    else:
        df["cost"] = cfg["cost_roundtrip"]
    return df


def build_trades(df, pivot_window, horizon, min_gap):
    """ຄືນ DataFrame trade: side, fwd, cost, net - ອີງ RSI regular divergence.
    Cooldown ແລະ entry-at-next-open ຄືກັນກັບ script ອື່ນ."""
    close = df["close"].values
    open_ = df["open"].values
    rsi = df["rsi"].values
    cost = df["cost"].values
    n = len(close)

    is_high, is_low = find_pivots(df["close"], pivot_window)
    high_idx = np.where(is_high)[0]
    low_idx = np.where(is_low)[0]
    confirm_offset = pivot_window

    raw_side = np.zeros(n)
    for j in range(1, len(high_idx)):
        t1, t2 = high_idx[j - 1], high_idx[j]
        if t2 - t1 < min_gap:
            continue
        if not (np.isfinite(rsi[t1]) and np.isfinite(rsi[t2])):
            continue
        if close[t2] > close[t1] and rsi[t2] < rsi[t1]:      # bearish divergence
            ct = t2 + confirm_offset
            if ct < n:
                raw_side[ct] = -1.0
    for j in range(1, len(low_idx)):
        t1, t2 = low_idx[j - 1], low_idx[j]
        if t2 - t1 < min_gap:
            continue
        if not (np.isfinite(rsi[t1]) and np.isfinite(rsi[t2])):
            continue
        if close[t2] < close[t1] and rsi[t2] > rsi[t1]:      # bullish divergence
            ct = t2 + confirm_offset
            if ct < n:
                raw_side[ct] = 1.0

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
# STATS (ຄືກັນກັບ spike_momentum_test.py)
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
    best = None
    if verbose:
        print("\n=== IN-SAMPLE scan: divergence mean (bps) | HAC t ===")
    for pw in cfg["pivot_windows"]:
        for h in cfg["horizons"]:
            tr = build_trades(df_is, pw, h, cfg["min_swing_gap"])
            net = tr["net"].values
            if len(net) < 40:
                continue
            mu, t = newey_west_t(net, h)
            if verbose:
                print(f"  pivot_w={pw} h={h:>2}: n={len(net):>5}  mean={bps(mu):>6.2f}bps  t={t:>5.2f}")
            if mu > 0 and np.isfinite(t) and (best is None or t > best["t"]):
                best = dict(pw=pw, h=h, t=t, mu=mu, n=len(net))
    return best


def evaluate_oos(df_oos, cfg, pw, h):
    tr = build_trades(df_oos, pw, h, cfg["min_swing_gap"])
    net = tr["net"].values
    mu, t = newey_west_t(net, h)
    lo, hi = block_boot_ci(net, cfg["block_len"], cfg["n_boot"])
    pdir = perm_direction_pvalue(tr["fwd"].values, tr["cost"].values, mu, cfg["n_perm"]) if len(net) else np.nan

    sig = np.isfinite(t) and abs(t) >= cfg["min_abs_t"] and mu > 0 and np.isfinite(lo) and lo > 0
    sig_perm = np.isfinite(pdir) and pdir <= cfg["max_perm_p"]
    enough = len(net) >= cfg["min_trades"]

    return dict(pw=pw, h=h, n=len(net), mu=mu, t=t, lo=lo, hi=hi, pdir=pdir,
                sig=sig, sig_perm=sig_perm, enough=enough,
                passed=(mu > 0 and sig and sig_perm and enough))


def print_oos_report(m, cfg, label="OUT-OF-SAMPLE"):
    print(f"\n=== {label} validation ===")
    print(f"  param: pivot_window={m['pw']}, horizon={m['h']}")
    print(f"  divergence trades: n={m['n']:>4}  mean={bps(m['mu']):>6.2f}bps  HAC t={m['t']:>5.2f}")
    print(f"  bootstrap 95% CI : [{bps(m['lo']):.2f}, {bps(m['hi']):.2f}] bps")
    print(f"  direction-perm p : {m['pdir']:.4f}  (H0: ທິດສຸ່ມກໍ່ໄດ້ຜົນຄືກັນ)")

    print("\n=== SCORECARD (OOS) ===")
    print(f"  [{'PASS' if m['mu'] > 0 else 'FAIL'}] mean > 0")
    print(f"  [{'PASS' if m['sig'] else 'FAIL'}] significant (HAC t>= {cfg['min_abs_t']}, CI>0)")
    print(f"  [{'PASS' if m['sig_perm'] else 'FAIL'}] direction ດີກວ່າສຸ່ມ (perm p<= {cfg['max_perm_p']})")
    print(f"  [{'PASS' if m['enough']  else 'FAIL'}] trades >= {cfg['min_trades']}")

    if m["passed"]:
        print("\n>>> VERDICT: RSI divergence ມີ edge ຈິງ (ຜ່ານ OOS) — ຄຸ້ມພິຈາລະນາໃສ່ໃນ EA.")
    else:
        print("\n>>> VERDICT: RSI divergence ບໍ່ພິສູດວ່າຊ່ວຍ.")


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
            print("  IS: ບໍ່ພົບ (pivot_w,h) ທີ່ mean>0 -> FAIL")
            reports.append(dict(passed=False, n=0, mu=np.nan, pw=None, h=None))
            continue

        print(f"  IS best: pivot_w={best['pw']} h={best['h']} (HAC t={best['t']:.2f})")
        m = evaluate_oos(df_oos, cfg, best["pw"], best["h"])
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
            "pivot_window": m.get("pw"), "horizon": m.get("h"),
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
        df = make_synthetic(cfg["synth_n"], cfg["synth_seed"], cfg["synth_mode"], cfg["rsi_period"])
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
        print("\n>>> IS: ບໍ່ພົບ (pivot_w,h) ທີ່ mean>0. VERDICT: RSI divergence ບໍ່ຊ່ວຍ.")
        return
    print(f"\nBest IS combo: pivot_w={best['pw']}, h={best['h']} (HAC t={best['t']:.2f}, mean={bps(best['mu']):.1f}bps)")
    m = evaluate_oos(df_oos, cfg, best["pw"], best["h"])
    print_oos_report(m, cfg)


if __name__ == "__main__":
    main()
