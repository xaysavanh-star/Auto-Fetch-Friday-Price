#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_fade_zone_filter.py
=======================================================================
ຄຳຖາມ: "spike-fade ໃກ້ zone (ແນວຮັບ/ແນວຕ້ານ) ຊະນະ spike-fade ທົ່ວໄປ ຫຼືບໍ່?"
Does conditioning spike-fade on proximity to a support/resistance zone add edge?

ນີ້ຄືການເອົາໄອເດຍ DZV (zone) ມາພິສູດແບບວິທະຍາສາດ ແທນທີ່ຈະໂຫຼດ indicator black box.

Zone = swing high/low ຫຼ້າສຸດ ຄຳນວນແບບ causal (rolling max/min ຂອງແທ່ງອະດີດ, shift(1)).
Proximity = |close - zone| / ATR  (ໜ່ວຍເປັນ ATR).

ໂຕທົດສອບຫຼັກ (near-vs-far):
   ແບ່ງ trade spike-fade ອອກເປັນ "ໃກ້ zone" vs "ໄກ zone"
   H0: mean(near) = mean(far)   -> zone ບໍ່ໃຫ້ຂໍ້ມູນເພີ່ມ
   H1: mean(near) > mean(far)   -> zone filter ມີຄຸນຄ່າ
ໃຊ້: Welch t-test + label-permutation test + HAC t ຂອງ subset + OOS validation.

ໝາຍເຫດ: script ນີ້ standalone (ບໍ່ຕ້ອງ import ໄຟລ໌ອື່ນ).
=======================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(2025)

CONFIG = {
    "use_synthetic": False,
    "data_path": "XAUUSD_M15.csv",

    "vol_window": 100,        # z-score vol window
    "zone_window": 50,        # swing high/low lookback (ນິຍາມ zone, ໃຊ້ສະເພາະ zone_def="swing")
    "atr_period": 14,
    "use_log_returns": True,

    # zone_def: ນິຍາມ resistance/support ແນວໃດ
    #   'swing' -> rolling max/min ຂອງ zone_window ແທ່ງ (ຄ່າເລີ່ມຕົ້ນ, ຄືເກົ່າ)
    #   'round' -> round-number level ໃກ້ໆລາຄາ (psychological level, ຫ່າງ round_step)
    #   'daily' -> high/low ຂອງມື້ກ່ອນໜ້າ (previous calendar day, causal)
    "zone_def": "round",
    "round_step": 10.0,       # ໄລຍະ round number (ເຊັ່ນ 1900, 1910, ... ສຳລັບ gold)

    # cost: ຖ້າມີ column "spread" ຈິງ (ຂໍ້ມູນ real) ຈະໃຊ້ spread*point_size/close
    # ຕໍ່ trade (scale ຕາມລາຄາ ແລະ ອິງ spread ຈິງ) ແທນ flat bps; ໃຊ້ cost_roundtrip
    # ເປັນ fallback ສະເພາະ synthetic (ບໍ່ມີ column spread)
    "cost_roundtrip": 0.00025,
    "point_size": 0.01,       # 1 point = 0.01 (XAUUSD ອີງລາຄາ 2 ຫຼັກທົດສະນິຍົມ ໃນ CSV)

    # grids (ນ້ອຍໆ ເພື່ອຄຸມ multiple-testing)
    "z_thresholds": [2.0, 2.5, 3.0],
    "horizons":     [3, 5, 10],
    "zone_tols":    [0.5, 1.0, 1.5],   # "ໃກ້" = dist <= tol (ATR units)

    "is_fraction": 0.70,      # ໃຊ້ສະເພາະຕອນ use_walk_forward=False
    "use_walk_forward": True,
    "n_folds": 6,             # expanding walk-forward -> n_folds-1 OOS folds
    "n_perm": 2000,
    "n_boot": 2000,
    "block_len": 20,

    # verdict
    "min_abs_t": 3.0,
    "max_perm_p": 0.01,
    "min_trades": 150,

    # ຖ້າລອງ zone_def ຫຼາຍແບບ (swing/round/daily...) ໃນ session ດຽວ, ໃສ່ຈຳນວນ
    # ທີ່ລອງທັງໝົດຢູ່ນີ້ - script ຈະຄິດ Bonferroni-adjusted p threshold ໃຫ້ ເພື່ອ
    # ເຕືອນວ່າລອງຫຼາຍວິທີເທົ່າໃດ ໂອກາດບັງເອີນ PASS ຈະສູງຂຶ້ນເທົ່ານັ້ນ
    "n_hypotheses_tried": 3,

    "results_csv": "walk_forward_log.csv",   # ບັນທຶກຜົນທຸກ fold ໄວ້ໃນໄຟລ໌ນີ້ (append)

    # synthetic: 'none' | 'uniform' | 'zone'
    # synth_n ໃຫຍ່ຂຶ້ນ (360k) ເພື່ອໃຫ້ແຕ່ລະ walk-forward fold ມີ trade ພຽງພໍ
    # ຮອດ min_trades=150 ໄດ້ຈິງ (fold ນ້ອຍເກີນໄປເຮັດໃຫ້ scorecard FAIL ຈາກ
    # n ບໍ່ພໍ ບໍ່ແມ່ນຈາກບໍ່ມີ edge)
    "synth_mode": "zone",
    "synth_n": 360000,
    "synth_seed": 11,
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


def make_synthetic(n, seed, mode, zone_window=50, fade_strength=1.2, tol_true=0.8):
    """
    ສ້າງ data ປອມ. mode:
      'none'    -> ບໍ່ມີ reversion (ຄວນໄດ້ NO EDGE)
      'uniform' -> reversion ຫຼັງ jump ທຸກຄັ້ງ (edge ມີ ແຕ່ 'zone filter' ບໍ່ຄວນຊ່ວຍ)
      'zone'    -> reversion ສະເພາະ jump ທີ່ຕົກໃກ້ swing high/low (zone filter ຄວນຊ່ວຍ)
    """
    rng = np.random.default_rng(seed)
    base_vol = 0.0009
    rets = rng.normal(0, base_vol, n)
    jump = rng.random(n) < 0.012
    jdir = rng.choice([-1.0, 1.0], n)
    jmag = 0.010 * (0.5 + rng.random(n))
    rets = rets + jump * jdir * jmag

    # base path (ກ່ອນ reversion) ເພື່ອຕັດສິນ near/far
    base_close = 2000.0 * np.exp(np.cumsum(rets))
    s = pd.Series(base_close)
    rec_high = s.rolling(zone_window).max().shift(1).values
    rec_low = s.rolling(zone_window).min().shift(1).values
    atr_proxy = pd.Series(np.abs(rets)).rolling(14).mean().shift(1).values * base_close

    if mode in ("uniform", "zone"):
        for t in np.where(jump)[0]:
            if t + 5 >= n:
                continue
            do_rev = True
            if mode == "zone":
                a = atr_proxy[t]
                if not np.isfinite(a) or a <= 0:
                    do_rev = False
                else:
                    if jdir[t] > 0:   # jump ຂຶ້ນ -> ໃກ້ resistance?
                        d = abs(base_close[t] - rec_high[t]) / a
                    else:             # jump ລົງ -> ໃກ້ support?
                        d = abs(base_close[t] - rec_low[t]) / a
                    do_rev = np.isfinite(d) and (d <= tol_true)
            if do_rev:
                for k in range(1, 6):
                    rets[t + k] -= fade_strength * jdir[t] * jmag[t] / 5.0

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


def add_zone_levels(df, cfg):
    """ຄິດໄລ່ rec_high/rec_low (resistance/support) ຕາມ zone_def, ທັງໝົດ causal."""
    zdef = cfg["zone_def"]
    if zdef == "swing":
        df["rec_high"] = df["high"].rolling(cfg["zone_window"]).max().shift(1)
        df["rec_low"] = df["low"].rolling(cfg["zone_window"]).min().shift(1)
    elif zdef == "round":
        step = cfg["round_step"]
        df["rec_high"] = np.ceil(df["close"] / step) * step   # round level ຖັດໄປຂ້າງເທິງ (resistance)
        df["rec_low"] = np.floor(df["close"] / step) * step   # round level ຖັດໄປຂ້າງລຸ່ມ (support)
    elif zdef == "daily":
        daily_hi = df["high"].resample("1D").max().shift(1)   # high ຂອງມື້ກ່ອນໜ້າ (ຮູ້ໄດ້ຕັ້ງແຕ່ຕົ້ນມື້ນີ້)
        daily_lo = df["low"].resample("1D").min().shift(1)
        df["rec_high"] = daily_hi.reindex(df.index, method="ffill")
        df["rec_low"] = daily_lo.reindex(df.index, method="ffill")
    else:
        raise ValueError(f"zone_def ບໍ່ຮູ້ຈັກ: {zdef}")
    return df


def prepare(df, cfg):
    close = df["close"]
    ret = np.log(close).diff() if cfg["use_log_returns"] else close.pct_change()
    sigma = ret.rolling(cfg["vol_window"]).std().shift(1)     # past-only
    df = df.copy()
    df["ret"] = ret
    df["z"] = ret / sigma
    df["atr"] = compute_atr(df, cfg["atr_period"])
    df = add_zone_levels(df, cfg)
    if "spread" in df.columns:
        # cost ອີງ spread ຈິງຂອງແທ່ງນັ້ນ, scale ຕາມລະດັບລາຄາ (ແທນ flat bps ຄົງທີ່
        # ຕະຫຼອດ 23 ປີ ທີ່ລາຄາປ່ຽນຈາກ ~340 ຫາ ~4600, ຄວາມກວ້າງ spread ເປັນ % ຕ່າງກັນຫຼາຍ)
        df["cost"] = (df["spread"] * cfg["point_size"] / df["close"]).clip(lower=0)
    else:
        df["cost"] = cfg["cost_roundtrip"]
    return df


def build_trades(df, z_thr, horizon):
    """ຄືນ DataFrame ຂອງ trade: side, net return, dist (ໄລຍະຫ່າງ zone / ATR).

    ກົດ cooldown: ຫຼັງ signal ຖືກຮັບ 1 ຄັ້ງ, ບໍ່ຮັບ signal ໃໝ່ຈົນກວ່າຈະຜ່ານ
    `horizon` ແທ່ງ - ກັນບໍ່ໃຫ້ vol cluster ດຽວກັນ ຖືກນັບເປັນຫຼາຍ trade ທີ່ overlap
    ໄລຍະເວລາກັນເອງ (ບໍ່ independent, ເຮັດໃຫ້ n ພອງ ແລະ t-stat ບິດເບືອນ).
    Entry ໃຊ້ open ຂອງແທ່ງຖັດຈາກ signal (ບໍ່ແມ່ນ close ຂອງແທ່ງ signal ເອງ)
    ເພາະໃນຄວາມເປັນຈິງ signal ຮູ້ໄດ້ພຽງຫຼັງແທ່ງປິດ. cost ອ່ານຈາກ df["cost"]
    (ຄິດໄລ່ໄວ້ໃນ prepare(), ອີງ spread ຈິງ ຫຼື flat fallback ສຳລັບ synthetic).
    """
    close = df["close"].values
    open_ = df["open"].values
    z = df["z"].values
    atr = df["atr"].values
    rhi = df["rec_high"].values
    rlo = df["rec_low"].values
    cost = df["cost"].values
    n = len(close)

    raw_side = np.zeros(n)
    raw_side[z >= z_thr] = -1.0   # spike ຂຶ້ນ -> short (fade to resistance)
    raw_side[z <= -z_thr] = +1.0  # spike ລົງ -> long  (fade to support)

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

    dist = np.full(n, np.nan)
    up = side == -1.0
    dn = side == +1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        dist[up] = np.abs(close[up] - rhi[up]) / atr[up]  # short: ໄລຍະຫາ resistance
        dist[dn] = np.abs(close[dn] - rlo[dn]) / atr[dn]  # long : ໄລຍະຫາ support

    mask = (side != 0) & np.isfinite(fwd) & np.isfinite(z) & np.isfinite(dist) & np.isfinite(atr) & (atr > 0) & np.isfinite(cost)
    idx = np.where(mask)[0]
    net = side[idx] * fwd[idx] - cost[idx]
    return pd.DataFrame({"pos": idx, "side": side[idx], "net": net, "dist": dist[idx]})


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


def perm_diff_pvalue(near, far, n_perm):
    """label-permutation: p ວ່າ mean(near) > mean(far) ໂດຍບັງເອີນ."""
    near = np.asarray(near, float); far = np.asarray(far, float)
    obs = near.mean() - far.mean()
    pool = np.concatenate([near, far]); n1 = len(near)
    cnt = 0
    for _ in range(n_perm):
        RNG.shuffle(pool)
        if (pool[:n1].mean() - pool[n1:].mean()) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


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
    """ໃນ IS: ຫາ (z,h,tol) ທີ່ near-vs-far diff ແຮງທີ່ສຸດ (Welch t)."""
    best = None
    if verbose:
        print("\n=== IN-SAMPLE scan: near-minus-far (bps) | Welch t ===")
    for z in cfg["z_thresholds"]:
        for h in cfg["horizons"]:
            tr = build_trades(df_is, z, h)
            if len(tr) < 60:
                continue
            for tol in cfg["zone_tols"]:
                near = tr.loc[tr["dist"] <= tol, "net"].values
                far = tr.loc[tr["dist"] > tol, "net"].values
                if len(near) < 30 or len(far) < 30:
                    continue
                diff = near.mean() - far.mean()
                wt = stats.ttest_ind(near, far, equal_var=False).statistic
                if verbose:
                    print(f"  z={z} h={h:>2} tol={tol}: n_near={len(near):>4} "
                          f"diff={bps(diff):>6.1f}  t={wt:>5.2f}")
                if diff > 0 and (best is None or wt > best["wt"]):
                    best = dict(z=z, h=h, tol=tol, wt=wt, diff=diff, n_near=len(near))
    return best


def evaluate_oos(df_oos, cfg, z, h, tol):
    """ຄິດໄລ່ metrics ຂອງ OOS (ບໍ່ print), ຄືນ dict ເພື່ອໃຫ້ walk-forward aggregate ໄດ້."""
    tr = build_trades(df_oos, z, h)
    near = tr.loc[tr["dist"] <= tol, "net"].values
    far = tr.loc[tr["dist"] > tol, "net"].values
    allt = tr["net"].values

    mu_all, t_all = newey_west_t(allt, h)
    mu_near, t_near = newey_west_t(near, h)
    diff, pdiff = perm_diff_pvalue(near, far, cfg["n_perm"])
    lo, hi = block_boot_ci(near, cfg["block_len"], cfg["n_boot"])

    improves = np.isfinite(mu_near) and np.isfinite(mu_all) and (mu_near > mu_all)
    sig_self = np.isfinite(t_near) and abs(t_near) >= cfg["min_abs_t"] and mu_near > 0 and np.isfinite(lo) and lo > 0
    sig_diff = (diff > 0) and (pdiff <= cfg["max_perm_p"])
    enough = len(near) >= cfg["min_trades"]

    return dict(z=z, h=h, tol=tol, n_all=len(allt), mu_all=mu_all, t_all=t_all,
                n_near=len(near), mu_near=mu_near, t_near=t_near,
                n_far=len(far), mu_far=(far.mean() if len(far) else np.nan),
                diff=diff, pdiff=pdiff, lo=lo, hi=hi,
                improves=improves, sig_self=sig_self, sig_diff=sig_diff, enough=enough,
                passed=(improves and sig_self and sig_diff and enough))


def print_oos_report(m, cfg, label="OUT-OF-SAMPLE"):
    print(f"\n=== {label} validation ===")
    print(f"  param: z={m['z']}, horizon={m['h']}, zone_tol={m['tol']} ATR")
    print(f"  baseline (all spike-fade) : n={m['n_all']:>4}  mean={bps(m['mu_all']):>6.2f}bps  HAC t={m['t_all']:>5.2f}")
    print(f"  filtered (near zone only) : n={m['n_near']:>4}  mean={bps(m['mu_near']):>6.2f}bps  HAC t={m['t_near']:>5.2f}")
    print(f"  far zone (rejected)       : n={m['n_far']:>4}  mean={bps(m['mu_far']):>6.2f}bps")
    print(f"  near - far difference     : {bps(m['diff']):>6.2f}bps  (permutation p={m['pdiff']:.4f})")
    print(f"  near bootstrap 95% CI     : [{bps(m['lo']):.2f}, {bps(m['hi']):.2f}] bps")

    print("\n=== SCORECARD (OOS) ===")
    print(f"  [{'PASS' if m['improves'] else 'FAIL'}] near-zone ດີກວ່າ baseline")
    print(f"  [{'PASS' if m['sig_self'] else 'FAIL'}] near-zone edge significant (HAC t>= {cfg['min_abs_t']}, CI>0)")
    print(f"  [{'PASS' if m['sig_diff'] else 'FAIL'}] near > far significant (perm p<= {cfg['max_perm_p']})")
    print(f"  [{'PASS' if m['enough']  else 'FAIL'}] trades >= {cfg['min_trades']}")

    if m["passed"]:
        print("\n>>> VERDICT: zone filter ເພີ່ມ edge ຈິງ (ຜ່ານ OOS) — ຄຸ້ມໃສ່ໃນ EA.")
    else:
        print("\n>>> VERDICT: zone filter ບໍ່ພິສູດວ່າຊ່ວຍ — ບໍ່ຄວນເພີ່ມຄວາມຊັບຊ້ອນນີ້.")


def run_walk_forward(df, cfg):
    """Expanding walk-forward: IS = ທຸກແທ່ງກ່ອນ fold, OOS = fold ນັ້ນ.
    ໃຫ້ n_folds-1 OOS window ອິດສະຫຼະຈາກກັນຕາມເວລາ, ແທນ split 70/30 ດຽວ
    (ຊ່ວຍເບິ່ງວ່າ edge ຄົງທີ່ຂ້າມ regime ຫຼືເປັນຄວາມບັງເອີນຂອງຊ່ວງໃດຊ່ວງໜຶ່ງ)."""
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
            print("  IS: ບໍ່ພົບ (z,h,tol) ທີ່ near>far -> FAIL")
            reports.append(dict(passed=False, n_near=0, diff=np.nan, mu_near=np.nan))
            continue

        print(f"  IS best: z={best['z']} h={best['h']} tol={best['tol']} (Welch t={best['wt']:.2f})")
        m = evaluate_oos(df_oos, cfg, best["z"], best["h"], best["tol"])
        print_oos_report(m, cfg, label=f"Fold {k} OOS")
        m["fold"] = k
        reports.append(m)

    n_pass = sum(1 for m in reports if m.get("passed"))
    total = len(reports)
    diffs = [float(bps(m["diff"])) for m in reports if np.isfinite(m.get("diff", np.nan))]
    print(f"\n{'='*70}\nWALK-FORWARD SUMMARY: {n_pass}/{total} fold ຜ່ານ scorecard ຄົບທຸກເງື່ອນໄຂ")
    if diffs:
        print(f"near-far diff ຂອງແຕ່ລະ fold (bps): {[round(d, 1) for d in diffs]}")

    n_hyp = max(1, cfg["n_hypotheses_tried"])
    adj_p = cfg["max_perm_p"] / n_hyp
    print(f"\nໝາຍເຫດ: ຖ້າລອງ zone_def ຫຼາຍແບບໃນ session ນີ້ ({n_hyp} ຄັ້ງ), p threshold ທີ່ຖືກຕ້ອງກວ່າ "
          f"(Bonferroni) ຄື {adj_p:.5f} ແທນ {cfg['max_perm_p']} — ຢ່າເຊື່ອ fold ໃດທີ່ p ຢູ່ລະຫວ່າງ {adj_p:.5f} ຫາ {cfg['max_perm_p']}")

    if n_pass >= max(1, int(np.ceil(total * 0.6))):
        print(">>> OVERALL: edge ຄົງທີ່ຢູ່ຫຼາຍ fold — ໜ້າເຊື່ອຖືກວ່າ split ດຽວ")
    else:
        print(">>> OVERALL: edge ບໍ່ຄົງທີ່ຂ້າມ fold — ຢ່າເອົາໃສ່ EA ຈາກຜົນນີ້ຢ່າງດຽວ")
    print("=" * 70)

    # ບັນທຶກຜົນທຸກ fold ໃສ່ CSV (append) - ເປີດເບິ່ງໄດ້ໃນ Excel
    rows = []
    for m in reports:
        rows.append({
            "run_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "zone_def": cfg["zone_def"],
            "fold": m.get("fold"),
            "z": m.get("z"), "horizon": m.get("h"), "zone_tol": m.get("tol"),
            "n_near": m.get("n_near"), "n_far": m.get("n_far"),
            "mu_near_bps": bps(m.get("mu_near")), "t_near": m.get("t_near"),
            "diff_bps": bps(m.get("diff")), "perm_p": m.get("pdiff"),
            "passed": m.get("passed", False),
        })
    out_path = cfg["results_csv"]
    header = not os.path.exists(out_path)
    pd.DataFrame(rows).to_csv(out_path, mode="a", header=header, index=False)
    print(f"ບັນທຶກຜົນ {len(rows)} fold ໃສ່ {out_path} ແລ້ວ (append)")


def main():
    cfg = CONFIG
    if cfg["use_synthetic"]:
        print(f">>> SYNTHETIC (mode={cfg['synth_mode']}) — ໃຊ້ validate ໂຕທົດສອບ, ຜົນບໍ່ມີຄວາມໝາຍທາງເງິນ")
        df = make_synthetic(cfg["synth_n"], cfg["synth_seed"], cfg["synth_mode"], cfg["zone_window"])
    else:
        print(f">>> REAL data: {cfg['data_path']}")
        df = load_ohlcv(cfg["data_path"])

    df = prepare(df, cfg)
    print(f"Bars: {len(df):,} | {df.index[0]} -> {df.index[-1]}")
    print(f"Zone def: {cfg['zone_def']}" + (f" (step={cfg['round_step']})" if cfg["zone_def"] == "round" else ""))
    print(f"Cost: {'ອິງ spread ຈິງ (bps ສະເລ່ຍ=' + format(bps(df['cost'].mean()), '.2f') + ')' if 'spread' in df.columns else 'flat ' + format(bps(cfg['cost_roundtrip']), '.2f') + 'bps'}")

    if cfg["use_walk_forward"]:
        run_walk_forward(df, cfg)
        return

    split = int(len(df) * cfg["is_fraction"])
    df_is, df_oos = df.iloc[:split].copy(), df.iloc[split:].copy()

    best = scan_is(df_is, cfg, verbose=True)
    if best is None:
        print("\n>>> IS: ບໍ່ພົບ (z,h,tol) ທີ່ near>far. VERDICT: zone filter ບໍ່ຊ່ວຍ.")
        return
    print(f"\nBest IS combo: z={best['z']}, h={best['h']}, tol={best['tol']} "
          f"(Welch t={best['wt']:.2f}, diff={bps(best['diff']):.1f}bps)")
    m = evaluate_oos(df_oos, cfg, best["z"], best["h"], best["tol"])
    print_oos_report(m, cfg)


if __name__ == "__main__":
    main()
