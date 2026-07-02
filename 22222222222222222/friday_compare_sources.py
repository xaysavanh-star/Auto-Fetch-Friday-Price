#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
friday_compare_sources.py
=======================================================================
ປຽບທຽບລາຄາ 2 ແຫຼ່ງ ຕໍ່ອາທິດ (week_of ດຽວກັນ):
   - friday_auto_price_log.csv    : ລາຄາ gold spot ສາທາລະນະ, ດຶງອັດຕະໂນມັດ
                                     ໂດຍ scheduled task (Friday 14:00 / 23:33 ICT)
   - friday_paper_trade_log.csv   : ລາຄາ broker ຈິງຂອງທ່ານ, ໃສ່ດ້ວຍມື

ຄິດໄລ່ net_bps ຂອງທັງ 2 ແຫຼ່ງ ແລະ ສະແດງຄວາມແຕກຕ່າງ - ຖ້າໃກ້ຄຽງກັນ
ໝາຍຄວາມວ່າແຫຼ່ງສາທາລະນະໃຊ້ແທນໄດ້ໃນອະນາຄົດ (ບໍ່ຕ້ອງໃສ່ດ້ວຍມືອີກ).
=======================================================================
"""

import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

AUTO_LOG = "friday_auto_price_log.csv"
MANUAL_LOG = "friday_paper_trade_log.csv"
SPREAD_POINTS = 30.0   # spread ສະເລ່ຍ XM Ultra Low Micro ຂອງທ່ານ (ຢູ່ໃນຊ່ວງ 25-35)
POINT_SIZE = 0.01


def cost_bps_at_price(entry_price):
    return (SPREAD_POINTS * POINT_SIZE / entry_price) * 10000.0


def bps(x):
    return np.nan if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def main():
    try:
        auto = pd.read_csv(AUTO_LOG)
    except FileNotFoundError:
        auto = pd.DataFrame(columns=["week_of", "fetch_time_utc", "leg", "price", "source"])

    try:
        manual = pd.read_csv(MANUAL_LOG)
    except FileNotFoundError:
        manual = pd.DataFrame(columns=["week_of", "entry_price", "exit_price", "net_bps"])

    if len(auto) == 0:
        print("ຍັງບໍ່ມີຂໍ້ມູນໃນ friday_auto_price_log.csv — scheduled task ຍັງບໍ່ໄດ້ run ຫຼືຍັງບໍ່ຮອດວັນສຸກ")
        return

    auto_wide = auto.pivot_table(index="week_of", columns="leg", values="price", aggfunc="first")
    if "entry" not in auto_wide.columns or "exit" not in auto_wide.columns:
        print("ຍັງມີພຽງ entry ຫຼື exit ຢ່າງດຽວ — ລໍຖ້າຄົບ 2 ຝັ່ງກ່ອນ")
        print(auto_wide)
        return

    auto_wide = auto_wide.dropna(subset=["entry", "exit"])
    raw_bps = (auto_wide["exit"] / auto_wide["entry"] - 1.0) * 10000.0
    auto_wide["auto_net_bps"] = raw_bps - auto_wide["entry"].apply(cost_bps_at_price)
    auto_wide = auto_wide.reset_index()

    print(f"{'='*70}\nປຽບທຽບແຫຼ່ງອັດຕະໂນມັດ (public) vs broker ຂອງທ່ານ (manual)\n{'='*70}")
    merged = auto_wide.merge(manual[["week_of", "net_bps"]], on="week_of", how="outer", suffixes=("", "_manual"))
    merged = merged.rename(columns={"net_bps": "manual_net_bps"})

    print(f"{'week_of':<12}{'auto(bps)':<12}{'broker(bps)':<13}{'diff':<10}")
    print("-" * 50)
    diffs = []
    for _, r in merged.iterrows():
        a = r.get("auto_net_bps")
        m = r.get("manual_net_bps")
        a_txt = f"{a:.2f}" if pd.notna(a) else "n/a"
        m_txt = f"{m:.2f}" if pd.notna(m) else "n/a"
        d_txt = ""
        if pd.notna(a) and pd.notna(m):
            d = a - m
            diffs.append(d)
            d_txt = f"{d:+.2f}"
        print(f"{str(r['week_of']):<12}{a_txt:<12}{m_txt:<13}{d_txt:<10}")

    if diffs:
        print(f"\nຄວາມແຕກຕ່າງສະເລ່ຍ (auto - broker): {np.mean(diffs):.2f}bps ("
              f"n={len(diffs)} ອາທິດທີ່ມີທັງ 2 ແຫຼ່ງ)")
        if abs(np.mean(diffs)) < 3.0 and len(diffs) >= 4:
            print(">> ໃກ້ຄຽງກັນພໍສົມຄວນ — ອາດໃຊ້ແຫຼ່ງອັດຕະໂນມັດແທນການໃສ່ດ້ວຍມືໄດ້ໃນອະນາຄົດ")
        elif len(diffs) < 4:
            print(">> ຍັງນ້ອຍເກີນໄປທີ່ຈະສະຫຼຸບ — ຕ້ອງການຢ່າງໜ້ອຍ 4+ ອາທິດ")
        else:
            print(">> ຕ່າງກັນພໍສົມຄວນ — ຄວນໃຊ້ລາຄາ broker ຈິງຕໍ່ໄປ (ໜ້າເຊື່ອຖືກວ່າ)")
    else:
        print("\nຍັງບໍ່ມີອາທິດໃດທີ່ມີທັງ 2 ແຫຼ່ງພ້ອມກັນ")


if __name__ == "__main__":
    main()
