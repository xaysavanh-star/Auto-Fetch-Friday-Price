#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
friday_log_helper.py
=======================================================================
ຊ່ວຍຄິດໄລ່ ແລະ ສະຫຼຸບ friday_paper_trade_log.csv ໃຫ້ອັດຕະໂນມັດ.
ບໍ່ຕ້ອງຄິດເລກເອງ - ພຽງແຕ່ໃສ່ entry_price ແລະ exit_price ໃນ CSV ແລ້ວ run
script ນີ້.

ວິທີໃຊ້:
   1. ເປີດ friday_paper_trade_log.csv ດ້ວຍ Excel
   2. ໃສ່ແຖວໃໝ່: week_of, entry_date, entry_time, entry_price,
      exit_date, exit_time, exit_price (ປ່ອຍ net_bps ຫວ່າງໄວ້)
   3. Save ແລ້ວປິດ Excel (ສຳຄັນ - Excel ຕ້ອງປິດກ່ອນ run script)
   4. Run: python friday_log_helper.py
   5. Script ຈະຄິດໄລ່ net_bps ໃຫ້ອັດຕະໂນມັດ (ຫັກ cost ຈິງ ຄິດຈາກ spread
      XM Ultra Low Micro ~30 points ຂອງທ່ານ ຫານດ້ວຍລາຄາຈິງຂອງ trade ນັ້ນ
      - ບໍ່ໃຊ້ຄ່າ bps ຄົງທີ່ ເພາະ cost ເປັນ % ປ່ຽນຕາມລາຄາ gold)
=======================================================================
"""

import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = "friday_paper_trade_log.csv"
SPREAD_POINTS = 30.0   # spread ສະເລ່ຍ XM Ultra Low Micro ຂອງທ່ານ (ຢູ່ໃນຊ່ວງ 25-35)
POINT_SIZE = 0.01      # 1 point = 0.01 (XAUUSD 2 ຫຼັກທົດສະນິຍົມ)


def cost_bps_at_price(entry_price):
    """cost ເປັນ bps ປ່ຽນຕາມລາຄາ gold ຕອນນັ້ນ (30 points ຄົງທີ່ ແຕ່ % ຕ່າງກັນ)."""
    return (SPREAD_POINTS * POINT_SIZE / entry_price) * 10000.0


def main():
    try:
        df = pd.read_csv(LOG_PATH)
    except FileNotFoundError:
        print(f"ບໍ່ພົບ {LOG_PATH} — ໃຫ້ສ້າງໄຟລ໌ນີ້ກ່ອນ (ຫຼືກັບໄປໃຊ້ໄຟລ໌ template ເດີມ)")
        return

    if len(df) == 0:
        print("ຍັງບໍ່ມີຂໍ້ມູນໃນ log — ໃສ່ trade ທຳອິດກ່ອນ")
        return

    updated = False
    for i, row in df.iterrows():
        if pd.notna(row.get("entry_price")) and pd.notna(row.get("exit_price")) and pd.isna(row.get("net_bps")):
            raw_bps = (row["exit_price"] / row["entry_price"] - 1.0) * 10000.0
            cost = cost_bps_at_price(row["entry_price"])
            net_bps = raw_bps - cost
            df.at[i, "net_bps"] = round(net_bps, 2)
            updated = True
            print(f"  ຄິດໄລ່ {row.get('week_of', i)}: raw={raw_bps:.2f}bps - cost{cost:.2f}bps(@{row['entry_price']:.2f}) = {net_bps:.2f}bps")

    if updated:
        df.to_csv(LOG_PATH, index=False)
        print(f"\nບັນທຶກ net_bps ໃໝ່ໃສ່ {LOG_PATH} ແລ້ວ")

    valid = df.dropna(subset=["net_bps"])
    n = len(valid)
    print(f"\n{'='*60}\nສະຫຼຸບ paper-trade Friday effect\n{'='*60}")
    print(f"ຈຳນວນ trade ທີ່ບັນທຶກແລ້ວ: {n}")

    if n == 0:
        print("ຍັງບໍ່ມີ trade ຄົບຖ້ວນ (ຕ້ອງມີທັງ entry_price ແລະ exit_price)")
        return

    mean_bps = valid["net_bps"].mean()
    win_rate = (valid["net_bps"] > 0).mean() * 100
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Mean: {mean_bps:.2f}bps ຕໍ່ trade")

    if n < 20:
        print(f"\n>> ຍັງນ້ອຍເກີນໄປ (n={n} < 20) — ຢືນຢັນຫຍັງບໍ່ໄດ້ເທື່ອ, ບັນທຶກຕໍ່ໄປ")
    else:
        # ອະນຸໂລມ HAC t ແບບງ່າຍ (ບໍ່ adjust autocorrelation ເພາະ n ນ້ອຍ ແລະ 1 trade/ອາທິດ ບໍ່ overlap)
        se = valid["net_bps"].std(ddof=1) / np.sqrt(n)
        t = mean_bps / se if se > 0 else np.nan
        print(f"t-stat ປະມານ: {t:.2f}")
        if mean_bps > 0 and t >= 2.0:
            print(">> ຍັງສອດຄ່ອງກັບ backtest (ບວກ, t ພໍໃຊ້ໄດ້) - ຕິດຕາມຕໍ່ໄປ")
        elif mean_bps > 0:
            print(">> ຍັງເປັນບວກ ແຕ່ t ຍັງອ່ອນ - ຕິດຕາມຕໍ່ໄປ, ຢ່າຫາກໍ່ຕັດສິນໃຈ")
        else:
            print(">> ເປັນລົບ - ບໍ່ສອດຄ່ອງກັບ backtest, ຄວນທົບທວນ ຫຼືຢຸດຕິດຕາມ effect ນີ້")


if __name__ == "__main__":
    main()
