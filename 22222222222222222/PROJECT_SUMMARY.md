# ສະຫຼຸບໂຄງການ: ຫາ Trading Edge ໃນ XAUUSD M15

ອັບເດດຫຼ້າສຸດ: 2026-07-02
ຂໍ້ມູນທີ່ໃຊ້: `XAUUSD_M15.csv` (2003-05-05 ຫາ 2026-04-30, 553,823 ແທ່ງ)

## ວິທີການທົດສອບ (ໃຊ້ຄືກັນທຸກ script)

- Cost ອີງ spread ຈິງ (column `<SPREAD>` ໃນ CSV, scale ຕາມລາຄາ)
- Causal ທັງໝົດ (ບໍ່ look-ahead) — feature ທຸກໂຕໃຊ້ພຽງຂໍ້ມູນອະດີດ
- Cooldown ກັນ signal ຊ້ອນກັນ (trade ບໍ່ independent)
- ວັດແທກດ້ວຍ: Newey-West HAC t-test, permutation test, block bootstrap CI
- ຢືນຢັນ pipeline ຖືກຕ້ອງດ້ວຍ synthetic data ກ່ອນທົດສອບຂໍ້ມູນຈິງທຸກຄັ້ງ
- Walk-forward (expanding window) ແທນ split ດຽວ

## ຜົນ: 9 ໄອເດຍ, 0 ຜ່ານ

| # | Script | ໄອເດຍ | ຜົນ |
|---|---|---|---|
| 1 | `spike_fade_zone_filter.py` | Zone-proximity filter (swing/round/daily) ຕໍ່ spike-fade | ❌ |
| 2 | `spike_fade_zone_filter.py` | Spike-fade baseline (z-score) | ❌ ຂາດທຶນມີນັຍສຳຄັນ |
| 3 | `spike_momentum_test.py` | Momentum ຕາມທິດ spike | ❌ ຂາດທຶນມີນັຍສຳຄັນ |
| 4 | `rsi_divergence_test.py` | RSI Divergence (regular) | ❌ ຂາດທຶນມີນັຍສຳຄັນ |
| 5 | `barrier_exit_test.py` | ATR SL/TP exit (fade + momentum) | ❌ ຂາດທຶນມີນັຍສຳຄັນ ທັງ 2 ທິດ |
| 6 | `winrate_by_timeframe.py` | Win-rate ຂ້າມ M15/M30/H1/H4/D1 | ❌ ບໍ່ມີ TF ໃດເກີນ ~52% |
| 7 | `macd_crossover_test.py` | MACD crossover | ❌ ຂາດທຶນມີນັຍສຳຄັນ |
| 8 | `macd_rsi_trend_confluence_test.py` | MACD+RSI+Trend confluence | ❌ ຂາດທຶນຮ້າຍແຮງກວ່າ MACD ດ່ຽວ |
| 9 | `session_effect_test.py` | Hour-of-day (ຫັກ cost ຈິງ) | ❌ edge ຫາຍໄປຫຼັງຫັກ cost |

**ສະຫຼຸບ**: ບໍ່ມີ classical technical indicator (ດ່ຽວ ຫຼື ປະສົມ) ໃຫ້ edge ໃນ XAUUSD M15 ຫຼັງຫັກ cost ຈິງ. ຢ່າສ້າງ EA ຈາກໂຕໃດໃນ 9 ໂຕນີ້.

## ໄອເດຍທີ 10: Friday Effect — ໜ້າສົນໃຈ ແຕ່ຍັງບໍ່ພິສູດເຕັມທີ່

**Scripts**: `session_effect_test.py` (ພົບ), `friday_effect_test.py` (ຂຸດເລິກ), `friday_session_zone_test.py` (ຫາ window)

**ຫຼັກຖານທີ່ຜ່ານ**:
- Full-sample vs 0: n=1182, mean=10.33bps, HAC t=3.14, CI=[4.79,16.12] ✅
- Friday vs ວັນອື່ນ (permutation): diff=10.71bps, p=0.0010 ✅
- **19/23 ປີ ເປັນບວກ** — sign test p=0.0013 ✅ (ຫຼັກຖານແຮງທີ່ສຸດ)
- ບໍ່ໄດ້ມາຈາກ NFP (ວັນສຸກທຳອິດຂອງເດືອນ ອ່ອນກວ່າວັນສຸກທຳມະດາ)
- ກົນໄກ: drift ເລີ່ມແຕ່ວັນພະຫັດ, ບໍ່ແມ່ນ weekend gap (Fri close→Mon open ອ່ອນສຸດ)
- ຊ່ວງທີ່ concentrate ແຮງສຸດ: **ບ່າຍ-ແລງວັນສຸກ (12:00-23:45)**, ເຊົ້າວັນສຸກບໍ່ມີ edge ເລີຍ

**ຫຼັກຖານທີ່ຍັງບໍ່ຜ່ານ**:
- Walk-forward (temporal 2-fold split): OOS t=2.06, 1.29 — ບໍ່ຮອດ 3.0 (ອາດເປັນຂໍ້ຈຳກັດ sample size, ບໍ່ແມ່ນ signal ບໍ່ຈິງ — ພຽງ ~1182 ວັນສຸກໃນ 23 ປີ)
- ຈຳກັດ window ໃຫ້ແຄບລົງ (ບ່າຍ-ແລງ) ອາດເປັນ overfitting ຈາກການລອງ 6 window (ບໍ່ໄດ້ confirm ເພີ່ມ, ພຽງແຕ່ຊີ້ທິດ)

**ບົດສະຫຼຸບ**: ໄດ້ຂຸດຂໍ້ມູນປະຫວັດສາດອອກເກືອບໝົດແລ້ວ. ບໍ່ມີທາງພິສູດຕື່ມໄດ້ອີກຈາກ backtest — ຂັ້ນຕອນທີ່ເຫຼືອຄື **ຕິດຕາມ forward/paper trade**.

## ໄຟລ໌ອື່ນ

- `barrier_exit_walk_forward_log.csv`, `macd_crossover_walk_forward_log.csv`, ແລະອື່ນໆ — ຜົນລະອຽດແຕ່ລະ fold ຂອງແຕ່ລະການທົດລອງ
- `winrate_by_timeframe.csv` — ຕາຕະລາງ win-rate ຂ້າມ timeframe
- `friday_paper_trade_log.csv` — ໃຊ້ບັນທຶກຜົນ paper-trade ດ້ວຍມື (ຈາກ broker ຈິງ)
- `friday_auto_price_log.csv` — ບັນທຶກອັດຕະໂນມັດຈາກ scheduled task (ລາຄາ public)
- `friday_compare_sources.py` — ປຽບທຽບ 2 ແຫຼ່ງ
- `friday_log_helper.py` — ຄິດໄລ່ net_bps ໃຫ້ອັດຕະໂນມັດ (cost ອີງ SPREAD_POINTS=30, dynamic ຕາມລາຄາ)

## ຂໍ້ມູນ broker ຂອງຜູ້ໃຊ້ (XM)

- Account: **XM Ultra Low Micro**
- Symbol: **GOLDm#** (ບໍ່ແມ່ນ XAUUSD — ຊື່ຕ່າງກັນ underlying ດຽວກັນ), digits=2, tick=0.01
- Spread: **ລອຍຕົວ** (float), ສະເລ່ຍ ~29-30 points (ຢືນຢັນກົງກັບ SPREAD_POINTS=30 ທີ່ໃຊ້ໃນ script) — ອາດປ່ຽນແປງຕາມສະພາບຕະຫຼາດ, ບໍ່ຄົງທີ່ 100%
- ຊົ່ວໂມງເທຣດ: ຈັນ-ພະຫັດ 01:00-23:59, **ວັນສຸກ 01:00-23:58** (ປິດໄວກວ່າ 1 ນາທີ)
- Volume ຕ່ຳສຸດ: **0.1 lot** (step 0.01) — ບໍ່ແມ່ນ 0.01 ຄືທີ່ເຄີຍສົມມຸດ (ແກ້ໄຂແລ້ວ)
- ຂະໜາດບັນຊີທີ່ວາງແຜນໃຊ້: **$150**

## ແຜນ position sizing ສຳລັບບັນຊີ $150 (ໃຊ້ຫຼັງ paper-trade ຢືນຢັນແລ້ວເທົ່ານັ້ນ)

⚠️ **ຢ່າໃຊ້ເງິນຈິງ ຈົນກວ່າ paper-trade ຈະຮອດ 20-30 ອາທິດ (~6 ເດືອນ) ແລະ ຍັງເປັນບວກສອດຄ່ອງ** — ຕົກລົງກັນແລ້ວ 2026-07-02

ເມື່ອຮອດເວລາ:
- Position size: **0.1 lot** (ຕ່ຳສຸດທີ່ broker ອະນຸຍາດ) ຕໍ່ Friday, ບໍ່ scale/ບໍ່ເພີ່ມ
- Notional ~$400+ (ຂຶ້ນກັບລາຄາ gold ຕອນນັ້ນ), margin ໃຊ້ໜ້ອຍຫຼາຍ (~0.5% ຂອງບັນຊີ)
- **Stop-loss ~$50-75** ຫ່າງຈາກ entry (risk ~3-5% ຂອງບັນຊີຕໍ່ trade) — ບໍ່ມີໃນ backtest ເດີມ, ເພີ່ມເພື່ອ risk management ຈິງ
- ຫ້າມມີຫຼາຍກວ່າ 1 position ພ້ອມກັນ

## ຂັ້ນຕອນຖັດໄປ: Paper-trade Friday Effect

1. ທຸກວັນສຸກ, ບັນທຶກລາຄາ `GOLDm#` ຊ່ວງ ~12:00 ແລະ close ຊ່ວງ **23:55** (ກ່ອນຕະຫຼາດປິດ 23:58) ລົງ `friday_paper_trade_log.csv`
2. Run `python friday_log_helper.py` ຄິດ bps ອັດຕະໂນມັດ
3. ຢ່າໃຊ້ເງິນຈິງ ຈົນກວ່າຈະມີຢ່າງໜ້ອຍ 20-30 ຈຸດຂໍ້ມູນໃໝ່ (~6 ເດືອນ) ທີ່ຍັງເປັນບວກສອດຄ່ອງກັບ backtest
4. Run `python friday_compare_sources.py` ເປັນລະຍະ ເບິ່ງວ່າແຫຼ່ງອັດຕະໂນມັດ vs broker ໃກ້ຄຽງກັນບໍ່
5. ຖ້າຢາກໄດ້ຫຼັກຖານແຮງກວ່ານີ້ໄວກວ່າ: ລອງທົດສອບ Friday effect ນີ້ໃນ instrument ອື່ນ (EUR/USD, Silver) ຖ້າມີຂໍ້ມູນ
