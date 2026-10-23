# Stage 0 — Data pipeline & leakage-safe splitting

Implements Stage 0 of `PRPD_Staged_Implementation_Plan.docx`.

> For the classify / fault library / results dashboard web tool built on top of this pipeline's final
> models, see [`webapp/README.md`](webapp/README.md).

## Run it

```bash
python3 -m prpd_data_prep.cli \
  --dataset-root "../Dataset" \
  --calibration-csv "../batch_random_slice_summary.csv" \
  --output-dir "./manifests"
```

Writes `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv` to `--output-dir`,
each with columns `filepath, fault_type, sensor_id, session_id, calibration_factor`.

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

## What it does, per the plan's decisions

- Asserts `Dataset/` contains exactly the 13 decided fault classes (1, 2, 3, 4, 5,
  5i, 6, 7 — includes 7A, 8, 9, 10, 12, 13) and fails loudly, listing the mismatch,
  if a class is missing or an unexpected `fault_*` folder is present (e.g. a
  leftover compound-fault folder).
- Derives `fault_type` and `sensor_id` from the `Dataset/fault_<code>/sensor_<n>/`
  directory structure, not from `batch_random_slice_summary.csv`'s own
  `fault_label` column (which has known parsing bugs).
- Derives `session_id` from the filename by stripping the `Sensor <n>_` prefix and
  the trailing `_<slice_index>` suffix, so all three sensors' simultaneous
  captures — and every random slice of one recording — collapse to one session.
- Excludes (and reports, doesn't silently drop) any stray file whose name matches
  the out-of-scope `12A`/`12B` codes.
- Joins `calibration_factor` from the CSV by PNG filename, and warns when a
  fault_type's sessions fall entirely on one side of the ~10 Aug 2026 HFCT
  recalibration (a possible amplitude confound).
- Splits by session (no `session_id` crosses train/val/test); a fault_type with
  fewer than 3 independent sessions falls back to a classical random 70/15/15
  split at the image level instead, which is reported explicitly per class.
- Verifies zero filepath/session_id overlap across splits before writing anything.

## Layout

```
prpd_data_prep/
  config.py       # taxonomy + split constants, decided in the plan's Dataset Audit
  scan.py         # directory scan, session_id parsing, taxonomy assertion
  calibration.py  # CSV join + calibration-era confound check
  split.py        # session-level split with image-level fallback
  integrity.py    # overlap checks + per-split reporting
  cli.py          # orchestrates the above, CLI entry point
tests/
  test_scan.py    # session parsing, stray-file detection, taxonomy assertion
  test_split.py   # overlap-check logic (synthetic data) + split behavior
```
