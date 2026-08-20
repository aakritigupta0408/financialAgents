# Fix: Ledger Must Show Both Arms (ctl + t2)

## Context

The user reports the treatment isn't visible in the ledger. Server-side data shows the opposite skew: the last-8 rows per horizon are currently ALL t2 rows. Root cause: `ledgerRows()` in `site/live_online.html` slices the last 8 rows in **file/insertion order**, so whichever arm was batch-appended most recently monopolizes the table (t2's 72-row backfill landed after control's rows). Additionally the user's browser may be caching the pre-t2 page JS (python http.server sends no cache headers), hiding the new Arm column entirely.

## Changes — `site/live_online.html` only

1. In `draw()` / `ledgerRows()`: sort the horizon's rows by `target_ts` **descending** (then ctl before t2 at the same timestamp) *before* slicing, and take the most recent **6 slots × both arms (12 rows)** — so every slot shows its ctl and t2 predictions adjacent, regardless of append order. Remove the now-misleading `slice(-8).reverse()`.
2. Add `<meta http-equiv="cache-control" content="no-store">` in `<head>` to stop the browser from serving stale page JS from python http.server.

## Verification

1. Reload page with hard refresh once; ledger tables show alternating ctl/t2 rows for the same target times, Arm column visible.
2. `python3 - <<…` sanity: simulate the new sort+slice on `results/prediction_log.jsonl` — both arms present in the top 12 for each horizon.
3. Commit + push.
