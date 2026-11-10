# Point-in-time inputs and portable exports

The ordinary official/market loaders now capture content-addressed snapshots under
`data/cache/vintages`. Official snapshots retain RF. Available Kenneth French ZIP
payloads are preserved by hash and linked to their normalized snapshot. The wrapper
and yfinance expose normalized frames, not original HTTP payloads. Existing cache
imports are first observed **now**, regardless of dates or old download metadata.
`publication_at: null` means the publisher's actual release time is unknown.

Objects and observation receipts are immutable, atomically created, and SHA-256
verified on read. Keep the entire store, including `receipts/`, when moving it.
This is local integrity checking, not externally signed or independently timestamped
proof; deletion of the entire receipt history cannot be detected without a backup.
An identical consecutive ingestion reuses its snapshot and observation receipt.
Changed values create new snapshots; prior observations remain replayable.

## Commands

Run from this repo with its installed Python environment:

```bash
python -m ff5_predictor.vintage_cli --store data/cache/vintages capture --config config/nowcast/latest.yaml
python -m ff5_predictor.vintage_cli --store data/cache/vintages replay --config config/nowcast/latest.yaml --as-of 2026-11-05T22:00:00Z --mode observed
python -m ff5_predictor.vintage_cli --store data/cache/vintages export --as-of 2026-11-05T22:00:00Z --mode observed --output data/nowcasts/kairos.json
python -m ff5_predictor.vintage_cli --store data/cache/vintages score --prediction-id MANIFEST_HASH --output data/nowcasts/scores.json
```

Use timestamps after your first capture for observed replay. `capture` follows the
configured loader cache/refresh policy, so it can download. Replay, scoring and
export read local snapshots only. Replay never replaces the mutable latest cache.
To include a frozen nowcast in an export, add `--prediction-id MANIFEST_HASH`.
The export timestamp must be after actual issuance, and its mode must match the
prediction's mode. The latest run metadata contains `prediction_manifest_id`.
Scoring writes a new immutable score object each time; the requested output file
is only a convenience copy.

The same controls work in normal configs:

```yaml
point_in_time:
  store_dir: data/cache/vintages
  decision_at: '2026-11-05T22:00:00Z'
  mode: observed
  delay_days: 45
```

* `observed`: only snapshots observed by the decision timestamp. Missing historical
  observations fail explicitly. Training features, targets and preprocessing use
  only the eligible frames. This does not reconstruct historical publication vintages.
* `retrospective`: latest stored vintage, truncated at the decision date. This is
  the compatibility default for ordinary workflows, and is explicitly labeled.
* `approximate_delay`: latest stored vintage with official rows withheld by the
  configured calendar-day delay. This is a simulation, never exact historical evidence.

Market daily bars are usable only after 16:15 America/New_York on their session
date, and after observation in observed mode. DST is respected. Weekends are
excluded. Capture removes unfinished sessions: a morning snapshot cannot become a
completed daily bar merely because the replay clock later passes the close. A new
post-close observation is required. The policy deliberately delays early-close sessions until 16:15; it does
not maintain an exchange holiday calendar or support non-US exchange close times.
Use this store for the configured US-traded daily ETF/equity inputs. Historical
adjustments in Yahoo prices remain revisions, not reconstructed past adjustments.

Each issued prediction references immutable normalized source snapshots, an eligible
row manifest, resolved config, full training and inference inputs, predictions, and
all fitted model artifacts. These survive mutable `latest` run copies.
First-observed and latest-revision scores are separate rows with target snapshot
IDs. First-observed means the first version captured locally, not a proven first
publisher release. Historical replay runs are issued now, not backdated.

## Kairos JSON contract, schema version 1

The local export is ordinary JSON, independent of Python imports. Validate
`schema_version == 1`, `factor_order == ['Mkt-RF','SMB','HML','RMW','CMA','Mom']`,
USD currency, daily frequency, decimal units, return basis and availability mode
before importing. Each `records` item has:

* `date`: YYYY-MM-DD.
* `values`: the six named decimal factor returns plus `RF`, a decimal for official
  history and `null` for nowcasts. Never replace a missing RF with zero.
* `source`: `official` or `nowcast`.
* `available_at`: UTC timestamp, with first locally observed value availability
  for official data and actual issuance time for nowcasts.
* `snapshot_id`: official normalized content hash or issued-prediction manifest hash.
* `model_id`: null for official data; prediction manifest hash for nowcasts.

`provenance` includes `generated_at`, `decision_at`, `dataset_id`, snapshot hashes,
source receipt, currency, units, frequency, explicit return basis and availability
mode. `Mkt-RF` is an excess market return; the other factors are long-short returns.
RF is a separate daily risk-free return. The embedded prediction manifest contains
model hashes, config, feature names and input references. Official history takes
precedence over a nowcast once observed.

No calibrated covariance is exported yet: `uncertainty: null` and
`uncertainty_status: not_calibrated` are intentional. Consumers must not infer a
zero covariance or a confidence interval. Training artifacts are Parquet and model
artifacts are Python-specific, but the downstream JSON contract is portable.

Current limits: observed replay requires fundamentals disabled because those inputs
are not snapshotted. Python model objects are frozen with joblib; cross-version
loading is not guaranteed. Raw Yahoo
HTTP responses, publication feeds, exchange calendars and signed external receipt
storage are deferred. Existing historical backtest commands remain retrospective;
use the replay command for observed-vintage claims.

## Automated Kairos publication

Run from this repository with its installed environment:

```bash
.venv/bin/python -m ff5_predictor.publish_feed \
  --output /home/zach/projects/kairos/backend/data/factors/feed.json \
  --kairos-backend /home/zach/projects/kairos/backend --timeout 900
```

The command captures official and market snapshots through `vintage_cli capture`,
then exports observed data through `vintage_cli export`. It does not train nowcasts.
An optional `--prediction-id` includes an already-issued immutable prediction;
`--skip-capture` only exports existing snapshots. The consumer's own bounded
parser validates the candidate in its virtual environment. Empty or invalid feeds,
command failures and timeouts retain the prior output. A nonblocking flock returns
75 for an overlapping writer; other failures return 1 and successful publication
returns 0. Feed and health files use same-directory temporary files, fsync and
atomic rename. Consumers must not write the output directly. Feed and health are
separate atomic files, so a crash between them can leave old health until next run.

`deploy/kairos-factor-feed.service` and `.timer` are user-systemd units configured
for the current checkout locations. Adjust paths before deploying elsewhere.
To install for this user, without restarting Kairos:

```bash
install -d ~/.config/systemd/user
install -m 644 deploy/kairos-factor-feed.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kairos-factor-feed.timer
```

Set `KAIROS_FACTOR_FEED_PATH` in Kairos's backend environment to the output path.
The supplied timer runs weekdays at 23:30 UTC with up to five minutes jitter;
`Persistent=true` catches missed runs. The publisher bounds its subprocess work
to 900 seconds and systemd bounds the whole process to 960 seconds. The timer
requires a running user manager; unattended boot operation needs the operator's
existing user-service policy. The units are supplied, not automatically enabled.
