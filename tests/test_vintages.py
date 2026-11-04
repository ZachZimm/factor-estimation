import json

import pandas as pd
import pytest

from ff5_predictor.vintages import (
    FACTOR_ORDER, SnapshotStore, close_at, export_bundle, freeze_run, score,
)


def factors(value=0.01):
    return pd.DataFrame({k: [value] for k in FACTOR_ORDER + ['RF']},
                        index=pd.to_datetime(['2026-10-30']))


def test_revision_isolation_idempotence_and_modes(tmp_path):
    store = SnapshotStore(tmp_path)
    first = store.ingest(factors(), 'official', '2026-11-02T12:00Z')
    assert store.ingest(factors(), 'official', '2026-11-02T13:00Z') == first
    assert len(store.receipts('official')) == 1
    before = store.select('official', '2026-11-02T14:00Z')
    store.ingest(factors(0.02), 'official', '2026-11-03T12:00Z')
    pd.testing.assert_frame_equal(before, store.select('official', '2026-11-02T14:00Z'))
    assert store.select('official', '2026-11-02T14:00Z', 'retrospective').iloc[0, 0] == .02
    assert store.select('official', '2026-11-02T14:00Z', 'approximate_delay').empty
    with pytest.raises(ValueError, match='cannot be reconstructed'):
        store.select('official', '2026-11-01T14:00Z')


def test_close_and_dst(tmp_path):
    store = SnapshotStore(tmp_path)
    market = pd.DataFrame({'SPY_close': [99., 100.]},
                          index=pd.to_datetime(['2026-10-30', '2026-11-02']))
    store.ingest(market, 'market', '2026-11-02T12:00Z')
    assert len(store.select('market', '2026-11-02T21:14Z')) == 1
    assert len(store.select('market', '2026-11-02T21:15Z')) == 1
    market.loc[pd.Timestamp('2026-11-02'), 'SPY_close'] = 102.
    store.ingest(market, 'market', '2026-11-02T21:16Z')
    assert len(store.select('market', '2026-11-02T21:15Z')) == 1
    final = store.select('market', '2026-11-02T21:16Z')
    assert len(final) == 2
    assert final.iloc[-1]['SPY_close'] == 102.
    assert close_at('2026-07-02').hour == 20
    assert close_at('2026-11-02').hour == 21
    with pytest.raises(ValueError, match='timezone'):
        store.select('market', '2026-11-02')


def test_intraday_only_capture_cannot_become_daily_history(tmp_path):
    store = SnapshotStore(tmp_path)
    market = pd.DataFrame({'SPY_close': [100.]}, index=pd.to_datetime(['2026-11-02']))
    with pytest.raises(ValueError, match='No completed market sessions'):
        store.ingest(market, 'market', '2026-11-02T12:00Z')
    assert store.receipts('market') == []


@pytest.mark.parametrize('missing', [True, False])
def test_integrity(tmp_path, missing):
    store = SnapshotStore(tmp_path)
    key = store.ingest(factors(), 'official')
    path = tmp_path / 'objects' / key
    if missing:
        path.unlink()
    else:
        path.write_text('{}')
    with pytest.raises(ValueError, match='snapshot|Snapshot'):
        store.select('official')


def test_frozen_inputs_scoring_and_export(tmp_path):
    store = SnapshotStore(tmp_path)
    official = factors()
    market = pd.DataFrame({'SPY_close': [100.]}, index=official.index)
    predictions = pd.DataFrame([{'date': '2026-10-30', 'model_type': 'ridge',
                                 **{f'pred_{k}': .015 for k in FACTOR_ORDER}}])
    config = {'data': {'cache_dir': str(tmp_path)}, 'point_in_time': {'store_dir': str(tmp_path)}}
    key = freeze_run(config, official, market, predictions, market, official)
    frozen = store.get_bytes(key)
    store.ingest(factors(.02), 'official')
    _, scored = score(store, key)
    assert scored['records'][0]['error'] == pytest.approx(.005)
    assert scored['records'][6]['error'] == pytest.approx(-.005)
    assert store.get_bytes(key) == frozen
    bundle = export_bundle(store)
    assert bundle['schema_version'] == 1
    assert bundle['factor_order'] == FACTOR_ORDER
    assert bundle['records'][0]['values']['RF'] == .02
    json.dumps(bundle, allow_nan=False)


def test_legacy_import_never_backdates(tmp_path):
    from ff5_predictor.vintages import load_snapshot
    config = {'data': {'cache_dir': str(tmp_path)}}
    loaded = load_snapshot(config, 'official', lambda _: factors())
    store = SnapshotStore(tmp_path / 'vintages')
    assert pd.Timestamp(loaded.attrs['receipt']['observed_at']) > pd.Timestamp('2026-11-01T00:00Z')
    config['point_in_time'] = {'decision_at': '2026-10-31T23:00Z', 'mode': 'observed'}
    with pytest.raises(ValueError, match='cannot be reconstructed'):
        load_snapshot(config, 'official', lambda _: pytest.fail('Replay must not load cache/network'))


def test_nowcast_export_null_rf_and_rejects_mode_mismatch(tmp_path):
    store = SnapshotStore(tmp_path)
    official = factors()
    market = pd.DataFrame({'SPY_close': [101.]}, index=pd.to_datetime(['2026-11-02']))
    predictions = pd.DataFrame([{'date': '2026-11-02', 'model_type': 'ridge',
                                 **{f'pred_{k}': .015 for k in FACTOR_ORDER}}])
    config = {'data': {'cache_dir': str(tmp_path)}, 'point_in_time': {'store_dir': str(tmp_path)}}
    key = freeze_run(config, official, market, predictions, market, official)
    bundle = export_bundle(store, mode='retrospective', prediction_id=key)
    assert bundle['records'][-1]['values']['RF'] is None
    assert bundle['records'][-1]['source'] == 'nowcast'
    assert bundle['provenance']['uncertainty'] is None
    with pytest.raises(ValueError, match='modes differ'):
        export_bundle(store, mode='observed', prediction_id=key)
    with pytest.raises(ValueError, match='issued after'):
        export_bundle(store, '2026-11-01T22:00Z', mode='retrospective', prediction_id=key)


def test_replay_does_not_replace_latest_cache(tmp_path):
    from ff5_predictor.data_famafrench import load_ff5
    from ff5_predictor.vintages import store_for
    config = {'data': {'cache_dir': str(tmp_path)}, 'point_in_time': {
        'decision_at': '2026-11-02T22:00Z', 'mode': 'observed'}}
    store = store_for(config)
    store.ingest(factors(), 'official', '2026-11-02T12:00Z')
    store.ingest(factors(.02), 'official', '2026-11-03T12:00Z')
    path = tmp_path / 'ff5_daily_decimal.parquet'
    path.write_bytes(b'newer cache sentinel')
    assert load_ff5(config).iloc[0, 0] == .01
    assert path.read_bytes() == b'newer cache sentinel'
