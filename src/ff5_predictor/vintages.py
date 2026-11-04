"""Content-addressed inputs and append-only observation receipts.

Observation time is local evidence, never a claim about publisher release time.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

FACTOR_ORDER = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
MODES = ('observed', 'retrospective', 'approximate_delay')


def utc(value=None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz='UTC')
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError('Decision/observation timestamps must include a timezone')
    return result.tz_convert('UTC')


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def close_at(date) -> pd.Timestamp:
    # Conservative US equities policy: 16:15 New York, including early-close days.
    # Delaying early-close availability is safe; weekends are never eligible.
    return (pd.Timestamp(date).normalize().tz_localize('America/New_York')
            + pd.Timedelta(hours=16, minutes=15)).tz_convert('UTC')


class SnapshotStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / 'objects').mkdir(parents=True, exist_ok=True)

    def put_bytes(self, payload: bytes) -> str:
        key = hashlib.sha256(payload).hexdigest()
        path = self.root / 'objects' / key
        if path.exists():
            self.get_bytes(key)
            return key
        fd, name = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(name, path)  # Atomic create, never replace an existing object.
            except FileExistsError:
                self.get_bytes(key)
        finally:
            os.unlink(name)
        return key

    def get_bytes(self, key: str) -> bytes:
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid snapshot hash')
        try:
            payload = (self.root / 'objects' / key).read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f'Missing snapshot object: {key}') from exc
        if hashlib.sha256(payload).hexdigest() != key:
            raise ValueError(f'Snapshot integrity failure: {key}')
        return payload

    def put(self, value) -> str:
        return self.put_bytes(canonical(value))

    def get(self, key):
        return json.loads(self.get_bytes(key))

    def receipts(self, dataset):
        directory = self.root / 'receipts' / dataset
        result = []
        for path in directory.glob('*.json'):
            obj = self.get(path.stem)
            if canonical(obj) != path.read_bytes() or obj['dataset'] != dataset:
                raise ValueError('Receipt integrity failure')
            self.get_bytes(obj['snapshot_id'])
            for key in obj.get('raw_hashes', []):
                self.get_bytes(key)
            result.append(obj)
        return sorted(result, key=lambda x: (x['observed_at'], x['receipt_id'] if 'receipt_id' in x else x['snapshot_id']))

    def ingest(self, frame, dataset, observed_at=None, raw=(), source=None):
        if dataset not in ('official', 'market'):
            raise ValueError('Unsupported dataset')
        frame = frame.sort_index().copy()
        if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError('Snapshots require nonempty unique dates and columns')
        payload = {'columns': list(frame.columns), 'rows': [
            [str(pd.Timestamp(date).date()), *[None if pd.isna(v) else float(v) for v in row]]
            for date, row in zip(frame.index, frame.to_numpy())]}
        snapshot_id = self.put(payload)
        history = self.receipts(dataset)
        now = utc(observed_at).isoformat()
        if history and now < history[-1]['observed_at']:
            raise ValueError('Cannot backdate ingestion before existing observations')
        if history and history[-1]['snapshot_id'] == snapshot_id:
            return snapshot_id
        receipt = {'dataset': dataset, 'snapshot_id': snapshot_id, 'observed_at': now,
                   'publication_at': None, 'source': source or 'normalized_loader_output',
                   'raw_hashes': [self.put_bytes(value) for value in raw],
                   'availability_policy': 'observed_and_1615_America/New_York' if dataset == 'market' else 'first_observed'}
        key = self.put(receipt)
        folder = self.root / 'receipts' / dataset
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (key + '.json')
        try:
            with path.open('xb') as f:
                f.write(canonical(receipt))
        except FileExistsError:
            pass
        return snapshot_id

    def frame(self, snapshot_id):
        obj = self.get(snapshot_id)
        return pd.DataFrame([r[1:] for r in obj['rows']], columns=obj['columns'],
                            index=pd.DatetimeIndex([r[0] for r in obj['rows']], name='date'))

    def select(self, dataset, decision_at=None, mode='observed', delay_days=45, required_columns=()):
        if mode not in MODES:
            raise ValueError(f'Availability mode must be one of {MODES}')
        decision = utc(decision_at)
        history = self.receipts(dataset)
        history = [r for r in history if set(required_columns).issubset(self.get(r['snapshot_id'])['columns'])]
        eligible = [r for r in history if utc(r['observed_at']) <= decision] if mode == 'observed' else history
        if not eligible:
            raise ValueError(f'No {dataset} snapshot observed by {decision.isoformat()}; historical vintages cannot be reconstructed')
        receipt = eligible[-1]
        frame = self.frame(receipt['snapshot_id'])
        if dataset == 'market':
            frame = frame.loc[[d.weekday() < 5 and close_at(d) <= decision for d in frame.index]]
        else:
            limit = decision.tz_localize(None).normalize()
            if mode == 'approximate_delay':
                if delay_days < 0:
                    raise ValueError('delay_days must be nonnegative')
                limit -= pd.Timedelta(days=delay_days)
            frame = frame.loc[frame.index < limit]  # Never use a same-day official return.
        frame.attrs['snapshot_id'] = receipt['snapshot_id']
        frame.attrs['receipt'] = receipt
        frame.attrs['availability_mode'] = mode
        return frame

    def dataset(self, official, market, decision_at, mode):
        return self.put({'kind': 'dataset', 'schema_version': 1, 'decision_at': utc(decision_at).isoformat(),
                         'availability_mode': mode,
                         'snapshot_ids': {k: f.attrs['snapshot_id'] for k, f in [('official', official), ('market', market)]},
                         'eligible_dates': {k: [str(d.date()) for d in f.index] for k, f in [('official', official), ('market', market)]}})


def store_for(config):
    fallback = Path(config.get('nowcast', {}).get('output_dir', 'data/cache')) / 'cache'
    cache_dir = Path(config.get('data', {}).get('cache_dir', fallback))
    root = config.get('point_in_time', {}).get('store_dir', cache_dir / 'vintages')
    return SnapshotStore(root)


def load_snapshot(config, dataset, loader):
    options = config.get('point_in_time', {})
    store = store_for(config)
    decision = options.get('decision_at')
    mode = options.get('mode', 'retrospective')
    if mode == 'observed' and config.get('fundamentals', {}).get('enabled', False):
        raise ValueError('Observed replay requires fundamentals disabled: fundamentals are not snapshotted')
    if decision is None:
        frame = loader(config)
        if "snapshot_id" not in frame.attrs:
            store.ingest(frame, dataset, source='loader_or_legacy_cache_observed_now')
    # Explicit decisions are read-only: never download or overwrite caches during replay.
    required = [f'{t.upper()}_close' for t in config['data'].get('tickers', [])] if dataset == 'market' else []
    result = store.select(dataset, decision, mode, int(options.get('delay_days', 45)), required)
    if decision is None:
        result.attrs.update(frame.attrs)
    start, end = config['data'].get('start_date'), config['data'].get('end_date')
    if start:
        result = result.loc[pd.Timestamp(start):]
    if end:
        result = result.loc[:pd.Timestamp(end)]
    return result


def freeze_run(config, official, market, predictions, features, train, model_paths=(), fitted_models=None):
    store = store_for(config)
    options = config.get('point_in_time', {})
    for name, frame in [('official', official), ('market', market)]:
        if 'snapshot_id' not in frame.attrs:
            frame.attrs['snapshot_id'] = store.ingest(frame, name)
    issued = utc().isoformat()
    dataset_id = store.dataset(official, market, options.get('decision_at') or issued,
                               options.get('mode', 'retrospective'))
    def table(frame):
        buffer = io.BytesIO()
        frame.to_parquet(buffer)
        return store.put_bytes(buffer.getvalue())
    import joblib
    model_hashes = {p.name: store.put_bytes(p.read_bytes()) for p in model_paths}
    for name, model in (fitted_models or {}).items():
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        model_hashes[name] = store.put_bytes(buffer.getvalue())
    manifest = {'kind': 'issued_prediction', 'schema_version': 1, 'issued_at': issued,
                'dataset_id': dataset_id, 'config': config, 'features': list(features.columns),
                'feature_input_id': table(features), 'training_input_id': table(train),
                'predictions_id': table(predictions),
                'model_hashes': model_hashes}
    return store.put(manifest)


def score(store, prediction_id):
    issued = store.get(prediction_id)
    predictions = pd.read_parquet(io.BytesIO(store.get_bytes(issued['predictions_id'])))
    history = store.receipts('official')
    frames = [(r, store.frame(r['snapshot_id'])) for r in history]
    rows = []
    for _, row in predictions.iterrows():
        date = pd.Timestamp(row['date']).normalize()
        matches = [(r, f.loc[date]) for r, f in frames if date in f.index]
        for label, match in [('first_observed', matches[:1]), ('latest_revision', matches[-1:])]:
            if not match:
                continue
            receipt, actual = match[0]
            for factor in FACTOR_ORDER:
                if f'pred_{factor}' in row and factor in actual:
                    rows.append({'date': str(date.date()), 'factor': factor, 'model': row.get('model_type'),
                                 'target': label, 'actual': float(actual[factor]),
                                 'error': float(row[f'pred_{factor}'] - actual[factor]),
                                 'snapshot_id': receipt['snapshot_id']})
    result = {'prediction_id': prediction_id, 'scored_at': utc().isoformat(), 'records': rows}
    return store.put(result), result


def export_bundle(store, decision_at=None, mode='observed', prediction_id=None, delay_days=45):
    official = store.select('official', decision_at, mode, delay_days)
    if set(FACTOR_ORDER + ['RF']) - set(official.columns):
        raise ValueError('Export requires all six factors and published RF')
    receipt = official.attrs['receipt']
    history_frames = [(r, store.frame(r['snapshot_id'])) for r in store.receipts('official')]
    def first_seen(date, row):
        for r, f in history_frames:
            if utc(r['observed_at']) > utc(receipt['observed_at']):
                break
            if date in f.index and f.loc[date].reindex(FACTOR_ORDER + ['RF']).equals(row.reindex(FACTOR_ORDER + ['RF'])):
                return r['observed_at']
        return receipt['observed_at']
    records = [{'date': str(date.date()), 'values': {k: float(row[k]) for k in FACTOR_ORDER + ['RF']},
                'source': 'official', 'available_at': first_seen(date, row),
                'snapshot_id': receipt['snapshot_id'], 'model_id': None} for date, row in official.iterrows()]
    prediction_manifest = None
    dataset_id = store.put({'kind': 'export_dataset', 'official_snapshot_id': receipt['snapshot_id'],
                            'decision_at': utc(decision_at).isoformat(), 'availability_mode': mode})
    if prediction_id:
        prediction_manifest = store.get(prediction_id)
        if utc(prediction_manifest['issued_at']) > utc(decision_at):
            raise ValueError('Prediction was issued after export decision timestamp')
        dataset_id = prediction_manifest['dataset_id']
        dataset = store.get(dataset_id)
        for key in dataset['snapshot_ids'].values():
            store.get_bytes(key)
        for name in ('feature_input_id', 'training_input_id', 'predictions_id'):
            store.get_bytes(prediction_manifest[name])
        for key in prediction_manifest['model_hashes'].values():
            store.get_bytes(key)
        if dataset['availability_mode'] != mode:
            raise ValueError('Prediction and export availability modes differ')
        predictions = pd.read_parquet(io.BytesIO(store.get_bytes(prediction_manifest['predictions_id'])))
        primary = prediction_manifest['config'].get('nowcast', {}).get('primary_model', 'ridge')
        for _, row in predictions.loc[predictions['model_type'] == primary].iterrows():
            date = str(pd.Timestamp(row['date']).date())
            if date in set(official.index.strftime('%Y-%m-%d')):
                continue
            if close_at(date) > utc(decision_at):
                raise ValueError('Prediction uses a session not yet closed')
            records.append({'date': date, 'values': {**{k: float(row[f'pred_{k}']) for k in FACTOR_ORDER}, 'RF': None},
                            'source': 'nowcast', 'available_at': prediction_manifest['issued_at'],
                            'snapshot_id': prediction_id, 'model_id': prediction_id})
    return {'schema_version': 1, 'factor_order': FACTOR_ORDER, 'records': sorted(records, key=lambda r: r['date']),
            'provenance': {'generated_at': utc().isoformat(), 'decision_at': utc(decision_at).isoformat(),
                           'dataset_id': dataset_id, 'snapshot_hashes': sorted(set([receipt['snapshot_id']] + (list(dataset['snapshot_ids'].values()) if prediction_id else []))),
                           'currency': 'USD', 'units': 'decimal', 'frequency': 'daily',
                           'return_basis': 'Mkt-RF excess over RF; other factors long-short; RF risk-free daily return',
                           'availability_mode': mode, 'publication_time_known': False,
                           'source': receipt, 'prediction_manifest': prediction_manifest,
                           'uncertainty': None, 'uncertainty_status': 'not_calibrated'}}
