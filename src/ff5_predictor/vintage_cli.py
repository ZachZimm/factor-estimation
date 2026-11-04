"""Local-only point-in-time commands: python -m ff5_predictor.vintage_cli."""
import argparse
import json
from pathlib import Path

import yaml

from ff5_predictor.vintages import MODES, SnapshotStore, export_bundle, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', default='data/cache/vintages')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('export', 'replay'):
        p = commands.add_parser(command)
        p.add_argument('--as-of', required=True, help='Timezone-aware decision timestamp')
        p.add_argument('--mode', choices=MODES, default='observed')
        p.add_argument('--delay-days', type=int, default=45)
        if command == 'export':
            p.add_argument('--prediction-id')
            p.add_argument('--output', required=True)
        else:
            p.add_argument('--config', required=True)
    p = commands.add_parser('capture')
    p.add_argument('--config', required=True)
    p = commands.add_parser('score')
    p.add_argument('--prediction-id', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    store = SnapshotStore(args.store)
    if args.command == 'capture':
        from ff5_predictor.config import load_config
        from ff5_predictor.data_famafrench import load_ff5
        from ff5_predictor.data_yfinance import load_market_data
        config = load_config(args.config)
        config['point_in_time'] = {'store_dir': args.store}
        print(json.dumps({k: loader(config).attrs['snapshot_id'] for k, loader in [('official', load_ff5), ('market', load_market_data)]}))
        return
    if args.command == 'export':
        result = export_bundle(store, args.as_of, args.mode, args.prediction_id, args.delay_days)
    elif args.command == 'score':
        key, result = score(store, args.prediction_id)
        result = {**result, 'score_id': key}
    else:
        from ff5_predictor.latest_nowcast import run_latest_nowcast
        from ff5_predictor.config import load_config
        config = load_config(args.config)
        config['point_in_time'] = {'store_dir': args.store, 'decision_at': args.as_of,
                                   'mode': args.mode, 'delay_days': args.delay_days}
        result = run_latest_nowcast(config)
        print(json.dumps({'run_dir': str(result.run_dir),
                          'prediction_manifest_id': result.metadata['prediction_manifest_id']}))
        return
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
