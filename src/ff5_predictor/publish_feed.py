"""Bounded capture/export with a locked, validated, atomic Kairos feed publication."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def atomic_json(path, payload):
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, 0o640)
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def publish(output, build, validate):
    """Callbacks execute under the same advisory writer lock; failed builds retain output."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output) + '.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        health = Path(str(output) + '.health.json')
        started = datetime.now(timezone.utc).isoformat()
        try:
            with tempfile.TemporaryDirectory(prefix='.feed-', dir=output.parent) as work:
                candidate = Path(work) / 'candidate.json'
                build(candidate)
                if candidate.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError('Feed size limit')
                validate(candidate)
                payload = json.loads(candidate.read_text())
                if not payload.get('records'):
                    raise ValueError('Empty feed')
                atomic_json(output, payload)
            atomic_json(health, dict(status='ok', attempted_at=started,
                                    completed_at=datetime.now(timezone.utc).isoformat()))
            return 0
        except Exception as exc:
            # Never publish exception messages, paths, subprocess output, or credentials.
            atomic_json(health, dict(status='failed', attempted_at=started,
                                    error_type=type(exc).__name__,
                                    completed_at=datetime.now(timezone.utc).isoformat()))
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/nowcast/latest.yaml')
    parser.add_argument('--store', default='data/cache/vintages')
    parser.add_argument('--output', required=True)
    parser.add_argument('--kairos-backend', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--prediction-id', help='Optional existing immutable prediction; never trains models')
    parser.add_argument('--skip-capture', action='store_true', help='Export existing snapshots only')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        parser.error('timeout must be 1..3600 seconds')
    deadline = time.monotonic() + args.timeout

    def run(command, **kwargs):
        subprocess.run(command, check=True, timeout=max(.01, deadline - time.monotonic()),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)

    def build(candidate):
        base = [sys.executable, '-m', 'ff5_predictor.vintage_cli', '--store', args.store]
        if not args.skip_capture:
            run([*base, 'capture', '--config', args.config])
        command = [*base, 'export', '--as-of', datetime.now(timezone.utc).isoformat(),
                   '--mode', 'observed', '--output', str(candidate)]
        if args.prediction_id:
            command += ['--prediction-id', args.prediction_id]
        run(command)

    def validate(candidate):
        backend = args.kairos_backend.resolve()
        run([str(backend / 'venv/bin/python'), '-m', 'server.validate_factor_feed', str(candidate)], cwd=backend)

    return publish(args.output, build, validate)


if __name__ == '__main__':
    raise SystemExit(main())
