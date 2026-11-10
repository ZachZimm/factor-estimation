import fcntl
import json
import pytest
from ff5_predictor.publish_feed import publish


def test_publish_and_retain(tmp_path):
    output = tmp_path / 'feed.json'
    def build(path):
        path.write_text('{"records":[1]}')
    assert publish(output, build, lambda p: None) == 0
    before = output.read_bytes()
    def invalid(path):
        raise ValueError('malformed feed')
    assert publish(output, build, invalid) == 1
    assert output.read_bytes() == before
    assert json.loads((tmp_path / 'feed.json.health.json').read_text())['status'] == 'failed'
    assert publish(output, lambda p: p.write_text('{'), lambda p: None) == 1
    assert output.read_bytes() == before


def test_overlapping_writer(tmp_path):
    output = tmp_path / 'feed.json'
    with open(str(output) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert publish(output, lambda p: pytest.fail('overlap'), lambda p: None) == 75
    assert not output.exists()


def test_empty_and_oversized(tmp_path):
    output = tmp_path / 'feed.json'
    assert publish(output, lambda p: p.write_text('{"records":[]}'), lambda p: None) == 1
    def large(path):
        with path.open('wb') as handle:
            handle.truncate(32 * 1024 * 1024 + 1)
    assert publish(output, large, lambda p: pytest.fail('oversized validation')) == 1
    assert not output.exists()
