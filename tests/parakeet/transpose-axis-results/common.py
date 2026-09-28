"""Publication IO; outputs must be new and finite JSON values are required."""
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def csvfile(path, rows):
    with path.open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def application():
    base = Path(__file__).resolve().parents[3]/'artifacts/parakeet-transpose-axis-app-amd-20260928'
    assert pin(base/'closed.json')['sha256'] == '2e4741a67772d71e4442630d7f9cf751dfba2a52e12631fffed3b2db62fefd85'
    proof = read(base/'closed.json')
    assert proof['passed'] and proof['admitted'] and proof['analysis'] == pin(base/'analysis.json')
    value = read(base/'analysis.json')
    assert value['passed'] and value['performance']['admitted']
    return value
