"""Record both independent assignment schedules once, before new inference."""
from pathlib import Path
import argparse
import hashlib
import json
import secrets
import subprocess

from design import PROTOCOL, COHORTS, assignment_schedule, ESTIMATOR

ROOT = Path(__file__).resolve().parents[3]


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Assignment output exists; do not select another draw'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Commit tools before drawing'
    draws = {phase: [secrets.randbelow(6) for _ in range(COHORTS*10)] for phase in ('aa', 'compare')}
    schedules = {phase: assignment_schedule(values, phase) for phase, values in draws.items()}
    record = dict(protocol=PROTOCOL, source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  mechanism='One secrets.randbelow(6) draw independently for every cohort/case/policy/phase; no balancing or seed selection.',
                  cohorts=COHORTS, draws=draws, schedules=schedules,
                  sources={p.name: pin(p) for p in Path(__file__).parent.glob('*.py')}, estimator=pin(ESTIMATOR),
                  inference_started=False, scope='Frozen assignments only; full payload and campaign validation remain required.')
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(dict(protocol=PROTOCOL, assignment_draws=1200, workers_per_phase=1800, manifest=pin(args.output))))


if __name__ == '__main__':
    main()
