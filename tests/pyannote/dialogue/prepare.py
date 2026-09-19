"""Prepare the pinned, annotated dialogue and reuse the complete pipeline lane.

Reads an existing pyannote.audio git checkout; downloads nothing. All outputs
are new files. Binary audio remains in the caller's ignored artifact directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import wave

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(reference_source, output):
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    pins = json.loads((here / 'pins.json').read_text(encoding='utf-8'))
    if np.__version__ != pins['versions']['numpy']:
        raise ValueError('Use pinned NumPy ' + pins['versions']['numpy'])
    output.mkdir(parents=True, exist_ok=False)
    assets = output / 'assets'
    assets.mkdir()
    revision = pins['corpus']['revision']
    for name in ('sample.wav', 'sample.rttm'):
        data = subprocess.check_output(['git', 'show', revision + ':src/pyannote/audio/sample/' + name], cwd=reference_source)
        path = assets / name
        path.write_bytes(data)
        if sha(path) != pins['corpus']['assets'][name]:
            raise ValueError('Upstream asset digest: ' + name)
    with wave.open(str(assets / 'sample.wav'), 'rb') as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) != (1, 2, 16000, 480000):
            raise ValueError('Expected mono PCM16 16 kHz thirty-second sample')
        pcm = np.frombuffer(source.readframes(source.getnframes()), '<i2').astype(np.float32) / np.float32(32768)
    labels = []
    for line in (assets / 'sample.rttm').read_text().splitlines():
        fields = line.split()
        if len(fields) != 10 or fields[0] != 'SPEAKER':
            raise ValueError('Unexpected RTTM annotation')
        start = float(fields[3])
        labels.append([start, start + float(fields[4]), fields[7]])
    cases = []
    for pin, (start, end) in zip(pins['cases'], pins['corpus']['fixed_windows'], strict=True):
        name = pin['name']
        path = assets / (name + '-pcm.npy')
        data = pcm[start * 16000:end * 16000]
        np.save(path, data, allow_pickle=False)
        if sha(path) != pin['pcm_sha256'] or len(data) != pin['samples']:
            raise ValueError('Prepared PCM identity: ' + name)
        intervals = [[max(s, start) - start, min(e, end) - start, k]
                     for s, e, k in labels if min(e, end) > max(s, start)]
        cases.append(dict(name=name, start=start, end=end, pcm=path.name, pcm_sha256=sha(path),
                          samples=len(data), windows=max(1, end-start-9), annotations=intervals))
    manifest = dict(source='https://github.com/pyannote/pyannote-audio', revision=revision,
                    upstream_path='src/pyannote/audio/sample',
                    assets={p.name: sha(p) for p in sorted(assets.iterdir())},
                    sample_rate=16000, decode='PCM16 little endian / 32768 float32', cases=cases)
    path = assets / 'audio.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if sha(path) != pins['corpus']['manifest_sha256']:
        raise ValueError('Prepared corpus manifest identity')
    # A separate directory binds this corpus's pins and recipe resources into
    # the shared runners. The original five-case lane keeps its own pins.
    lane = output / 'recipe'
    lane.mkdir()
    common = here.parent / 'diarization'
    for name in ('generate_reference.py', 'native_rules.py', 'evidence.py', 'audit.py',
                 'Program.cs', 'Detail.cs', 'Evidence.cs', 'test_refusals.py', 'requirements.txt'):
        shutil.copyfile(common / name, lane / name)
    shutil.copyfile(here / 'pins.json', lane / 'pins.json')
    for name in ('DiarizationReplay.csproj', 'DiarizationDetail.csproj'):
        project = (common / name).read_text(encoding='utf-8')
        project = project.replace('../../../src/Lokad.Onnx.Data/Lokad.Onnx.Data.csproj',
                                  (root / 'src/Lokad.Onnx.Data/Lokad.Onnx.Data.csproj').as_posix())
        project = project.replace('../../Shared/NpySupport.cs', (root / 'tests/Shared/NpySupport.cs').as_posix())
        # XML attribute values may contain special characters in workspace paths.
        from xml.sax.saxutils import escape
        project = project.replace(root.as_posix(), escape(root.as_posix(), {'"': '&quot;'}))
        (lane / name).write_text(project, encoding='utf-8')
    record = dict(corpus_manifest_sha256=sha(path), pins_sha256=sha(here / 'pins.json'),
                  preparation_sha256=sha(Path(__file__)),
                  recipes={p.name: sha(p) for p in sorted(lane.iterdir())})
    (output / 'preparation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('Prepared four fixed cases and shared replay recipes:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.reference_source.resolve(), args.output.resolve())
