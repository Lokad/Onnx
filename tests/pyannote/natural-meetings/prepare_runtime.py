"""Bind prepared audio to existing qualified product and native source assets."""
from pathlib import Path
import argparse
import hashlib
import wave
import numpy as np
from common import CORE, DATA, pin, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    base = a.artifact.resolve()
    prepared = read(base / 'preparation.json')
    assert prepared['passed']
    for section in ['inputs', 'originals']:
        for name, wanted in prepared[section].items():
            assert pin(base / section / name) == wanted
    old = read(root / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json')
    receipt = read(root / 'artifacts/audio-ort-baseline-v2-20260919/receipt.json')
    assert receipt['complete'] and receipt['all_owned_processes_terminal']
    manifest = {key: old[key] for key in ['models', 'native_assets', 'upstream', 'pins']}
    assert manifest['pins'] == read(root / 'tests/pyannote/dialogue/pins.json')
    for item in list(manifest['models'].values()) + list(manifest['native_assets'].values()) + list(manifest['upstream'].values()):
        assert pin(root / item['path']) == {k:item[k] for k in ['bytes','sha256']}
    for key, item in manifest['models'].items():
        assert item['sha256'] == manifest['pins']['models'][key]
    for name, item in manifest['upstream'].items():
        assert hashlib.sha256((root / item['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == manifest['pins']['source_hashes'][name]
    remote_paths = ['artifacts/segmented-conv-20260918/models/pyannote-segmentation/segmentation/model.onnx',
                    'artifacts/pyannote-embedding-20260919/models/embedding_encoder.onnx',
                    'artifacts/wespeaker-api-20260919/reference/projection.onnx',
                    'artifacts/pyannote-clustering-20260919/reference/prepared.json']
    for key, remote in zip(['segmentation','encoder','projection','plda'],remote_paths,strict=True):
        manifest['models'][key]['amd_path'] = remote
    native_sources = {}
    for name in ['tests/audio/comparison/native_adapters.py','tests/pyannote/diarization/native_rules.py']:
        assert pin(root / name) == receipt['frozen_payload'][name]
        native_sources[name] = dict(path=name, **pin(root/name))
    cases = read(base / 'inputs/dataset.json')['cases']
    case_fields = ['name','samples','path','pcm_sha256','wav']
    selected = [{key:c[key] for key in case_fields} for c in cases]
    with wave.open(str(base / 'inputs' / selected[0]['path']),'rb') as source:
        pcm = np.frombuffer(source.readframes(480000),dtype='<i2').astype(np.float32)/np.float32(32768)
    selected.append(dict(selected[0],name='ES2004a-recovery30',samples=480000,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()))
    qualified = root / 'artifacts/asr-multilingual-amd-20260920/collected/bin'
    assert pin(qualified/'Lokad.Onnx.dll')['sha256'] == CORE and pin(qualified/'Lokad.Onnx.Data.dll')['sha256'] == DATA
    manifest.update(schema=1, family='pyannote', cases=selected, native_sources=native_sources,core_sha256=CORE,data_sha256=DATA,
                    input_manifest_sha256=pin(base/'inputs/dataset.json')['sha256'],native_host='Windows i7-14700KF CPU2',managed_host='AMD EPYC 9V74 CPU2',
                    limits=dict(rss=8*1024**3,seconds=3600,available=1024**3,preflight=8*1024**3),
                    accuracy_scope='Two fixed natural ten-minute meeting excerpts; descriptive replay times, no repeated latency comparison')
    write(base / 'manifest.json', manifest)
    print('Bound three calls, four models, native source identities and current qualified product.')


if __name__ == '__main__':
    main()
