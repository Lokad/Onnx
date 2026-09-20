"""Download original pinned AMI references and preserve unmodified PCM16 crops."""
from pathlib import Path
import argparse
import hashlib
import shutil
import urllib.request
import wave
import numpy as np
import soundfile as sf
from common import REVISION, MEETINGS, DURATION, annotations, coverage, pin, select, write


def download(url, path):
    partial = path.with_suffix(path.suffix + '.partial')
    assert not path.exists() and not partial.exists()
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as stream:
        headers = dict(response.headers)
        shutil.copyfileobj(response, stream)
    if 'Content-Length' in headers:
        assert partial.stat().st_size == int(headers['Content-Length'])
    partial.rename(path)
    return dict(url=url, headers=headers, **pin(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    a = p.parse_args()
    base = a.artifact.resolve()
    base.mkdir(parents=True)
    originals = base / 'originals'
    originals.mkdir()
    inputs = base / 'inputs'
    inputs.mkdir()
    raw = f'https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/{REVISION}/'
    sources = {}
    for name, suffix in [('test.meetings.txt', 'lists/test.meetings.txt'), ('setup-README.md', 'README.md'), ('setup-LICENSE', 'LICENSE')]:
        sources[name] = download(raw + suffix, originals / name)
    sources['ami-download.html'] = download('https://groups.inf.ed.ac.uk/ami/download/', originals / 'ami-download.html')
    assert select((originals / 'test.meetings.txt').read_text()) == MEETINGS
    cases = []
    for meeting in MEETINGS:
        for suffix, directory in [('rttm', 'only_words/rttms'), ('uem', 'uems')]:
            name = meeting + '.' + suffix
            sources[name] = download(raw + directory + '/test/' + name, originals / name)
        name = meeting + '.Mix-Headset.wav'
        sources[name] = download(f'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{name}', originals / name)
        with wave.open(str(originals / name), 'rb') as source:
            assert (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) == (1, 2, 16000, 'NONE')
            assert source.getnframes() >= DURATION * 16000
            original_samples = source.getnframes()
            original_bytes = source.readframes(DURATION * 16000)
        assert len(original_bytes) == DURATION * 32000
        pcm = np.frombuffer(original_bytes, dtype='<i2').astype(np.float32) / np.float32(32768)
        independently_decoded, rate = sf.read(originals / name, frames=DURATION * 16000, dtype='float32')
        assert rate == 16000 and independently_decoded.shape == pcm.shape and independently_decoded.tobytes() == pcm.tobytes()
        path = inputs / (meeting + '-600s.wav')
        with wave.open(str(path), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(original_bytes)
        labels = annotations((originals / (meeting + '.rttm')).read_text(), (originals / (meeting + '.uem')).read_text(), meeting)
        counts = coverage(labels)
        assert counts['speakers'] >= 3 and counts['overlap_seconds'] > 0
        row = dict(name=meeting, seconds=DURATION, samples=DURATION * 16000, path=path.name, pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                   wav=pin(path), original_samples=original_samples, intervals=labels, coverage=counts)
        cases.append(row)
        print(meeting, counts, flush=True)
    manifest = dict(schema=1, revision=REVISION, cases=cases, selection='First test session-a at ES and IS; first600seconds; mixed headset',
                    audio_license='CC-BY-4.0', reference_setup_license='Apache-2.0', sources=sources,
                    scoring=dict(collar_seconds=0, overlap_included=True, uem_seconds=DURATION, label_assignment='optimal one-to-one', annotations='only_words'))
    write(inputs / 'dataset.json', manifest)
    write(base / 'preparation.json', dict(passed=True, inputs={p.name: pin(p) for p in inputs.iterdir()}, originals={p.name: pin(p) for p in originals.iterdir()},
                                       source={p.name: pin(p) for p in Path(__file__).parent.iterdir() if p.suffix == '.py'}, numpy=np.__version__, soundfile=sf.__version__))


if __name__ == '__main__':
    main()
