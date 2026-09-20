"""Pinned corpus policy and small, side-effect-free evidence helpers."""
from pathlib import Path
import hashlib
import json
import unicodedata
import numpy as np

REVISION = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
LOCALES = [('en_us', 'en'), ('fr_fr', 'fr'), ('de_de', 'de'), ('es_419', 'es'), ('it_it', 'it')]
BANDS = [(4, 10), (12, 25)]
CORE = '187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4'
DATA = '809242b58725c6ae47514cc3908ef59ffafae6be36bb6e2fba20144d9a975af5'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def select(inventory):
    if len({r['filename'] for r in inventory}) != len(inventory):
        raise ValueError('Duplicate source filename')
    genders = sorted({r['gender'] for r in inventory})
    if not genders or not set(genders) <= {0, 1}:
        raise ValueError('Missing or invalid gender labels')
    selected = []
    for round_index in range(2):
        gender = genders[round_index % len(genders)]
        for low, high in BANDS:
            used_ids = {r['id'] for r in selected}
            eligible = [r for r in inventory if r['gender'] == gender and low * 16000 <= r['samples'] <= high * 16000
                        and r['id'] not in used_ids]
            if not eligible:
                raise ValueError('Missing gender/duration stratum')
            selected.append(min(eligible, key=lambda r: (r['id'], r['filename'])))
    return selected


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold().translate(str.maketrans({'\u2018': "'", '\u2019': "'", '\u02bc': "'"}))
    chars = [c if c.isalnum() or (c == "'" and 0 < i < len(text) - 1 and text[i-1].isalnum() and text[i+1].isalnum()) else ' '
             for i, c in enumerate(text)]
    return ' '.join(''.join(chars).split())


def variants(pcm, locale, filename):
    if pcm.dtype != np.float32 or pcm.ndim != 1 or len(pcm) < 2 or not np.isfinite(pcm).all() or not np.any(pcm) or np.abs(pcm).max() > 1:
        raise ValueError('Source PCM must be nonzero finite normalized mono float32')
    seed = int.from_bytes(hashlib.sha256((REVISION + '/' + locale + '/' + filename).encode('utf-8')).digest()[:8], 'little')
    original = pcm.astype(np.float64)
    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(len(pcm))
    noise -= noise.mean()
    signal_power = float(np.mean(original * original))
    noise *= np.sqrt(signal_power / (10 * np.mean(noise * noise)))
    gain = min(1., .99 / max(float(np.abs(original).max()), float(np.abs(original + noise).max())))
    clean = (original * gain).astype(np.float32)
    noisy = ((original + noise) * gain).astype(np.float32)
    actual_noise = noisy.astype(np.float64) - clean.astype(np.float64)
    actual_snr = float(10 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean(actual_noise ** 2)))
    if not np.isfinite(actual_snr) or not np.isfinite(clean).all() or not np.isfinite(noisy).all() or abs(actual_snr - 10) > .001 or max(float(np.abs(clean).max()), float(np.abs(noisy).max())) > 1:
        raise ValueError('Noise construction contract failed')
    return clean, noisy, dict(seed=seed, gain=gain, source_power=signal_power, actual_snr_db=actual_snr)
