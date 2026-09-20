"""Two separately structured double-precision fixed-coefficient frontends."""
import math
from common import np, STAGES

def validate(samples, window, mel):
    assert samples.dtype == window.dtype == mel.dtype == np.float32
    assert samples.ndim == 1 and 400 <= samples.size <= 480000
    assert np.isfinite(samples).all() and np.max(np.abs(samples)) <= 1
    assert window.shape == (400,) and mel.shape == (80, 256)
    assert np.isfinite(window).all() and np.isfinite(mel).all() and (mel >= 0).all()

def numpy_route(samples, window, mel):
    validate(samples, window, mel)
    scaled = samples.astype(np.float64) * 32768
    frames = np.lib.stride_tricks.sliding_window_view(scaled, 400)[::160].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    previous = np.concatenate((frames[:, :1], frames[:, :-1]), axis=1)
    emphasized = frames - float(np.float32(.97)) * previous
    windowed = np.pad(emphasized * window.astype(np.float64), ((0, 0), (0, 112)))
    transform = np.fft.rfft(windowed, axis=1)
    power = np.abs(transform) ** 2
    energy = power[:, :256] @ mel.astype(np.float64).T
    raw = np.log(np.maximum(energy, float(np.finfo(np.float32).eps)))
    features = raw - raw.mean(axis=0, keepdims=True)
    return dict(zip(STAGES, (windowed, transform.real, transform.imag, power, energy, raw, features[None])))

def torch_route(samples, window, mel):
    import torch
    validate(samples, window, mel)
    waveform = torch.from_numpy(samples.copy()).to(dtype=torch.float64)
    framed = (waveform * 32768).unfold(0, 400, 160)
    dc = framed - torch.mean(framed, dim=1).unsqueeze(1)
    preemphasis = dc.clone()
    preemphasis[:, 1:] -= float(np.float32(.97)) * dc[:, :-1]
    preemphasis[:, 0] -= float(np.float32(.97)) * dc[:, 0]
    windowed = torch.zeros((framed.size(0), 512), dtype=torch.float64)
    windowed[:, :400] = preemphasis * torch.tensor(window, dtype=torch.float64)
    spectrum = torch.fft.rfft(windowed, n=512, dim=1)
    power = spectrum.abs().square()
    weights = torch.zeros((80, 257), dtype=torch.float64)
    weights[:, :256] = torch.tensor(mel, dtype=torch.float64)
    energy = torch.mm(power, weights.transpose(0, 1))
    raw = energy.clamp_min(float(np.finfo(np.float32).eps)).log()
    features = raw - torch.mean(raw, dim=0)
    return {name: value.numpy().copy() for name, value in zip(STAGES,
            (windowed, spectrum.real, spectrum.imag, power, energy, raw, features.unsqueeze(0)))}

def scalar_window(samples, window, frame):
    values = [float(x) * 32768 for x in samples[160 * frame:160 * frame + 400]]
    mean = math.fsum(values) / 400
    centered = [value - mean for value in values]
    return np.asarray([(value - float(np.float32(.97)) * centered[max(0, i - 1)]) * float(window[i])
                       for i, value in enumerate(centered)] + [0.] * 112)

def direct_tables():
    # Exact periodic reduction avoids introducing large-angle error in the oracle.
    real = [math.cos(-2 * math.pi * j / 512) for j in range(512)]
    imaginary = [math.sin(-2 * math.pi * j / 512) for j in range(512)]
    for index, a, b in [(0, 1., 0.), (128, 0., -1.), (256, -1., 0.), (384, 0., 1.)]:
        real[index], imaginary[index] = a, b
    return ([[real[(k * j) % 512] for j in range(512)] for k in range(257)],
            [[imaginary[(k * j) % 512] for j in range(512)] for k in range(257)])

def direct_fourier(windowed, tables):
    values = windowed.tolist()
    return tuple(np.asarray([math.fsum(a * b for a, b in zip(values, row)) for row in table]) for table in tables)
