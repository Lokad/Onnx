"""Generate isolated rounding groups from the exact unchanged product source."""
import hashlib

SOURCE_SHA='801be94d80e99d897b2067093dd6ad4514e6b9a5cd91d8cd50cf14a77bb77bf1'
VARIANTS=('Original','Frame','Spectrum','Output','All')


def replace_once(text, old, new):
    assert text.count(old)==1,repr(old)
    return text.replace(old,new)


def generate(data, variant):
    assert hashlib.sha256(data).hexdigest()==SOURCE_SHA,'Product source drift'
    assert variant in VARIANTS
    text=data.decode('utf-8').replace('\r\n','\n')
    text=replace_once(text,'namespace Lokad.Onnx;','namespace PrecisionProbe;\nusing Lokad.Onnx;')
    text=replace_once(text,'public static class WeSpeakerAudio\n{',f'public static class {variant}\n{{\n    static {variant}() {{ }}')
    text=replace_once(text,'static readonly float[] Window = CreateWindow();','static readonly float[] Window = Tables.Window.ToArray();')
    text=replace_once(text,'static readonly float[] MelWeights = CreateMelWeights();','static readonly float[] MelWeights = Tables.Mel.ToArray();')
    if variant in ('Frame','All'):
        for old,new in [('var centered = new float[WindowSize];','var centered = new double[WindowSize];'),
                        ('float mean = (float)(total / WindowSize);','double mean = total / WindowSize;'),
                        ('float previous = .97f * centered','double previous = .97f * centered'),
                        ('float value = (centered[j] - previous) * Window[j];','double value = (centered[j] - previous) * Window[j];')]:
            text=replace_once(text,old,new)
    if variant in ('Spectrum','All'):
        for old,new in [('var powers = new float[FourierSize / 2 + 1];','var powers = new double[FourierSize / 2 + 1];'),
                        ('float real = (float)spectrum[k].Real, imaginary = (float)spectrum[k].Imaginary;','double real = spectrum[k].Real, imaginary = spectrum[k].Imaginary;'),
                        ('float magnitude = (float)Math.Sqrt((double)real * real + (double)imaginary * imaginary);','double magnitude = Math.Sqrt(real * real + imaginary * imaginary);')]:
            text=replace_once(text,old,new)
    if variant in ('Output','All'):
        for old,new in [('var output = result.Buffer.Span;','var output = new double[frames * MelBins];'),
                        ('MathF.Log(Math.Max(Epsilon, (float)energy))','Math.Log(Math.Max((double)Epsilon, energy))'),
                        ('float mean = (float)(total / frames);','double mean = total / frames;')]:
            text=replace_once(text,old,new)
        text=replace_once(text,'        return result;\n    }\n\n    static void Fourier',
                          '        for (int i = 0; i < output.Length; i++) result.Buffer.Span[i] = (float)output[i];\n        return result;\n    }\n\n    static void Fourier')
    text=replace_once(text,'            Fourier(spectrum);','            Capture.Window(frame, spectrum);\n            Fourier(spectrum);\n            Capture.Fourier(frame, spectrum);')
    text=replace_once(text,'            for (int mel = 0; mel < MelBins; mel++)','            Capture.Power(frame, powers);\n            for (int mel = 0; mel < MelBins; mel++)')
    statement=next(line for line in text.splitlines() if 'output[frame * MelBins + mel] = Math' in line)
    text=replace_once(text,statement,'                Capture.Energy(frame, mel, energy);\n'+statement+'\n                Capture.Log(frame, mel, output[frame * MelBins + mel]);')
    text=replace_once(text,'        return result;\n    }\n\n    static void Fourier','        Capture.Features(result.Buffer.Span);\n        return result;\n    }\n\n    static void Fourier')
    return text
