import argparse
from shared import *
from calculation import ideal_decimal, ideal_double, calculate

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); spec = read(base / 'manifest.json'); verify(spec['files'])
    ps = psutil_module(); process = ps.Process(); assert process.cpu_affinity() == [0] and openblas_threads() == 1
    for name, row in spec['captures'].items():
        capture = read(base / name / 'result.json'); assert capture['complete'] and capture['data']['sha256'] == row['data']['sha256']
    for name in ['Window', 'MelWeights']: assert pin(base / 'five' / (name + '.f32')) == pin(base / 'dialogue' / (name + '.f32'))
    weights = dict(native=np.load(ROOT / spec['native_mel']).astype(np.float64),
                   managed=np.fromfile(base / 'dialogue/MelWeights.f32', dtype='<f4').reshape(80, 256).astype(np.float64), ideal=ideal_decimal())
    double = ideal_double(); assert np.max(np.abs(double - weights['ideal'])) <= CONTROL_LIMIT
    output = base / 'analysis'; output.mkdir(exist_ok=False)
    for name, value in weights.items():
        with (output / (name + '-weights.npy')).open('xb') as stream: np.save(stream, value, allow_pickle=False)
    with (output / 'ideal-double-weights.npy').open('xb') as stream: np.save(stream, double, allow_pickle=False)
    records = []
    for case in spec['cases']:
        power = np.load(WINDOWS / 'numpy' / case['name'] / 'power.npy', allow_pickle=False)
        directory = output / case['name']; directory.mkdir(); rows = {}
        for setting, coefficients in weights.items():
            stages = calculate(power, coefficients); entries = {}
            for stage, values in stages.items():
                path = directory / (setting + '-' + stage + '.npy')
                with path.open('xb') as stream: np.save(stream, values, allow_pickle=False)
                entries[stage] = dict(shape=list(values.shape), pin=pin(path))
            rows[setting] = entries
        records.append(dict(name=case['name'], rows=rows)); print(case['name'], flush=True)
    write(output / 'result.json', dict(complete=True, manifest=pin(base / 'manifest.json'), records=records,
          runtime=dict(pid=process.pid, birth=process.create_time(), affinity=process.cpu_affinity(), numpy=np.__version__,
                       blas_threads=openblas_threads(), libraries=libraries(process), native_ort_loaded=any('onnxruntime' in r.path.lower() for r in process.memory_maps()))))

if __name__ == '__main__': main()
