import argparse, shutil, subprocess
from shared import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); base.mkdir(parents=True, exist_ok=False); (base / 'app').mkdir()
    psutil_module().Process().cpu_affinity([0]); files = {}
    def bind(path, wanted=None):
        actual = pin(path)
        if wanted is not None: assert actual == wanted, str(path)
        files[rel(path)] = actual
    receipt = read(WINDOWS / 'closed.json'); assert pin(WINDOWS / 'closed.json')['sha256'] == WINDOW_RECEIPT
    bind(WINDOWS / 'closed.json'); prior = read(WINDOWS / 'manifest.json')
    for name, wanted in receipt['files'].items(): bind(WINDOWS / name, wanted)
    for name, wanted in receipt['reports'].items(): bind(ROOT / name, wanted)
    for name, wanted in prior['files'].items(): bind(ROOT / name, wanted)
    for name, wanted in prior['numeric'].items(): assert pin(name) == wanted
    assert prior['interpreter'] == pin(sys.executable)
    captures = {}
    for name, folder in CAPTURES.items():
        for path in (ROOT / folder).iterdir():
            if path.is_file(): bind(path)
        captures[name] = dict(directory=folder, data=pin(ROOT / folder / 'Lokad.Onnx.Data.dll'))
    diagnostic = ROOT / 'artifacts/wespeaker-frontend-20260919/diagnostic/result'
    for name in ['window.f32', 'mel.f32']: bind(diagnostic / name)
    folder = Path(__file__).parent
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', rel(folder)], cwd=ROOT, check=True)
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', rel(folder)], cwd=ROOT).strip()
    for path in folder.iterdir():
        if path.is_file(): bind(path)
    for name in ['Capture.csproj', 'Program.cs']:
        shutil.copyfile(folder / name, base / 'app' / name); bind(base / 'app' / name)
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(folder / 'test_calculation.py')], capture_output=True, text=True, timeout=60)
    write(base / 'tests.json', dict(code=tests.returncode, stdout=tests.stdout, stderr=tests.stderr)); assert tests.returncode == 0
    spec = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), prior=rel(WINDOWS / 'manifest.json'),
                files=files, captures=captures, diagnostic=rel(diagnostic), cases=prior['cases'], native_mel=prior['mel'],
                numeric=prior['numeric'], interpreter=prior['interpreter'], tests=pin(base / 'tests.json'), control_limit=CONTROL_LIMIT,
                original_limit=ORIGINAL_LIMIT, limits=LIMITS)
    write(base / 'manifest.json', spec)
    print(json.dumps(dict(files=len(files), cases=len(spec['cases']), manifest=pin(base / 'manifest.json'))))

if __name__ == '__main__': main()
