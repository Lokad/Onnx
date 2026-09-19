"""Bounded arithmetic-only check, with inherited CPU affinity and code capture."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, time
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    base = args.output.resolve(); base.mkdir(parents=True, exist_ok=False)
    parent = psutil.Process(); prior = parent.cpu_affinity()
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    env.update(DOTNET_JitDisasm='SoftmaxReduction.Kernels:*', DOTNET_JitStdOutFile=str(base / 'codegen.txt'))
    command = ['dotnet', str(binary), str(base / 'check.json'), '--condition']
    identity = dict(command=command, started=time.time(), samples=[], cpu=args.cpu)
    try:
        with (base / 'check.log').open('x', encoding='utf-8') as log:
            parent.cpu_affinity([args.cpu])
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            parent.cpu_affinity([0])
            process = psutil.Process(child.pid); birth = process.create_time()
            identity.update(pid=child.pid, create_time=birth)
            start = time.monotonic()
            try:
                while child.poll() is None:
                    try:
                        assert process.create_time() == birth and process.cpu_affinity() == [args.cpu]
                        rss = process.memory_info().rss
                        identity['samples'].append(dict(seconds=time.monotonic()-start, rss=rss))
                        assert rss < 1024**3 and time.monotonic()-start < 60
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(.1)
            finally:
                if child.poll() is None:
                    assert process.create_time() == birth and process.is_running()
                    child.kill()
                identity.update(code=child.wait(), ended=time.time())
                identity['sha256'] = {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(base.rglob('*')) if p.is_file()}
                identity['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
                with (base / 'identity.json').open('x', encoding='utf-8') as f:
                    json.dump(identity, f, indent=2); f.write('\n')
    finally:
        parent.cpu_affinity(prior)
    assert identity['code'] == 0 and not psutil.pid_exists(child.pid)
    print((base / 'check.log').read_text(encoding='utf-8'))
    print('Terminal', child.pid, 'peak', max(s['rss'] for s in identity['samples']))


if __name__ == '__main__':
    main()
