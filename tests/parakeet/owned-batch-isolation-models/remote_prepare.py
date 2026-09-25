"""Reuse qualified full-model inputs and consumers with exact product identities."""
from pathlib import Path
import copy
import json
import os
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE/'payload.json').exists()
    stage = read(BASE/'stage.json')
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for terminal in stage['terminals']:
        assert pin(terminal['remote']) == pin(BASE/terminal['local'])
        receipt = read(terminal['remote'])
        assert receipt['terminal'] and receipt['code'] == 0 and not any(live(i) for i in receipt['identities'])
    graph = read(BASE/'evidence/graphs-closed.json')
    assert graph['passed'] and graph['admitted'] and graph['all_controls_passed']
    assert graph['files']['analysis.json'] == pin(BASE/'evidence/graphs-analysis.json')
    assert graph['generator'] == pin(BASE/'evidence/graphs-generator.py')
    combined = read(BASE/'evidence/graphs-analysis.json')
    assert graph['source_closures'] == combined['source_closures'] == dict(
        original_graphs=pin(BASE/'evidence/original-graphs-closed.json'),
        short_e5_correction=pin(BASE/'evidence/short-e5-closed.json'))
    assert not read(BASE/'evidence/original-graphs-closed.json')['admitted']
    corrected = read(BASE/'evidence/short-e5-closed.json')
    assert corrected['passed'] and corrected['admitted'] and corrected['all_controls_passed']
    assert combined['original_graph_failure_preserved'] and len(combined['performance']) == 8
    assert all(row['qualified'] for row in combined['performance'])
    assert combined['products']['candidate']['Lokad.Onnx.dll'] == stage['identities']['candidate']['Lokad.Onnx.dll']
    build = read(BASE/'evidence/build-analysis.json')
    assert build['product'] == stage['identities']['candidate']
    assert build['compiled_review'] == pin(BASE/'evidence/build-review.json')
    assert read(BASE/'evidence/build-review.json')['release_dispatcher_restored']
    assert read(BASE/'evidence/parent-analysis.json')['identities']['candidate'] == stage['identities']['selected']
    for name, wanted in stage['external'].items(): assert pin(name) == wanted, name
    for name, value in stage['links'].items():
        destination = BASE/name
        assert destination.resolve().is_relative_to(BASE.resolve()) and not destination.exists()
        source = Path(value['source']); assert pin(source) == value['identity'], name
        destination.parent.mkdir(parents=True, exist_ok=True); os.link(source, destination)
    for role in ['selected','candidate']:
        folder = BASE/'runtimes'/role
        for name, wanted in stage['identities'][role].items(): assert pin(folder/name) == wanted, name
        for name, wanted in stage['consumers'].items(): assert pin(folder/(name+'.dll')) == wanted, name
        manifest = copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'],
                        data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'],
                        product_source=stage['labels'][role])
        (BASE/'manifests').mkdir(exist_ok=True)
        save(BASE/'manifests'/(role+'-parakeet.json'), manifest)
    payload = dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,
                   **{name:stage[name] for name in ['previous_owner','identities','consumers','external','interpreter','failed_release_controls','release_admitted']},
                   scope='Full unchanged Parakeet correctness: 784 arrays and 20 public clips per product/instruction mode; no performance score.',
                   files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'})
    save(BASE/'payload.json',payload); verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__ == '__main__':
    main()
