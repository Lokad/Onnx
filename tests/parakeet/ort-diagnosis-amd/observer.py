"""Observe the pinned native consumer without changing its decoding or graph outputs."""
import ast
import hashlib
import json
from pathlib import Path
import time


def instrument(source):
    """Insert one installation hook after the original adapter is loaded."""
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    expected = ast.dump(ast.parse('module_spec.loader.exec_module(module)').body[0])
    locations = [i for i, node in enumerate(main.body) if ast.dump(node) == expected]
    assert len(locations) == 1, 'Native consumer changed: review instrumentation'
    index = locations[0] + 1
    main.body.insert(index, ast.parse('diagnostic.install(module)').body[0])
    restored = ast.parse(ast.unparse(tree))
    restored_main = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    del restored_main.body[index]
    assert ast.dump(restored) == ast.dump(ast.parse(source)), 'Unexpected consumer change'
    return ast.fix_missing_locations(tree)


class Observer:
    def __init__(self, manifest, output, mode):
        assert mode in ('control', 'profile')
        self.manifest, self.output, self.mode = manifest, output, mode
        self.sessions, self.setup, self.requests, self.calls = {}, [], [], []
        self.active = None

    def install(self, adapter):
        assert not self.sessions
        ort = adapter.ort
        self.ort = ort
        original_session = ort.InferenceSession
        original_graph = adapter.Parakeet.graph
        original_call = adapter.Parakeet.__call__
        graphs = {str(Path(self.manifest['models'][v]['path']).resolve()): k
                  for k, v in self.manifest['graphs'].items()}

        def session(path, options, providers):
            name = graphs[str(Path(path).resolve())]
            assert name not in self.sessions
            assert providers == ['CPUExecutionProvider']
            assert options.intra_op_num_threads == options.inter_op_num_threads == 1
            assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
            assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            assert not options.enable_profiling and not options.optimized_model_filepath
            for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
                assert options.get_session_config_entry(key) == '0'
            if self.mode == 'profile':
                options.enable_profiling = True
                options.profile_file_prefix = str(self.output / name)
            start, cpu = time.perf_counter_ns(), time.process_time_ns()
            value = original_session(path, options, providers=providers)
            elapsed, used = time.perf_counter_ns()-start, time.process_time_ns()-cpu
            self.sessions[name] = value
            self.setup.append(dict(graph=name, wall_ns=elapsed, cpu_ns=used,
                inputs=[dict(name=v.name, shape=v.shape, type=v.type) for v in value.get_inputs()],
                outputs=[dict(name=v.name, shape=v.shape, type=v.type) for v in value.get_outputs()],
                profile_start_ns=value.get_profiling_start_time_ns() if self.mode == 'profile' else None))
            return value

        def tensors(values):
            return {name: dict(shape=list(v.shape), dtype=str(v.dtype)) for name, v in values.items()}

        def graph(model, name, feeds):
            assert self.active is not None
            inputs = tensors(feeds)
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            result = original_graph(model, name, feeds)
            used, end = time.thread_time_ns()-cpu, time.perf_counter_ns()
            self.calls.append(dict(request=self.active, graph=name, start_ns=start, end_ns=end,
                                   cpu_ns=used, inputs=inputs, outputs=tensors(result)))
            return result

        def call(model, pcm):
            assert self.active is None
            index = len(self.requests)
            self.active = index
            first = len(self.calls)
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            try:
                result = original_call(model, pcm)
                used, end = time.thread_time_ns()-cpu, time.perf_counter_ns()
                self.requests.append(dict(index=index, name=self.manifest['cases'][index % 20]['name'],
                    iteration=index//20, start_ns=start, end_ns=end, cpu_ns=used,
                    first_call=first, calls=len(self.calls)-first, decoder_calls=result['decoder_calls']))
                return result
            finally:
                self.active = None

        ort.InferenceSession = session
        adapter.Parakeet.graph, adapter.Parakeet.__call__ = graph, call

    def finish(self):
        assert len(self.requests) == 80 and len(self.calls) == 4960
        profiles = {}
        if self.mode == 'profile':
            for name, session in self.sessions.items():
                path = Path(session.end_profiling()).resolve()
                assert path.parent == self.output.resolve() and path.is_file()
                with path.open('rb') as stream:
                    profiles[name] = dict(file=path.name, bytes=path.stat().st_size,
                        sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
        value = dict(mode=self.mode, setup=self.setup, requests=self.requests, calls=self.calls,
                     profiles=profiles, build_info=self.ort.get_build_info(),
                     available_providers=self.ort.get_available_providers(),
                     graph_outputs_changed=False, original_consumer_hook_count=1)
        with (self.output/'observation.json').open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)


def run(native, assets, manifest, output, mode):
    import sys
    manifest_value = json.loads(manifest.read_text(encoding='utf8'))
    assert manifest_value['family'] == 'parakeet' and len(manifest_value['cases']) == 20
    assert sum(c['expected']['decoder_calls'] for c in manifest_value['cases']) == 1200
    output.mkdir()
    observer = Observer(manifest_value, output, mode)
    sys.path.insert(0, str(native.parent))
    namespace = dict(__name__='native_diagnostic_consumer', __file__=str(native), diagnostic=observer)
    exec(compile(instrument(native.read_text(encoding='utf8')), str(native), 'exec'), namespace)
    sys.argv = [str(native), str(assets), str(manifest), str(output/'requests'), 'timing']
    namespace['main']()
    observer.finish()


if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 6
    run(*(Path(p).resolve() for p in sys.argv[1:5]), sys.argv[5])
