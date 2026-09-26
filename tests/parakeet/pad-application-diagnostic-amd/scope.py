"""Exact source transformations and compiled safety gates before inference."""
import ast
import json


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def consumer(source):
    source = replace_once(source, 'PhaseConsumer.Initialize();',
                          'PhaseConsumer.Initialize();\nTraceStartup.Wait(output);')
    source = source.replace('if(pass>=warmup && sampled)', 'if(sampled)')
    assert source.count('if(sampled)Require(DiagnosticEvents.Log.IsEnabled()') == 2
    source = replace_once(source, '    if(pass>=warmup)DiagnosticEvents.Log.Boundary("begin",c.Name,pass);',
                          '    TraceStartup.Anchor();\n    DiagnosticEvents.Log.Boundary("begin",c.Name,pass);')
    source = replace_once(source, '    if(pass>=warmup)DiagnosticEvents.Log.Boundary("end",c.Name,pass);',
                          '    DiagnosticEvents.Log.Boundary("end",c.Name,pass);\n    TraceStartup.Anchor();')
    return replace_once(source, '\nreturn 0;', '\nTraceStartup.Save(output);\nreturn 0;')


def supervisor(source):
    source = replace_once(source, "spawn('worker',command,build_env if build else env,cpu)",
        "spawn('worker',command,job_environment(name,spec,build_env if build else env),cpu)")
    source = source.replace("BASE/name/'ready.json'", "BASE/name/'startup-ready.json'")
    assert source.count("BASE/name/'startup-ready.json'") == 3
    source = replace_once(source, '      members=[]', '''      if name.endswith('-capture') and (BASE/name/'ready.json').exists() and not (BASE/name/'release.json').exists():
       ready=read(BASE/name/'ready.json');identity=row['processes']['worker']
       assert ready['pid']==identity['pid'] and ready['warmup_records']==20 and 'collector' in children
       assert children['collector'].poll() is None and live(identity)
       save(BASE/name/'release.json',dict(pid=identity['pid'],sampled=True))
      members=[]''')
    for name in ['startup-ready.json', 'ready.json', 'release.json']:
        source = source.replace("BASE/name/'"+name+"'", "BASE/name/'requests/"+name+"'")
    return source


def attribute_module(source):
    tree = ast.parse(source)
    function, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'attribute']
    return ('from collections import Counter\nfrom protocol import read\n'
            "PHASES={'nemo128.onnx':'frontend','encoder-model.onnx':'encoder','decoder_joint-model.onnx':'decoder'}\n\n"
            + ast.get_source_segment(source, function) + '\n')


def compiled(value, role, spec, built, reference_observer):
    assert value['inventory_complete'] and role in ['current', 'candidate']
    core, data, runner = value['observations']
    assert [r['assembly'] for r in value['observations']] == ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll', 'SampledAudio.dll']
    for row in value['observations']:
        assert row['public_surface_equal'] and not row['removed']
        assert all(row['method_flags_after'][k] == v for k, v in row['method_flags_before'].items())
    assert core['before_sha256'] == spec['products']['current']['Lokad.Onnx.dll']['sha256']
    assert core['after_sha256'] == spec['products'][role]['Lokad.Onnx.dll']['sha256']
    assert core['methods'] == 3179
    if role == 'current':
        assert core['unchanged_methods'] == 3179 and not core['differences'] and not core['added']
    else:
        assert core['unchanged_methods'] == 3178
        assert len(core['differences']) == 1 and '::Pad::' in core['differences'][0]
        assert len(core['added']) == 1 and '::PadDispatch::' in core['added'][0]
        # Compare changed/new bodies to the previously reviewed rejected build.
        retained = spec['candidate_methods']
        assert core['candidate_methods'] == retained
    assert data['before_sha256'] == spec['products']['current']['Lokad.Onnx.Data.dll']['sha256']
    assert data['after_sha256'] == spec['observer']['sha256']
    for key in ['normalized_methods', 'method_flags_before', 'method_flags_after', 'candidate_methods',
                'public_surface', 'methods', 'unchanged_methods', 'added', 'removed', 'differences']:
        assert data[key] == reference_observer[key], key
    assert runner['before_sha256'] == spec['original_consumer']['sha256']
    assert runner['after_sha256'] == built['consumer']['sha256']
    assert runner['methods'] == 164 and runner['unchanged_methods'] == 163
    assert runner['differences'] == ['Program::<Main>$::Int32 <Main>$(System.String[])']
    assert runner['added'] and all(k.startswith(('TraceStartup', 'ParakeetClock', '<>f__AnonymousType')) for k in runner['added'])
    return dict(passed=True, role=role, core_methods=3179, data_methods=697,
                original_consumer_methods=164, unchanged_consumer_methods=163,
                resolved_operand_bindings=True, existing_implementation_flags_equal=True)
