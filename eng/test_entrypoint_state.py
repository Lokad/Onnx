"""Exercise real PowerShell entrypoints with isolated roots and command stubs."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
PWSH = shutil.which('pwsh')


@unittest.skipUnless(PWSH, 'PowerShell 7 is required')
class E5EnvironmentTests(unittest.TestCase):
    def run_gate(self, scenario, initial):
        with tempfile.TemporaryDirectory(prefix='lonnx-entrypoint-') as directory:
            root = Path(directory).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            eng = root / 'eng'
            eng.mkdir()
            script = (ROOT / 'eng/test-e5.ps1').read_text(encoding='utf-8')
            (eng / 'test-e5.ps1').write_text(script, encoding='utf-8')
            (eng / 'restore-tokenizers.ps1').write_text("$global:LASTEXITCODE = 0\n", encoding='utf-8')
            if scenario != 'missing-assets':
                assets = root / 'models/multilingual-e5-small'
                assets.mkdir(parents=True)
                for name in ['model.onnx', 'sentencepiece.bpe.model']:
                    (assets / name).write_bytes(b'synthetic fixture, never parsed')
            model_hash = re.search(r'\$expectedModel = "([A-F0-9]+)"', script).group(1)
            tokenizer_hash = re.search(r'\$expectedTokenizer = "([A-F0-9]+)"', script).group(1)
            wrapper = r'''
$ErrorActionPreference = 'Stop'
$before = $env:DOTNET_JitOSR
$global:ObservedFlags = [Collections.Generic.List[string]]::new()
function mockpython {
    $global:ObservedFlags.Add($env:DOTNET_JitOSR)
    if ($env:LOKAD_ENTRYPOINT_CASE -eq 'child-throws' -and $args[0] -eq '-c') { throw 'fixture child failure' }
    $global:LASTEXITCODE = if ($env:LOKAD_ENTRYPOINT_CASE -eq 'native-fails' -and $args[0] -eq '-m') { 7 } else { 0 }
    if ($args[0] -eq '--version') { 'Python fixture' }
}
function dotnet {
    $global:ObservedFlags.Add($env:DOTNET_JitOSR)
    $outputIndex = [Array]::IndexOf($args, '--output')
    if ($outputIndex -ge 0) { '{}' | Set-Content -LiteralPath $args[$outputIndex + 1] }
    $global:LASTEXITCODE = 0
}
function Get-FileHash($Path, $Algorithm) {
    $hash = if ($Path -like '*model.onnx') { '__MODEL__' }
        elseif ($Path -like '*sentencepiece.bpe.model') { '__TOKENIZER__' } else { 'A' * 64 }
    [pscustomobject]@{ Hash = $hash }
}
$caught = $null
try { & (Join-Path $PSScriptRoot 'eng/test-e5.ps1') -Python mockpython; $code = $LASTEXITCODE }
catch { $caught = $_.Exception.Message; $code = 17 }
@{ before=$before; after=$env:DOTNET_JitOSR; code=$code; caught=$caught; flags=@($global:ObservedFlags) } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'state.json')
exit 0
'''.replace('__MODEL__', model_hash).replace('__TOKENIZER__', tokenizer_hash)
            runner = root / 'wrapper.ps1'
            runner.write_text(wrapper, encoding='utf-8')
            env = os.environ.copy()
            env['LOKAD_ENTRYPOINT_CASE'] = scenario
            if initial is None:
                env.pop('DOTNET_JitOSR', None)
            else:
                env['DOTNET_JitOSR'] = initial
            result = subprocess.run([PWSH, '-NoProfile', '-NonInteractive', '-File', str(runner)],
                                    env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads((root / 'state.json').read_text(encoding='utf-8-sig'))

    def check_scenario(self, scenario, code):
        for initial in (None, '1'):
            with self.subTest(initial=initial):
                state = self.run_gate(scenario, initial)
                self.assertEqual(state['code'], code, state)
                self.assertEqual(state['before'], initial)
                self.assertEqual(state['after'], initial, state)
                self.assertTrue(all(flag == '0' for flag in state['flags']), state)
                if scenario == 'success':
                    self.assertGreaterEqual(len(state['flags']), 7)

    def test_missing_assets_restore_environment(self):
        self.check_scenario('missing-assets', 1)

    def test_thrown_child_failure_restores_environment(self):
        self.check_scenario('child-throws', 17)

    def test_native_failure_restores_environment(self):
        self.check_scenario('native-fails', 1)

    def test_success_restores_environment_and_children_get_deterministic_setting(self):
        self.check_scenario('success', 0)


@unittest.skipUnless(PWSH and shutil.which('git'), 'PowerShell 7 and Git are required')
class BenchmarkPreparationTests(unittest.TestCase):
    def run_preparation(self, script_name, choice):
        with tempfile.TemporaryDirectory(prefix='lonnx-preparation-') as directory:
            root = Path(directory).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            eng = root / 'eng'
            eng.mkdir()
            for name in ['build-common-campaign.ps1', 'build-embeddinggemma2-benchmark.ps1']:
                shutil.copyfile(ROOT / 'eng' / name, eng / name)
            core = root / 'src/Lokad.Onnx'
            core.mkdir(parents=True)
            (core / 'Lokad.Onnx.csproj').write_text('<Project />', encoding='utf-8')
            data = root / 'src/Lokad.Onnx.Data'
            data.mkdir()
            (data / 'EmbeddingGemma2Fixture.cs').write_text('// fixture, never compiled', encoding='utf-8')
            (root / 'global.json').write_text('{}', encoding='utf-8')

            def git(*args):
                result = subprocess.run(['git', '-c', 'commit.gpgsign=false', '-c', 'user.name=Fixture',
                                         '-c', 'user.email=fixture@example.invalid', *args], cwd=root,
                                        text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout.strip()

            git('init', '-q')
            git('add', '-f', 'src', 'global.json')
            git('commit', '-qm', 'integrated fixture')
            integrated = git('rev-parse', 'HEAD')
            adapter = root / 'tools/Lokad.Onnx.Protobuf'
            adapter.mkdir(parents=True)
            (adapter / 'Lokad.Onnx.Protobuf.csproj').write_text('<Project />', encoding='utf-8')
            git('add', '-f', 'tools')
            git('commit', '-qm', 'separate fixture')
            separate = git('rev-parse', 'HEAD')
            refs = {
                'missing': (None, None), 'invalid': ('no-such-ref', None),
                'mixed': (integrated, separate), 'integrated': (integrated, integrated),
                'separate': (separate, None)
            }
            baseline, candidate = refs[choice]
            wrapper = r'''
$ErrorActionPreference = 'Stop'
$global:DotnetCalls = [Collections.Generic.List[string]]::new()
function dotnet {
    $global:DotnetCalls.Add(($args -join ' '))
    if ($args[0] -eq '--version') { '10.0.204' }
    else {
        $index = [Array]::IndexOf($args, '-o')
        $destination = $args[$index + 1]
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        'fixture core, never executed' | Set-Content (Join-Path $destination 'Lokad.Onnx.dll')
    }
    $global:LASTEXITCODE = 0
}
$parameters = @{ Output = (Join-Path $PSScriptRoot 'output') }
if ($env:FIXTURE_BASELINE) { $parameters.L0Ref = $env:FIXTURE_BASELINE }
if ($env:FIXTURE_CANDIDATE) { $parameters.L1Ref = $env:FIXTURE_CANDIDATE }
$caught = $null
try { & (Join-Path $PSScriptRoot ('eng/' + $env:FIXTURE_SCRIPT)) @parameters; $code = $LASTEXITCODE }
catch { $caught = $_.Exception.Message; $code = 17 }
@{ code=$code; caught=$caught; outputExists=(Test-Path $parameters.Output); dotnetCalls=@($global:DotnetCalls) } |
    ConvertTo-Json | Set-Content (Join-Path $PSScriptRoot 'state.json')
exit 0
'''
            runner = root / 'wrapper.ps1'
            runner.write_text(wrapper, encoding='utf-8')
            env = os.environ.copy()
            env.update(FIXTURE_SCRIPT=script_name, FIXTURE_BASELINE=baseline or '', FIXTURE_CANDIDATE=candidate or '')
            result = subprocess.run([PWSH, '-NoProfile', '-NonInteractive', '-File', str(runner)],
                                    env=env, text=True, capture_output=True, timeout=25)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((root / 'state.json').read_text(encoding='utf-8-sig'))
            if state['code'] == 0:
                state['layouts'] = [json.loads((root / 'output' / role / 'build.json').read_text(encoding='utf-8-sig'))['importer_layout']
                                    for role in ['L0', 'L1']]
            return state

    def check_rejection(self, choice):
        for script in ['build-common-campaign.ps1', 'build-embeddinggemma2-benchmark.ps1']:
            with self.subTest(script=script):
                state = self.run_preparation(script, choice)
                self.assertNotEqual(state['code'], 0, state)
                self.assertFalse(state['outputExists'], state)
                self.assertEqual(state['dotnetCalls'], [], state)
                if choice == 'missing':
                    self.assertIn('L0Ref', state['caught'])

    def test_baseline_is_required_before_any_work(self):
        self.check_rejection('missing')

    def test_invalid_ref_does_not_create_output_or_start_dotnet(self):
        self.check_rejection('invalid')

    def test_mixed_layout_does_not_create_output_or_start_dotnet(self):
        self.check_rejection('mixed')

    def test_same_era_refs_reach_existing_preparation_paths(self):
        for script in ['build-common-campaign.ps1', 'build-embeddinggemma2-benchmark.ps1']:
            for layout in ['integrated', 'separate']:
                with self.subTest(script=script, layout=layout):
                    state = self.run_preparation(script, layout)
                    self.assertEqual(state['code'], 0, state)
                    self.assertTrue(state['outputExists'])
                    self.assertGreater(len(state['dotnetCalls']), 2)
                    self.assertEqual(state['layouts'], [layout, layout])


if __name__ == '__main__':
    unittest.main()
