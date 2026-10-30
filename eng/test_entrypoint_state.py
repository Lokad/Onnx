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
    def run_gate(self, scenario, initial, failure=None, require_intrinsics=False):
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
$global:ModeCalls = @{}
$global:ReferenceRan = $false
function mockpython {
    $global:ObservedFlags.Add($env:DOTNET_JitOSR)
    if ($args[0] -eq '-m') { $global:ReferenceRan = $true }
    if ($env:LOKAD_ENTRYPOINT_CASE -eq 'child-throws' -and $args[0] -eq '-c') { throw 'fixture child failure' }
    $global:LASTEXITCODE = if ($env:LOKAD_ENTRYPOINT_CASE -eq 'native-fails' -and $args[0] -eq '-m') { 7 } else { 0 }
    if ($args[0] -eq '--version') { 'Python fixture' }
}
function dotnet {
    $global:ObservedFlags.Add($env:DOTNET_JitOSR)
    $outputIndex = [Array]::IndexOf($args, '--output')
    if ($outputIndex -ge 0) { '{}' | Set-Content -LiteralPath $args[$outputIndex + 1] }
    $global:LASTEXITCODE = 0
    $modeIndex = [Array]::IndexOf($args, '--mode')
    if ($modeIndex -ge 0) {
        $mode = $args[$modeIndex + 1]
        $global:ModeCalls[$mode] = [int]$global:ModeCalls[$mode] + 1
        if ($mode -eq $env:LOKAD_ENTRYPOINT_FAIL_MODE -and $global:ModeCalls[$mode] -eq [int]$env:LOKAD_ENTRYPOINT_FAIL_ATTEMPT) {
            $global:LASTEXITCODE = [int]$env:LOKAD_ENTRYPOINT_FAIL_EXIT
        }
    }
}
function Get-FileHash($Path, $Algorithm) {
    $hash = if ($Path -like '*model.onnx') { '__MODEL__' }
        elseif ($Path -like '*sentencepiece.bpe.model') { '__TOKENIZER__' } else { 'A' * 64 }
    [pscustomobject]@{ Hash = $hash }
}
$caught = $null
try {
    & (Join-Path $PSScriptRoot 'eng/test-e5.ps1') -Python mockpython -RequireIntrinsics:($env:LOKAD_ENTRYPOINT_REQUIRE -eq '1')
    $code = $LASTEXITCODE
}
catch { $caught = $_.Exception.Message; $code = 17 }
$artifactPath = Join-Path $PSScriptRoot 'artifacts/e5/dotnet-intrinsics.json'
$artifact = if (Test-Path -LiteralPath $artifactPath) { Get-Content -LiteralPath $artifactPath -Raw | ConvertFrom-Json } else { $null }
@{ before=$before; after=$env:DOTNET_JitOSR; code=$code; caught=$caught; flags=@($global:ObservedFlags)
   intrinsics=$artifact; referenceRan=$global:ReferenceRan; modeCalls=$global:ModeCalls } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'state.json')
exit 0
'''.replace('__MODEL__', model_hash).replace('__TOKENIZER__', tokenizer_hash)
            runner = root / 'wrapper.ps1'
            runner.write_text(wrapper, encoding='utf-8')
            env = os.environ.copy()
            env['LOKAD_ENTRYPOINT_CASE'] = scenario
            mode, attempt, exit_code = failure or ('', 0, 0)
            env.update(LOKAD_ENTRYPOINT_FAIL_MODE=mode, LOKAD_ENTRYPOINT_FAIL_ATTEMPT=str(attempt),
                       LOKAD_ENTRYPOINT_FAIL_EXIT=str(exit_code), LOKAD_ENTRYPOINT_REQUIRE='1' if require_intrinsics else '0')
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

    def check_runner_exit(self, mode, attempt, exit_code, required, expected):
        for initial in (None, '1'):
            with self.subTest(mode=mode, attempt=attempt, exit_code=exit_code, required=required, initial=initial):
                state = self.run_gate('success', initial, (mode, attempt, exit_code), required)
                self.assertEqual(state['code'], expected, state)
                self.assertEqual(state['before'], initial)
                self.assertEqual(state['after'], initial, state)
                self.assertTrue(all(flag == '0' for flag in state['flags']), state)
                skipped = mode == 'intrinsics' and attempt == 1 and exit_code == 3 and not required
                self.assertEqual(bool((state['intrinsics'] or {}).get('unsupported')), skipped, state)
                self.assertEqual(state['referenceRan'], expected == 0, state)
                if skipped:
                    self.assertEqual(state['modeCalls']['intrinsics'], 1)

    def test_optional_intrinsics_skips_only_explicit_unavailable_exit(self):
        for code in (0, 3, 4, 17):
            self.check_runner_exit('intrinsics', 1, code, False, 0 if code in (0, 3) else 1)

    def test_required_intrinsics_rejects_first_run_failures(self):
        for code in (3, 4, 17):
            self.check_runner_exit('intrinsics', 1, code, True, 1)

    def test_repeat_failures_never_become_hardware_skips(self):
        for code in (3, 4, 17):
            self.check_runner_exit('intrinsics', 2, code, False, 1)

    def test_other_modes_never_skip_failure_exits(self):
        for mode in ('scalar', 'simd'):
            for code in (3, 4):
                self.check_runner_exit(mode, 1, code, False, 1)


@unittest.skipUnless(PWSH and shutil.which('git'), 'PowerShell 7 and Git are required')
class BenchmarkPreparationTests(unittest.TestCase):
    def run_preparation(self, script_name, choice, framework='net11.0'):
        with tempfile.TemporaryDirectory(prefix='lonnx-preparation-') as directory:
            root = Path(directory).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            eng = root / 'eng'
            eng.mkdir()
            for name in ['build-common-campaign.ps1', 'build-embeddinggemma2-benchmark.ps1']:
                shutil.copyfile(ROOT / 'eng' / name, eng / name)
            core = root / 'src/Lokad.Onnx'
            core.mkdir(parents=True)
            (core / 'Lokad.Onnx.csproj').write_text('<Project><PropertyGroup><TargetFrameworks>net10.0;net11.0</TargetFrameworks></PropertyGroup></Project>', encoding='utf-8')
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
            (core / 'Lokad.Onnx.csproj').write_text('<Project><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup></Project>', encoding='utf-8')
            git('add', '-f', 'src', 'global.json')
            git('commit', '-qm', 'net10 fixture')
            net10 = git('rev-parse', 'HEAD')
            (core / 'Lokad.Onnx.csproj').write_text('<Project><PropertyGroup><TargetFrameworks>net10.0;net11.0</TargetFrameworks></PropertyGroup></Project>', encoding='utf-8')
            git('add', '-f', 'src')
            git('commit', '-qm', 'integrated fixture')
            integrated = git('rev-parse', 'HEAD')
            (core / 'Fixture.cs').write_text('// candidate fixture, never compiled', encoding='utf-8')
            git('add', '-f', 'src')
            git('commit', '-qm', 'candidate fixture')
            candidate_head = git('rev-parse', 'HEAD')
            refs = {
                'missing': (None, None), 'invalid': ('no-such-ref', None),
                'integrated': (integrated, integrated),
                'default-candidate': (integrated, None), 'unsupported': (net10, None)
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
$parameters = @{ Output = (Join-Path $PSScriptRoot 'output'); Framework = $env:FIXTURE_FRAMEWORK }
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
            env.update(FIXTURE_SCRIPT=script_name, FIXTURE_BASELINE=baseline or '', FIXTURE_CANDIDATE=candidate or '', FIXTURE_FRAMEWORK=framework)
            result = subprocess.run([PWSH, '-NoProfile', '-NonInteractive', '-File', str(runner)],
                                    env=env, text=True, capture_output=True, timeout=25)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((root / 'state.json').read_text(encoding='utf-8-sig'))
            if state['code'] == 0:
                state['builds'] = [json.loads((root / 'output' / role / 'build.json').read_text(encoding='utf-8-sig'))
                                   for role in ['L0', 'L1']]
                state['expected_sources'] = [baseline, candidate or candidate_head]
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

    def test_unsupported_framework_does_not_create_output_or_start_dotnet(self):
        self.check_rejection('unsupported')

    def test_net10_selection_reaches_all_builds(self):
        for script in ['build-common-campaign.ps1', 'build-embeddinggemma2-benchmark.ps1']:
            with self.subTest(script=script):
                state = self.run_preparation(script, 'integrated', framework='net10.0')
                self.assertEqual(state['code'], 0, state)
                for command in state['dotnetCalls']:
                    if command.startswith('build '):
                        self.assertIn('-f net10.0', command)

    def test_pinned_and_default_candidate_refs_reach_preparation_paths(self):
        for script in ['build-common-campaign.ps1', 'build-embeddinggemma2-benchmark.ps1']:
            for choice in ['integrated', 'default-candidate']:
                with self.subTest(script=script, choice=choice):
                    state = self.run_preparation(script, choice)
                    self.assertEqual(state['code'], 0, state)
                    self.assertTrue(state['outputExists'])
                    self.assertGreater(len(state['dotnetCalls']), 2)
                    self.assertEqual([b['source_sha'] for b in state['builds']], state['expected_sources'])
                    self.assertEqual([b['target_framework'] for b in state['builds']], ['net11.0', 'net11.0'])
                    self.assertTrue(all('Lokad.Onnx.dll' in b['runtime_files'] for b in state['builds']))


if __name__ == '__main__':
    unittest.main()
