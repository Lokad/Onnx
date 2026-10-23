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


if __name__ == '__main__':
    unittest.main()
