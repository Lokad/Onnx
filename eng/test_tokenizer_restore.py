"""Exercise the qualified-content guard against real, isolated NuGet restores."""
import hashlib
import html
import json
import pathlib
import subprocess
import tempfile
import unittest
import uuid
import zipfile


REPO = pathlib.Path(__file__).resolve().parent.parent
GUARD = REPO / 'eng/verify-tokenizers.ps1'
ASSEMBLY = 'lib/net10.0/Lokad.Tokenizers.dll'


class TokenizerRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='onnx-tokenizer-restore-')
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.project = self.root / 'Consumer.csproj'
        self.project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup><ItemGroup><PackageReference Include="Lokad.Tokenizers" Version="[0.2.0]" /></ItemGroup></Project>')
        self.cache = self.root / 'packages'
        self.assets = self.root / 'obj/project.assets.json'

    def package(self, feed, assembly=b'qualified assembly', description='qualified manifest'):
        folder = self.root / feed
        folder.mkdir(exist_ok=True)
        path = folder / 'Lokad.Tokenizers.0.2.0.nupkg'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('Lokad.Tokenizers.nuspec', f'<package><metadata><id>Lokad.Tokenizers</id><version>0.2.0</version><authors>Test</authors><description>{description}</description></metadata></package>')
            archive.writestr(ASSEMBLY, assembly)
            archive.writestr(f'package/services/metadata/core-properties/{uuid.uuid4().hex}.psmdcp', 'generated archive metadata')
        return path

    def restore(self, package):
        config = self.root / 'nuget.config'
        config.write_text(f'<configuration><packageSources><clear/><add key="qualified" value="{html.escape(package.parent.as_posix(), quote=True)}"/></packageSources></configuration>')
        command = ['dotnet', 'restore', str(self.project), '--configfile', str(config), '--packages', str(self.cache),
                   '--force-evaluate', '--tl:off', '-v', 'minimal', '-p:NuGetAudit=false']
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def verify(self, package):
        return subprocess.run(['pwsh', '-NoProfile', '-File', str(GUARD), '-PackagePath', str(package), '-AssetsPath', str(self.assets)],
                              capture_output=True, text=True, timeout=30)

    def test_rejects_different_same_version_cache_after_qualified_only_restore(self):
        old = self.package('old', assembly=b'unqualified assembly')
        qualified = self.package('qualified')
        self.restore(old)
        self.restore(qualified)
        cached = self.cache / 'lokad.tokenizers/0.2.0' / ASSEMBLY
        self.assertEqual(b'unqualified assembly', cached.read_bytes())
        result = self.verify(qualified)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('content differs', result.stderr)
        self.assertIn('NUGET_PACKAGES', result.stderr)
        self.assertEqual(b'unqualified assembly', cached.read_bytes())

    def test_matching_restore_and_repeat_restore_report_consumed_identity(self):
        qualified = self.package('qualified')
        for _ in range(2):
            self.restore(qualified)
            result = self.verify(qualified)
            self.assertEqual(0, result.returncode, result.stderr)
            receipt = json.loads(result.stdout)
            self.assertEqual(hashlib.sha256(b'qualified assembly').hexdigest(), receipt['assemblySha256'])
            self.assertEqual((self.cache / 'lokad.tokenizers/0.2.0').resolve(), pathlib.Path(receipt['resolvedPackageDirectory']))

    def test_repacking_metadata_can_change_without_invalidating_content(self):
        first = self.package('first')
        second = self.package('second')
        self.assertNotEqual(first.read_bytes(), second.read_bytes())
        self.restore(first)
        self.restore(second)
        result = self.verify(second)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(hashlib.sha256(second.read_bytes()).hexdigest(), json.loads(result.stdout)['packageSha256'])

    def test_same_assembly_with_another_dependency_manifest_is_rejected(self):
        old = self.package('old', description='another dependency manifest')
        qualified = self.package('qualified')
        self.restore(old)
        self.restore(qualified)
        result = self.verify(qualified)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('content differs', result.stderr)
        self.assertIn('nuspec', result.stderr)

    def test_checks_extracted_assembly_instead_of_only_cached_archive(self):
        qualified = self.package('qualified')
        self.restore(qualified)
        (self.cache / 'lokad.tokenizers/0.2.0' / ASSEMBLY).write_bytes(b'changed extracted assembly')
        result = self.verify(qualified)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('content differs', result.stderr)

    def test_missing_resolved_dependency_is_an_explicit_failure(self):
        qualified = self.package('qualified')
        self.restore(qualified)
        assets = json.loads(self.assets.read_text())
        assets['libraries'].pop('Lokad.Tokenizers/0.2.0')
        self.assets.write_text(json.dumps(assets))
        result = self.verify(qualified)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('must resolve Lokad.Tokenizers', result.stderr)


if __name__ == '__main__':
    unittest.main()
