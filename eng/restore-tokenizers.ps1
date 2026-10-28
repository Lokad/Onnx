param([string]$SourceDirectory)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$revision = '912c4c923376928bcc3d7ba28734b2d2c9af88c3'
function Invoke-Checked([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Tokenizer dependency step failed ($LASTEXITCODE)" }
}
if (!$SourceDirectory) {
    $sibling = [IO.Path]::GetFullPath((Join-Path $root '../Tokenizers'))
    if (Test-Path -LiteralPath (Join-Path $sibling '.git')) {
        $head = (git -C $sibling rev-parse HEAD).Trim()
        if ($LASTEXITCODE -eq 0 -and $head -eq $revision) { $SourceDirectory = $sibling }
    }
}
if (!$SourceDirectory) {
    $SourceDirectory = Join-Path $root "artifacts/dependencies/Tokenizers-$revision"
    if (!(Test-Path -LiteralPath $SourceDirectory)) {
        Invoke-Checked { git clone --no-checkout 'https://github.com/Lokad/Tokenizers.git' $SourceDirectory }
        Invoke-Checked { git -C $SourceDirectory checkout --detach $revision }
    }
}
$SourceDirectory = [IO.Path]::GetFullPath($SourceDirectory)
$head = (git -C $SourceDirectory rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $revision) { throw "Expected Tokenizers commit $revision at $SourceDirectory" }
$changes = @(git -C $SourceDirectory status --porcelain --untracked-files=normal)
if ($LASTEXITCODE -ne 0 -or $changes.Count -gt 0) { throw 'Build the qualified Tokenizers commit from a clean checkout.' }
$project = Join-Path $SourceDirectory 'src/Lokad.Tokenizers.csproj'
$feed = Join-Path $root 'artifacts/dependencies/nuget'
$receipt = Join-Path $feed 'tokenizers-receipt.json'
if (Test-Path -LiteralPath $receipt) { Remove-Item -LiteralPath $receipt }
# SDK selection follows the working directory, even for an absolute project
# path. Respect the dependency's global.json when our root SDK changes.
Push-Location $SourceDirectory
try {
    Invoke-Checked { dotnet restore $project --locked-mode --tl:off -v minimal }
    Invoke-Checked { dotnet pack $project -c Release --no-restore --tl:off --nologo -v minimal -o $feed }
} finally { Pop-Location }
$configuration = Join-Path $root 'artifacts/dependencies/nuget.config'
$escapedFeed = [Security.SecurityElement]::Escape($feed)
@"
<configuration>
  <packageSources>
    <clear />
    <add key="qualified-tokenizers" value="$escapedFeed" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
"@ | Set-Content -LiteralPath $configuration -Encoding utf8
Invoke-Checked { dotnet restore Lokad.Onnx.slnx --configfile $configuration --tl:off -v minimal }
$package = Join-Path $feed 'Lokad.Tokenizers.0.2.0.nupkg'
$verified = & (Join-Path $PSScriptRoot 'verify-tokenizers.ps1') -PackagePath $package -AssetsPath (Join-Path $root 'src/Lokad.Onnx.Data/obj/project.assets.json') |
    ConvertFrom-Json
@{ revision = $revision; version = '0.2.0'; packageSha256 = $verified.packageSha256
    resolvedPackageDirectory = $verified.resolvedPackageDirectory
    assemblySha256 = $verified.assemblySha256; manifestSha256 = $verified.manifestSha256 } |
    ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding utf8
Write-Host "PASS qualified Tokenizers assembly and manifest restored from $revision"
