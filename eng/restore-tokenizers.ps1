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
Invoke-Checked { dotnet restore $project --locked-mode --tl:off -v minimal }
Invoke-Checked { dotnet pack $project -c Release --no-restore --tl:off --nologo -v minimal -o $feed }
Invoke-Checked { dotnet restore Lokad.Onnx.slnx --source $feed --source 'https://api.nuget.org/v3/index.json' --tl:off -v minimal }
$package = Join-Path $feed 'Lokad.Tokenizers.0.2.0.nupkg'
@{ revision = $revision; version = '0.2.0'; packageSha256 = (Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash.ToLowerInvariant() } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $feed 'tokenizers-receipt.json') -Encoding utf8
Write-Host "PASS qualified Tokenizers package restored from $revision"
