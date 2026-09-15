# Restore git-ignored local model assets from Hugging Face (recovery lane).
# Source of truth: tests/Lokad.Onnx.Backend.Tests/ModelManifest.json
# (repo, path, revision, sha256, bytes per asset). Verifies every file after
# placement and fails on any mismatch. Cache-friendly: repeat runs reuse the
# HF hub cache, so only missing blobs download.
# Run: pwsh -NoProfile -File eng/restore-models.ps1
param([string]$Manifest = "tests/Lokad.Onnx.Backend.Tests/ModelManifest.json")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Fail($msg) { Write-Host "FAIL restore-models: $msg"; exit 1 }
if (!(Get-Command hf -ErrorAction SilentlyContinue)) { Fail "hf CLI not found (see hf-cli skill)" }
$man = Get-Content $Manifest -Raw | ConvertFrom-Json
$stage = Join-Path $root "artifacts/model-stage"
New-Item -ItemType Directory -Force $stage | Out-Null
foreach ($m in $man.models) {
  $dest = Join-Path $root $m.file
  $need = $true
  if (Test-Path $dest) {
    $h = (Get-FileHash $dest -Algorithm SHA256).Hash
    $s = (Get-Item $dest).Length
    if ($h -eq $m.sha256 -and $s -eq $m.bytes) { Write-Host ("OK cached: " + $m.file); $need = $false }
    else { Write-Host ("STALE: " + $m.file + " hash/bytes differ, re-fetching") }
  }
  if ($need) {
    $repo = $m.source.repo
    $inc = $m.source.path
    $rev = $m.source.revision
    $args = @("download", $repo, "--include", $inc, "--local-dir", $stage)
    if ($rev) { $args += @("--revision", $rev) }
    Write-Host ("FETCH " + $repo + "@" + ($rev ? $rev : "main") + " :: " + $inc)
    & hf @args
    if ($LASTEXITCODE -ne 0) { Fail ("hf download failed for " + $m.file) }
    $src = Join-Path $stage ($inc -replace "/", [System.IO.Path]::DirectorySeparatorChar)
    if (!(Test-Path $src)) { Fail ("downloaded file not found: " + $src) }
    New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
    Copy-Item $src $dest -Force
    $h = (Get-FileHash $dest -Algorithm SHA256).Hash
    $s = (Get-Item $dest).Length
    if ($h -ne $m.sha256) { Fail ("sha mismatch after fetch: " + $m.file) }
    if ($s -ne $m.bytes) { Fail ("byte mismatch after fetch: " + $m.file) }
    Write-Host ("PASS " + $m.file)
  }
}
Write-Host "PASS restore-models: all assets verified"