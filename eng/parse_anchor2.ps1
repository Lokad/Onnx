# Winner-neutral anchor gate for the L0/L1 campaign lane (E01/E02 M1).
# Usage: pwsh -NoProfile -File eng/parse_anchor2.ps1 <anchor.log> [-SpreadCap 0.30]
# Exit 0 (LAUNCH) only when both Lokad and ORT Mm30Up summaries parse with
# (Max-Min)/Median spread below the cap. Differences from eng/parse_anchor.ps1:
# stats blocks are paired to their benchmark headers instead of assuming
# summary order, and the Lokad/ORT ratio is recorded but NEVER gated, so a
# genuine future Lokad win below 1.0 cannot fail the gate.
param([string]$Log, [double]$SpreadCap = 0.30)
if (-not $Log -or -not (Test-Path $Log)) { Write-Host "ABORT: missing log"; exit 1 }
$med = @{}
$cur = ""
$detailed = $false
foreach ($line in (Get-Content $Log)) {
  if ($line -match "\* Detailed results \*") { $detailed = $true }
  if (-not $detailed) { continue }
  if ($line -match "- (Lokad|ORT) session") { $cur = $Matches[1] }
  $m = [regex]::Match($line, "Min = ([0-9.]+) us, Q1 = [0-9.]+ us, Median = ([0-9.]+) us, Q3 = [0-9.]+ us, Max = ([0-9.]+) us")
  if ($m.Success -and ($cur -eq "Lokad" -or $cur -eq "ORT") -and -not $med.ContainsKey($cur)) {
    $med[$cur] = @{ "min" = [double]$m.Groups[1].Value; "med" = [double]$m.Groups[2].Value; "max" = [double]$m.Groups[3].Value }
  }
}
if (-not $med.ContainsKey("Lokad") -or -not $med.ContainsKey("ORT")) { Write-Host "ABORT: need both Lokad and Ort summaries"; exit 1 }
$lokSpread = ($med["Lokad"]["max"] - $med["Lokad"]["min"]) / $med["Lokad"]["med"]
$ortSpread = ($med["ORT"]["max"] - $med["ORT"]["min"]) / $med["ORT"]["med"]
$ratio = $med["Lokad"]["med"] / $med["ORT"]["med"]
Write-Host ("lokad median=" + $med["Lokad"]["med"] + " spread=" + $lokSpread.ToString("P1") + " ort median=" + $med["ORT"]["med"] + " spread=" + $ortSpread.ToString("P1") + " ratio=" + $ratio.ToString("F3") + " (recorded, never gated)")
if ($lokSpread -ge $SpreadCap) { Write-Host "ABORT: lokad spread too wide"; exit 1 }
if ($ortSpread -ge $SpreadCap) { Write-Host "ABORT: ort spread too wide"; exit 1 }
Write-Host "LAUNCH"
exit 0