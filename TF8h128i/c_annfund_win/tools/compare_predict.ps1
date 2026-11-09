param(
    [Parameter(Mandatory = $true)][string]$Code,
    [Parameter(Mandatory = $true)][string]$LinuxPredictFile,
    [string]$WinExe = ""
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $WinExe) {
    $WinExe = Join-Path $Root "c_annfund_win\build\Release\annfund.exe"
}
if (-not (Test-Path $WinExe)) { throw "missing $WinExe" }
if (-not (Test-Path $LinuxPredictFile)) { throw "missing $LinuxPredictFile" }

$winOut = Join-Path $env:TEMP "pred_win_$Code.txt"
Push-Location $Root
try {
    & $WinExe --predict $Code | Tee-Object -FilePath $winOut
} finally {
    Pop-Location
}

function Load-PredRows([string]$path) {
    $rows = @()
    foreach ($line in Get-Content $path) {
        if ($line -match '^(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)') {
            $rows += ,@([double]$Matches[2], [double]$Matches[3], [double]$Matches[4])
        }
    }
    return $rows
}

$a = Load-PredRows $LinuxPredictFile
$b = Load-PredRows $winOut
if ($a.Count -eq 0 -or $a.Count -ne $b.Count) {
    throw "empty/mismatch linux=$($a.Count) win=$($b.Count)"
}
$rel = 0.0
for ($i = 0; $i -lt $a.Count; $i++) {
    for ($j = 0; $j -lt 3; $j++) {
        $u = $a[$i][$j]
        $v = $b[$i][$j]
        $den = [Math]::Max([Math]::Abs($u), 1e-8)
        $r = [Math]::Abs($u - $v) / $den
        if ($r -gt $rel) { $rel = $r }
    }
}
Write-Host "max_rel $rel"
if ($rel -ge 1e-3) { exit 1 }
exit 0
