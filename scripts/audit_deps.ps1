# Audits the pinned requirements (feature 3, Section N #104). Run from anywhere:
#
#     powershell -ExecutionPolicy Bypass -File scripts\audit_deps.ps1            # audit
#     powershell -ExecutionPolicy Bypass -File scripts\audit_deps.ps1 -Install   # first time: set up pip-audit
#
# Three checks, each its own line in the result; exit code 1 if any fails:
#   1. requirements.txt is exactly what .local/venv has (scripts/pin_requirements.py --check), so what is
#      audited is what the tests ran on.
#   2. pip-audit against two advisory sources, PyPI's and OSV's, for every pin. The file pins every package
#      the app installs, so --no-deps --disable-pip audits it as written, without installing anything.
#   3. No pin is yanked or younger than 7 days (scripts/pin_requirements.py --releases).
#
# pip-audit lives in its own venv, .local/tools/audit-venv, so it never adds packages to the app's venv,
# and its cache is under .local/ too (nothing on C:). A finding that can't be fixed yet goes in $Explained
# below with the reason, and is then passed to pip-audit as --ignore-vuln; nothing is ignored without one.
param([switch]$Install)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$AppPython = Join-Path $Root ".local\venv\Scripts\python.exe"
$AuditVenv = Join-Path $Root ".local\tools\audit-venv"
$AuditPython = Join-Path $AuditVenv "Scripts\python.exe"
$Cache = Join-Path $Root ".local\tools\pip-audit-cache"
$Requirements = Join-Path $Root "requirements.txt"
$PipAuditVersion = "2.10.1"

# Advisory id -> why it's accepted for now (and what would end it). Empty: every finding is fixed (N §4).
$Explained = @{
}

# pip's and pip-audit's temporary files go to .local/tmp, not C:
$Tmp = Join-Path $Root ".local\tmp"
New-Item -ItemType Directory -Force $Tmp | Out-Null
$env:TMP = $Tmp; $env:TEMP = $Tmp; $env:PIP_NO_CACHE_DIR = "1"

if (-not (Test-Path $AppPython)) { Write-Host "[audit] no .local\venv: create it and install requirements.txt first"; exit 2 }
if (-not (Test-Path $AuditPython)) {
    if (-not $Install) { Write-Host "[audit] pip-audit isn't set up: run this script once with -Install"; exit 2 }
    & $AppPython -m venv $AuditVenv
    & $AuditPython -m pip install --no-cache-dir --quiet "pip-audit==$PipAuditVersion"
    if ($LASTEXITCODE -ne 0) { Write-Host "[audit] installing pip-audit failed"; exit 2 }
}

$failed = @()
Push-Location $Root
try {
    Write-Host "[audit] 1/3 requirements.txt against .local\venv"
    & $AppPython scripts\pin_requirements.py --check
    if ($LASTEXITCODE -ne 0) { $failed += "pins differ from .local\venv" }

    $ignore = @()
    foreach ($id in $Explained.Keys) { $ignore += @("--ignore-vuln", $id); Write-Host "[audit] accepted for now: $id ($($Explained[$id]))" }
    foreach ($service in @("pypi", "osv")) {
        Write-Host "[audit] 2/3 pip-audit, $service advisories"
        & $AuditPython -m pip_audit -r $Requirements --no-deps --disable-pip --cache-dir $Cache --progress-spinner off -s $service @ignore
        if ($LASTEXITCODE -ne 0) { $failed += "pip-audit ($service) found vulnerabilities, or couldn't run" }
    }

    Write-Host "[audit] 3/3 yanked or brand-new pins"
    & $AppPython scripts\pin_requirements.py --releases
    if ($LASTEXITCODE -ne 0) { $failed += "a pin is yanked or under the 7-day wait" }
}
finally { Pop-Location }

if ($failed.Count) {
    Write-Host "[audit] FAILED: $($failed -join '; ')"
    exit 1
}
Write-Host "[audit] OK: pins match the venv, no known vulnerabilities (PyPI, OSV), none yanked or brand-new."
exit 0
