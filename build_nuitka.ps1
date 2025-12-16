<# 
Build FMRPreview with Nuitka.
Prereqs (64-bit): Python 3.13 environment with Nuitka (and zstandard for onefile compression),
and the VS 2022 Build Tools (MSVC + Windows 10/11 SDK).

Examples:
  # Folder layout (fast startup)
  ./build_nuitka.ps1

  # Single-file exe, no console window
  ./build_nuitka.ps1 -OneFile

  # Include Data folder contents if present
  ./build_nuitka.ps1 -IncludeDataDir
#>

param(
    [switch]$OneFile = $false,
    [switch]$NoConsole = $true,
    [string]$Python = "python",
    [string]$Config = "FMRPreview.cfg",
    [switch]$IncludeDataDir = $false,
    [string]$DataDir = "Data",
    [switch]$NoClean = $false
)

$cmdBase = @($Python, "-m", "nuitka")

$argsList = @(
    "FMRPreview.py",
    "--standalone",
    "--report=build-report.json",
    "--plugin-enable=pyqt5",
    "--include-qt-plugins=sensible",
    "--include-data-file=$Config=$Config",
    "--assume-yes-for-downloads"
)

if (-not $NoClean) {
    $argsList += "--remove-output"
}

if ($IncludeDataDir -and (Test-Path -LiteralPath $DataDir)) {
    $argsList += "--include-data-dir=$DataDir=$DataDir"
}

if ($OneFile) {
    $argsList += "--onefile"
}

if ($NoConsole) {
    $argsList += "--windows-console-mode=disable"
}

Write-Host "Invoking:" ($cmdBase + $argsList -join " ")

& $Python "-m" "nuitka" @argsList

if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed with exit code $LASTEXITCODE"
}
