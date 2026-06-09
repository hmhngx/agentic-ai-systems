# Install repo git hooks from .githooks/ (run once per clone).
$ErrorActionPreference = "Stop"

$repoRoot = git rev-parse --show-toplevel
if (-not $repoRoot) {
    Write-Error "Not inside a git repository."
}

$hooksDir = Join-Path $repoRoot ".git\hooks"
$sourceDir = Join-Path $repoRoot ".githooks"

if (-not (Test-Path $sourceDir)) {
    Write-Error "Missing .githooks directory at $sourceDir"
}

Get-ChildItem $sourceDir -File | ForEach-Object {
    $dest = Join-Path $hooksDir $_.Name
    Copy-Item $_.FullName $dest -Force
    Write-Host "Installed $($_.Name) -> $dest"
}

Write-Host "Git hooks installed. docs/ is blocked at commit and push time."
