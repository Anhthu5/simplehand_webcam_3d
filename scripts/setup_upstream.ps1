$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
$vendorDir = Join-Path $projectDir 'third_party/simpleHand'
$referenceCommit = 'c3e6abc55512d712c4ae9b2695e529607f73b00f'
if (-not (Test-Path -LiteralPath $vendorDir)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectDir 'third_party') | Out-Null
    git clone https://github.com/patienceFromZhou/simpleHand.git $vendorDir
    if ($LASTEXITCODE -ne 0) { throw 'Git clone failed' }
    git -C $vendorDir checkout --detach $referenceCommit
    if ($LASTEXITCODE -ne 0) { throw 'Could not checkout reference commit' }
} else {
    Write-Host 'Existing repository preserved. Checking its revision.'
}
$actualCommit = git -C $vendorDir rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $referenceCommit) {
    throw 'Unexpected upstream revision. Use a fresh third_party/simpleHand directory.'
}
$manoFile = Join-Path $vendorDir 'models/MANO_RIGHT_C.pkl'
if (-not (Test-Path -LiteralPath $manoFile)) { throw "Missing asset: $manoFile" }
Write-Host "Official source ready at $vendorDir"
Write-Host 'Next: download the trained checkpoint following README.md.'
