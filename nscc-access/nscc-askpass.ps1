$ErrorActionPreference = 'Stop'

$credentialPath = $env:NSCC_CREDENTIAL_FILE
if ([string]::IsNullOrWhiteSpace($credentialPath)) {
    throw 'NSCC_CREDENTIAL_FILE is not set.'
}

$promptText = ($args -join ' ')
$passwordField = if ($promptText -match '172\.21\.26\.100|NTU-JumpHost') {
    'NTU_PASSWORD'
} elseif ($promptText -match 'aspire2antu\.nscc\.sg|ASPIRE2A') {
    'NSCC_PASSWORD'
} else {
    throw 'AskPass refused an unrecognised SSH host prompt.'
}

$passwordLine = [System.IO.File]::ReadLines($credentialPath) |
    Where-Object { $_ -like "$passwordField=*" } |
    Select-Object -First 1

if ($null -eq $passwordLine) {
    throw "$passwordField is missing."
}

[Console]::Out.Write($passwordLine.Substring($passwordField.Length + 1))
