$ErrorActionPreference = 'Stop'

$accessDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $accessDirectory
$credentialPath = Join-Path $projectRoot 'credentials.env'
$transcriptPath = Join-Path $accessDirectory 'nscc-login-transcript.txt'

$fields = @{}
foreach ($line in [System.IO.File]::ReadAllLines($credentialPath)) {
    if ($line -match '^\s*#' -or $line -notmatch '=') {
        continue
    }
    $parts = $line.Split('=', 2)
    $fields[$parts[0].Trim()] = $parts[1]
}

$nsccUser = $fields['NSCC_USERNAME'].Trim().ToLowerInvariant()
$ntuUser = $nsccUser.ToUpperInvariant()
$nsccHost = $fields['NSCC_HOST'].Trim()

if ($nsccUser -notmatch '^[a-z0-9._-]+$') {
    throw 'The configured NSCC username is invalid.'
}
if ($nsccHost -notmatch '^[A-Za-z0-9.-]+$') {
    throw 'The configured NSCC host is invalid.'
}

$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$jumpKnownHosts = 'C:\Users\Raymond\.ssh\known_hosts_ntu_jump'
$inspection = "printf '\n=== LOGIN NODE ===\n'; hostname; date; uname -srmo; " +
    "printf '\n=== PROJECT ALLOCATION ===\n'; myprojects; " +
    "printf '\n=== PERSONAL USAGE ===\n'; myusage; " +
    "printf '\n=== STORAGE QUOTA ===\n'; myquota; " +
    "printf '\n=== PBS QUEUES ===\n'; qstat -Q; " +
    "printf '\n=== SOFTWARE MODULE SAMPLE ===\n'; module avail 2>&1 | head -80; " +
    "printf '\n=== READY ===\n'; exec bash -l"
$onwardCommand = "ssh -tt -o StrictHostKeyChecking=accept-new ${nsccUser}@${nsccHost} '$inspection'"

Clear-Host
Write-Host 'NSCC ASPIRE2A+ through the NTU Jump Host' -ForegroundColor Cyan
Write-Host ''
Write-Host 'Prompt 1: enter your NTU organizational/SSO password.' -ForegroundColor Yellow
Write-Host 'Prompt 2: enter your NSCC password.' -ForegroundColor Yellow
Write-Host 'Passwords remain invisible and are not written to the transcript.'
Write-Host 'After the inspection completes, leave the window open and return to Codex.'
Write-Host ''

Start-Transcript -LiteralPath $transcriptPath -Force | Out-Null
try {
    & $ssh -tt `
        -o ConnectTimeout=20 `
        -o PubkeyAuthentication=no `
        -o PreferredAuthentications=keyboard-interactive `
        -o StrictHostKeyChecking=yes `
        -o "UserKnownHostsFile=$jumpKnownHosts" `
        -o KexAlgorithms=ecdh-sha2-nistp256 `
        -c aes256-ctr `
        -m hmac-sha2-256 `
        "${ntuUser}@172.21.26.100" `
        $onwardCommand
}
finally {
    Stop-Transcript | Out-Null
    Write-Host ''
    Write-Host "Session ended. Transcript: $transcriptPath"
    Read-Host 'Press Enter to close'
}
