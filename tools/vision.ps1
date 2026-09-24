<#
.SYNOPSIS
    Alle wiederkehrenden Handgriffe von SADA Vision - ueber Docker.

.DESCRIPTION
    Auf den Arbeitsrechnern gibt es kein Python. Jeder Befehl laeuft deshalb
    in einem Container, und die Aufrufe sind lang genug, dass man sie
    falsch tippt. Dieses Skript ist die kurze Fassung.

        .\tools\vision.ps1 build          Dienst-Image bauen
        .\tools\vision.ps1 test           Tests und ruff
        .\tools\vision.ps1 serve          Dienst auf 127.0.0.1:8080
        .\tools\vision.ps1 stop           Dienst anhalten
        .\tools\vision.ps1 dataset        CrackSeg9k holen und entpacken
        .\tools\vision.ps1 train          Training starten (im Hintergrund)
        .\tools\vision.ps1 watch          Trainingsfortschritt ansehen
        .\tools\vision.ps1 evaluate       Bestes Modell bewerten
        .\tools\vision.ps1 export         Nach ONNX ausliefern
        .\tools\vision.ps1 probe <datei>  Ein Foto durchschicken, Vorschau holen

.NOTES
    --shm-size=2g beim Training ist Pflicht: Dockers Vorgabe von 64 MB
    reicht den DataLoader-Arbeitern nicht, und der Lauf stirbt mitten drin
    mit "Bus error".
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('build', 'test', 'serve', 'stop', 'dataset', 'train', 'watch',
                 'evaluate', 'export', 'probe', 'help')]
    [string]$Command = 'help',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Run = "runs/crack_unet_r18"
$Config = "training/configs/crack_unet_r18.yaml"

function Invoke-Service {
    param([string[]]$Arguments)
    docker run --rm -v "${Root}:/app" sada-vision:test @Arguments
}

function Invoke-Training {
    param([string[]]$Arguments, [switch]$Detached, [string]$Name)
    $common = @('--shm-size=2g', '-v', "${Root}:/work", '-w', '/work')
    if ($Detached) {
        docker run -d --name $Name @common sada-vision:train @Arguments
    } else {
        docker run --rm @common sada-vision:train @Arguments
    }
}

switch ($Command) {
    'build' {
        docker build -f deploy/Dockerfile -t sada-vision:dev $Root
        docker build -f deploy/Dockerfile --target dev -t sada-vision:test $Root
        docker build -f deploy/Dockerfile.train -t sada-vision:train $Root
    }

    'test' {
        Invoke-Service -Arguments @('ruff', 'check', 'src', 'tests', 'training')
        Invoke-Service -Arguments @('pytest', '-q')
    }

    'serve' {
        docker rm -f sada-vision 2>$null | Out-Null
        docker run -d --name sada-vision `
            -p 127.0.0.1:8080:8080 `
            -v "${Root}\models:/models:ro" `
            sada-vision:dev
        Write-Host "Dienst laeuft: http://127.0.0.1:8080/docs"
        Write-Host "Markervorlage: http://127.0.0.1:8080/tools/marker.png?size_mm=60"
    }

    'stop' { docker rm -f sada-vision }

    'dataset' {
        Invoke-Training -Arguments @('python', 'training/datasets/crackseg9k.py',
                                     '--out', 'data/crackseg9k')
    }

    'train' {
        docker rm -f sada-vision-train 2>$null | Out-Null
        $argv = @('python', 'training/train.py', '--config', $Config) + $Rest
        Invoke-Training -Detached -Name 'sada-vision-train' -Arguments $argv
        Write-Host "Laeuft im Hintergrund. Fortschritt: .\tools\vision.ps1 watch"
    }

    'watch' { docker logs -f --tail 40 sada-vision-train }

    'evaluate' {
        $argv = @('python', 'training/evaluate.py', "$Run/best.pt",
                  '--data', 'data/crackseg9k/test') + $Rest
        Invoke-Training -Arguments $argv
    }

    'export' {
        $argv = @('python', 'training/export_onnx.py', "$Run/best.pt",
                  '--out', 'models/crack_unet_r18.onnx') + $Rest
        Invoke-Training -Arguments $argv
    }

    'probe' {
        if (-not $Rest -or -not (Test-Path $Rest[0])) {
            Write-Error "Bitte eine Bilddatei angeben: .\tools\vision.ps1 probe foto.jpg"
            exit 1
        }
        $image = (Resolve-Path $Rest[0]).Path
        $extra = if ($Rest.Count -gt 1) { $Rest[1..($Rest.Count - 1)] } else { @() }
        $form = @('-F', "image=@$image") + ($extra | ForEach-Object { '-F'; $_ })

        curl.exe -s @form http://127.0.0.1:8080/api/v1/detect/crack -o befund.json
        curl.exe -s @form http://127.0.0.1:8080/api/v1/preview/crack -o vorschau.png
        Write-Host "Geschrieben: befund.json, vorschau.png"
    }

    default { Get-Help $PSCommandPath -Detailed }
}
