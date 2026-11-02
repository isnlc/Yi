param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [int]$MaxPages = 20
)

$ErrorActionPreference = "Stop"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

Add-Type -AssemblyName System.Runtime.WindowsRuntime

$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime]

function Await-Operation {
    param(
        [Parameter(Mandatory = $true)]$Operation,
        [Parameter(Mandatory = $true)][Type]$ResultType
    )

    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object {
            $_.Name -eq "AsTask" -and
            $_.IsGenericMethod -and
            $_.GetParameters().Count -eq 1
        } |
        Select-Object -First 1
    $task = $method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

function Await-Action {
    param([Parameter(Mandatory = $true)]$Action)

    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object {
            $_.Name -eq "AsTask" -and
            -not $_.IsGenericMethod -and
            $_.GetParameters().Count -eq 1
        } |
        Select-Object -First 1
    $task = $method.Invoke($null, @($Action))
    $task.Wait()
}

function Read-OcrFromStream {
    param(
        [Parameter(Mandatory = $true)]$Stream,
        [Parameter(Mandatory = $true)]$Engine
    )

    $decoder = Await-Operation `
        ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($Stream)) `
        ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await-Operation `
        ($decoder.GetSoftwareBitmapAsync()) `
        ([Windows.Graphics.Imaging.SoftwareBitmap])
    try {
        $result = Await-Operation `
            ($Engine.RecognizeAsync($bitmap)) `
            ([Windows.Media.Ocr.OcrResult])
        return $result.Text
    }
    finally {
        if ($null -ne $bitmap) {
            $bitmap.Dispose()
        }
    }
}

try {
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    if (-not [System.IO.File]::Exists($resolvedPath)) {
        throw "文件不存在。"
    }

    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    if ($null -eq $engine) {
        throw "Windows OCR 引擎不可用。"
    }

    $file = Await-Operation `
        ([Windows.Storage.StorageFile]::GetFileFromPathAsync($resolvedPath)) `
        ([Windows.Storage.StorageFile])
    $extension = [System.IO.Path]::GetExtension($resolvedPath).ToLowerInvariant()
    $sections = New-Object System.Collections.Generic.List[object]

    if ($extension -eq ".pdf") {
        $document = Await-Operation `
            ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) `
            ([Windows.Data.Pdf.PdfDocument])
        $pageCount = [Math]::Min([int]$document.PageCount, [Math]::Max(1, $MaxPages))

        for ($index = 0; $index -lt $pageCount; $index++) {
            $page = $document.GetPage($index)
            $stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
            try {
                Await-Action ($page.RenderToStreamAsync($stream))
                $stream.Seek(0)
                $text = Read-OcrFromStream -Stream $stream -Engine $engine
                if (-not [string]::IsNullOrWhiteSpace($text)) {
                    $sections.Add([pscustomobject]@{
                        label = "第 $($index + 1) 页"
                        text = $text.Trim()
                    })
                }
            }
            finally {
                $stream.Dispose()
                $page.Dispose()
            }
        }
    }
    else {
        $stream = Await-Operation `
            ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) `
            ([Windows.Storage.Streams.IRandomAccessStream])
        try {
            $text = Read-OcrFromStream -Stream $stream -Engine $engine
            if (-not [string]::IsNullOrWhiteSpace($text)) {
                $sections.Add([pscustomobject]@{
                    label = "图片"
                    text = $text.Trim()
                })
            }
        }
        finally {
            $stream.Dispose()
        }
    }

    [pscustomobject]@{
        ok = $true
        sections = $sections
    } | ConvertTo-Json -Depth 4 -Compress
}
catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
