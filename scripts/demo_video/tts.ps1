param([string]$Dir)
# Render each narration line to a WAV file with the offline Windows SAPI voice.
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 (Join-Path $Dir "narration.json") | ConvertFrom-Json
$out = Join-Path $Dir "audio"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($it in $items) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice("Microsoft David Desktop")
    $s.Rate = 0
    $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $s.SetOutputToWaveFile((Join-Path $out ($it.id + ".wav")), $fmt)
    $s.Speak($it.tts)
    $s.Dispose()
}
"done"
