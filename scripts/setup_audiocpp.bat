@echo off
setlocal EnableExtensions

rem Clone, build, and install the audio.cpp models used by this project.
rem Usage: setup_audiocpp.bat [cpu^|cuda^|vulkan]

set "backend=%~1"
if not defined backend set "backend=cpu"
if /i not "%backend%"=="cpu" if /i not "%backend%"=="cuda" if /i not "%backend%"=="vulkan" (
    echo Usage: %~nx0 [cpu^|cuda^|vulkan] 1>&2
    exit /b 2
)

where git >nul 2>nul || (echo ERROR: git was not found on PATH. 1>&2 & exit /b 1)
where cmake >nul 2>nul || (echo ERROR: cmake was not found on PATH. 1>&2 & exit /b 1)
where python >nul 2>nul || (echo ERROR: Python was not found on PATH. 1>&2 & exit /b 1)
where powershell.exe >nul 2>nul || (echo ERROR: Windows PowerShell was not found. 1>&2 & exit /b 1)

for %%I in ("%~dp0..") do set "repo_root=%%~fI"
set "audiocpp_dir=%USERPROFILE%\github\audio.cpp"
set "vendor_dir=%repo_root%\vendor\audio.cpp"

if exist "%audiocpp_dir%\." (
    echo "%audiocpp_dir%" already exists, skipping clone.
) else (
    echo Cloning audio.cpp...
    if not exist "%USERPROFILE%\github" mkdir "%USERPROFILE%\github"
    if errorlevel 1 exit /b 1
    git clone https://github.com/0xShug0/audio.cpp.git "%audiocpp_dir%"
    if errorlevel 1 exit /b 1
)

if not exist "%vendor_dir%\." (
    if not exist "%repo_root%\vendor" mkdir "%repo_root%\vendor"
    if errorlevel 1 exit /b 1
    mklink /J "%vendor_dir%" "%audiocpp_dir%"
    if errorlevel 1 exit /b 1
)

if not exist "%vendor_dir%\server.json" copy /Y "%repo_root%\server.json" "%vendor_dir%\server.json" >nul
if errorlevel 1 exit /b 1

if not exist "%audiocpp_dir%\scripts\build_windows.ps1" (
    echo ERROR: audio.cpp's scripts\build_windows.ps1 was not found. Update the audio.cpp clone and retry. 1>&2
    exit /b 1
)

set "AUDIOCPP_SERVER_CONFIG=%vendor_dir%\server.json"
set "AUDIOCPP_BACKEND=%backend%"
python -c "import json, os; p=os.environ['AUDIOCPP_SERVER_CONFIG']; c=json.load(open(p, encoding='utf-8-sig')); c['backend']=os.environ['AUDIOCPP_BACKEND']; json.dump(c, open(p, 'w', encoding='utf-8'), indent=2); print()"
if errorlevel 1 exit /b 1

set "jobs=%NUMBER_OF_PROCESSORS%"
if not defined jobs set "jobs=4"
echo Building audio.cpp for Windows (%backend%)...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%audiocpp_dir%\scripts\build_windows.ps1" -Preset "windows-%backend%-release" -ModelSet custom -Models "chatterbox,qwen3_asr,nemotron_asr,personaplex,qwen3_tts,stable_audio" -Target audiocpp_cli -Jobs %jobs%
if errorlevel 1 exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%audiocpp_dir%\scripts\build_windows.ps1" -Preset "windows-%backend%-release" -ModelSet custom -Models "chatterbox,qwen3_asr,nemotron_asr,personaplex,qwen3_tts,stable_audio" -Target audiocpp_server -Jobs %jobs%
if errorlevel 1 exit /b 1

pushd "%audiocpp_dir%"
if errorlevel 1 exit /b 1
echo Installing models...
for %%M in (chatterbox_q8_0 nemotron_asr_q8_0 qwen3_tts_1_7b_voicedesign_q8_0 stable_audio_3_small_music_q8_0 stable_audio_3_small_sfx_q8_0) do (
    python tools\model_manager_v2.py install %%M
    if errorlevel 1 (
        popd
        exit /b 1
    )
)
popd

echo Done. Start the server from the repository root with:
echo   vendor\audio.cpp\build\windows-%backend%-release\bin\audiocpp_server.exe --config vendor\audio.cpp\server.json
exit /b 0