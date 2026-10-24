@echo off
setlocal EnableDelayedExpansion
title AI2-THOR Embodied Agent  --  Test Launcher
cd /d "%~dp0"

rem ===========================================================================
rem  One-click launcher for the MSc thesis system.
rem  Double-click this file.  It does the boring parts:
rem    - checks Ollama is up on Windows
rem    - finds the CURRENT WSL -> Windows IP (it changes on every reboot, and a
rem      stale one is the #1 cause of "the agent stopped planning") and repairs
rem      sim/gui_config.json if it drifted
rem    - starts the web console inside WSL with the GPU display env vars
rem    - waits for the port, then opens the console, the trace viewer and the
rem      test guide in your browser
rem  Nothing here touches your research data.  The only file it may write is
rem  sim/gui_config.json (and it backs that up first).
rem ===========================================================================

set "WSLDISTRO=Ubuntu-22.04"
set "WSLSIM=/mnt/d/Personal/Master's Project/sim"
set "VENVPY=/home/arash/thor-env/bin/python"
set "PORT=8000"

:MENU
cls
echo ===========================================================================
echo    AI2-THOR EMBODIED AGENT  -  TEST LAUNCHER
echo ===========================================================================
echo.
echo    [1]  START EVERYTHING  (web console + simulator)
echo         Needs the GPU and WSL. This is the one you want for hands-on
echo         testing - typing commands like "find the apple and slice it".
echo.
echo    [N]  START WITHOUT THE LLM  (web console + simulator, Ollama not used)
echo         Drive by hand, check the dataset, read traces. Nothing contacts
echo         Ollama; switch the planner on later with the LLM chip in the topbar.
echo.
echo    [2]  OFFLINE CHECKS ONLY  (no GPU, no simulator, ~2 minutes)
echo         Runs all six self-test suites. Proves retrieval, memory, tracing
echo         and the eval harness are healthy without booting Unity.
echo.
echo    [3]  OPEN THE TEST GUIDE  (HOW-TO-TEST.html)
echo.
echo    [4]  STOP EVERYTHING  (shut the console down, free the GPU)
echo.
echo    [5]  DIAGNOSE  (what is up, what is not, and why)
echo.
echo    [Q]  Quit
echo.
set "CHOICE="
set /p "CHOICE=   Pick one and press Enter:  "
if /i "!CHOICE!"=="1" goto START_ALL
if /i "!CHOICE!"=="N" goto START_NOLLM
if /i "!CHOICE!"=="2" goto OFFLINE
if /i "!CHOICE!"=="3" goto GUIDE
if /i "!CHOICE!"=="4" goto STOPALL
if /i "!CHOICE!"=="5" goto DIAGNOSE
if /i "!CHOICE!"=="Q" exit /b 0
goto MENU

rem ===========================================================================
:START_NOLLM
set "NOLLM=1"
set "LLMFLAG=--no-llm"
cls
echo ===========================================================================
echo    STARTING WITHOUT THE LLM  ^(turn it on later with the LLM chip^)
echo ===========================================================================
echo.
goto START_BODY

:START_ALL
set "NOLLM=0"
set "LLMFLAG="
cls
echo ===========================================================================
echo    STARTING THE FULL SYSTEM
echo ===========================================================================
echo.

:START_BODY

rem ---- 1. is the console already running? ----------------------------------
call :PORTUP
if "!PORTOPEN!"=="1" (
    echo  [i] The console is ALREADY RUNNING on port %PORT%.
    echo.
    rem  The web files are re-read from disk on every request but webapp.py is loaded ONCE, so a console
    rem  started before a code change serves the NEW page on an OLD server: new panels stay empty and their
    rem  buttons never appear. /status reports the running server's API version; compare it with the file.
    call :STALECHECK
    set "AGAIN="
    if "!STALE!"=="1" (
        echo  [!!] It is OLDER than the code on disk ^(server v!RUNAPI!, webapp.py v!FILEAPI!^).
        echo       New panels will stay EMPTY until it is restarted.
        echo.
        set /p "AGAIN=     Restart it? [Y/n]:  "
        if "!AGAIN!"=="" set "AGAIN=y"
    ) else (
        set /p "AGAIN=     Restart it? [y/N]:  "
    )
    if /i "!AGAIN!"=="y" (
        call :KILLWEBAPP
        timeout /t 2 /nobreak >nul
    ) else (
        echo  [ok] Reusing the console that is already up.
        rem  a reused console keeps its LLM switch, so set it to what was asked for
        if "!NOLLM!"=="1" (
            powershell -NoProfile -Command "try{Invoke-RestMethod -Uri 'http://127.0.0.1:%PORT%/llm?on=0' -TimeoutSec 8 | Out-Null}catch{}" >nul 2>nul
            echo  [ok] LLM switched OFF on the running console.
        ) else (
            echo       ^(if its LLM chip says off, click it to turn the planner on^)
        )
        goto OPENBROWSERS
    )
)
if "!PORTOPEN!"=="2" (
    echo  [!!] Port %PORT% is taken by something that is NOT this console
    echo       ^(it did not answer /status^). Probably a leftover web server.
    echo.
    echo       Find it with:   powershell -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen ^| Select OwningProcess"
    echo       then:           powershell -Command "Get-Process -Id ^<pid^>"
    echo.
    echo       The console cannot start until that port is free.
    echo.
    pause
    goto MENU
)

rem ---- 2. Ollama on Windows -------------------------------------------------
echo  [1/5] Checking Ollama on Windows...
if "!NOLLM!"=="1" (
    echo        skipped - starting without the LLM
    goto AFTEROLLAMA
)
call :OLLAMACHECK
if not "!OLLAMAOK!"=="1" (
    echo.
    echo  [!!] Ollama is NOT responding at http://127.0.0.1:11434
    echo.
    echo       The simulator will still start and you can drive the robot by
    echo       hand, but it will not PLAN anything - no model, no plan.
    echo.
    echo       Fix: open the Ollama app, or run  ollama serve  in a terminal.
    echo.
    set "GOON="
    set /p "GOON=     Continue without the planner? [y/N]:  "
    if /i not "!GOON!"=="y" goto MENU
) else (
    echo        ok - Ollama %OLLAMAVER% is up, %OLLAMAMODELS% models installed
)
:AFTEROLLAMA

rem ---- 3. WSL distro + the host IP that always drifts ----------------------
echo  [2/5] Checking WSL and the WSL-^>Windows route...
set "HOSTIP="
for /f "tokens=3" %%a in ('wsl -d %WSLDISTRO% --exec sh -c "ip route show default" 2^>nul') do set "HOSTIP=%%a"
if "!HOSTIP!"=="" (
    echo.
    echo  [!!] Could not reach WSL distro "%WSLDISTRO%".
    echo       Check it exists:   wsl --list --verbose
    echo       Note the DEFAULT distro is usually docker-desktop, which will not work.
    echo.
    pause
    goto MENU
)
echo        ok - WSL reaches Windows at !HOSTIP!

echo  [3/5] Making sure the agent points at the right Ollama...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$p='sim/gui_config.json'; $want='http://!HOSTIP!:11434';" ^
  "try{ $j=Get-Content $p -Raw | ConvertFrom-Json }catch{ Write-Host '        (could not read gui_config.json - skipping)'; exit 0 };" ^
  "if($j.endpoint -ne $want){" ^
  "  Copy-Item $p ($p + '.bak') -Force;" ^
  "  $j.endpoint=$want;" ^
  "  ($j | ConvertTo-Json -Depth 6) | Set-Content $p -Encoding utf8;" ^
  "  Write-Host ('        REPAIRED - endpoint was stale, now ' + $want + '  (backup: gui_config.json.bak)')" ^
  "} else { Write-Host ('        ok - already ' + $want) }"

rem ---- 4. can WSL actually see Ollama? -------------------------------------
if "!NOLLM!"=="1" goto LAUNCH
wsl -d %WSLDISTRO% --exec sh -c "curl -s -m 5 http://!HOSTIP!:11434/api/version >/dev/null" 2>nul
if errorlevel 1 (
    echo.
    echo  [!!] WSL cannot reach Ollama at !HOSTIP!:11434 even though Windows can.
    echo       Ollama is probably bound to 127.0.0.1 only.
    echo       Fix: set the env var  OLLAMA_HOST=0.0.0.0  and restart Ollama.
    echo.
    set "GOON="
    set /p "GOON=     Continue anyway? [y/N]:  "
    if /i not "!GOON!"=="y" goto MENU
) else (
    echo        ok - WSL can reach Ollama
)

rem ---- 5. launch the console ------------------------------------------------
:LAUNCH
echo  [4/5] Starting the web console inside WSL ^(this opens a second window^)...
echo        First boot downloads/starts the Unity build - it can take a minute.
start "AI2-THOR web console  -  close this window to stop the agent" wsl -d %WSLDISTRO% --cd "%WSLSIM%" -- env MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 %VENVPY% webapp.py !LLMFLAG!

echo  [5/5] Waiting for the console to answer on port %PORT%...
set "WAITED=0"
:WAITLOOP
timeout /t 3 /nobreak >nul
set /a WAITED+=3
call :PORTUP
if "!PORTOPEN!"=="1" goto UP
if !WAITED! GEQ 150 (
    echo.
    echo  [!!] Port %PORT% never opened after !WAITED! seconds.
    echo       Look at the second window - the real error is printed there.
    echo       Most common causes:
    echo         - another AI2-THOR controller already has the GPU ^(only one fits in 8 GB^)
    echo         - DISPLAY=:0 not available ^(WSLg not running^)
    echo.
    pause
    goto MENU
)
echo        ...still starting ^(!WAITED!s^)
goto WAITLOOP

:UP
echo        ok - console is up after !WAITED! seconds

:OPENBROWSERS
echo.
echo ===========================================================================
echo    READY
echo ===========================================================================
echo.
echo    Console .......  http://127.0.0.1:%PORT%
echo    Trace viewer ..  http://127.0.0.1:%PORT%/trace
echo    Test guide ....  HOW-TO-TEST.html
echo.
echo    Try this first, in the command box of the console:
echo.
echo        find the apple in the room and slice it
echo.
echo    Then open the trace viewer and look at what that command wrote to the
echo    knowledge base, and which action caused each write.
echo.
start "" "http://127.0.0.1:%PORT%"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:%PORT%/trace"
if exist "HOW-TO-TEST.html" (
    timeout /t 1 /nobreak >nul
    start "" "HOW-TO-TEST.html"
) else (
    echo    [i] HOW-TO-TEST.html is not here yet - it is still being generated.
)
echo.
echo    Leave this window open. Press any key to return to the menu.
pause >nul
goto MENU

rem ===========================================================================
:OFFLINE
cls
echo ===========================================================================
echo    OFFLINE CHECKS  -  no GPU, no simulator, nothing is modified
echo ===========================================================================
echo.
where python >nul 2>nul
if errorlevel 1 (
    echo  [!!] No "python" on PATH. These checks need Windows Python 3.10+.
    pause
    goto MENU
)
set "FAILED=0"
call :RUNTEST "retrieval  (agentic RAG)"      sim\_agentic_rag_selftest.py
call :RUNTEST "tracing    (provenance)"       sim\_trace_selftest.py
call :RUNTEST "evaluation (SR/PC/TDS)"        sim\_eval_selftest.py
call :RUNTEST "self-improving loop"           sim\_phase7_selftest.py
call :RUNTEST "human feedback"                sim\_phase6_selftest.py
call :RUNTEST "bandit wiring"                 sim\_phase6a_selftest.py
echo.
echo ---------------------------------------------------------------------------
if "!FAILED!"=="0" (
    echo    ALL SUITES PASSED - the non-simulator half of the system is healthy.
) else (
    echo    !FAILED! SUITE^(S^) FAILED - see the log paths above.
)
echo ---------------------------------------------------------------------------
echo.
pause
goto MENU

:RUNTEST
set "LABEL=%~1"
set "SCRIPT=%~2"
set "LOG=%TEMP%\thesis_test_%~n2.log"
<nul set /p "=   %LABEL% ... "
python "%SCRIPT%" > "%LOG%" 2>&1
if errorlevel 1 (
    echo FAIL      ^(log: %LOG%^)
    set /a FAILED+=1
) else (
    for /f "tokens=*" %%l in ('findstr /r /c:"PASS" /c:"checks:" "%LOG%"') do set "LASTLINE=%%l"
    echo PASS
)
exit /b 0

rem ===========================================================================
:GUIDE
if exist "HOW-TO-TEST.html" (
    start "" "HOW-TO-TEST.html"
) else (
    echo.
    echo  [!!] HOW-TO-TEST.html is not here yet.
    echo       It is generated by the test-guide workflow; ask Claude for it.
    echo.
    pause
)
goto MENU

rem ===========================================================================
:STOPALL
cls
echo Stopping the web console and freeing the GPU...
call :KILLWEBAPP
echo Done.
echo.
pause
goto MENU

:KILLWEBAPP
wsl -d %WSLDISTRO% --exec pkill -f webapp.py >nul 2>nul
wsl -d %WSLDISTRO% --exec pkill -f thor-Linux64 >nul 2>nul
exit /b 0

rem ===========================================================================
:DIAGNOSE
cls
echo ===========================================================================
echo    DIAGNOSE
echo ===========================================================================
echo.
call :OLLAMACHECK
if "!OLLAMAOK!"=="1" (echo    Ollama on Windows ....... UP  ^(v%OLLAMAVER%, %OLLAMAMODELS% models^)) else (echo    Ollama on Windows ....... DOWN  - open the Ollama app)

wsl -d %WSLDISTRO% --exec true >nul 2>nul
if errorlevel 1 (echo    WSL %WSLDISTRO% ....... UNREACHABLE) else (echo    WSL %WSLDISTRO% ....... OK)

set "HOSTIP="
for /f "tokens=3" %%a in ('wsl -d %WSLDISTRO% --exec sh -c "ip route show default" 2^>nul') do set "HOSTIP=%%a"
if "!HOSTIP!"=="" (echo    WSL -^> Windows IP ....... unknown) else (echo    WSL -^> Windows IP ....... !HOSTIP!)

if not "!HOSTIP!"=="" (
    wsl -d %WSLDISTRO% --exec sh -c "curl -s -m 5 http://!HOSTIP!:11434/api/version >/dev/null" 2>nul
    if errorlevel 1 (echo    WSL can reach Ollama .... NO   - set OLLAMA_HOST=0.0.0.0 and restart Ollama) else (echo    WSL can reach Ollama .... YES)
)

powershell -NoProfile -Command "try{$j=Get-Content 'sim/gui_config.json' -Raw|ConvertFrom-Json; Write-Host ('    Configured endpoint ..... ' + $j.endpoint + '   model: ' + $j.model)}catch{Write-Host '    Configured endpoint ..... unreadable'}"

call :PORTUP
if "!PORTOPEN!"=="1" (
    echo    Web console ............. UP on http://127.0.0.1:%PORT%
) else if "!PORTOPEN!"=="2" (
    echo    Web console ............. port %PORT% TAKEN by another process ^(not the console^)
) else (
    echo    Web console ............. not running
)

wsl -d %WSLDISTRO% --exec test -x %VENVPY% >nul 2>nul
if errorlevel 1 (echo    WSL venv python ......... MISSING at %VENVPY%) else (echo    WSL venv python ......... OK)

if exist "sim\trace.db" (
    for %%f in ("sim\trace.db") do echo    Trace store ............. %%~zf bytes
) else (
    echo    Trace store ............. not created yet ^(runs once you use the agent^)
)
echo.
pause
goto MENU

rem ===========================================================================
:OLLAMACHECK
rem NOTE: 127.0.0.1, never "localhost". PowerShell resolves localhost to ::1 first
rem and Ollama binds IPv4 only, so the localhost form TIMES OUT instead of failing
rem fast - which looks exactly like "Ollama is down" when it is perfectly healthy.
set "OLLAMAOK=0"
set "OLLAMAVER="
set "OLLAMAMODELS="
for /f "tokens=1,2 delims=|" %%a in ('powershell -NoProfile -Command "try{$v=(Invoke-RestMethod -Uri http://127.0.0.1:11434/api/version -TimeoutSec 5).version; $m=(Invoke-RestMethod -Uri http://127.0.0.1:11434/api/tags -TimeoutSec 5).models.Count; Write-Output ($v + '|' + $m)}catch{Write-Output ''}" 2^>nul') do (
    set "OLLAMAVER=%%a"
    set "OLLAMAMODELS=%%b"
)
if not "!OLLAMAVER!"=="" set "OLLAMAOK=1"
exit /b 0

:STALECHECK
rem STALE=1 when the running console reports an older API version than sim\webapp.py declares.
set "STALE=0"
set "RUNAPI=?"
set "FILEAPI=?"
for /f "tokens=1,2,3 delims=|" %%a in ('powershell -NoProfile -Command "try{$run=[int]((Invoke-RestMethod -Uri http://127.0.0.1:%PORT%/status -TimeoutSec 8).api); $m=[regex]::Match((Get-Content 'sim/webapp.py' -Raw),'WEBAPP_API\s*=\s*(\d+)'); $file=[int]$m.Groups[1].Value; $s=0; if($m.Success -and $run -lt $file){$s=1}; Write-Output ([string]$s + '|' + $run + '|' + $file)}catch{Write-Output '0|?|?'}" 2^>nul') do (
    set "STALE=%%a"
    set "RUNAPI=%%b"
    set "FILEAPI=%%c"
)
exit /b 0

:PORTUP
rem Three-state, because "the port is open" does NOT mean "our console is there":
rem   PORTOPEN=1  our console answered /status with real JSON
rem   PORTOPEN=2  something else owns the port (another server, a stale process)
rem   PORTOPEN=0  nothing is listening
set "PORTOPEN=0"
for /f %%a in ('powershell -NoProfile -Command "$c=New-Object Net.Sockets.TcpClient; $open=$false; try{$c.Connect('127.0.0.1',%PORT%); $open=$c.Connected}catch{}finally{$c.Close()}; if(-not $open){'0'; exit}; try{$r=Invoke-WebRequest -Uri http://127.0.0.1:%PORT%/status -TimeoutSec 8 -UseBasicParsing; if($r.StatusCode -eq 200 -and $r.Content -match '\"fps\"'){'1'}else{'2'}}catch{'2'}" 2^>nul') do set "PORTOPEN=%%a"
exit /b 0
