# POA Services Manager - Complete Native Windows + WSL Setup
# Handles Redis (WSL), InfluxDB (Native Windows via NSSM), Grafana (Native Windows)
# Version: 2.0 with integrated NSSM configuration

$ErrorActionPreference = "Continue"

function Log-OK  { param($m) Write-Host ("[OK ] " + $m) -ForegroundColor Green }
function Log-ERR { param($m) Write-Host ("[ERR] " + $m) -ForegroundColor Red }
function Log-INF { param($m) Write-Host ("[INF] " + $m) -ForegroundColor Cyan }
function Log-WRN { param($m) Write-Host ("[WRN] " + $m) -ForegroundColor Yellow }

# Test WSL availability
function Test-WSL {
  try {
    wsl -l -q | Out-Null
    return $true
  } catch {
    Log-ERR "WSL not available. Install WSL and Ubuntu first."
    return $false
  }
}

# ---------- Redis (WSL Ubuntu) ----------
function Start-Redis {
  if (-not (Test-WSL)) { return }
  Log-INF "Starting Redis in WSL (Ubuntu)..."
  try {
    wsl -d Ubuntu -- sudo service redis-server start | Out-Null
    if ($LASTEXITCODE -eq 0) { Log-OK "Redis started (WSL)." } else { Log-WRN "Redis start returned non-zero exit code." }
  } catch { Log-ERR "Redis start error: $_" }
}

function Stop-Redis {
  if (-not (Test-WSL)) { return }
  Log-INF "Stopping Redis in WSL (Ubuntu)..."
  try {
    wsl -d Ubuntu -- sudo service redis-server stop | Out-Null
    Log-OK "Redis stopped (WSL)."
  } catch { Log-ERR "Redis stop error: $_" }
}

function Test-Redis {
  if (-not (Test-WSL)) { return $false }
  Log-INF "Testing Redis (127.0.0.1:6379) via WSL redis-cli..."
  try {
    $pong = wsl -d Ubuntu -- redis-cli -h 127.0.0.1 ping 2>&1
    if ($pong -match "^PONG$") { Log-OK "Redis responded PONG."; return $true }
    Log-ERR "Redis ping failed: $pong"; return $false
  } catch { Log-ERR "Redis test error: $_"; return $false }
}

function Get-RedisInfo {
  if (-not (Test-WSL)) { return }
  Log-INF "Redis INFO (server/clients)..."
  try {
    wsl -d Ubuntu -- bash -c "redis-cli -h 127.0.0.1 info server | egrep 'redis_version|uptime_in_seconds'; redis-cli -h 127.0.0.1 info clients | egrep 'connected_clients'" 2>&1
  } catch { Log-ERR "Redis INFO error: $_" }
}

function redis-cli { wsl -d Ubuntu -- redis-cli @args }
Set-Alias rcli redis-cli

# ---------- InfluxDB (Native Windows Service via NSSM) ----------
function Configure-InfluxDBNative {
    param(
        [string]$NssmExe      = "C:\Tools\nssm\nssm-2.24\win64\nssm.exe",
        [string]$ServiceName  = "InfluxDB",
        [string]$InfluxExe    = "C:\influxdata\influxdb-1.8.10-1\influxd.exe",
        [string]$ConfigPath   = "C:\influxdata\config.yml",
        [string]$LogDir       = "C:\influxdata\logs",
        [string]$StdoutLog    = "C:\influxdata\logs\influxd.out.log",
        [string]$StderrLog    = "C:\influxdata\logs\influxd.err.log",
        [string]$StartMode    = "SERVICE_AUTO_START",
        [int]$RotateSeconds   = 86400,
        [int]$RotateBytes     = 10485760
    )

    try {
        if (-not (Test-Path $NssmExe)) { Log-ERR "NSSM not found at $NssmExe"; return $false }
        if (-not (Test-Path $InfluxExe)) { Log-ERR "influxd.exe not found at $InfluxExe"; return $false }
        if (-not (Test-Path $ConfigPath)) { Log-ERR "Config file not found at $ConfigPath"; return $false }

        # Ensure log directory
        New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

        Log-INF "Removing existing InfluxDB service if present..."
        & $NssmExe remove $ServiceName confirm | Out-Null

        Log-INF "Installing InfluxDB service '$ServiceName'..."
        & $NssmExe install $ServiceName $InfluxExe -config $ConfigPath

        Log-INF "Configuring service parameters..."
        & $NssmExe set $ServiceName Start $StartMode
        & $NssmExe set $ServiceName AppDirectory (Split-Path $InfluxExe -Parent)
        & $NssmExe set $ServiceName AppStdout $StdoutLog
        & $NssmExe set $ServiceName AppStderr $StderrLog
        & $NssmExe set $ServiceName AppRotateFiles 1
        & $NssmExe set $ServiceName AppRotateOnline 1
        & $NssmExe set $ServiceName AppRotateSeconds $RotateSeconds
        & $NssmExe set $ServiceName AppRotateBytes $RotateBytes

        Log-INF "Starting InfluxDB service..."
        Start-Service $ServiceName
        Start-Sleep -Seconds 3

        $svc = Get-Service $ServiceName -ErrorAction Stop
        Log-OK "InfluxDB service status: $($svc.Status)"
        return $true

    } catch {
        Log-ERR "Configure-InfluxDBNative failed: $_"
        return $false
    }
}

function Start-InfluxDB {
  Log-INF "Starting InfluxDB (native Windows service)..."
  try {
    $svc = Get-Service -Name "InfluxDB","influxdb","influxdb2","influxdb64" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($svc) {
      if ($svc.Status -ne "Running") { 
        Start-Service -Name $svc.Name
        Log-OK "InfluxDB Windows service started ($($svc.Name))." 
      } else { 
        Log-OK "InfluxDB Windows service already running ($($svc.Name))." 
      }
      return $true
    } else {
      Log-WRN "InfluxDB Windows service not found. Run 'Configure InfluxDB Service' from menu first."
      return $false
    }
  } catch {
    Log-ERR "InfluxDB start error: $_"
    return $false
  }
}

function Stop-InfluxDB {
  Log-INF "Stopping InfluxDB Windows service..."
  try {
    $svc = Get-Service -Name "InfluxDB","influxdb","influxdb2","influxdb64" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($svc) {
      if ($svc.Status -ne "Stopped") { 
        Stop-Service -Name $svc.Name -Force
        Log-OK "InfluxDB Windows service stopped ($($svc.Name))." 
      } else { 
        Log-OK "InfluxDB Windows service already stopped ($($svc.Name))." 
      }
    } else {
      Log-WRN "No InfluxDB Windows service found."
    }
  } catch {
    Log-ERR "InfluxDB stop error: $_"
  }
}

function Test-InfluxDB {
  Log-INF "Testing InfluxDB http://localhost:8086 ..."
  try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8086/ping" -TimeoutSec 5 -ErrorAction Stop
    if ($resp.StatusCode -eq 204) { 
      Log-OK "InfluxDB /ping responded 204."
      try {
        $health = Invoke-RestMethod -Uri "http://localhost:8086/health" -TimeoutSec 5
        Log-OK ("InfluxDB health: " + $health.status)
        return $true
      } catch {
        Log-WRN "InfluxDB /health check failed: $_"
        return $true  # ping worked, so service is up
      }
    } else { 
      Log-WRN "InfluxDB /ping unexpected code: $($resp.StatusCode)"
      return $false
    }
  } catch { 
    Log-ERR "InfluxDB /ping failed: $_"
    return $false
  }
}

function Debug-InfluxDB {
  Write-Host ""
  Write-Host "INFLUXDB DEBUG" -ForegroundColor Yellow
  Write-Host "==============" -ForegroundColor Yellow
  
  Log-INF "Windows services matching 'influx':"
  Get-Service -Name "*influx*" -ErrorAction SilentlyContinue | Format-Table -AutoSize
  
  Log-INF "InfluxDB processes (Windows):"
  Get-Process -Name "*influx*" -ErrorAction SilentlyContinue | Format-Table -AutoSize
  
  $null = Test-InfluxDB
  
  # Check NSSM service details if InfluxDB service exists
  try {
    $svc = Get-Service -Name "InfluxDB" -ErrorAction Stop
    Log-INF "NSSM service configuration:"
    $nssm = "C:\Tools\nssm\nssm-2.24\win64\nssm.exe"
    if (Test-Path $nssm) {
      Write-Host "  Application: " -NoNewline; & $nssm get InfluxDB Application
      Write-Host "  Parameters:  " -NoNewline; & $nssm get InfluxDB AppParameters  
      Write-Host "  Directory:   " -NoNewline; & $nssm get InfluxDB AppDirectory
      Write-Host "  Stdout Log:  " -NoNewline; & $nssm get InfluxDB AppStdout
      Write-Host "  Stderr Log:  " -NoNewline; & $nssm get InfluxDB AppStderr
    } else {
      Log-WRN "NSSM executable not found at $nssm"
    }
    
    # Check recent service logs if they exist
    $errLog = "C:\influxdata\logs\influxd.err.log"
    if (Test-Path $errLog) {
      Log-INF "Recent error log (last 10 lines):"
      Get-Content -Tail 10 $errLog | ForEach-Object { Write-Host "    $_" }
    }
    
    # Check port usage
    Log-INF "Port 8086 and 8088 usage:"
    $ports = netstat -abno | Select-String ":808[68]"
    if ($ports) { $ports | ForEach-Object { Write-Host "    $_" } }
    else { Write-Host "    No processes using ports 8086 or 8088" }
    
  } catch {
    Log-WRN "InfluxDB service not found or NSSM unavailable: $_"
  }
}


function influx-cli { wsl -d Ubuntu -- influx @args }
Set-Alias icli influx-cli

# ---------- Grafana (Native Windows Service) ----------
function Start-Grafana {
  Log-INF "Starting Grafana Windows service..."
  try {
    $svc = Get-Service -Name "Grafana" -ErrorAction Stop
    if ($svc.Status -ne "Running") { Start-Service -Name "Grafana"; Log-OK "Grafana started." } else { Log-OK "Grafana already running." }
  } catch { Log-ERR "Grafana start error or service not found: $_" }
}

function Stop-Grafana {
  Log-INF "Stopping Grafana Windows service..."
  try {
    $svc = Get-Service -Name "Grafana" -ErrorAction Stop
    if ($svc.Status -ne "Stopped") { Stop-Service -Name "Grafana"; Log-OK "Grafana stopped." } else { Log-OK "Grafana already stopped." }
  } catch { Log-ERR "Grafana stop error or service not found: $_" }
}

function Test-Grafana {
  Log-INF "Testing Grafana http://localhost:3000 ..."
  try {
    $resp = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 10 -ErrorAction Stop
    if ($resp.StatusCode -eq 200) { Log-OK "Grafana web UI responded 200."; return $true }
    Log-WRN "Grafana unexpected status: $($resp.StatusCode)"; return $false
  } catch { Log-ERR "Grafana web test failed: $_"; return $false }
}

# ---------- Status & Debug Functions ----------
function Get-ServiceStatus {
  Write-Host ""
  Write-Host "POA SERVICES STATUS" -ForegroundColor Yellow
  Write-Host "===================" -ForegroundColor Yellow

  Write-Host "`nRedis (WSL Ubuntu):"
  if (Test-Redis) { Get-RedisInfo }

  Write-Host "`nInfluxDB (Windows Service):"
  try {
    $influx = Get-Service -Name "InfluxDB","influxdb","influxdb2","influxdb64" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($influx) {
      Write-Host ("  Service status: " + $influx.Status)
    } else {
      Log-WRN "  InfluxDB service not installed."
    }
  } catch {
    Log-ERR "  Error checking InfluxDB service."
  }
  if (Test-InfluxDB) { Write-Host "  HTTP API: http://localhost:8086" -ForegroundColor Green }

  Write-Host "`nGrafana (Windows Service):"
  try {
    $g = Get-Service -Name "Grafana" -ErrorAction Stop
    Write-Host ("  Service status: " + $g.Status)
  } catch {
    Log-ERR "  Grafana service not found."
  }
  if (Test-Grafana) { Write-Host "  Web UI: http://localhost:3000 (default admin/admin)" -ForegroundColor Green }
}

function Debug-Redis {
  Write-Host ""
  Write-Host "REDIS DEBUG" -ForegroundColor Yellow
  Write-Host "===========" -ForegroundColor Yellow
  if (Test-WSL) {
    Log-INF "WSL Ubuntu kernel:"; wsl -d Ubuntu -- uname -a
    Log-INF "Redis service status:"; wsl -d Ubuntu -- sudo service redis-server status
    Log-INF "Redis processes (WSL):"; wsl -d Ubuntu -- bash -c "ps aux | grep [r]edis"
  }
  $null = Test-Redis
}

function Debug-InfluxDB {
  Write-Host ""
  Write-Host "INFLUXDB DEBUG" -ForegroundColor Yellow
  Write-Host "==============" -ForegroundColor Yellow
  Log-INF "Windows services matching 'influx':"
  Get-Service -Name "*influx*" -ErrorAction SilentlyContinue | Format-Table -AutoSize
  Log-INF "InfluxDB processes (Windows):"
  Get-Process -Name "*influx*" -ErrorAction SilentlyContinue | Format-Table -AutoSize
  $null = Test-InfluxDB
  
  # Check NSSM service details if InfluxDB service exists
  try {
    $svc = Get-Service -Name "InfluxDB" -ErrorAction Stop
    Log-INF "NSSM service details:"
    $nssm = "C:\Tools\nssm\nssm-2.24\win64\nssm.exe"
    if (Test-Path $nssm) {
      & $nssm dump InfluxDB
    }
  } catch {
    Log-WRN "InfluxDB service not found or NSSM unavailable."
  }
}

function Debug-Grafana {
  Write-Host ""
  Write-Host "GRAFANA DEBUG" -ForegroundColor Yellow
  Write-Host "=============" -ForegroundColor Yellow
  Log-INF "Grafana Windows service:"
  Get-Service -Name "Grafana" -ErrorAction SilentlyContinue | Format-List
  Log-INF "Grafana processes:"
  Get-Process -Name "*grafana*" -ErrorAction SilentlyContinue | Format-Table -AutoSize
  $null = Test-Grafana
}

# ---------- Service Orchestration ----------
function Initialize-POAServices {
  Write-Host ""
  Write-Host "INITIALIZING POA SERVICES" -ForegroundColor Yellow
  Write-Host "=========================" -ForegroundColor Yellow
  
  Start-Redis
  if (-not (Start-InfluxDB)) {
    Log-WRN "InfluxDB service not found. Consider running 'Configure InfluxDB Service' first."
  }
  Start-Grafana
  
  Log-INF "Waiting 5 seconds for services to come up..."
  Start-Sleep -Seconds 5
  Get-ServiceStatus
}

function Stop-POAServices {
  Write-Host ""
  Write-Host "STOPPING POA SERVICES" -ForegroundColor Yellow
  Write-Host "=====================" -ForegroundColor Yellow
  Stop-Grafana
  Stop-InfluxDB
  Stop-Redis
}

function Show-QuickCommands {
  Write-Host ""
  Write-Host "QUICK COMMANDS" -ForegroundColor Yellow
  Write-Host "=============="
  Write-Host "Redis (WSL):"
  Write-Host "  rcli ping"
  Write-Host "  rcli set test hello"
  Write-Host "  rcli get test"
  Write-Host ""
  Write-Host "InfluxDB:"
  Write-Host "  Invoke-WebRequest http://localhost:8086/ping"
  Write-Host "  Invoke-RestMethod http://localhost:8086/health"
  Write-Host ""
  Write-Host "Grafana:"
  Write-Host "  Start-Service Grafana"
  Write-Host "  Get-Service Grafana"
  Write-Host "  Open http://localhost:3000"
  Write-Host ""
  Write-Host "Service Management:"
  Write-Host "  Get-Service InfluxDB"
  Write-Host "  Restart-Service InfluxDB"
}

function Show-Menu {
  Write-Host ""
  Write-Host "POA SERVICES MANAGER" -ForegroundColor Yellow
  Write-Host "===================="
  Write-Host "1) Initialize All Services"
  Write-Host "2) Service Status"
  Write-Host "3) Debug Redis"
  Write-Host "4) Debug InfluxDB"
  Write-Host "5) Debug Grafana"
  Write-Host "6) Stop All Services"
  Write-Host "7) Configure InfluxDB Service (NSSM)"
  Write-Host "8) Quick Commands Help"
  Write-Host "0) Exit"
}

function Start-POAServiceManager {
  do {
    Show-Menu
    $choice = Read-Host "Select option"
    switch ($choice) {
      "1" { Initialize-POAServices }
      "2" { Get-ServiceStatus }
      "3" { Debug-Redis }
      "4" { Debug-InfluxDB }
      "5" { Debug-Grafana }
      "6" { Stop-POAServices }
      "7" { 
        Write-Host ""
        Write-Host "CONFIGURING INFLUXDB WINDOWS SERVICE" -ForegroundColor Yellow
        Write-Host "====================================" -ForegroundColor Yellow
        if (Configure-InfluxDBNative) {
          Log-OK "InfluxDB service configured successfully."
          Log-INF "Testing service..."
          if (Test-InfluxDB) {
            Log-OK "InfluxDB is healthy and responding."
          }
        } else {
          Log-ERR "InfluxDB service configuration failed."
        }
      }
      "8" { Show-QuickCommands }
      "0" { Log-INF "Exiting..."; break }
      default { Log-WRN "Invalid option." }
    }
    if ($choice -ne "0") {
      Write-Host ""
      Write-Host "Press any key to continue..." -ForegroundColor Gray
      $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
  } while ($choice -ne "0")
}

# ---------- Auto-run when executed directly ----------
Write-Host ">>> POA Services Manager v2.0 starting..." -ForegroundColor Yellow
Initialize-POAServices
Start-POAServiceManager
Write-Host ">>> POA Services Manager done." -ForegroundColor Yellow
