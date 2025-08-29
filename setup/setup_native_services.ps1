#!/usr/bin/env powershell
# OP Trading Platform – Native Windows Service Setup
# Run as Administrator

# Read environment variables
$POA_HOME   = $Env:POA_HOME   | ForEach-Object { if ($_ -ne "") { $_ } else { "C:\POA" } }
$LOG_DIR    = $Env:POA_LOGS    | ForEach-Object { if ($_ -ne "") { $_ } else { "$POA_HOME\logs" } }
$DATA_DIR   = $Env:POA_DATA    | ForEach-Object { if ($_ -ne "") { $_ } else { "$POA_HOME\data" } }
$CONFIG_DIR = $Env:POA_CONFIG  | ForEach-Object { if ($_ -ne "") { $_ } else { "$POA_HOME\config" } }

Write-Host "🚀 Setting up POA Trading Platform Native Services..." -ForegroundColor Green
Write-Host "🔧 Using POA_HOME=`"$POA_HOME`"" -ForegroundColor Cyan

# Create directories
foreach ($dir in @($LOG_DIR, $DATA_DIR, $CONFIG_DIR)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created directory $dir"
    } else {
        Write-Host "ℹ️  Directory $dir already exists"
    }
}

# Start Redis service
Write-Host "📊 Starting Redis service..." -ForegroundColor Yellow
Start-Service -Name Redis -ErrorAction SilentlyContinue
if ((Get-Service -Name Redis).Status -eq 'Running') {
    Write-Host "✅ Redis is running"
} else {
    Write-Host "❌ Redis failed to start" -ForegroundColor Red
}

# Start InfluxDB service
Write-Host "📈 Starting InfluxDB service..." -ForegroundColor Yellow
Start-Service -Name influxdb -ErrorAction SilentlyContinue
if ((Get-Service -Name influxdb).Status -eq 'Running') {
    Write-Host "✅ InfluxDB is running"
} else {
    Write-Host "❌ InfluxDB failed to start" -ForegroundColor Red
}

# Start Grafana service
Write-Host "📊 Starting Grafana service..." -ForegroundColor Yellow
Start-Service -Name grafana -ErrorAction SilentlyContinue
if ((Get-Service -Name grafana).Status -eq 'Running') {
    Write-Host "✅ Grafana is running"
} else {
    Write-Host "❌ Grafana failed to start" -ForegroundColor Red
}

# Start Prometheus (standalone executable)
Write-Host "📉 Starting Prometheus..." -ForegroundColor Yellow
$prometheusExe = "C:\prometheus\prometheus.exe"
if (Test-Path $prometheusExe) {
    Start-Process -FilePath $prometheusExe -ArgumentList '--config.file=C:\prometheus\prometheus.yml' -WindowStyle Hidden
    Write-Host "✅ Prometheus started"
} else {
    Write-Host "❌ Prometheus executable not found at $prometheusExe" -ForegroundColor Red
}

Write-Host "🎉 All native services setup complete!"
Write-Host "🌐 Service URLs:"
Write-Host "   • Redis:        http://localhost:6379"
Write-Host "   • InfluxDB:     http://localhost:8086"
Write-Host "   • Grafana:      http://localhost:3000"
Write-Host "   • Prometheus:   http://localhost:9090"
