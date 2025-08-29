# Start-WSLRedis.ps1

# Start Redis in Ubuntu WSL if not running
wsl -d Ubuntu -- bash -c '
if ! pgrep -x "redis-server" > /dev/null
then
    sudo service redis-server start
    echo "Redis server started"
else
    echo "Redis server is already running"
fi
'

# Open an interactive WSL shell session after starting Redis
wsl -d Ubuntu
