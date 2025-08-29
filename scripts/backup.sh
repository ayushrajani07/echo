#!/bin/bash
# OP Trading Platform - Backup Script
# Version: 3.1.2

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="op_trading_backup_${TIMESTAMP}"

echo "Starting backup: ${BACKUP_NAME}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup InfluxDB data
if [ -d "/source/influxdb" ]; then
    echo "Backing up InfluxDB data..."
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/influxdb_${TIMESTAMP}.tar.gz" -C /source influxdb/
fi

# Backup Redis data
if [ -d "/source/redis" ]; then
    echo "Backing up Redis data..."
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/redis_${TIMESTAMP}.tar.gz" -C /source redis/
fi

# Backup Grafana data
if [ -d "/source/grafana" ]; then
    echo "Backing up Grafana data..."
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/grafana_${TIMESTAMP}.tar.gz" -C /source grafana/
fi

# Create manifest
echo "Backup created: ${TIMESTAMP}" > "${BACKUP_DIR}/${BACKUP_NAME}/manifest.txt"
echo "Components: InfluxDB, Redis, Grafana" >> "${BACKUP_DIR}/${BACKUP_NAME}/manifest.txt"

# Cleanup old backups (keep last 30 days)
find "${BACKUP_DIR}" -name "op_trading_backup_*" -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true

echo "Backup completed: ${BACKUP_NAME}"
