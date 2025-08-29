#!/usr/bin/env python3
"""
OP TRADING PLATFORM - POSITION MONITOR
======================================
Version: 3.1.2 - Enhanced Position Monitor
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PositionMonitor:
    """Enhanced position monitor for options trading."""
    
    def __init__(self):
        self.positions = {}
        self.alerts = []
        self.thresholds = {
            "oi_change_percent": 15.0,
            "volume_spike_percent": 100.0,
            "price_impact_percent": 5.0
        }
    
    def monitor_positions(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor positions for significant changes."""
        try:
            alerts = []
            
            for symbol, data in current_data.items():
                # Check for significant OI changes
                if self._check_oi_change(symbol, data):
                    alerts.append({
                        "type": "OI_CHANGE",
                        "symbol": symbol,
                        "message": f"Significant OI change detected in {symbol}",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Check for volume spikes
                if self._check_volume_spike(symbol, data):
                    alerts.append({
                        "type": "VOLUME_SPIKE",
                        "symbol": symbol,
                        "message": f"Volume spike detected in {symbol}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Update position tracking
            self.positions.update(current_data)
            
            return {
                "alerts": alerts,
                "monitored_positions": len(current_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Position monitoring failed: {str(e)}")
            return {}
    
    def _check_oi_change(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Check for significant OI changes."""
        try:
            current_oi = data.get("oi", 0)
            previous_data = self.positions.get(symbol, {})
            previous_oi = previous_data.get("oi", 0)
            
            if previous_oi > 0:
                change_percent = abs((current_oi - previous_oi) / previous_oi) * 100
                return change_percent > self.thresholds["oi_change_percent"]
            
            return False
        except:
            return False
    
    def _check_volume_spike(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Check for volume spikes."""
        try:
            current_volume = data.get("volume", 0)
            previous_data = self.positions.get(symbol, {})
            previous_volume = previous_data.get("volume", 0)
            
            if previous_volume > 0:
                change_percent = ((current_volume - previous_volume) / previous_volume) * 100
                return change_percent > self.thresholds["volume_spike_percent"]
            
            return False
        except:
            return False
