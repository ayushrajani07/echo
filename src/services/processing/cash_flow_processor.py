#!/usr/bin/env python3
"""
OP TRADING PLATFORM - CASH FLOW PROCESSOR
=========================================
Version: 3.1.2 - Enhanced Cash Flow Processor
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class CashFlowProcessor:
    """Enhanced cash flow processor for options trading."""
    
    def __init__(self):
        self.cash_flows = {}
        self.flow_history = []
        
    def calculate_cash_flows(self, options_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cash flows from options data."""
        try:
            total_inflow = 0
            total_outflow = 0
            
            for option in options_data:
                price = option.get("last_price", 0)
                volume = option.get("volume", 0)
                
                # Estimate cash flow (simplified)
                cash_value = price * volume * 50  # Assuming lot size of 50
                
                # Classify as inflow or outflow based on price movement
                if option.get("net_change", 0) > 0:
                    total_inflow += cash_value
                else:
                    total_outflow += cash_value
            
            net_flow = total_inflow - total_outflow
            
            return {
                "total_cash_inflow": total_inflow,
                "total_cash_outflow": total_outflow,
                "net_cash_flow": net_flow,
                "buying_pressure": total_inflow / (total_inflow + total_outflow) if (total_inflow + total_outflow) > 0 else 0.5,
                "selling_pressure": total_outflow / (total_inflow + total_outflow) if (total_inflow + total_outflow) > 0 else 0.5,
                "market_sentiment": "BULLISH" if net_flow > 0 else "BEARISH" if net_flow < 0 else "NEUTRAL",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cash flow calculation failed: {str(e)}")
            return {}
    
    def process_position_changes(self, current_positions: Dict[str, Any], 
                               previous_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Process position changes between time periods."""
        try:
            position_changes = []
            
            for symbol in current_positions:
                current_oi = current_positions[symbol].get("oi", 0)
                previous_oi = previous_positions.get(symbol, {}).get("oi", 0)
                
                if previous_oi > 0:
                    oi_change = current_oi - previous_oi
                    oi_change_percent = (oi_change / previous_oi) * 100
                    
                    position_changes.append({
                        "symbol": symbol,
                        "oi_change": oi_change,
                        "oi_change_percent": oi_change_percent,
                        "volume": current_positions[symbol].get("volume", 0),
                        "price_impact": current_positions[symbol].get("net_change_percent", 0)
                    })
            
            return {
                "position_changes": position_changes,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Position change processing failed: {str(e)}")
            return {}
