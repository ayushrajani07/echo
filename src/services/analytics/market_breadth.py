#!/usr/bin/env python3
"""
OP TRADING PLATFORM - MARKET BREADTH ANALYTICS
==============================================
Version: 3.1.2 - Enhanced Market Breadth Analytics
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class MarketBreadthAnalytics:
    """Advanced market breadth analytics."""
    
    def __init__(self):
        self.breadth_data = {}
        
    def calculate_market_breadth(self, indices_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive market breadth metrics."""
        try:
            advances = 0
            declines = 0
            unchanged = 0
            total_indices = len(indices_data)
            
            for symbol, data in indices_data.items():
                net_change = data.get("net_change", 0)
                
                if net_change > 0:
                    advances += 1
                elif net_change < 0:
                    declines += 1
                else:
                    unchanged += 1
            
            # Calculate ratios
            advance_decline_ratio = advances / declines if declines > 0 else float('inf')
            advance_percentage = (advances / total_indices) * 100 if total_indices > 0 else 0
            
            # Determine market sentiment
            if advance_percentage > 60:
                market_sentiment = "BULLISH"
            elif advance_percentage < 40:
                market_sentiment = "BEARISH"
            else:
                market_sentiment = "NEUTRAL"
            
            return {
                "advances": advances,
                "declines": declines,
                "unchanged": unchanged,
                "total_indices": total_indices,
                "advance_decline_ratio": advance_decline_ratio,
                "advance_percentage": advance_percentage,
                "market_sentiment": market_sentiment,
                "breadth_score": self._calculate_breadth_score(advance_percentage),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market breadth calculation failed: {str(e)}")
            return {}
    
    def _calculate_breadth_score(self, advance_percentage: float) -> float:
        """Calculate a normalized breadth score (0-100)."""
        # Normalize advance percentage to a 0-100 scale
        return max(0, min(100, advance_percentage))
