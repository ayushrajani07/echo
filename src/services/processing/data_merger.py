#!/usr/bin/env python3
"""
OP TRADING PLATFORM - DATA MERGER
=================================
Version: 3.1.2 - Enhanced Data Merger
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class DataMerger:
    """Enhanced data merger for ATM aggregates and market data."""
    
    def __init__(self):
        self.merge_cache = {}
        
    def merge_atm_aggregates(self, overview_data: Dict[str, Any], 
                           atm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ATM aggregates with overview data."""
        try:
            merged_data = overview_data.copy()
            
            # Merge ATM data by index
            for index, atm_values in atm_data.items():
                if index in merged_data:
                    merged_data[index].update(atm_values)
                else:
                    merged_data[index] = atm_values
            
            logger.debug(f"Merged data for {len(merged_data)} indices")
            return merged_data
            
        except Exception as e:
            logger.error(f"Data merge failed: {str(e)}")
            return overview_data
    
    def merge_market_quotes(self, base_data: Dict[str, Any], 
                          quotes: Dict[str, Any]) -> Dict[str, Any]:
        """Merge market quotes with base data."""
        try:
            merged = base_data.copy()
            
            for symbol, quote_data in quotes.items():
                if symbol in merged:
                    merged[symbol].update({
                        "last_price": quote_data.get("last_price"),
                        "volume": quote_data.get("volume"),
                        "oi": quote_data.get("oi"),
                        "quote_timestamp": datetime.now().isoformat()
                    })
            
            return merged
            
        except Exception as e:
            logger.error(f"Quote merge failed: {str(e)}")
            return base_data
