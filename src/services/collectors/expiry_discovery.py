#!/usr/bin/env python3
"""
OP TRADING PLATFORM - EXPIRY DISCOVERY
======================================
Version: 3.1.2 - Enhanced Expiry Discovery
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional
import calendar

logger = logging.getLogger(__name__)

def discover_weeklies_for_index(pool_nfo: List[Dict], pool_bfo: List[Dict], 
                               index: str, atm: int, count: int = 2) -> List[date]:
    """Discover weekly expiry dates for an index."""
    try:
        combined_pool = pool_nfo + pool_bfo
        
        # Filter instruments for the index
        index_instruments = [
            inst for inst in combined_pool 
            if index.upper() in str(inst.get("name", "")).upper()
            and inst.get("instrument_type") in ["CE", "PE"]
        ]
        
        # Extract unique expiry dates
        expiry_dates = set()
        for inst in index_instruments:
            expiry_str = inst.get("expiry", "")
            if expiry_str:
                try:
                    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                    if expiry > date.today():
                        expiry_dates.add(expiry)
                except:
                    continue
        
        # Sort and return next few expiries
        sorted_expiries = sorted(expiry_dates)
        return sorted_expiries[:count]
        
    except Exception as e:
        logger.error(f"Weekly expiry discovery failed for {index}: {str(e)}")
        return []

def discover_monthlies_for_index(pool_nfo: List[Dict], pool_bfo: List[Dict],
                                index: str, atm: int) -> tuple:
    """Discover monthly expiry dates for an index."""
    try:
        combined_pool = pool_nfo + pool_bfo
        
        # Filter instruments
        index_instruments = [
            inst for inst in combined_pool
            if index.upper() in str(inst.get("name", "")).upper() 
            and inst.get("instrument_type") in ["CE", "PE"]
        ]
        
        # Find monthly expiries (typically last Thursday of month)
        monthly_expiries = []
        for inst in index_instruments:
            expiry_str = inst.get("expiry", "")
            if expiry_str:
                try:
                    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                    if expiry > date.today() and expiry.day >= 24:
                        monthly_expiries.append(expiry)
                except:
                    continue
        
        # Sort and get next two
        sorted_monthlies = sorted(set(monthly_expiries))
        this_month = sorted_monthlies[0] if len(sorted_monthlies) > 0 else None
        next_month = sorted_monthlies[1] if len(sorted_monthlies) > 1 else None
        
        return this_month, next_month
        
    except Exception as e:
        logger.error(f"Monthly expiry discovery failed for {index}: {str(e)}")
        return None, None
