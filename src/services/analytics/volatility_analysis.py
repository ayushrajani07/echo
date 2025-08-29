#!/usr/bin/env python3
"""
OP TRADING PLATFORM - VOLATILITY ANALYSIS
=========================================
Version: 3.1.2 - Enhanced Volatility Analysis
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class VolatilityAnalytics:
    """Advanced volatility analytics for options."""
    
    def __init__(self):
        self.volatility_data = {}
        self.historical_data = {}
        
    def analyze_implied_volatility(self, options_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze implied volatility patterns."""
        try:
            iv_data = []
            
            for option in options_data:
                iv = option.get("iv")
                if iv and iv > 0:
                    iv_data.append(iv)
            
            if not iv_data:
                return {}
            
            mean_iv = np.mean(iv_data)
            median_iv = np.median(iv_data)
            std_iv = np.std(iv_data)
            
            # Classify volatility regime
            if mean_iv > 0.3:
                volatility_regime = "HIGH"
            elif mean_iv > 0.2:
                volatility_regime = "MODERATE"
            else:
                volatility_regime = "LOW"
            
            return {
                "mean_iv": mean_iv,
                "median_iv": median_iv,
                "iv_std": std_iv,
                "iv_range": [min(iv_data), max(iv_data)],
                "volatility_regime": volatility_regime,
                "iv_percentiles": {
                    "25th": np.percentile(iv_data, 25),
                    "75th": np.percentile(iv_data, 75),
                    "90th": np.percentile(iv_data, 90)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"IV analysis failed: {str(e)}")
            return {}
    
    def calculate_volatility_skew(self, call_ivs: List[float], 
                                put_ivs: List[float]) -> Dict[str, Any]:
        """Calculate volatility skew between calls and puts."""
        try:
            if not call_ivs or not put_ivs:
                return {}
            
            avg_call_iv = np.mean(call_ivs)
            avg_put_iv = np.mean(put_ivs)
            
            skew = avg_put_iv - avg_call_iv
            skew_percentage = (skew / avg_call_iv) * 100 if avg_call_iv > 0 else 0
            
            # Interpret skew
            if skew_percentage > 5:
                skew_interpretation = "PUT_PREMIUM"
            elif skew_percentage < -5:
                skew_interpretation = "CALL_PREMIUM"
            else:
                skew_interpretation = "BALANCED"
            
            return {
                "avg_call_iv": avg_call_iv,
                "avg_put_iv": avg_put_iv,
                "volatility_skew": skew,
                "skew_percentage": skew_percentage,
                "skew_interpretation": skew_interpretation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Volatility skew calculation failed: {str(e)}")
            return {}
