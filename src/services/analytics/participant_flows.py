#!/usr/bin/env python3
"""
OP TRADING PLATFORM - PARTICIPANT FLOWS ANALYTICS
=================================================
Version: 3.1.2 - Enhanced Participant Flows Analytics
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ParticipantFlowAnalytics:
    """Advanced analytics for participant flows."""
    
    def __init__(self):
        self.flow_data = {}
        self.patterns = {}
        
    def analyze_flow_patterns(self, participant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze flow patterns across participants."""
        try:
            analysis = {}
            
            # FII Analysis
            if "FII" in participant_data:
                fii_data = participant_data["FII"]
                analysis["FII"] = {
                    "flow_direction": "BUYING" if fii_data.get("net_flow", 0) > 0 else "SELLING",
                    "flow_intensity": self._calculate_flow_intensity(fii_data.get("net_flow", 0)),
                    "sector_preference": self._analyze_sector_preference(fii_data.get("sector_allocation", {})),
                    "risk_appetite": self._assess_risk_appetite(fii_data)
                }
            
            # DII Analysis
            if "DII" in participant_data:
                dii_data = participant_data["DII"]
                analysis["DII"] = {
                    "flow_direction": "BUYING" if dii_data.get("net_flow", 0) > 0 else "SELLING",
                    "mutual_fund_bias": dii_data.get("mutual_fund_activity", 0),
                    "insurance_bias": dii_data.get("insurance_activity", 0),
                    "domestic_sentiment": self._assess_domestic_sentiment(dii_data)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Flow pattern analysis failed: {str(e)}")
            return {}
    
    def _calculate_flow_intensity(self, net_flow: float) -> str:
        """Calculate flow intensity based on net flow."""
        abs_flow = abs(net_flow)
        if abs_flow > 1000:
            return "VERY_HIGH"
        elif abs_flow > 500:
            return "HIGH"
        elif abs_flow > 100:
            return "MODERATE"
        else:
            return "LOW"
    
    def _analyze_sector_preference(self, sector_allocation: Dict[str, float]) -> str:
        """Analyze sector preference."""
        if not sector_allocation:
            return "BALANCED"
        
        max_sector = max(sector_allocation, key=sector_allocation.get)
        max_allocation = sector_allocation[max_sector]
        
        if max_allocation > 40:
            return f"CONCENTRATED_{max_sector}"
        else:
            return "DIVERSIFIED"
    
    def _assess_risk_appetite(self, participant_data: Dict[str, Any]) -> str:
        """Assess risk appetite based on participant behavior."""
        # Simplified risk assessment
        net_flow = abs(participant_data.get("net_flow", 0))
        activity_level = participant_data.get("activity_level", "MODERATE")
        
        if activity_level == "HIGH" and net_flow > 500:
            return "AGGRESSIVE"
        elif activity_level == "MODERATE":
            return "MODERATE"
        else:
            return "CONSERVATIVE"
    
    def _assess_domestic_sentiment(self, dii_data: Dict[str, Any]) -> str:
        """Assess domestic market sentiment."""
        net_flow = dii_data.get("net_flow", 0)
        mf_activity = dii_data.get("mutual_fund_activity", 0)
        
        if net_flow > 0 and mf_activity > 0:
            return "OPTIMISTIC"
        elif net_flow < 0:
            return "PESSIMISTIC"
        else:
            return "NEUTRAL"
