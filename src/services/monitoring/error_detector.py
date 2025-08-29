#!/usr/bin/env python3
"""
OP TRADING PLATFORM - ERROR DETECTOR
====================================
Version: 3.1.2 - Enhanced Error Detection
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import traceback

logger = logging.getLogger(__name__)

class ErrorDetector:
    """Enhanced error detection and classification system."""
    
    def __init__(self):
        self.error_history = []
        self.error_patterns = {}
        self.alert_thresholds = {
            "error_rate_per_minute": 5,
            "consecutive_errors": 3,
            "critical_error_threshold": 1
        }
    
    def detect_and_classify_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and classify errors with context."""
        try:
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "stack_trace": traceback.format_exc(),
                "severity": self._assess_severity(error),
                "category": self._categorize_error(error),
                "recovery_suggestions": self._get_recovery_suggestions(error)
            }
            
            # Add to history
            self.error_history.append(error_info)
            
            # Check if alert should be triggered
            if self._should_trigger_alert(error_info):
                error_info["alert_triggered"] = True
                self._trigger_alert(error_info)
            
            return error_info
            
        except Exception as e:
            logger.error(f"Error detection failed: {str(e)}")
            return {"error": "Error detection failed"}
    
    def _assess_severity(self, error: Exception) -> str:
        """Assess error severity."""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Critical errors
        if any(keyword in error_msg for keyword in ["system", "memory", "disk", "database"]):
            return "CRITICAL"
        
        # High severity
        if any(keyword in error_msg for keyword in ["token", "authentication", "connection"]):
            return "HIGH"
        
        # Medium severity
        if any(keyword in error_msg for keyword in ["timeout", "rate", "limit"]):
            return "MEDIUM"
        
        # Low severity (default)
        return "LOW"
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error type."""
        error_msg = str(error).lower()
        
        if "token" in error_msg or "auth" in error_msg:
            return "AUTHENTICATION"
        elif "network" in error_msg or "connection" in error_msg:
            return "NETWORK"
        elif "rate" in error_msg or "limit" in error_msg:
            return "RATE_LIMIT"
        elif "data" in error_msg or "json" in error_msg:
            return "DATA"
        elif "timeout" in error_msg:
            return "TIMEOUT"
        else:
            return "GENERAL"
    
    def _get_recovery_suggestions(self, error: Exception) -> List[str]:
        """Get recovery suggestions based on error type."""
        error_msg = str(error).lower()
        suggestions = []
        
        if "token" in error_msg:
            suggestions.extend([
                "Refresh authentication token",
                "Check API credentials",
                "Verify token permissions"
            ])
        elif "network" in error_msg:
            suggestions.extend([
                "Check internet connection",
                "Verify API endpoint availability",
                "Try alternative endpoints"
            ])
        elif "rate" in error_msg:
            suggestions.extend([
                "Reduce request frequency",
                "Implement exponential backoff",
                "Use batch operations"
            ])
        else:
            suggestions.extend([
                "Check application logs",
                "Retry with exponential backoff",
                "Contact support if issue persists"
            ])
        
        return suggestions
    
    def _should_trigger_alert(self, error_info: Dict[str, Any]) -> bool:
        """Determine if alert should be triggered."""
        severity = error_info.get("severity", "LOW")
        
        # Always alert on critical errors
        if severity == "CRITICAL":
            return True
        
        # Check error rate
        recent_errors = [
            e for e in self.error_history
            if datetime.fromisoformat(e["timestamp"]) > datetime.now() - timedelta(minutes=1)
        ]
        
        if len(recent_errors) > self.alert_thresholds["error_rate_per_minute"]:
            return True
        
        # Check consecutive errors
        if len(self.error_history) >= self.alert_thresholds["consecutive_errors"]:
            recent_consecutive = self.error_history[-self.alert_thresholds["consecutive_errors"]:]
            if all(e.get("severity") in ["HIGH", "CRITICAL"] for e in recent_consecutive):
                return True
        
        return False
    
    def _trigger_alert(self, error_info: Dict[str, Any]):
        """Trigger alert for error."""
        logger.error(f"ALERT TRIGGERED: {error_info['severity']} error detected - {error_info['error_message']}")
        # Add alert notification logic here
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and statistics."""
        try:
            total_errors = len(self.error_history)
            
            if total_errors == 0:
                return {"total_errors": 0}
            
            # Error counts by severity
            severity_counts = {}
            category_counts = {}
            
            for error in self.error_history:
                severity = error.get("severity", "UNKNOWN")
                category = error.get("category", "UNKNOWN")
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Recent error rate (last hour)
            recent_errors = [
                e for e in self.error_history
                if datetime.fromisoformat(e["timestamp"]) > datetime.now() - timedelta(hours=1)
            ]
            
            return {
                "total_errors": total_errors,
                "recent_errors_last_hour": len(recent_errors),
                "severity_breakdown": severity_counts,
                "category_breakdown": category_counts,
                "last_error": self.error_history[-1] if self.error_history else None,
                "error_rate_per_hour": len(recent_errors),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error summary generation failed: {str(e)}")
            return {"error": "Summary generation failed"}
