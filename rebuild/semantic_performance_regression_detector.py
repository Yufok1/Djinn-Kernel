"""
Semantic Performance Regression Detector - Advanced performance monitoring
Provides statistical analysis and automated regression detection with AI-enhanced insights
"""

import uuid
import numpy as np
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict
import json

# Import kernel dependencies
from violation_pressure_calculation import ViolationMonitor
from temporal_isolation_safety import TemporalIsolationManager
from monitoring_observability import MonitoringObservability

# Import semantic components
from semantic_data_structures import (
    RegressionSeverity, SemanticHealth, FormationPattern,
    EvolutionStage, SemanticMetrics
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_checkpoint_manager import SemanticCheckpointManager

class RegressionTrigger(Enum):
    """Types of regression detection triggers"""
    PERFORMANCE_THRESHOLD = "performance_threshold"
    STATISTICAL_ANOMALY = "statistical_anomaly"
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PREDICTIVE_MODEL = "predictive_model"

@dataclass
class PerformanceMetric:
    """Individual performance metric with statistical properties"""
    metric_name: str
    current_value: float
    baseline_value: float
    historical_values: List[float] = field(default_factory=list)
    statistical_mean: float = 0.0
    statistical_std: float = 0.0
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    regression_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

@dataclass
class RegressionAlert:
    """Regression detection alert"""
    alert_id: uuid.UUID
    trigger_type: RegressionTrigger
    severity: RegressionSeverity
    affected_metrics: List[str]
    regression_score: float
    statistical_significance: float
    baseline_checkpoint: uuid.UUID
    current_checkpoint: uuid.UUID
    detection_timestamp: datetime
    recommended_actions: List[str] = field(default_factory=list)
    auto_rollback_eligible: bool = False

@dataclass
class TrendAnalysis:
    """Statistical trend analysis result"""
    metric_name: str
    trend_type: str  # "linear", "exponential", "logarithmic", "polynomial"
    trend_strength: float  # -1.0 to 1.0
    projection_horizon: timedelta
    projected_values: List[Tuple[datetime, float]] = field(default_factory=list)
    confidence_level: float = 0.95

@dataclass
class AnomalyDetection:
    """Statistical anomaly detection result"""
    metric_name: str
    anomaly_type: str  # "outlier", "shift", "drift", "spike"
    severity_score: float
    detection_method: str
    statistical_data: Dict[str, float] = field(default_factory=dict)

class SemanticPerformanceRegressionDetector:
    """
    Advanced performance regression detector with statistical analysis
    Provides automated monitoring, prediction, and intelligent alerting
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 checkpoint_manager: SemanticCheckpointManager,
                 violation_monitor: ViolationMonitor,
                 temporal_isolation: TemporalIsolationManager,
                 monitoring: Optional[MonitoringObservability] = None):
        """
        Initialize performance regression detector
        
        Args:
            state_manager: Semantic state manager
            event_bridge: Event bridge for coordination
            checkpoint_manager: Checkpoint manager for baselines
            violation_monitor: Violation monitoring system
            temporal_isolation: Temporal isolation manager
            monitoring: Optional monitoring system
        """
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.checkpoint_manager = checkpoint_manager
        self.violation_monitor = violation_monitor
        self.temporal_isolation = temporal_isolation
        self.monitoring = monitoring
        
        # Performance tracking
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_baselines: Dict[uuid.UUID, Dict[str, float]] = {}
        self.current_metrics: Dict[str, PerformanceMetric] = {}
        
        # Statistical analysis
        self.trend_analyses: Dict[str, TrendAnalysis] = {}
        self.anomaly_detections: List[AnomalyDetection] = []
        self.regression_alerts: List[RegressionAlert] = []
        
        # Detection configuration
        self.detection_thresholds = {
            "formation_success_rate": {"critical": 0.15, "severe": 0.10, "warning": 0.05},
            "semantic_accuracy": {"critical": 0.20, "severe": 0.12, "warning": 0.06},
            "violation_pressure": {"critical": 0.25, "severe": 0.15, "warning": 0.08},
            "mathematical_consistency": {"critical": 0.18, "severe": 0.12, "warning": 0.07},
            "formation_latency_ms": {"critical": 2.0, "severe": 1.5, "warning": 1.2},
            "system_coherence": {"critical": 0.15, "severe": 0.10, "warning": 0.05}
        }
        
        # Statistical parameters
        self.confidence_level = 0.95
        self.anomaly_sensitivity = 2.5  # Standard deviations
        self.trend_window = 50  # Number of samples for trend analysis
        self.prediction_horizon = timedelta(hours=24)
        
        # Alert management
        self.alert_cooldown = timedelta(minutes=30)
        self.recent_alerts: Dict[str, datetime] = {}
        self.auto_rollback_enabled = True
        self.rollback_threshold_severity = RegressionSeverity.SEVERE
        
        # Thread safety
        self._detector_lock = threading.RLock()
        
        # Monitoring setup
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self._monitor_thread.start()
        
        # Subscribe to relevant events
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for automatic monitoring"""
        # Subscribe to state updates
        self.event_bridge.subscribe_semantic_event(
            "SEMANTIC_STATE_UPDATED",
            self._handle_state_update
        )
        
        # Subscribe to checkpoint events
        self.event_bridge.subscribe_semantic_event(
            "CHECKPOINT_CREATED",
            self._handle_checkpoint_created
        )
        
        # Subscribe to formation events
        self.event_bridge.subscribe_semantic_event(
            "FORMATION_COMPLETED",
            self._handle_formation_completed
        )
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics and trigger analysis
        
        Args:
            metrics: Current performance metrics
        """
        with self._detector_lock:
            timestamp = datetime.utcnow()
            
            # Update metric history
            for metric_name, value in metrics.items():
                self.metric_history[metric_name].append((timestamp, value))
                
                # Update current metric object
                if metric_name not in self.current_metrics:
                    self.current_metrics[metric_name] = PerformanceMetric(
                        metric_name=metric_name,
                        current_value=value,
                        baseline_value=value
                    )
                else:
                    self.current_metrics[metric_name].current_value = value
                
                # Update statistical properties
                self._update_metric_statistics(metric_name)
            
            # Trigger regression analysis
            self._detect_regressions(metrics)
            
            # Update trend analysis
            self._update_trend_analyses()
            
            # Check for anomalies
            self._detect_anomalies(metrics)
    
    def _update_metric_statistics(self, metric_name: str) -> None:
        """Update statistical properties for a metric"""
        metric = self.current_metrics[metric_name]
        
        # Get recent values
        recent_values = [value for _, value in list(self.metric_history[metric_name])[-100:]]
        
        if len(recent_values) >= 2:
            metric.statistical_mean = statistics.mean(recent_values)
            metric.statistical_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
            
            # Calculate confidence interval
            if metric.statistical_std > 0:
                margin = 1.96 * metric.statistical_std / (len(recent_values) ** 0.5)
                metric.confidence_interval = (
                    metric.statistical_mean - margin,
                    metric.statistical_mean + margin
                )
        
        # Update historical values
        metric.historical_values = recent_values[-50:]  # Keep last 50 values
    
    def _detect_regressions(self, current_metrics: Dict[str, float]) -> None:
        """Detect performance regressions using multiple methods"""
        
        # Get current baseline
        baseline = self._get_current_baseline()
        if not baseline:
            return
        
        detected_regressions = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline:
                continue
            
            baseline_value = baseline[metric_name]
            
            # Calculate regression scores using different methods
            threshold_regression = self._check_threshold_regression(metric_name, current_value, baseline_value)
            statistical_regression = self._check_statistical_regression(metric_name, current_value)
            trend_regression = self._check_trend_regression(metric_name)
            
            # Determine overall regression severity
            max_severity = None
            triggers = []
            
            if threshold_regression:
                severity, trigger = threshold_regression
                if max_severity is None or severity.value > max_severity.value:
                    max_severity = severity
                triggers.append(trigger)
            
            if statistical_regression:
                severity, trigger = statistical_regression
                if max_severity is None or severity.value > max_severity.value:
                    max_severity = severity
                triggers.append(trigger)
            
            if trend_regression:
                severity, trigger = trend_regression
                if max_severity is None or severity.value > max_severity.value:
                    max_severity = severity
                triggers.append(trigger)
            
            # Create regression alert if significant
            if max_severity and max_severity != RegressionSeverity.WARNING:
                alert = self._create_regression_alert(
                    metric_name, max_severity, triggers, baseline_value, current_value
                )
                detected_regressions.append(alert)
        
        # Process detected regressions
        for alert in detected_regressions:
            self._process_regression_alert(alert)
    
    def _check_threshold_regression(self, metric_name: str, current_value: float, baseline_value: float) -> Optional[Tuple[RegressionSeverity, RegressionTrigger]]:
        """Check for regression based on threshold analysis"""
        if metric_name not in self.detection_thresholds:
            return None
        
        thresholds = self.detection_thresholds[metric_name]
        
        # Calculate regression percentage
        if baseline_value == 0:
            return None
        
        if metric_name == "formation_latency_ms":
            # For latency, increase is bad
            regression = (current_value - baseline_value) / baseline_value
        else:
            # For other metrics, decrease is bad
            regression = (baseline_value - current_value) / baseline_value
        
        # Determine severity
        if regression >= thresholds["critical"]:
            return RegressionSeverity.CRITICAL, RegressionTrigger.PERFORMANCE_THRESHOLD
        elif regression >= thresholds["severe"]:
            return RegressionSeverity.SEVERE, RegressionTrigger.PERFORMANCE_THRESHOLD
        elif regression >= thresholds["warning"]:
            return RegressionSeverity.WARNING, RegressionTrigger.PERFORMANCE_THRESHOLD
        
        return None
    
    def _check_statistical_regression(self, metric_name: str, current_value: float) -> Optional[Tuple[RegressionSeverity, RegressionTrigger]]:
        """Check for regression based on statistical analysis"""
        if metric_name not in self.current_metrics:
            return None
        
        metric = self.current_metrics[metric_name]
        
        if metric.statistical_std == 0:
            return None
        
        # Calculate z-score
        z_score = abs(current_value - metric.statistical_mean) / metric.statistical_std
        
        # Determine if this is a significant deviation
        if z_score >= 3.0:  # 3 sigma
            return RegressionSeverity.CRITICAL, RegressionTrigger.STATISTICAL_ANOMALY
        elif z_score >= 2.5:  # 2.5 sigma
            return RegressionSeverity.SEVERE, RegressionTrigger.STATISTICAL_ANOMALY
        elif z_score >= 2.0:  # 2 sigma
            return RegressionSeverity.WARNING, RegressionTrigger.STATISTICAL_ANOMALY
        
        return None
    
    def _check_trend_regression(self, metric_name: str) -> Optional[Tuple[RegressionSeverity, RegressionTrigger]]:
        """Check for regression based on trend analysis"""
        if metric_name not in self.trend_analyses:
            return None
        
        trend = self.trend_analyses[metric_name]
        
        # Check if trend is strongly negative
        if metric_name == "formation_latency_ms":
            # For latency, positive trend is bad
            if trend.trend_strength > 0.7:
                return RegressionSeverity.SEVERE, RegressionTrigger.TREND_ANALYSIS
            elif trend.trend_strength > 0.5:
                return RegressionSeverity.WARNING, RegressionTrigger.TREND_ANALYSIS
        else:
            # For other metrics, negative trend is bad
            if trend.trend_strength < -0.7:
                return RegressionSeverity.SEVERE, RegressionTrigger.TREND_ANALYSIS
            elif trend.trend_strength < -0.5:
                return RegressionSeverity.WARNING, RegressionTrigger.TREND_ANALYSIS
        
        return None
    
    def _update_trend_analyses(self) -> None:
        """Update trend analysis for all metrics"""
        for metric_name in self.current_metrics.keys():
            if len(self.metric_history[metric_name]) >= self.trend_window:
                trend = self._calculate_trend_analysis(metric_name)
                self.trend_analyses[metric_name] = trend
    
    def _calculate_trend_analysis(self, metric_name: str) -> TrendAnalysis:
        """Calculate trend analysis for a specific metric"""
        history = list(self.metric_history[metric_name])[-self.trend_window:]
        
        # Extract values and time indices
        values = [value for _, value in history]
        time_indices = list(range(len(values)))
        
        # Calculate linear trend
        if len(values) >= 2:
            correlation = np.corrcoef(time_indices, values)[0, 1]
            trend_strength = correlation
            
            # Determine trend type (simplified to linear for now)
            trend_type = "linear"
            
            # Project future values
            projected_values = []
            if abs(correlation) > 0.3:  # Significant trend
                # Simple linear projection
                slope = np.polyfit(time_indices, values, 1)[0]
                last_time = history[-1][0]
                
                for hours in range(1, 25):  # Next 24 hours
                    future_time = last_time + timedelta(hours=hours)
                    projected_value = values[-1] + slope * hours
                    projected_values.append((future_time, projected_value))
        else:
            trend_strength = 0.0
            trend_type = "stable"
            projected_values = []
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_type=trend_type,
            trend_strength=trend_strength,
            projection_horizon=self.prediction_horizon,
            projected_values=projected_values,
            confidence_level=self.confidence_level
        )
    
    def _detect_anomalies(self, current_metrics: Dict[str, float]) -> None:
        """Detect statistical anomalies in performance metrics"""
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.current_metrics:
                continue
            
            metric = self.current_metrics[metric_name]
            
            if len(metric.historical_values) >= 10 and metric.statistical_std > 0:
                # Z-score anomaly detection
                z_score = abs(current_value - metric.statistical_mean) / metric.statistical_std
                
                if z_score >= self.anomaly_sensitivity:
                    anomaly = AnomalyDetection(
                        metric_name=metric_name,
                        anomaly_type="outlier",
                        severity_score=z_score,
                        detection_method="z_score",
                        statistical_data={
                            "z_score": z_score,
                            "mean": metric.statistical_mean,
                            "std": metric.statistical_std,
                            "current_value": current_value
                        }
                    )
                    self.anomaly_detections.append(anomaly)
    
    def _create_regression_alert(self, metric_name: str, severity: RegressionSeverity, 
                                triggers: List[RegressionTrigger], baseline_value: float, 
                                current_value: float) -> RegressionAlert:
        """Create a regression alert"""
        
        # Calculate regression score
        regression_score = abs(current_value - baseline_value) / baseline_value if baseline_value > 0 else 1.0
        
        # Get current and baseline checkpoints
        current_checkpoint = self.checkpoint_manager.get_latest_checkpoint_id()
        baseline_checkpoint = self._get_baseline_checkpoint_id()
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(metric_name, severity, triggers)
        
        # Determine auto-rollback eligibility
        auto_rollback_eligible = (
            self.auto_rollback_enabled and 
            severity.value >= self.rollback_threshold_severity.value and
            baseline_checkpoint is not None
        )
        
        return RegressionAlert(
            alert_id=uuid.uuid4(),
            trigger_type=triggers[0] if triggers else RegressionTrigger.PERFORMANCE_THRESHOLD,
            severity=severity,
            affected_metrics=[metric_name],
            regression_score=regression_score,
            statistical_significance=self._calculate_statistical_significance(metric_name),
            baseline_checkpoint=baseline_checkpoint or uuid.uuid4(),
            current_checkpoint=current_checkpoint or uuid.uuid4(),
            detection_timestamp=datetime.utcnow(),
            recommended_actions=recommended_actions,
            auto_rollback_eligible=auto_rollback_eligible
        )
    
    def _generate_recommended_actions(self, metric_name: str, severity: RegressionSeverity, 
                                    triggers: List[RegressionTrigger]) -> List[str]:
        """Generate recommended actions for regression"""
        actions = []
        
        if severity == RegressionSeverity.CRITICAL:
            actions.append("IMMEDIATE: Stop all semantic operations")
            actions.append("IMMEDIATE: Initiate emergency rollback")
            actions.append("URGENT: Investigate root cause")
        elif severity == RegressionSeverity.SEVERE:
            actions.append("Create emergency checkpoint")
            actions.append("Consider rollback to last stable state")
            actions.append("Reduce semantic operation rate")
        else:
            actions.append("Monitor closely for further degradation")
            actions.append("Investigate potential causes")
            actions.append("Consider preventive checkpoint")
        
        # Metric-specific actions
        if metric_name == "formation_success_rate":
            actions.append("Review formation pattern quality")
            actions.append("Check semantic consistency")
        elif metric_name == "violation_pressure":
            actions.append("Increase VP monitoring frequency")
            actions.append("Activate additional safety protocols")
        
        return actions
    
    def _process_regression_alert(self, alert: RegressionAlert) -> None:
        """Process a regression alert"""
        with self._detector_lock:
            # Check cooldown
            if self._is_alert_in_cooldown(alert.affected_metrics[0]):
                return
            
            # Store alert
            self.regression_alerts.append(alert)
            self.recent_alerts[alert.affected_metrics[0]] = alert.detection_timestamp
            
            # Publish alert event
            self._publish_regression_alert(alert)
            
            # Execute automatic actions
            if alert.auto_rollback_eligible:
                self._execute_auto_rollback(alert)
            
            # Update monitoring
            if self.monitoring:
                self.monitoring.record_alert(
                    alert_type="performance_regression",
                    severity=alert.severity.value,
                    details=alert.__dict__
                )
    
    def _execute_auto_rollback(self, alert: RegressionAlert) -> None:
        """Execute automatic rollback for critical regression"""
        try:
            success = self.checkpoint_manager.execute_rollback(
                target_checkpoint_id=alert.baseline_checkpoint,
                reason=f"Auto-rollback: {alert.severity.value} regression in {alert.affected_metrics}",
                automatic=True
            )
            
            if success:
                # Publish rollback event
                self.event_bridge.publish_semantic_event({
                    "event_type": "AUTO_ROLLBACK_EXECUTED",
                    "alert_id": str(alert.alert_id),
                    "baseline_checkpoint": str(alert.baseline_checkpoint),
                    "regression_score": alert.regression_score,
                    "timestamp": datetime.utcnow().isoformat()
                })
        except Exception as e:
            print(f"Auto-rollback failed: {e}")
    
    def _publish_regression_alert(self, alert: RegressionAlert) -> None:
        """Publish regression alert event"""
        self.event_bridge.publish_semantic_event({
            "event_type": "PERFORMANCE_REGRESSION_DETECTED",
            "alert_id": str(alert.alert_id),
            "severity": alert.severity.value,
            "affected_metrics": alert.affected_metrics,
            "regression_score": alert.regression_score,
            "recommended_actions": alert.recommended_actions,
            "auto_rollback_eligible": alert.auto_rollback_eligible,
            "timestamp": alert.detection_timestamp.isoformat()
        })
    
    def _monitor_performance(self) -> None:
        """Background performance monitoring"""
        while self._monitoring_active:
            try:
                # Get current system health
                health = self.state_manager.get_current_health()
                
                # Convert health to metrics
                metrics = {
                    "formation_stability": health.formation_stability,
                    "semantic_stability": health.semantic_stability,
                    "system_coherence": health.system_coherence,
                    "evolution_progress": health.evolution_progress
                }
                
                # Update metrics
                self.update_performance_metrics(metrics)
                
                # Sleep
                threading.Event().wait(60.0)  # Check every minute
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
    
    def _get_current_baseline(self) -> Optional[Dict[str, float]]:
        """Get current performance baseline"""
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint_id()
        if latest_checkpoint:
            return self.checkpoint_manager.get_checkpoint_baseline(latest_checkpoint)
        return None
    
    def _get_baseline_checkpoint_id(self) -> Optional[uuid.UUID]:
        """Get baseline checkpoint ID"""
        return self.checkpoint_manager.get_latest_checkpoint_id()
    
    def _calculate_statistical_significance(self, metric_name: str) -> float:
        """Calculate statistical significance of current deviation"""
        if metric_name not in self.current_metrics:
            return 0.0
        
        metric = self.current_metrics[metric_name]
        
        if metric.statistical_std == 0:
            return 0.0
        
        z_score = abs(metric.current_value - metric.statistical_mean) / metric.statistical_std
        
        # Convert z-score to significance level (simplified)
        if z_score >= 3.0:
            return 0.999
        elif z_score >= 2.0:
            return 0.95
        elif z_score >= 1.0:
            return 0.68
        else:
            return 0.0
    
    def _is_alert_in_cooldown(self, metric_name: str) -> bool:
        """Check if alert is in cooldown period"""
        if metric_name not in self.recent_alerts:
            return False
        
        last_alert = self.recent_alerts[metric_name]
        return datetime.utcnow() - last_alert < self.alert_cooldown
    
    def _handle_state_update(self, event: Dict[str, Any]) -> None:
        """Handle semantic state update event"""
        # Extract metrics from state update if available
        if "performance_metrics" in event.get("payload", {}):
            metrics = event["payload"]["performance_metrics"]
            self.update_performance_metrics(metrics)
    
    def _handle_checkpoint_created(self, event: Dict[str, Any]) -> None:
        """Handle checkpoint creation event"""
        # Update performance baseline
        checkpoint_id = uuid.UUID(event.get("checkpoint_id", ""))
        if checkpoint_id:
            current_state = self.state_manager.current_state
            performance_metrics = {
                "formation_success_rate": current_state.get("formation_success_rate", 0.0),
                "semantic_accuracy": current_state.get("semantic_accuracy", 0.0),
                "system_coherence": current_state.get("system_coherence", 1.0)
            }
            self.performance_baselines[checkpoint_id] = performance_metrics
    
    def _handle_formation_completed(self, event: Dict[str, Any]) -> None:
        """Handle formation completion event"""
        # Extract formation metrics
        payload = event.get("payload", {})
        metrics = {
            "formation_latency_ms": payload.get("latency_ms", 0.0),
            "violation_pressure": payload.get("violation_pressure", 0.0),
            "mathematical_consistency": payload.get("mathematical_consistency", 1.0)
        }
        
        # Filter out zero/invalid values
        valid_metrics = {k: v for k, v in metrics.items() if v > 0}
        
        if valid_metrics:
            self.update_performance_metrics(valid_metrics)
    
    def get_regression_summary(self) -> Dict[str, Any]:
        """Get comprehensive regression summary"""
        with self._detector_lock:
            recent_alerts = [alert for alert in self.regression_alerts 
                           if (datetime.utcnow() - alert.detection_timestamp) < timedelta(hours=24)]
            
            return {
                "total_alerts": len(self.regression_alerts),
                "recent_alerts": len(recent_alerts),
                "critical_alerts": len([a for a in recent_alerts if a.severity == RegressionSeverity.CRITICAL]),
                "severe_alerts": len([a for a in recent_alerts if a.severity == RegressionSeverity.SEVERE]),
                "active_trends": len([t for t in self.trend_analyses.values() if abs(t.trend_strength) > 0.3]),
                "recent_anomalies": len([a for a in self.anomaly_detections if 
                                       (datetime.utcnow() - datetime.utcnow()) < timedelta(hours=1)]),
                "auto_rollbacks_executed": len([a for a in recent_alerts if a.auto_rollback_eligible]),
                "monitored_metrics": list(self.current_metrics.keys())
            }
    
    def get_trend_predictions(self, metric_name: str) -> Optional[List[Tuple[datetime, float]]]:
        """Get trend predictions for a specific metric"""
        if metric_name in self.trend_analyses:
            return self.trend_analyses[metric_name].projected_values
        return None
    
    def shutdown(self) -> None:
        """Shutdown performance regression detector"""
        self._monitoring_active = False
        self._monitor_thread.join(timeout=5)
        
        # Final summary
        summary = self.get_regression_summary()
        print(f"Performance Regression Detector shutdown. Summary: {summary}")
