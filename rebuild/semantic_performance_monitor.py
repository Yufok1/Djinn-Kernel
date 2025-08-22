# rebuild/semantic_performance_monitor.py
"""
Semantic Performance Monitor - Observability and Health Tracking
Provides dedicated performance monitoring for semantic capabilities and contributes to kernel Reflection Index
"""

import uuid
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import asyncio
import statistics
import math

# Import kernel dependencies
from uuid_anchor_mechanism import UUIDanchor
from event_driven_coordination import DjinnEventBus
from violation_pressure_calculation import ViolationMonitor
from trait_convergence_engine import TraitConvergenceEngine

# Import semantic components
from semantic_data_structures import (
    SemanticTrait, MathematicalTrait, FormationPattern,
    SemanticComplexity, TraitCategory, SemanticViolation, SemanticEvent
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager
from semantic_transcendence import SemanticTranscendence, TranscendenceLevel, LearningStrategy
from semantic_codex_amendment import SemanticCodexAmendment, AmendmentStatus

class PerformanceMetric(Enum):
    """Types of semantic performance metrics"""
    SEMANTIC_ACCURACY = "semantic_accuracy"           # Accuracy of semantic understanding
    LEARNING_VELOCITY = "learning_velocity"           # Speed of semantic learning
    TRANSCENDENCE_PROGRESS = "transcendence_progress" # Progress toward transcendence
    FOUNDATION_UTILIZATION = "foundation_utilization" # Usage of semantic foundation
    GOVERNANCE_EFFICIENCY = "governance_efficiency"   # Efficiency of governance processes
    EVOLUTION_STABILITY = "evolution_stability"       # Stability of evolutionary processes
    REFLECTION_INDEX = "reflection_index"             # Overall semantic reflection index

class HealthStatus(Enum):
    """Health status levels"""
    OPTIMAL = "optimal"           # Peak performance
    HEALTHY = "healthy"           # Good performance
    DEGRADED = "degraded"         # Reduced performance
    CRITICAL = "critical"         # Poor performance
    EMERGENCY = "emergency"       # System failure

@dataclass
class SemanticPerformanceSnapshot:
    """Snapshot of semantic performance at a point in time"""
    snapshot_id: uuid.UUID
    timestamp: datetime
    semantic_accuracy: float
    learning_velocity: float
    transcendence_progress: float
    foundation_utilization: float
    governance_efficiency: float
    evolution_stability: float
    reflection_index: float
    health_status: HealthStatus
    performance_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    system_alerts: List[str] = field(default_factory=list)

@dataclass
class PerformanceTrend:
    """Trend analysis for performance metrics"""
    metric_type: PerformanceMetric
    current_value: float
    trend_direction: str  # "improving", "stable", "declining"
    trend_magnitude: float
    prediction_horizon: int  # hours
    confidence_level: float
    contributing_factors: List[str] = field(default_factory=list)

@dataclass
class ReflectionIndex:
    """Kernel's global reflection index incorporating semantic performance"""
    index_id: uuid.UUID
    timestamp: datetime
    overall_score: float
    semantic_component: float
    mathematical_component: float
    governance_component: float
    evolution_component: float
    stability_score: float
    coherence_score: float
    transcendence_score: float
    health_indicators: Dict[str, HealthStatus] = field(default_factory=dict)
    performance_insights: List[str] = field(default_factory=list)

@dataclass
class PerformanceAlert:
    """Alert for performance issues"""
    alert_id: uuid.UUID
    alert_type: str
    severity: str  # "info", "warning", "critical", "emergency"
    message: str
    affected_metrics: List[PerformanceMetric]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False

class SemanticPerformanceMonitor:
    """
    Semantic Performance Monitor
    Provides comprehensive performance tracking and observability for semantic capabilities
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 checkpoint_manager: SemanticCheckpointManager,
                 transcendence_engine: SemanticTranscendence,
                 governance_system: SemanticCodexAmendment,
                 uuid_anchor: UUIDanchor,
                 trait_convergence: TraitConvergenceEngine):
        
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.checkpoint_manager = checkpoint_manager
        self.transcendence_engine = transcendence_engine
        self.governance_system = governance_system
        self.uuid_anchor = uuid_anchor
        self.trait_convergence = trait_convergence
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.current_snapshot: Optional[SemanticPerformanceSnapshot] = None
        self.performance_trends: Dict[PerformanceMetric, PerformanceTrend] = {}
        
        # Reflection index
        self.reflection_index_history: deque = deque(maxlen=100)
        self.current_reflection_index: Optional[ReflectionIndex] = None
        
        # Alerting system
        self.active_alerts: Dict[uuid.UUID, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        
        # Performance thresholds
        self.performance_thresholds = {
            PerformanceMetric.SEMANTIC_ACCURACY: {'warning': 0.7, 'critical': 0.5},
            PerformanceMetric.LEARNING_VELOCITY: {'warning': 0.3, 'critical': 0.1},
            PerformanceMetric.TRANSCENDENCE_PROGRESS: {'warning': 0.2, 'critical': 0.05},
            PerformanceMetric.FOUNDATION_UTILIZATION: {'warning': 0.9, 'critical': 0.95},
            PerformanceMetric.GOVERNANCE_EFFICIENCY: {'warning': 0.6, 'critical': 0.4},
            PerformanceMetric.EVOLUTION_STABILITY: {'warning': 0.7, 'critical': 0.5},
            PerformanceMetric.REFLECTION_INDEX: {'warning': 0.6, 'critical': 0.4}
        }
        
        # Monitoring state
        self.monitoring_active = True
        self.monitoring_interval = 30  # seconds
        
        # Thread safety
        self._monitor_lock = threading.RLock()
        
        # Background monitoring
        self._monitor_thread = threading.Thread(target=self._continuous_monitoring, daemon=True)
        self._monitor_thread.start()
        
        # Register event handlers
        self.event_bridge.register_handler("SEMANTIC_PERFORMANCE_UPDATE", self._handle_performance_update)
        self.event_bridge.register_handler("TRANSCENDENCE_LEVEL_CHANGE", self._handle_transcendence_change)
        self.event_bridge.register_handler("GOVERNANCE_AMENDMENT_COMPLETED", self._handle_governance_change)
        self.event_bridge.register_handler("SEMANTIC_VIOLATION_DETECTED", self._handle_violation)
        
        print(f"📊 SemanticPerformanceMonitor initialized with {len(self.performance_thresholds)} metrics")
    
    def capture_performance_snapshot(self) -> SemanticPerformanceSnapshot:
        """
        Capture current performance snapshot
        
        Returns:
            Current performance snapshot
        """
        with self._monitor_lock:
            # Gather performance data from all components
            semantic_accuracy = self._calculate_semantic_accuracy()
            learning_velocity = self._calculate_learning_velocity()
            transcendence_progress = self._calculate_transcendence_progress()
            foundation_utilization = self._calculate_foundation_utilization()
            governance_efficiency = self._calculate_governance_efficiency()
            evolution_stability = self._calculate_evolution_stability()
            
            # Calculate reflection index
            reflection_index = self._calculate_reflection_index(
                semantic_accuracy, learning_velocity, transcendence_progress,
                foundation_utilization, governance_efficiency, evolution_stability
            )
            
            # Determine health status
            health_status = self._determine_health_status(reflection_index)
            
            # Check for anomalies
            performance_anomalies = self._detect_performance_anomalies(
                semantic_accuracy, learning_velocity, transcendence_progress,
                foundation_utilization, governance_efficiency, evolution_stability
            )
            
            # Generate alerts
            system_alerts = self._generate_system_alerts(
                semantic_accuracy, learning_velocity, transcendence_progress,
                foundation_utilization, governance_efficiency, evolution_stability
            )
            
            # Create snapshot
            snapshot = SemanticPerformanceSnapshot(
                snapshot_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                semantic_accuracy=semantic_accuracy,
                learning_velocity=learning_velocity,
                transcendence_progress=transcendence_progress,
                foundation_utilization=foundation_utilization,
                governance_efficiency=governance_efficiency,
                evolution_stability=evolution_stability,
                reflection_index=reflection_index,
                health_status=health_status,
                performance_anomalies=performance_anomalies,
                system_alerts=system_alerts
            )
            
            # Store snapshot
            self.current_snapshot = snapshot
            self.performance_history.append(snapshot)
            
            # Update trends
            self._update_performance_trends(snapshot)
            
            # Update reflection index
            self._update_reflection_index(snapshot)
            
            # Publish performance event
            self._publish_performance_event(snapshot)
            
            return snapshot
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status"""
        with self._monitor_lock:
            if not self.current_snapshot:
                self.capture_performance_snapshot()
            
            return {
                'current_snapshot': asdict(self.current_snapshot),
                'performance_trends': {metric.value: asdict(trend) for metric, trend in self.performance_trends.items()},
                'reflection_index': asdict(self.current_reflection_index) if self.current_reflection_index else None,
                'active_alerts': len(self.active_alerts),
                'health_status': self.current_snapshot.health_status.value if self.current_snapshot else 'unknown'
            }
    
    def get_reflection_index(self) -> ReflectionIndex:
        """Get current reflection index"""
        with self._monitor_lock:
            if not self.current_reflection_index:
                self.capture_performance_snapshot()
            return self.current_reflection_index
    
    def get_performance_trends(self, metric_type: PerformanceMetric = None) -> Dict[str, Any]:
        """Get performance trends"""
        with self._monitor_lock:
            if metric_type:
                return asdict(self.performance_trends.get(metric_type))
            else:
                return {metric.value: asdict(trend) for metric, trend in self.performance_trends.items()}
    
    def acknowledge_alert(self, alert_id: uuid.UUID) -> bool:
        """Acknowledge a performance alert"""
        with self._monitor_lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                return True
            return False
    
    def resolve_alert(self, alert_id: uuid.UUID) -> bool:
        """Resolve a performance alert"""
        with self._monitor_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                return True
            return False
    
    def _calculate_semantic_accuracy(self) -> float:
        """Calculate semantic accuracy"""
        # Analyze semantic trait quality and coherence
        try:
            # Get recent semantic operations
            recent_operations = len(self.performance_history) if self.performance_history else 1
            
            # Simulate accuracy based on system state
            base_accuracy = 0.85
            learning_bonus = min(0.1, recent_operations * 0.001)
            foundation_penalty = 0.05 if self._is_over_dependent_on_foundation() else 0.0
            
            accuracy = base_accuracy + learning_bonus - foundation_penalty
            return min(1.0, max(0.0, accuracy))
        except Exception as e:
            print(f"Error calculating semantic accuracy: {e}")
            return 0.5
    
    def _calculate_learning_velocity(self) -> float:
        """Calculate learning velocity"""
        try:
            # Get transcendence engine status
            transcendence_status = self.transcendence_engine.get_evolution_status()
            return transcendence_status['evolution_state']['learning_velocity']
        except Exception as e:
            print(f"Error calculating learning velocity: {e}")
            return 0.0
    
    def _calculate_transcendence_progress(self) -> float:
        """Calculate transcendence progress"""
        try:
            # Get transcendence engine status
            transcendence_status = self.transcendence_engine.get_evolution_status()
            independence_score = transcendence_status['evolution_state']['independence_score']
            synthesis_capability = transcendence_status['evolution_state']['synthesis_capability']
            
            # Combine independence and synthesis
            progress = (independence_score * 0.6) + (synthesis_capability * 0.4)
            return min(1.0, max(0.0, progress))
        except Exception as e:
            print(f"Error calculating transcendence progress: {e}")
            return 0.0
    
    def _calculate_foundation_utilization(self) -> float:
        """Calculate foundation utilization"""
        try:
            # Get transcendence engine status
            transcendence_status = self.transcendence_engine.get_evolution_status()
            return transcendence_status['evolution_state']['foundation_dependency']
        except Exception as e:
            print(f"Error calculating foundation utilization: {e}")
            return 0.5
    
    def _calculate_governance_efficiency(self) -> float:
        """Calculate governance efficiency"""
        try:
            # Get governance system status
            governance_status = self.governance_system.get_constitutional_status()
            
            # Calculate efficiency based on active amendments and metrics
            active_amendments = governance_status['active_amendments']
            total_amendments = governance_status['governance_metrics']['total_amendments_proposed']
            
            if total_amendments == 0:
                return 1.0
            
            # Efficiency decreases with too many active amendments
            efficiency = 1.0 - (active_amendments / max(total_amendments, 1))
            return min(1.0, max(0.0, efficiency))
        except Exception as e:
            print(f"Error calculating governance efficiency: {e}")
            return 0.5
    
    def _calculate_evolution_stability(self) -> float:
        """Calculate evolution stability"""
        try:
            # Analyze recent performance history for stability
            if len(self.performance_history) < 2:
                return 0.8  # Default stability for new systems
            
            recent_snapshots = list(self.performance_history)[-5:]
            reflection_indices = [snapshot.reflection_index for snapshot in recent_snapshots]
            
            if len(reflection_indices) < 2:
                return 0.8
            
            # Calculate coefficient of variation (lower is more stable)
            mean_ri = statistics.mean(reflection_indices)
            std_ri = statistics.stdev(reflection_indices) if len(reflection_indices) > 1 else 0
            
            if mean_ri == 0:
                return 0.8
            
            cv = std_ri / mean_ri
            stability = max(0.0, 1.0 - cv)
            return min(1.0, stability)
        except Exception as e:
            print(f"Error calculating evolution stability: {e}")
            return 0.5
    
    def _calculate_reflection_index(self, 
                                  semantic_accuracy: float,
                                  learning_velocity: float,
                                  transcendence_progress: float,
                                  foundation_utilization: float,
                                  governance_efficiency: float,
                                  evolution_stability: float) -> float:
        """Calculate overall reflection index"""
        # Weighted combination of all metrics
        weights = {
            'semantic_accuracy': 0.25,
            'learning_velocity': 0.20,
            'transcendence_progress': 0.20,
            'foundation_utilization': 0.15,
            'governance_efficiency': 0.10,
            'evolution_stability': 0.10
        }
        
        reflection_index = (
            semantic_accuracy * weights['semantic_accuracy'] +
            learning_velocity * weights['learning_velocity'] +
            transcendence_progress * weights['transcendence_progress'] +
            (1.0 - foundation_utilization) * weights['foundation_utilization'] +  # Inverted
            governance_efficiency * weights['governance_efficiency'] +
            evolution_stability * weights['evolution_stability']
        )
        
        return min(1.0, max(0.0, reflection_index))
    
    def _determine_health_status(self, reflection_index: float) -> HealthStatus:
        """Determine overall health status"""
        if reflection_index >= 0.8:
            return HealthStatus.OPTIMAL
        elif reflection_index >= 0.6:
            return HealthStatus.HEALTHY
        elif reflection_index >= 0.4:
            return HealthStatus.DEGRADED
        elif reflection_index >= 0.2:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.EMERGENCY
    
    def _detect_performance_anomalies(self, 
                                    semantic_accuracy: float,
                                    learning_velocity: float,
                                    transcendence_progress: float,
                                    foundation_utilization: float,
                                    governance_efficiency: float,
                                    evolution_stability: float) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        anomalies = []
        
        # Check each metric against thresholds
        metrics = {
            PerformanceMetric.SEMANTIC_ACCURACY: semantic_accuracy,
            PerformanceMetric.LEARNING_VELOCITY: learning_velocity,
            PerformanceMetric.TRANSCENDENCE_PROGRESS: transcendence_progress,
            PerformanceMetric.FOUNDATION_UTILIZATION: foundation_utilization,
            PerformanceMetric.GOVERNANCE_EFFICIENCY: governance_efficiency,
            PerformanceMetric.EVOLUTION_STABILITY: evolution_stability
        }
        
        for metric, value in metrics.items():
            thresholds = self.performance_thresholds[metric]
            
            if value <= thresholds['critical']:
                anomalies.append({
                    'metric': metric.value,
                    'severity': 'critical',
                    'value': value,
                    'threshold': thresholds['critical'],
                    'description': f'{metric.value} below critical threshold'
                })
            elif value <= thresholds['warning']:
                anomalies.append({
                    'metric': metric.value,
                    'severity': 'warning',
                    'value': value,
                    'threshold': thresholds['warning'],
                    'description': f'{metric.value} below warning threshold'
                })
        
        return anomalies
    
    def _generate_system_alerts(self,
                              semantic_accuracy: float,
                              learning_velocity: float,
                              transcendence_progress: float,
                              foundation_utilization: float,
                              governance_efficiency: float,
                              evolution_stability: float) -> List[str]:
        """Generate system alerts"""
        alerts = []
        
        # Check for critical issues
        if semantic_accuracy < 0.5:
            alerts.append("Critical: Semantic accuracy below 50%")
        
        if learning_velocity < 0.1:
            alerts.append("Warning: Learning velocity very low")
        
        if foundation_utilization > 0.95:
            alerts.append("Warning: Over-dependent on semantic foundation")
        
        if governance_efficiency < 0.4:
            alerts.append("Critical: Governance efficiency compromised")
        
        if evolution_stability < 0.5:
            alerts.append("Warning: Evolution stability declining")
        
        return alerts
    
    def _update_performance_trends(self, snapshot: SemanticPerformanceSnapshot):
        """Update performance trends"""
        metrics = {
            PerformanceMetric.SEMANTIC_ACCURACY: snapshot.semantic_accuracy,
            PerformanceMetric.LEARNING_VELOCITY: snapshot.learning_velocity,
            PerformanceMetric.TRANSCENDENCE_PROGRESS: snapshot.transcendence_progress,
            PerformanceMetric.FOUNDATION_UTILIZATION: snapshot.foundation_utilization,
            PerformanceMetric.GOVERNANCE_EFFICIENCY: snapshot.governance_efficiency,
            PerformanceMetric.EVOLUTION_STABILITY: snapshot.evolution_stability,
            PerformanceMetric.REFLECTION_INDEX: snapshot.reflection_index
        }
        
        for metric_type, current_value in metrics.items():
            # Get recent values for trend calculation
            recent_values = []
            for hist_snapshot in list(self.performance_history)[-10:]:
                if metric_type == PerformanceMetric.SEMANTIC_ACCURACY:
                    recent_values.append(hist_snapshot.semantic_accuracy)
                elif metric_type == PerformanceMetric.LEARNING_VELOCITY:
                    recent_values.append(hist_snapshot.learning_velocity)
                elif metric_type == PerformanceMetric.TRANSCENDENCE_PROGRESS:
                    recent_values.append(hist_snapshot.transcendence_progress)
                elif metric_type == PerformanceMetric.FOUNDATION_UTILIZATION:
                    recent_values.append(hist_snapshot.foundation_utilization)
                elif metric_type == PerformanceMetric.GOVERNANCE_EFFICIENCY:
                    recent_values.append(hist_snapshot.governance_efficiency)
                elif metric_type == PerformanceMetric.EVOLUTION_STABILITY:
                    recent_values.append(hist_snapshot.evolution_stability)
                elif metric_type == PerformanceMetric.REFLECTION_INDEX:
                    recent_values.append(hist_snapshot.reflection_index)
            
            if len(recent_values) >= 2:
                # Calculate trend
                trend_direction = "stable"
                trend_magnitude = 0.0
                
                if len(recent_values) >= 3:
                    # Simple linear trend
                    first_half = recent_values[:len(recent_values)//2]
                    second_half = recent_values[len(recent_values)//2:]
                    
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    
                    if second_avg > first_avg * 1.05:
                        trend_direction = "improving"
                        trend_magnitude = (second_avg - first_avg) / first_avg
                    elif second_avg < first_avg * 0.95:
                        trend_direction = "declining"
                        trend_magnitude = (first_avg - second_avg) / first_avg
                
                # Create trend
                trend = PerformanceTrend(
                    metric_type=metric_type,
                    current_value=current_value,
                    trend_direction=trend_direction,
                    trend_magnitude=trend_magnitude,
                    prediction_horizon=24,  # 24 hours
                    confidence_level=0.7,
                    contributing_factors=self._identify_contributing_factors(metric_type, current_value)
                )
                
                self.performance_trends[metric_type] = trend
    
    def _update_reflection_index(self, snapshot: SemanticPerformanceSnapshot):
        """Update reflection index"""
        # Calculate component scores
        semantic_component = snapshot.semantic_accuracy
        mathematical_component = snapshot.evolution_stability
        governance_component = snapshot.governance_efficiency
        evolution_component = snapshot.transcendence_progress
        
        # Calculate derived scores
        stability_score = snapshot.evolution_stability
        coherence_score = snapshot.semantic_accuracy
        transcendence_score = snapshot.transcendence_progress
        
        # Create reflection index
        reflection_index = ReflectionIndex(
            index_id=uuid.uuid4(),
            timestamp=snapshot.timestamp,
            overall_score=snapshot.reflection_index,
            semantic_component=semantic_component,
            mathematical_component=mathematical_component,
            governance_component=governance_component,
            evolution_component=evolution_component,
            stability_score=stability_score,
            coherence_score=coherence_score,
            transcendence_score=transcendence_score,
            health_indicators={
                'semantic_health': snapshot.health_status,
                'evolution_health': HealthStatus.HEALTHY if snapshot.evolution_stability > 0.7 else HealthStatus.DEGRADED,
                'governance_health': HealthStatus.HEALTHY if snapshot.governance_efficiency > 0.6 else HealthStatus.DEGRADED
            },
            performance_insights=self._generate_performance_insights(snapshot)
        )
        
        self.current_reflection_index = reflection_index
        self.reflection_index_history.append(reflection_index)
    
    def _identify_contributing_factors(self, metric_type: PerformanceMetric, value: float) -> List[str]:
        """Identify factors contributing to metric value"""
        factors = []
        
        if metric_type == PerformanceMetric.SEMANTIC_ACCURACY:
            if value > 0.8:
                factors.append("Strong semantic foundation")
                factors.append("Effective learning patterns")
            elif value < 0.6:
                factors.append("Semantic foundation dependency")
                factors.append("Learning pattern instability")
        
        elif metric_type == PerformanceMetric.LEARNING_VELOCITY:
            if value > 0.5:
                factors.append("Active transcendence engine")
                factors.append("Efficient pattern discovery")
            elif value < 0.2:
                factors.append("Limited learning opportunities")
                factors.append("Foundation dependency")
        
        elif metric_type == PerformanceMetric.TRANSCENDENCE_PROGRESS:
            if value > 0.5:
                factors.append("Successful knowledge synthesis")
                factors.append("Growing independence")
            elif value < 0.2:
                factors.append("Foundation dependency")
                factors.append("Limited autonomous synthesis")
        
        return factors
    
    def _generate_performance_insights(self, snapshot: SemanticPerformanceSnapshot) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        if snapshot.semantic_accuracy > 0.8:
            insights.append("Semantic understanding is highly accurate")
        
        if snapshot.learning_velocity > 0.5:
            insights.append("Learning velocity is strong")
        
        if snapshot.transcendence_progress > 0.5:
            insights.append("Making good progress toward transcendence")
        
        if snapshot.foundation_utilization < 0.3:
            insights.append("Successfully reducing foundation dependency")
        
        if snapshot.governance_efficiency > 0.8:
            insights.append("Governance system is highly efficient")
        
        if snapshot.evolution_stability > 0.8:
            insights.append("Evolutionary processes are stable")
        
        return insights
    
    def _is_over_dependent_on_foundation(self) -> bool:
        """Check if system is over-dependent on foundation"""
        if not self.current_snapshot:
            return False
        return self.current_snapshot.foundation_utilization > 0.9
    
    def _publish_performance_event(self, snapshot: SemanticPerformanceSnapshot):
        """Publish performance event"""
        try:
            performance_event = SemanticEvent(
                event_uuid=uuid.uuid4(),
                event_type='SEMANTIC_PERFORMANCE_UPDATE',
                timestamp=snapshot.timestamp,
                payload={
                    'reflection_index': snapshot.reflection_index,
                    'health_status': snapshot.health_status.value,
                    'semantic_accuracy': snapshot.semantic_accuracy,
                    'learning_velocity': snapshot.learning_velocity,
                    'transcendence_progress': snapshot.transcendence_progress
                }
            )
            self.event_bridge.publish_semantic_event(performance_event)
        except Exception as e:
            print(f"Error publishing performance event: {e}")
    
    def _continuous_monitoring(self):
        """Continuous performance monitoring"""
        while self.monitoring_active:
            try:
                # Capture performance snapshot
                self.capture_performance_snapshot()
                
                # Check for alerts
                self._check_for_alerts()
                
                # Sleep for monitoring interval
                import time
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in continuous monitoring: {e}")
                import time
                time.sleep(60)  # Longer sleep on error
    
    def _check_for_alerts(self):
        """Check for performance alerts"""
        if not self.current_snapshot:
            return
        
        # Check for critical anomalies
        for anomaly in self.current_snapshot.performance_anomalies:
            if anomaly['severity'] == 'critical':
                alert = PerformanceAlert(
                    alert_id=uuid.uuid4(),
                    alert_type='performance_critical',
                    severity='critical',
                    message=anomaly['description'],
                    affected_metrics=[PerformanceMetric(anomaly['metric'])],
                    recommended_actions=['Investigate performance degradation', 'Check system resources']
                )
                self.active_alerts[alert.alert_id] = alert
    
    def _handle_performance_update(self, event_data: Dict[str, Any]):
        """Handle performance update events"""
        print(f"📊 Performance update received: RI={event_data.get('reflection_index', 'N/A'):.2f}")
    
    def _handle_transcendence_change(self, event_data: Dict[str, Any]):
        """Handle transcendence change events"""
        print(f"📊 Transcendence change detected: {event_data.get('new_level', 'unknown')}")
    
    def _handle_governance_change(self, event_data: Dict[str, Any]):
        """Handle governance change events"""
        print(f"📊 Governance change detected: {event_data.get('amendment_type', 'unknown')}")
    
    def _handle_violation(self, event_data: Dict[str, Any]):
        """Handle violation events"""
        print(f"📊 Violation detected: {event_data.get('violation_type', 'unknown')}")

def main():
    """Main function to demonstrate semantic performance monitoring"""
    print("📊 SEMANTIC PERFORMANCE MONITOR - OBSERVABILITY & HEALTH")
    print("=" * 60)
    
    # Setup mock dependencies (reuse from previous modules)
    class MockUUIDanchor:
        def anchor_trait(self, word):
            import uuid
            return uuid.uuid5(uuid.NAMESPACE_DNS, str(word))
    
    class MockDjinnEventBus:
        def __init__(self):
            self.events = []
            self.handlers = {}
            self.subscriptions = {}
        def register_handler(self, event_type, handler):
            self.handlers[event_type] = handler
        def subscribe(self, event_type, handler):
            self.subscriptions[event_type] = handler
        def publish(self, event_data):
            self.events.append(event_data)
    
    class MockViolationMonitor:
        def calculate_violation_pressure(self, trait_data):
            return 0.5
    
    class MockTraitConvergenceEngine:
        def calculate_convergence_stability(self, trait_data):
            return 0.7
    
    class MockTemporalIsolationManager:
        def create_isolation_context(self):
            return "test_context"
    
    class MockLocalSemanticDatabase:
        def __init__(self):
            self.traits = {}
        def get_trait(self, trait_id):
            return self.traits.get(trait_id)
        def store_trait(self, trait):
            self.traits[trait.trait_uuid] = trait
    
    class MockMathematicalSemanticAPI:
        def __init__(self):
            self.queries = []
        def query_semantic_database(self, query_type, query_data):
            self.queries.append((query_type, query_data))
            return {'result': 'mock_result', 'confidence': 0.8}
        def get_semantic_guidance(self, context):
            return {'guidance': 'mock_guidance', 'confidence': 0.7}
    
    # Setup components
    uuid_anchor = MockUUIDanchor()
    event_bus = MockDjinnEventBus()
    violation_monitor = MockViolationMonitor()
    trait_convergence = MockTraitConvergenceEngine()
    temporal_isolation = MockTemporalIsolationManager()
    semantic_database = MockLocalSemanticDatabase()
    semantic_api = MockMathematicalSemanticAPI()
    
    state_manager = SemanticStateManager(event_bus, uuid_anchor, violation_monitor)
    event_bridge = SemanticEventBridge(event_bus, state_manager, violation_monitor, temporal_isolation)
    semantic_violation_monitor = SemanticViolationMonitor(violation_monitor, temporal_isolation, state_manager, event_bridge)
    checkpoint_manager = SemanticCheckpointManager(state_manager, event_bridge, semantic_violation_monitor, uuid_anchor)
    
    transcendence_engine = SemanticTranscendence(
        state_manager, event_bridge, semantic_violation_monitor,
        checkpoint_manager, uuid_anchor, trait_convergence,
        semantic_database, semantic_api
    )
    
    governance_system = SemanticCodexAmendment(
        state_manager, event_bridge, semantic_violation_monitor,
        checkpoint_manager, transcendence_engine, uuid_anchor, trait_convergence
    )
    
    # Create performance monitor
    performance_monitor = SemanticPerformanceMonitor(
        state_manager, event_bridge, semantic_violation_monitor,
        checkpoint_manager, transcendence_engine, governance_system,
        uuid_anchor, trait_convergence
    )
    
    print("✅ Performance monitoring system initialized")
    
    # Demonstrate performance monitoring
    print("\n📊 Capturing performance snapshots...")
    
    for i in range(3):
        snapshot = performance_monitor.capture_performance_snapshot()
        print(f"Snapshot {i+1}:")
        print(f"   Reflection Index: {snapshot.reflection_index:.3f}")
        print(f"   Health Status: {snapshot.health_status.value}")
        print(f"   Semantic Accuracy: {snapshot.semantic_accuracy:.3f}")
        print(f"   Learning Velocity: {snapshot.learning_velocity:.3f}")
        print(f"   Alerts: {len(snapshot.system_alerts)}")
        
        import time
        time.sleep(2)
    
    # Get performance status
    status = performance_monitor.get_performance_status()
    print(f"\n📈 Performance Status:")
    print(f"   Overall Health: {status['health_status']}")
    print(f"   Active Alerts: {status['active_alerts']}")
    
    # Get reflection index
    reflection_index = performance_monitor.get_reflection_index()
    print(f"\n🔍 Reflection Index:")
    print(f"   Overall Score: {reflection_index.overall_score:.3f}")
    print(f"   Semantic Component: {reflection_index.semantic_component:.3f}")
    print(f"   Evolution Component: {reflection_index.evolution_component:.3f}")
    print(f"   Governance Component: {reflection_index.governance_component:.3f}")
    
    print("\n🎯 PERFORMANCE MONITORING ACTIVE!")
    print("The kernel now has comprehensive observability and health tracking!")

if __name__ == "__main__":
    main()
