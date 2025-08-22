"""
Semantic Violation Monitor - VP calculation extension for semantic operations
Monitors semantic stability and triggers safety mechanisms
"""

import uuid
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict

# Import kernel dependencies
from violation_pressure_calculation import ViolationMonitor, ViolationClass
from temporal_isolation_safety import TemporalIsolationManager
from event_driven_coordination import DjinnEventBus

# Import semantic components
from semantic_data_structures import (
    FormationPattern, SemanticViolation, SemanticHealth,
    RegressionSeverity, FormationType, EvolutionStage
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge

class SemanticViolationType(Enum):
    """Types of semantic violations"""
    FORMATION_FAILURE = "formation_failure"
    CONSISTENCY_VIOLATION = "consistency_violation"
    PATTERN_DIVERGENCE = "pattern_divergence"
    EVOLUTION_REGRESSION = "evolution_regression"
    INDEPENDENCE_LOSS = "independence_loss"
    COHERENCE_BREAKDOWN = "coherence_breakdown"
    STABILITY_VIOLATION = "stability_violation"

@dataclass
class SemanticVPMetrics:
    """Metrics for semantic violation pressure calculation"""
    formation_success_rate: float = 1.0
    mathematical_consistency: float = 1.0
    pattern_stability: float = 1.0
    evolution_coherence: float = 1.0
    independence_level: float = 0.0
    semantic_entropy: float = 0.0
    
    def calculate_composite_vp(self) -> float:
        """Calculate composite violation pressure"""
        # Core VP formula adapted for semantic operations
        # VP = (1 - success_rate) * (1 - consistency) * (1 + entropy) * (1 - stability)
        base_vp = (1 - self.formation_success_rate) * (1 - self.mathematical_consistency)
        entropy_factor = 1 + self.semantic_entropy
        stability_factor = 2 - self.pattern_stability - self.evolution_coherence
        
        return min(1.0, base_vp * entropy_factor * stability_factor)

class SemanticViolationMonitor:
    """
    Extended violation monitor for semantic operations
    Integrates with kernel's VP calculation system
    """
    
    def __init__(self,
                 violation_monitor: ViolationMonitor,
                 temporal_isolation: TemporalIsolationManager,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge):
        """
        Initialize semantic violation monitor
        
        Args:
            violation_monitor: Core kernel violation monitor
            temporal_isolation: Temporal isolation system
            state_manager: Semantic state manager
            event_bridge: Semantic event bridge
        """
        # Kernel integrations
        self.violation_monitor = violation_monitor
        self.temporal_isolation = temporal_isolation
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        
        # Violation tracking
        self.violations: Dict[uuid.UUID, SemanticViolation] = {}
        self.violation_history: deque = deque(maxlen=1000)
        self.violation_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Metrics tracking
        self.current_metrics = SemanticVPMetrics()
        self.metrics_history: deque = deque(maxlen=100)
        self.rolling_vp: deque = deque(maxlen=50)
        
        # Thresholds
        self.vp_thresholds = {
            "warning": 0.5,
            "critical": 0.7,
            "emergency": 0.9
        }
        
        # Pattern analysis
        self.pattern_buffer: deque = deque(maxlen=100)
        self.divergence_tracking: Dict[FormationType, float] = {}
        
        # Thread safety
        self._monitor_lock = threading.RLock()
        
        # Monitoring thread
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._continuous_monitoring, daemon=True)
        self._monitor_thread.start()
    
    def calculate_semantic_vp(self, pattern: FormationPattern) -> Tuple[float, SemanticVPMetrics]:
        """
        Calculate violation pressure for semantic formation
        
        Args:
            pattern: Formation pattern to analyze
            
        Returns:
            Tuple of (vp_value, metrics)
        """
        with self._monitor_lock:
            # Update pattern buffer
            self.pattern_buffer.append(pattern)
            
            # Calculate formation success rate
            recent_patterns = list(self.pattern_buffer)
            success_rate = sum(1 for p in recent_patterns if p.formation_success) / len(recent_patterns) if recent_patterns else 0
            
            # Calculate mathematical consistency
            consistency = pattern.mathematical_consistency
            
            # Calculate pattern stability
            stability = self._calculate_pattern_stability(pattern.formation_type)
            
            # Calculate evolution coherence
            coherence = self._calculate_evolution_coherence()
            
            # Get independence level from state
            independence = self.state_manager.current_state.get("independence_level", 0.0)
            
            # Calculate semantic entropy
            entropy = self._calculate_semantic_entropy(recent_patterns)
            
            # Create metrics
            metrics = SemanticVPMetrics(
                formation_success_rate=success_rate,
                mathematical_consistency=consistency,
                pattern_stability=stability,
                evolution_coherence=coherence,
                independence_level=independence,
                semantic_entropy=entropy
            )
            
            # Calculate composite VP
            vp = metrics.calculate_composite_vp()
            
            # Apply pattern-specific modifiers
            vp = self._apply_formation_modifiers(vp, pattern)
            
            # Update current metrics
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            self.rolling_vp.append(vp)
            
            # Check for violations
            if vp > self.vp_thresholds["warning"]:
                self._create_violation(pattern, vp, metrics)
            
            return vp, metrics
    
    def _calculate_pattern_stability(self, formation_type: FormationType) -> float:
        """Calculate stability for specific formation type"""
        if formation_type not in self.divergence_tracking:
            return 1.0
        
        divergence = self.divergence_tracking[formation_type]
        # Convert divergence to stability (inverse relationship)
        return max(0.0, 1.0 - divergence)
    
    def _calculate_evolution_coherence(self) -> float:
        """Calculate coherence of semantic evolution"""
        # Get recent health from state manager
        health = self.state_manager.get_current_health()
        
        # Coherence based on system health
        return health.system_coherence
    
    def _calculate_semantic_entropy(self, patterns: List[FormationPattern]) -> float:
        """Calculate entropy in semantic formations"""
        if not patterns:
            return 0.0
        
        # Calculate variance in VP values
        vp_values = [p.violation_pressure for p in patterns]
        if len(vp_values) < 2:
            return 0.0
        
        mean_vp = sum(vp_values) / len(vp_values)
        variance = sum((vp - mean_vp) ** 2 for vp in vp_values) / len(vp_values)
        
        # Normalize to 0-1 range
        entropy = min(1.0, math.sqrt(variance))
        
        return entropy
    
    def _apply_formation_modifiers(self, base_vp: float, pattern: FormationPattern) -> float:
        """Apply formation-specific modifiers to VP"""
        # Higher complexity formations get slight VP reduction
        complexity_modifier = {
            FormationType.CHARACTER: 1.0,
            FormationType.WORD: 0.95,
            FormationType.SENTENCE: 0.9,
            FormationType.DIALOGUE: 0.85
        }
        
        modifier = complexity_modifier.get(pattern.formation_type, 1.0)
        
        # Apply evolution stage modifier
        current_stage = self.state_manager.current_state.get("evolution_stage", EvolutionStage.INITIALIZATION.value)
        if current_stage == EvolutionStage.SEMANTIC_TRANSCENDENCE.value:
            modifier *= 0.8  # Reduce VP for advanced stages
        elif current_stage == EvolutionStage.INITIALIZATION.value:
            modifier *= 1.2  # Increase VP for early stages
        
        return min(1.0, base_vp * modifier)
    
    def _create_violation(self, 
                         pattern: FormationPattern,
                         vp: float,
                         metrics: SemanticVPMetrics) -> SemanticViolation:
        """Create and track semantic violation"""
        # Determine severity based on VP
        if vp > self.vp_thresholds["emergency"]:
            severity = RegressionSeverity.CRITICAL
        elif vp > self.vp_thresholds["critical"]:
            severity = RegressionSeverity.SEVERE
        else:
            severity = RegressionSeverity.WARNING
        
        # Determine violation type
        violation_type = self._determine_violation_type(pattern, metrics)
        
        # Create violation
        violation = SemanticViolation(
            violation_uuid=uuid.uuid4(),
            violation_type=violation_type,
            severity=severity,
            violation_pressure=vp,
            formation_pattern=pattern,
            timestamp=datetime.utcnow(),
            mathematical_consistency=metrics.mathematical_consistency
        )
        
        # Track violation
        self.violations[violation.violation_uuid] = violation
        self.violation_history.append(violation)
        self.violation_patterns[violation_type].append(vp)
        
        # Register with kernel violation monitor
        self.violation_monitor.add_violation(
            entity_id=str(violation.violation_uuid),
            violation_class=ViolationClass.SEMANTIC,
            severity=vp,
            description=f"Semantic {violation_type}: {severity.value}"
        )
        
        # Trigger safety response if needed
        if severity in [RegressionSeverity.SEVERE, RegressionSeverity.CRITICAL]:
            self._trigger_safety_response(violation)
        
        return violation
    
    def _determine_violation_type(self, 
                                 pattern: FormationPattern,
                                 metrics: SemanticVPMetrics) -> str:
        """Determine specific type of violation"""
        if not pattern.formation_success:
            return SemanticViolationType.FORMATION_FAILURE.value
        elif metrics.mathematical_consistency < 0.5:
            return SemanticViolationType.CONSISTENCY_VIOLATION.value
        elif metrics.pattern_stability < 0.3:
            return SemanticViolationType.PATTERN_DIVERGENCE.value
        elif metrics.evolution_coherence < 0.3:
            return SemanticViolationType.COHERENCE_BREAKDOWN.value
        else:
            return SemanticViolationType.STABILITY_VIOLATION.value
    
    def _trigger_safety_response(self, violation: SemanticViolation) -> None:
        """Trigger safety response for severe violations"""
        if violation.severity == RegressionSeverity.CRITICAL:
            # Immediate isolation
            self.temporal_isolation.isolate_operation(
                operation_id=str(violation.violation_uuid),
                reason=f"Critical semantic violation: {violation.violation_type}",
                duration=timedelta(minutes=10)
            )
            
            # Force checkpoint
            self.state_manager.save_semantic_state(
                state_update={"critical_violation": asdict(violation)},
                create_checkpoint=True
            )
            
        elif violation.severity == RegressionSeverity.SEVERE:
            # Monitor closely
            self.violation_monitor.increase_monitoring_frequency(
                entity_id=str(violation.violation_uuid),
                frequency_multiplier=2.0
            )
    
    def _continuous_monitoring(self) -> None:
        """Continuous monitoring thread"""
        while self._monitoring_active:
            try:
                # Calculate rolling averages
                if self.rolling_vp:
                    avg_vp = sum(self.rolling_vp) / len(self.rolling_vp)
                    
                    # Check for sustained high VP
                    if avg_vp > self.vp_thresholds["critical"]:
                        self._handle_sustained_high_vp(avg_vp)
                
                # Analyze violation patterns
                self._analyze_violation_patterns()
                
                # Update divergence tracking
                self._update_divergence_tracking()
                
                # Sleep before next iteration
                threading.Event().wait(5.0)  # Check every 5 seconds
                
            except Exception as e:
                # Log error but continue monitoring
                print(f"Error in semantic monitoring: {e}")
    
    def _handle_sustained_high_vp(self, avg_vp: float) -> None:
        """Handle sustained high violation pressure"""
        # Create system-level violation
        system_violation = SemanticViolation(
            violation_uuid=uuid.uuid4(),
            violation_type="SUSTAINED_HIGH_VP",
            severity=RegressionSeverity.SEVERE,
            violation_pressure=avg_vp,
            formation_pattern=None,
            timestamp=datetime.utcnow(),
            mathematical_consistency=self.current_metrics.mathematical_consistency
        )
        
        # Track and respond
        self.violations[system_violation.violation_uuid] = system_violation
        self._trigger_safety_response(system_violation)
        
        # Publish event
        self.event_bridge.publish_semantic_event(system_violation)
    
    def _analyze_violation_patterns(self) -> None:
        """Analyze patterns in violations"""
        with self._monitor_lock:
            for violation_type, vp_values in self.violation_patterns.items():
                if len(vp_values) >= 10:
                    # Check for increasing trend
                    recent = vp_values[-10:]
                    if all(recent[i] <= recent[i+1] for i in range(9)):
                        # Monotonically increasing - dangerous pattern
                        self._handle_dangerous_pattern(violation_type, recent[-1])
    
    def _handle_dangerous_pattern(self, violation_type: str, current_vp: float) -> None:
        """Handle dangerous violation patterns"""
        # Create pattern violation
        pattern_violation = SemanticViolation(
            violation_uuid=uuid.uuid4(),
            violation_type=f"DANGEROUS_PATTERN_{violation_type}",
            severity=RegressionSeverity.CRITICAL,
            violation_pressure=current_vp,
            formation_pattern=None,
            timestamp=datetime.utcnow(),
            mathematical_consistency=0.0  # Pattern indicates breakdown
        )
        
        # Immediate safety response
        self._trigger_safety_response(pattern_violation)
    
    def _update_divergence_tracking(self) -> None:
        """Update divergence tracking for formation types"""
        with self._monitor_lock:
            # Group patterns by formation type
            type_groups = defaultdict(list)
            for pattern in self.pattern_buffer:
                type_groups[pattern.formation_type].append(pattern)
            
            # Calculate divergence for each type
            for formation_type, patterns in type_groups.items():
                if len(patterns) >= 5:
                    # Calculate divergence as variance in VP
                    vp_values = [p.violation_pressure for p in patterns]
                    mean_vp = sum(vp_values) / len(vp_values)
                    variance = sum((vp - mean_vp) ** 2 for vp in vp_values) / len(vp_values)
                    
                    # Store divergence
                    self.divergence_tracking[formation_type] = min(1.0, variance)
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get comprehensive violation summary"""
        with self._monitor_lock:
            # Calculate statistics
            total_violations = len(self.violation_history)
            
            severity_counts = defaultdict(int)
            type_counts = defaultdict(int)
            
            for violation in self.violation_history:
                severity_counts[violation.severity.value] += 1
                type_counts[violation.violation_type] += 1
            
            # Current status
            current_vp = self.rolling_vp[-1] if self.rolling_vp else 0.0
            avg_vp = sum(self.rolling_vp) / len(self.rolling_vp) if self.rolling_vp else 0.0
            
            return {
                "total_violations": total_violations,
                "severity_distribution": dict(severity_counts),
                "type_distribution": dict(type_counts),
                "current_vp": current_vp,
                "average_vp": avg_vp,
                "current_metrics": {
                    "formation_success_rate": self.current_metrics.formation_success_rate,
                    "mathematical_consistency": self.current_metrics.mathematical_consistency,
                    "pattern_stability": self.current_metrics.pattern_stability,
                    "evolution_coherence": self.current_metrics.evolution_coherence,
                    "independence_level": self.current_metrics.independence_level,
                    "semantic_entropy": self.current_metrics.semantic_entropy
                },
                "divergence_tracking": dict(self.divergence_tracking),
                "active_violations": len([v for v in self.violations.values() 
                                        if v.severity in [RegressionSeverity.SEVERE, RegressionSeverity.CRITICAL]])
            }
    
    def clear_violation(self, violation_id: uuid.UUID) -> bool:
        """Clear a resolved violation"""
        with self._monitor_lock:
            if violation_id in self.violations:
                del self.violations[violation_id]
                return True
            return False
    
    def adjust_thresholds(self, 
                          warning: Optional[float] = None,
                          critical: Optional[float] = None,
                          emergency: Optional[float] = None) -> None:
        """Adjust VP thresholds"""
        with self._monitor_lock:
            if warning is not None:
                self.vp_thresholds["warning"] = warning
            if critical is not None:
                self.vp_thresholds["critical"] = critical
            if emergency is not None:
                self.vp_thresholds["emergency"] = emergency
    
    def increase_monitoring_frequency(self, multiplier: float) -> None:
        """Increase monitoring frequency by specified multiplier"""
        with self._monitor_lock:
            self.monitoring_frequency *= multiplier
            # Adjust monitoring intervals if needed
            print(f"Monitoring frequency increased by {multiplier}x")
    
    def shutdown(self) -> None:
        """Shutdown violation monitor"""
        self._monitoring_active = False
        self._monitor_thread.join(timeout=5)
        
        # Final summary
        summary = self.get_violation_summary()
        print(f"Semantic Violation Monitor shutdown. Final summary: {summary}")
