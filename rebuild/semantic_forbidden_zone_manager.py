"""
Semantic Forbidden Zone Manager - Sandbox environment for experimental formations
Provides isolated execution space using μ-recursion chambers for safe experimentation
"""

import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict
import copy

# Import kernel dependencies
from forbidden_zone_management import ForbiddenZoneManager
from temporal_isolation_safety import TemporalIsolationManager
from violation_pressure_calculation import ViolationMonitor

# Import semantic components
from semantic_data_structures import (
    FormationPattern, SemanticViolation, SafetyNet,
    FormationType, RegressionSeverity, EvolutionStage
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager

class SandboxStatus(Enum):
    """Status of sandbox environment"""
    IDLE = "idle"
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    EVALUATING = "evaluating"
    INTEGRATING = "integrating"

class ExperimentType(Enum):
    """Types of semantic experiments"""
    NOVEL_FORMATION = "novel_formation"
    PATTERN_VARIATION = "pattern_variation"
    AGGRESSIVE_EVOLUTION = "aggressive_evolution"
    BOUNDARY_TESTING = "boundary_testing"
    RECOVERY_TESTING = "recovery_testing"

@dataclass
class SandboxEnvironment:
    """Isolated sandbox for semantic experimentation"""
    sandbox_id: uuid.UUID
    experiment_type: ExperimentType
    isolation_level: float  # 0.0 to 1.0 (1.0 = maximum isolation)
    state_snapshot: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    status: SandboxStatus = SandboxStatus.IDLE
    created_at: datetime = field(default_factory=datetime.utcnow)
    duration_limit: timedelta = field(default=timedelta(minutes=30))

@dataclass
class ExperimentResult:
    """Result of sandbox experiment"""
    experiment_id: uuid.UUID
    sandbox_id: uuid.UUID
    success: bool
    violation_pressure: float
    mathematical_consistency: float
    formations_tested: int
    formations_successful: int
    novel_patterns_discovered: List[FormationPattern] = field(default_factory=list)
    safety_violations: List[SemanticViolation] = field(default_factory=list)
    recommendation: str = ""
    integration_safe: bool = False

@dataclass
class QuarantineRecord:
    """Record of quarantined formation"""
    quarantine_id: uuid.UUID
    formation_pattern: FormationPattern
    reason: str
    violation_pressure: float
    quarantine_start: datetime
    quarantine_duration: timedelta
    release_conditions: Dict[str, Any] = field(default_factory=dict)
    released: bool = False
    release_time: Optional[datetime] = None

class SemanticForbiddenZoneManager:
    """
    Manages sandbox environments for safe semantic experimentation
    Integrates with kernel's Forbidden Zone infrastructure
    """
    
    def __init__(self,
                 forbidden_zone: ForbiddenZoneManager,
                 temporal_isolation: TemporalIsolationManager,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 checkpoint_manager: SemanticCheckpointManager):
        """
        Initialize forbidden zone manager
        
        Args:
            forbidden_zone: Kernel's forbidden zone manager
            temporal_isolation: Temporal isolation system
            state_manager: Semantic state manager
            event_bridge: Event bridge
            violation_monitor: Violation monitor
            checkpoint_manager: Checkpoint manager
        """
        # Kernel integrations
        self.forbidden_zone = forbidden_zone
        self.temporal_isolation = temporal_isolation
        
        # Semantic components
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.checkpoint_manager = checkpoint_manager
        
        # Sandbox management
        self.sandboxes: Dict[uuid.UUID, SandboxEnvironment] = {}
        self.active_sandboxes: Set[uuid.UUID] = set()
        self.sandbox_results: Dict[uuid.UUID, List[ExperimentResult]] = defaultdict(list)
        
        # Quarantine management
        self.quarantine_zone: Dict[uuid.UUID, QuarantineRecord] = {}
        self.quarantine_patterns: Set[str] = set()  # Hashes of quarantined patterns
        
        # Novel pattern discovery
        self.novel_patterns: List[FormationPattern] = []
        self.pattern_evaluations: Dict[str, float] = {}  # Pattern hash -> safety score
        
        # Safety thresholds
        self.safety_thresholds = {
            "max_vp": 0.95,              # Maximum VP before emergency stop
            "max_failures": 10,           # Max consecutive failures
            "min_consistency": 0.3,       # Minimum mathematical consistency
            "quarantine_vp": 0.8,        # VP threshold for quarantine
            "integration_threshold": 0.7  # Safety score for integration
        }
        
        # Thread safety
        self._zone_lock = threading.RLock()
        
        # Monitoring thread
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_sandboxes, daemon=True)
        self._monitor_thread.start()
    
    def create_sandbox(self,
                      experiment_type: ExperimentType,
                      isolation_level: float = 0.8,
                      constraints: Optional[Dict[str, Any]] = None) -> SandboxEnvironment:
        """
        Create isolated sandbox environment
        
        Args:
            experiment_type: Type of experiment
            isolation_level: Level of isolation (0.0 to 1.0)
            constraints: Optional constraints for experiment
            
        Returns:
            Created sandbox environment
        """
        with self._zone_lock:
            # Create checkpoint before sandbox
            checkpoint = self.checkpoint_manager.create_checkpoint(
                checkpoint_type="SANDBOX_BASELINE",
                description=f"Pre-sandbox checkpoint for {experiment_type.value}"
            )
            
            # Snapshot current state
            state_snapshot = copy.deepcopy(self.state_manager.current_state)
            
            # Create sandbox
            sandbox = SandboxEnvironment(
                sandbox_id=uuid.uuid4(),
                experiment_type=experiment_type,
                isolation_level=min(1.0, max(0.0, isolation_level)),
                state_snapshot=state_snapshot,
                constraints=constraints or {},
                status=SandboxStatus.IDLE
            )
            
            # Apply isolation constraints based on level
            if isolation_level > 0.7:
                sandbox.constraints.update({
                    "max_formations": 100,
                    "max_vp": 0.9,
                    "allow_state_modification": False,
                    "allow_external_access": False
                })
            elif isolation_level > 0.4:
                sandbox.constraints.update({
                    "max_formations": 500,
                    "max_vp": 0.95,
                    "allow_state_modification": True,
                    "allow_external_access": False
                })
            else:
                sandbox.constraints.update({
                    "max_formations": 1000,
                    "max_vp": 1.0,
                    "allow_state_modification": True,
                    "allow_external_access": True
                })
            
            # Store sandbox
            self.sandboxes[sandbox.sandbox_id] = sandbox
            
            # Register with kernel's forbidden zone
            self.forbidden_zone.register_zone(
                zone_id=str(sandbox.sandbox_id),
                zone_type="semantic_sandbox",
                isolation_level=isolation_level
            )
            
            return sandbox
    
    def activate_sandbox(self, sandbox_id: uuid.UUID) -> bool:
        """
        Activate sandbox for experimentation
        
        Args:
            sandbox_id: Sandbox to activate
            
        Returns:
            Success status
        """
        with self._zone_lock:
            if sandbox_id not in self.sandboxes:
                return False
            
            sandbox = self.sandboxes[sandbox_id]
            
            # Check if already active
            if sandbox_id in self.active_sandboxes:
                return True
            
            # Activate sandbox
            sandbox.status = SandboxStatus.ACTIVE
            self.active_sandboxes.add(sandbox_id)
            
            # Enter forbidden zone
            self.forbidden_zone.enter_zone(str(sandbox_id))
            
            # Publish activation event
            from semantic_data_structures import SemanticEvent
            activation_event = SemanticEvent(
                event_uuid=uuid.uuid4(),
                event_type="SANDBOX_ACTIVATED",
                timestamp=datetime.utcnow(),
                payload={
                    "sandbox_id": str(sandbox_id),
                    "experiment_type": sandbox.experiment_type.value,
                    "isolation_level": sandbox.isolation_level
                }
            )
            self.event_bridge.publish_semantic_event(activation_event)
            
            return True
    
    def execute_experiment(self,
                          sandbox_id: uuid.UUID,
                          formation_patterns: List[FormationPattern]) -> ExperimentResult:
        """
        Execute experiment in sandbox
        
        Args:
            sandbox_id: Sandbox for experiment
            formation_patterns: Patterns to test
            
        Returns:
            Experiment result
        """
        with self._zone_lock:
            if sandbox_id not in self.sandboxes:
                raise ValueError(f"Sandbox {sandbox_id} not found")
            
            sandbox = self.sandboxes[sandbox_id]
            
            if sandbox.status != SandboxStatus.ACTIVE:
                self.activate_sandbox(sandbox_id)
            
            # Create experiment result
            result = ExperimentResult(
                experiment_id=uuid.uuid4(),
                sandbox_id=sandbox_id,
                success=True,
                violation_pressure=0.0,
                mathematical_consistency=1.0,
                formations_tested=len(formation_patterns),
                formations_successful=0
            )
            
            # Test each pattern in isolation
            max_vp = 0.0
            min_consistency = 1.0
            consecutive_failures = 0
            
            for pattern in formation_patterns:
                # Check quarantine
                if self._is_quarantined(pattern):
                    continue
                
                # Test pattern
                test_result = self._test_pattern_safely(pattern, sandbox)
                
                if test_result["success"]:
                    result.formations_successful += 1
                    consecutive_failures = 0
                    
                    # Check for novel pattern
                    if self._is_novel_pattern(pattern):
                        result.novel_patterns_discovered.append(pattern)
                        self.novel_patterns.append(pattern)
                else:
                    consecutive_failures += 1
                    
                    # Check failure threshold
                    if consecutive_failures >= self.safety_thresholds["max_failures"]:
                        result.success = False
                        result.recommendation = "Experiment stopped due to excessive failures"
                        break
                
                # Track VP and consistency
                pattern_vp = test_result.get("violation_pressure", 0.0)
                pattern_consistency = test_result.get("mathematical_consistency", 1.0)
                
                max_vp = max(max_vp, pattern_vp)
                min_consistency = min(min_consistency, pattern_consistency)
                
                # Check safety thresholds
                if pattern_vp > self.safety_thresholds["max_vp"]:
                    # Emergency stop
                    result.success = False
                    result.recommendation = "Emergency stop due to critical VP"
                    self._quarantine_pattern(pattern, pattern_vp, "Critical VP exceeded")
                    break
                
                if pattern_consistency < self.safety_thresholds["min_consistency"]:
                    # Quarantine pattern
                    self._quarantine_pattern(pattern, pattern_vp, "Mathematical consistency violation")
                
                # Create violation if needed
                if pattern_vp > self.safety_thresholds["quarantine_vp"]:
                    violation = SemanticViolation(
                        violation_uuid=uuid.uuid4(),
                        violation_type="SANDBOX_VIOLATION",
                        severity=RegressionSeverity.WARNING,
                        violation_pressure=pattern_vp,
                        formation_pattern=pattern,
                        timestamp=datetime.utcnow(),
                        mathematical_consistency=pattern_consistency
                    )
                    result.safety_violations.append(violation)
            
            # Set final metrics
            result.violation_pressure = max_vp
            result.mathematical_consistency = min_consistency
            
            # Determine if safe for integration
            safety_score = self._calculate_safety_score(result)
            result.integration_safe = safety_score > self.safety_thresholds["integration_threshold"]
            
            if result.integration_safe:
                result.recommendation = "Experiment successful - safe for integration"
            elif result.success:
                result.recommendation = "Experiment completed but not safe for integration"
            
            # Store result
            self.sandbox_results[sandbox_id].append(result)
            
            # Update sandbox
            sandbox.results[str(result.experiment_id)] = {
                "success": result.success,
                "vp": result.violation_pressure,
                "consistency": result.mathematical_consistency,
                "novel_patterns": len(result.novel_patterns_discovered),
                "safety_violations": len(result.safety_violations),
                "integration_safe": result.integration_safe
            }
            
            return result
    
    def _test_pattern_safely(self,
                           pattern: FormationPattern,
                           sandbox: SandboxEnvironment) -> Dict[str, Any]:
        """Test pattern with safety constraints"""
        # Apply sandbox constraints
        if sandbox.constraints.get("max_vp", 1.0) < pattern.violation_pressure:
            return {"success": False, "reason": "VP exceeds sandbox limit"}
        
        # Calculate VP and metrics
        vp, metrics = self.violation_monitor.calculate_semantic_vp(pattern)
        
        # Simulate formation
        success = (vp < 0.7 and metrics.mathematical_consistency > 0.5)
        
        return {
            "success": success,
            "violation_pressure": vp,
            "mathematical_consistency": metrics.mathematical_consistency,
            "metrics": metrics
        }
    
    def _is_novel_pattern(self, pattern: FormationPattern) -> bool:
        """Check if pattern is novel"""
        pattern_hash = self._hash_pattern(pattern)
        
        # Check against existing patterns
        for existing in self.novel_patterns:
            if self._hash_pattern(existing) == pattern_hash:
                return False
        
        return True
    
    def _hash_pattern(self, pattern: FormationPattern) -> str:
        """Generate hash for pattern"""
        pattern_str = f"{pattern.formation_type.value}:{pattern.characters}:{pattern.word}:{pattern.sentence}"
        return hashlib.sha256(pattern_str.encode()).hexdigest()
    
    def _is_quarantined(self, pattern: FormationPattern) -> bool:
        """Check if pattern is quarantined"""
        pattern_hash = self._hash_pattern(pattern)
        return pattern_hash in self.quarantine_patterns
    
    def _quarantine_pattern(self,
                          pattern: FormationPattern,
                          vp: float,
                          reason: str) -> QuarantineRecord:
        """Quarantine dangerous pattern"""
        record = QuarantineRecord(
            quarantine_id=uuid.uuid4(),
            formation_pattern=pattern,
            reason=reason,
            violation_pressure=vp,
            quarantine_start=datetime.utcnow(),
            quarantine_duration=timedelta(hours=1),
            release_conditions={
                "min_system_stability": 0.8,
                "max_vp": 0.5,
                "approval_required": True
            }
        )
        
        # Store quarantine record
        self.quarantine_zone[record.quarantine_id] = record
        self.quarantine_patterns.add(self._hash_pattern(pattern))
        
        # Isolate pattern
        self.temporal_isolation.isolate_operation(
            operation_id=str(record.quarantine_id),
            reason=f"Pattern quarantine: {reason}",
            duration=record.quarantine_duration
        )
        
        return record
    
    def _calculate_safety_score(self, result: ExperimentResult) -> float:
        """Calculate safety score for experiment result"""
        # Weighted scoring
        vp_score = max(0, 1.0 - result.violation_pressure)
        consistency_score = result.mathematical_consistency
        success_rate = result.formations_successful / max(1, result.formations_tested)
        violation_penalty = 1.0 - (len(result.safety_violations) * 0.1)
        
        # Weighted average
        safety_score = (
            vp_score * 0.3 +
            consistency_score * 0.3 +
            success_rate * 0.2 +
            violation_penalty * 0.2
        )
        
        return max(0.0, min(1.0, safety_score))
    
    def integrate_safe_patterns(self,
                               sandbox_id: uuid.UUID,
                               experiment_id: uuid.UUID) -> bool:
        """
        Integrate safe patterns from experiment
        
        Args:
            sandbox_id: Sandbox containing experiment
            experiment_id: Experiment with safe patterns
            
        Returns:
            Success status
        """
        with self._zone_lock:
            # Find experiment result
            results = self.sandbox_results.get(sandbox_id, [])
            experiment_result = None
            
            for result in results:
                if result.experiment_id == experiment_id:
                    experiment_result = result
                    break
            
            if not experiment_result:
                return False
            
            if not experiment_result.integration_safe:
                return False
            
            # Exit sandbox
            sandbox = self.sandboxes[sandbox_id]
            sandbox.status = SandboxStatus.INTEGRATING
            
            # Integrate novel patterns
            integrated_count = 0
            for pattern in experiment_result.novel_patterns_discovered:
                # Evaluate pattern
                pattern_hash = self._hash_pattern(pattern)
                safety_score = self._evaluate_pattern_safety(pattern)
                
                if safety_score > self.safety_thresholds["integration_threshold"]:
                    # Safe to integrate
                    self.pattern_evaluations[pattern_hash] = safety_score
                    integrated_count += 1
                    
                    # Update state
                    self.state_manager.save_semantic_state({
                        "integrated_patterns": self.state_manager.current_state.get("integrated_patterns", 0) + 1,
                        "last_integration": datetime.utcnow().isoformat()
                    })
            
            # Deactivate sandbox
            self.active_sandboxes.discard(sandbox_id)
            sandbox.status = SandboxStatus.IDLE
            
            # Exit forbidden zone
            self.forbidden_zone.exit_zone(str(sandbox_id))
            
            # Publish integration event
            integration_event = SemanticEvent(
                event_uuid=uuid.uuid4(),
                event_type="PATTERNS_INTEGRATED",
                timestamp=datetime.utcnow(),
                payload={
                    "sandbox_id": str(sandbox_id),
                    "experiment_id": str(experiment_id),
                    "patterns_integrated": integrated_count
                }
            )
            self.event_bridge.publish_semantic_event(integration_event)
            
            return True
    
    def _evaluate_pattern_safety(self, pattern: FormationPattern) -> float:
        """Evaluate safety of pattern for integration"""
        # Calculate comprehensive safety score
        vp_safety = max(0, 1.0 - pattern.violation_pressure)
        consistency_safety = pattern.mathematical_consistency
        
        # Check historical performance
        pattern_hash = self._hash_pattern(pattern)
        historical_score = self.pattern_evaluations.get(pattern_hash, 0.5)
        
        # Combined safety score
        safety_score = (vp_safety * 0.4 + consistency_safety * 0.4 + historical_score * 0.2)
        
        return safety_score
    
    def release_quarantine(self, quarantine_id: uuid.UUID) -> bool:
        """
        Release pattern from quarantine
        
        Args:
            quarantine_id: Quarantine to release
            
        Returns:
            Success status
        """
        with self._zone_lock:
            if quarantine_id not in self.quarantine_zone:
                return False
            
            record = self.quarantine_zone[quarantine_id]
            
            # Check release conditions
            current_health = self.state_manager.get_current_health()
            
            if current_health.system_coherence < record.release_conditions.get("min_system_stability", 0.8):
                return False
            
            # Release from quarantine
            record.released = True
            record.release_time = datetime.utcnow()
            
            # Remove from quarantine patterns
            pattern_hash = self._hash_pattern(record.formation_pattern)
            self.quarantine_patterns.discard(pattern_hash)
            
            # Release isolation
            self.temporal_isolation.release_isolation(str(quarantine_id))
            
            return True
    
    def _monitor_sandboxes(self) -> None:
        """Monitor active sandboxes"""
        while self._monitoring_active:
            try:
                with self._zone_lock:
                    # Check sandbox timeouts
                    for sandbox_id in list(self.active_sandboxes):
                        sandbox = self.sandboxes.get(sandbox_id)
                        if sandbox:
                            elapsed = datetime.utcnow() - sandbox.created_at
                            if elapsed > sandbox.duration_limit:
                                # Timeout - deactivate
                                self.deactivate_sandbox(sandbox_id, "Timeout")
                    
                    # Check quarantine releases
                    for quarantine_id, record in list(self.quarantine_zone.items()):
                        if not record.released:
                            elapsed = datetime.utcnow() - record.quarantine_start
                            if elapsed > record.quarantine_duration:
                                # Try to release
                                self.release_quarantine(quarantine_id)
                
                # Sleep
                threading.Event().wait(30.0)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Error in sandbox monitoring: {e}")
    
    def deactivate_sandbox(self, sandbox_id: uuid.UUID, reason: str = "") -> bool:
        """Deactivate sandbox"""
        with self._zone_lock:
            if sandbox_id not in self.sandboxes:
                return False
            
            sandbox = self.sandboxes[sandbox_id]
            sandbox.status = SandboxStatus.IDLE
            self.active_sandboxes.discard(sandbox_id)
            
            # Exit forbidden zone
            self.forbidden_zone.exit_zone(str(sandbox_id))
            
            # Publish deactivation event
            deactivation_event = SemanticEvent(
                event_uuid=uuid.uuid4(),
                event_type="SANDBOX_DEACTIVATED",
                timestamp=datetime.utcnow(),
                payload={
                    "sandbox_id": str(sandbox_id),
                    "reason": reason
                }
            )
            self.event_bridge.publish_semantic_event(deactivation_event)
            
            return True
    
    def get_zone_summary(self) -> Dict[str, Any]:
        """Get comprehensive zone summary"""
        with self._zone_lock:
            return {
                "total_sandboxes": len(self.sandboxes),
                "active_sandboxes": len(self.active_sandboxes),
                "quarantined_patterns": len(self.quarantine_patterns),
                "novel_patterns_discovered": len(self.novel_patterns),
                "total_experiments": sum(len(results) for results in self.sandbox_results.values()),
                "integration_safe_patterns": sum(
                    1 for results in self.sandbox_results.values()
                    for result in results if result.integration_safe
                )
            }
    
    def shutdown(self) -> None:
        """Shutdown forbidden zone manager"""
        self._monitoring_active = False
        
        # Deactivate all sandboxes
        for sandbox_id in list(self.active_sandboxes):
            self.deactivate_sandbox(sandbox_id, "Shutdown")
        
        self._monitor_thread.join(timeout=5)
        
        # Final summary
        summary = self.get_zone_summary()
        print(f"Forbidden Zone Manager shutdown. Summary: {summary}")
