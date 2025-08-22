"""
Semantic Checkpoint Manager - Advanced checkpoint and safety net system
Provides comprehensive checkpoint management with A/B testing and rollback capabilities
"""

import uuid
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import deque, defaultdict
import statistics

# Import kernel dependencies
from uuid_anchor_mechanism import UUIDanchor
from event_driven_coordination import DjinnEventBus
from violation_pressure_calculation import ViolationMonitor

# Import semantic components
from semantic_data_structures import (
    SemanticCheckpoint, SemanticEvolutionBranch, EvolutionValidation,
    CheckpointType, EvolutionStage, EvolutionStrategy,
    RegressionSeverity, RollbackReason, SemanticGovernance
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor

@dataclass
class CheckpointValidation:
    """Validation result for checkpoint integrity"""
    checkpoint_id: uuid.UUID
    is_valid: bool
    hash_match: bool
    state_intact: bool
    parent_valid: bool
    error_details: Optional[str] = None

@dataclass
class EvolutionBranchComparison:
    """Comparison result between evolution branches"""
    branch_a_id: uuid.UUID
    branch_b_id: uuid.UUID
    performance_delta: Dict[str, float]
    winner: Optional[uuid.UUID] = None
    confidence: float = 0.0
    recommendation: str = ""

@dataclass
class RollbackDecision:
    """Decision record for state rollback"""
    decision_id: uuid.UUID
    reason: RollbackReason
    from_checkpoint: uuid.UUID
    to_checkpoint: uuid.UUID
    severity: RegressionSeverity
    automatic: bool
    governance_approval: Optional[bool] = None
    executed: bool = False
    execution_time: Optional[datetime] = None

class SemanticCheckpointManager:
    """
    Advanced checkpoint management with A/B testing and rollback
    Provides safety net for semantic evolution
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 uuid_anchor: UUIDanchor):
        """
        Initialize checkpoint manager
        
        Args:
            state_manager: Semantic state manager
            event_bridge: Event bridge for coordination
            violation_monitor: Violation monitoring system
            uuid_anchor: UUID anchoring mechanism
        """
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.uuid_anchor = uuid_anchor
        
        # Checkpoint management
        self.checkpoints: Dict[uuid.UUID, SemanticCheckpoint] = {}
        self.checkpoint_chain: List[uuid.UUID] = []
        self.checkpoint_validations: Dict[uuid.UUID, CheckpointValidation] = {}
        
        # Evolution branching for A/B testing
        self.evolution_branches: Dict[uuid.UUID, SemanticEvolutionBranch] = {}
        self.branch_comparisons: List[EvolutionBranchComparison] = []
        self.active_branches: Set[uuid.UUID] = set()
        
        # Rollback management
        self.rollback_decisions: Dict[uuid.UUID, RollbackDecision] = {}
        self.rollback_history: deque = deque(maxlen=100)
        
        # Performance baselines
        self.performance_baselines: Dict[uuid.UUID, Dict[str, float]] = {}
        self.regression_thresholds = {
            "formation_success_rate": 0.2,  # 20% drop triggers regression
            "semantic_accuracy": 0.15,       # 15% drop
            "formation_latency_ms": 2.0,     # 2x increase
            "mathematical_consistency": 0.25  # 25% drop
        }
        
        # Thread safety
        self._manager_lock = threading.RLock()
        
        # Monitoring thread
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_checkpoints, daemon=True)
        self._monitor_thread.start()
    
    def create_checkpoint(self,
                         checkpoint_type: CheckpointType = CheckpointType.MANUAL,
                         description: str = "",
                         force: bool = False) -> SemanticCheckpoint:
        """
        Create a new checkpoint
        
        Args:
            checkpoint_type: Type of checkpoint
            description: Optional description
            force: Force creation even if recent checkpoint exists
            
        Returns:
            Created checkpoint
        """
        with self._manager_lock:
            # Check if recent checkpoint exists (unless forced)
            if not force and self.checkpoint_chain:
                last_checkpoint_id = self.checkpoint_chain[-1]
                last_checkpoint = self.checkpoints.get(last_checkpoint_id)
                if last_checkpoint:
                    time_since_last = datetime.utcnow() - last_checkpoint.timestamp
                    if time_since_last < timedelta(minutes=1):
                        # Too recent, return existing
                        return last_checkpoint
            
            # Create checkpoint through state manager
            checkpoint = self.state_manager._create_checkpoint(
                state=self.state_manager.current_state.copy(),
                checkpoint_type=checkpoint_type,
                description=description
            )
            
            # Store in manager
            self.checkpoints[checkpoint.checkpoint_id] = checkpoint
            self.checkpoint_chain.append(checkpoint.checkpoint_id)
            
            # Validate immediately
            validation = self._validate_checkpoint(checkpoint)
            self.checkpoint_validations[checkpoint.checkpoint_id] = validation
            
            # Store performance baseline
            self.performance_baselines[checkpoint.checkpoint_id] = checkpoint.performance_metrics.copy()
            
            # Publish event
            from semantic_data_structures import SemanticCheckpointEvent
            checkpoint_event = SemanticCheckpointEvent(
                event_uuid=uuid.uuid4(),
                event_type="CHECKPOINT_CREATED",
                timestamp=datetime.utcnow(),
                checkpoint=checkpoint,
                checkpoint_type=checkpoint_type,
                trigger="manual"
            )
            self.event_bridge.publish_semantic_event(checkpoint_event)
            
            return checkpoint
    
    def create_evolution_branch(self,
                               base_checkpoint_id: uuid.UUID,
                               strategy: EvolutionStrategy,
                               parameters: Dict[str, Any]) -> SemanticEvolutionBranch:
        """
        Create evolution branch for A/B testing
        
        Args:
            base_checkpoint_id: Base checkpoint to branch from
            strategy: Evolution strategy to test
            parameters: Strategy parameters
            
        Returns:
            Created evolution branch
        """
        with self._manager_lock:
            # Validate base checkpoint
            if base_checkpoint_id not in self.checkpoints:
                raise ValueError(f"Base checkpoint {base_checkpoint_id} not found")
            
            # Create branch
            branch = SemanticEvolutionBranch(
                branch_uuid=uuid.uuid4(),
                base_checkpoint=base_checkpoint_id,
                strategy={
                    "type": strategy.value,
                    "parameters": parameters,
                    "created_at": datetime.utcnow().isoformat()
                },
                performance_metrics={},
                mathematical_consistency=1.0,
                adoption_decision=None,
                evolution_progress=0.0
            )
            
            # Store branch
            self.evolution_branches[branch.branch_uuid] = branch
            self.active_branches.add(branch.branch_uuid)
            
            # Restore base checkpoint for branch
            self.state_manager.restore_semantic_state(base_checkpoint_id)
            
            return branch
    
    def update_branch_metrics(self,
                             branch_id: uuid.UUID,
                             metrics: Dict[str, float]) -> None:
        """
        Update metrics for evolution branch
        
        Args:
            branch_id: Branch to update
            metrics: Performance metrics
        """
        with self._manager_lock:
            if branch_id not in self.evolution_branches:
                return
            
            branch = self.evolution_branches[branch_id]
            branch.performance_metrics.update(metrics)
            
            # Calculate evolution progress
            if branch.base_checkpoint in self.performance_baselines:
                baseline = self.performance_baselines[branch.base_checkpoint]
                progress = self._calculate_evolution_progress(baseline, metrics)
                branch.evolution_progress = progress
    
    def compare_evolution_branches(self,
                                  branch_a_id: uuid.UUID,
                                  branch_b_id: uuid.UUID) -> EvolutionBranchComparison:
        """
        Compare two evolution branches
        
        Args:
            branch_a_id: First branch
            branch_b_id: Second branch
            
        Returns:
            Comparison result
        """
        with self._manager_lock:
            branch_a = self.evolution_branches.get(branch_a_id)
            branch_b = self.evolution_branches.get(branch_b_id)
            
            if not branch_a or not branch_b:
                raise ValueError("Branch not found")
            
            # Calculate performance deltas
            performance_delta = {}
            for metric in branch_a.performance_metrics:
                if metric in branch_b.performance_metrics:
                    delta = branch_a.performance_metrics[metric] - branch_b.performance_metrics[metric]
                    performance_delta[metric] = delta
            
            # Determine winner with statistical confidence
            winner, confidence = self._determine_winner(branch_a, branch_b, performance_delta)
            
            # Create comparison
            comparison = EvolutionBranchComparison(
                branch_a_id=branch_a_id,
                branch_b_id=branch_b_id,
                performance_delta=performance_delta,
                winner=winner,
                confidence=confidence,
                recommendation=self._generate_recommendation(winner, confidence)
            )
            
            self.branch_comparisons.append(comparison)
            
            return comparison
    
    def _determine_winner(self,
                         branch_a: SemanticEvolutionBranch,
                         branch_b: SemanticEvolutionBranch,
                         performance_delta: Dict[str, float]) -> Tuple[Optional[uuid.UUID], float]:
        """Determine winning branch with confidence"""
        if not performance_delta:
            return None, 0.0
        
        # Calculate weighted score
        weights = {
            "formation_success_rate": 0.3,
            "semantic_accuracy": 0.3,
            "mathematical_consistency": 0.2,
            "formation_latency_ms": -0.2  # Negative weight for latency
        }
        
        score_a = 0.0
        score_b = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in performance_delta:
                delta = performance_delta[metric]
                if weight > 0:
                    # Higher is better
                    if delta > 0:
                        score_a += abs(weight)
                    else:
                        score_b += abs(weight)
                else:
                    # Lower is better
                    if delta < 0:
                        score_a += abs(weight)
                    else:
                        score_b += abs(weight)
                total_weight += abs(weight)
        
        if total_weight == 0:
            return None, 0.0
        
        # Normalize scores
        score_a /= total_weight
        score_b /= total_weight
        
        # Determine winner and confidence
        if score_a > score_b:
            confidence = (score_a - score_b) / max(score_a, score_b)
            return branch_a.branch_uuid, confidence
        elif score_b > score_a:
            confidence = (score_b - score_a) / max(score_a, score_b)
            return branch_b.branch_uuid, confidence
        else:
            return None, 0.0
    
    def _generate_recommendation(self, winner: Optional[uuid.UUID], confidence: float) -> str:
        """Generate recommendation based on comparison"""
        if winner is None:
            return "No clear winner - continue testing"
        elif confidence > 0.8:
            return f"Strong recommendation to adopt branch {winner}"
        elif confidence > 0.5:
            return f"Moderate recommendation to adopt branch {winner}"
        else:
            return "Weak preference - additional testing recommended"
    
    def adopt_evolution_branch(self, branch_id: uuid.UUID) -> bool:
        """
        Adopt winning evolution branch
        
        Args:
            branch_id: Branch to adopt
            
        Returns:
            Success status
        """
        with self._manager_lock:
            if branch_id not in self.evolution_branches:
                return False
            
            branch = self.evolution_branches[branch_id]
            
            # Mark as adopted
            branch.adoption_decision = True
            
            # Remove from active branches
            self.active_branches.discard(branch_id)
            
            # Create checkpoint from branch state
            checkpoint = self.create_checkpoint(
                checkpoint_type=CheckpointType.MILESTONE,
                description=f"Adopted evolution branch {branch_id}"
            )
            
            # Publish adoption event
            from semantic_data_structures import SemanticEvolutionEvent
            adoption_event = SemanticEvolutionEvent(
                event_uuid=uuid.uuid4(),
                event_type="EVOLUTION_BRANCH_ADOPTED",
                timestamp=datetime.utcnow(),
                evolution_type="branch_adoption",
                evolution_stage=EvolutionStage.EXPERIMENTATION,
                performance_delta=0.0,
                mathematical_consistency=True
            )
            self.event_bridge.publish_semantic_event(adoption_event)
            
            return True
    
    def detect_regression(self, current_metrics: Dict[str, float]) -> Optional[RegressionSeverity]:
        """
        Detect performance regression
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Regression severity if detected
        """
        with self._manager_lock:
            if not self.checkpoint_chain:
                return None
            
            # Get baseline from last stable checkpoint
            last_checkpoint_id = self.checkpoint_chain[-1]
            if last_checkpoint_id not in self.performance_baselines:
                return None
            
            baseline = self.performance_baselines[last_checkpoint_id]
            
            # Check each metric for regression
            max_severity = None
            
            for metric, threshold in self.regression_thresholds.items():
                if metric in baseline and metric in current_metrics:
                    baseline_value = baseline[metric]
                    current_value = current_metrics[metric]
                    
                    # Calculate regression
                    if baseline_value > 0:
                        if metric == "formation_latency_ms":
                            # For latency, increase is bad
                            regression = (current_value - baseline_value) / baseline_value
                        else:
                            # For other metrics, decrease is bad
                            regression = (baseline_value - current_value) / baseline_value
                        
                        # Determine severity
                        if regression > threshold * 2:
                            severity = RegressionSeverity.CRITICAL
                        elif regression > threshold * 1.5:
                            severity = RegressionSeverity.SEVERE
                        elif regression > threshold:
                            severity = RegressionSeverity.WARNING
                        else:
                            severity = None
                        
                        # Track maximum severity
                        if severity:
                            if max_severity is None or severity.value > max_severity.value:
                                max_severity = severity
            
            return max_severity
    
    def execute_rollback(self,
                        target_checkpoint_id: uuid.UUID,
                        reason: RollbackReason,
                        automatic: bool = False) -> bool:
        """
        Execute rollback to target checkpoint
        
        Args:
            target_checkpoint_id: Checkpoint to rollback to
            reason: Reason for rollback
            automatic: Whether rollback is automatic
            
        Returns:
            Success status
        """
        with self._manager_lock:
            # Validate target checkpoint
            if target_checkpoint_id not in self.checkpoints:
                return False
            
            validation = self._validate_checkpoint(self.checkpoints[target_checkpoint_id])
            if not validation.is_valid:
                return False
            
            # Create rollback decision
            current_checkpoint_id = self.checkpoint_chain[-1] if self.checkpoint_chain else None
            
            decision = RollbackDecision(
                decision_id=uuid.uuid4(),
                reason=reason,
                from_checkpoint=current_checkpoint_id,
                to_checkpoint=target_checkpoint_id,
                severity=RegressionSeverity.SEVERE,  # Default to severe
                automatic=automatic,
                governance_approval=None if automatic else True,
                executed=False
            )
            
            self.rollback_decisions[decision.decision_id] = decision
            
            # Execute rollback through state manager
            try:
                restored_state = self.state_manager.restore_semantic_state(target_checkpoint_id)
                
                # Mark as executed
                decision.executed = True
                decision.execution_time = datetime.utcnow()
                
                # Track in history
                self.rollback_history.append(decision)
                
                # Publish rollback event
                self.event_bridge.publish_semantic_event({
                    "event_type": "ROLLBACK_EXECUTED",
                    "decision_id": str(decision.decision_id),
                    "from_checkpoint": str(current_checkpoint_id) if current_checkpoint_id else None,
                    "to_checkpoint": str(target_checkpoint_id),
                    "reason": reason.value,
                    "automatic": automatic,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                return True
                
            except Exception as e:
                # Rollback failed
                decision.executed = False
                self.event_bridge.publish_semantic_event({
                    "event_type": "ROLLBACK_FAILED",
                    "decision_id": str(decision.decision_id),
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                return False
    
    def _validate_checkpoint(self, checkpoint: SemanticCheckpoint) -> CheckpointValidation:
        """Validate checkpoint integrity"""
        validation = CheckpointValidation(
            checkpoint_id=checkpoint.checkpoint_id,
            is_valid=True,
            hash_match=True,
            state_intact=True,
            parent_valid=True
        )
        
        # Verify hash
        calculated_hash = self._calculate_state_hash(checkpoint.semantic_state)
        if calculated_hash != checkpoint.mathematical_hash:
            validation.hash_match = False
            validation.is_valid = False
            validation.error_details = "Hash mismatch"
        
        # Verify state structure
        required_keys = ["semantic_version", "evolution_stage", "formation_success_rate"]
        for key in required_keys:
            if key not in checkpoint.semantic_state:
                validation.state_intact = False
                validation.is_valid = False
                validation.error_details = f"Missing required key: {key}"
                break
        
        # Verify parent if exists
        if checkpoint.parent_checkpoint:
            if checkpoint.parent_checkpoint not in self.checkpoints:
                validation.parent_valid = False
                validation.is_valid = False
                validation.error_details = "Parent checkpoint not found"
        
        return validation
    
    def _calculate_state_hash(self, state: Dict[str, Any]) -> str:
        """Calculate deterministic hash of state"""
        sorted_state = json.dumps(state, sort_keys=True)
        return hashlib.sha256(sorted_state.encode()).hexdigest()
    
    def _calculate_evolution_progress(self,
                                     baseline: Dict[str, float],
                                     current: Dict[str, float]) -> float:
        """Calculate evolution progress as percentage"""
        if not baseline or not current:
            return 0.0
        
        progress_scores = []
        
        for metric in baseline:
            if metric in current:
                baseline_value = baseline[metric]
                current_value = current[metric]
                
                if baseline_value > 0:
                    if metric == "formation_latency_ms":
                        # For latency, decrease is progress
                        progress = (baseline_value - current_value) / baseline_value
                    else:
                        # For other metrics, increase is progress
                        progress = (current_value - baseline_value) / baseline_value
                    
                    progress_scores.append(progress)
        
        if progress_scores:
            return sum(progress_scores) / len(progress_scores)
        return 0.0
    
    def _monitor_checkpoints(self) -> None:
        """Background monitoring thread"""
        while self._monitoring_active:
            try:
                with self._manager_lock:
                    # Check for regression
                    current_metrics = self.state_manager.current_state.get("performance_metrics", {})
                    severity = self.detect_regression(current_metrics)
                    
                    if severity and severity in [RegressionSeverity.SEVERE, RegressionSeverity.CRITICAL]:
                        # Find safe checkpoint to rollback to
                        safe_checkpoint = self._find_safe_checkpoint()
                        if safe_checkpoint:
                            # Automatic rollback for critical regression
                            if severity == RegressionSeverity.CRITICAL:
                                self.execute_rollback(
                                    safe_checkpoint,
                                    RollbackReason.AUTOMATIC_REGRESSION_PROTECTION,
                                    automatic=True
                                )
                    
                    # Clean up old branches
                    self._cleanup_old_branches()
                
                # Sleep before next check
                threading.Event().wait(10.0)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in checkpoint monitoring: {e}")
    
    def _find_safe_checkpoint(self) -> Optional[uuid.UUID]:
        """Find most recent safe checkpoint"""
        # Iterate backwards through checkpoint chain
        for checkpoint_id in reversed(self.checkpoint_chain[-10:]):  # Check last 10
            checkpoint = self.checkpoints.get(checkpoint_id)
            if checkpoint:
                validation = self._validate_checkpoint(checkpoint)
                if validation.is_valid:
                    # Check if checkpoint has good performance
                    if checkpoint_id in self.performance_baselines:
                        baseline = self.performance_baselines[checkpoint_id]
                        if baseline.get("formation_success_rate", 0) > 0.7:
                            return checkpoint_id
        return None
    
    def _cleanup_old_branches(self) -> None:
        """Clean up abandoned evolution branches"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        branches_to_remove = []
        for branch_id in list(self.active_branches):
            branch = self.evolution_branches.get(branch_id)
            if branch:
                created_at = datetime.fromisoformat(branch.strategy.get("created_at", datetime.utcnow().isoformat()))
                if created_at < cutoff_time and branch.adoption_decision is None:
                    branches_to_remove.append(branch_id)
        
        for branch_id in branches_to_remove:
            self.active_branches.discard(branch_id)
            del self.evolution_branches[branch_id]
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint summary"""
        with self._manager_lock:
            return {
                "total_checkpoints": len(self.checkpoints),
                "valid_checkpoints": sum(1 for v in self.checkpoint_validations.values() if v.is_valid),
                "active_branches": len(self.active_branches),
                "total_branches": len(self.evolution_branches),
                "rollback_count": len(self.rollback_history),
                "last_checkpoint": str(self.checkpoint_chain[-1]) if self.checkpoint_chain else None,
                "checkpoint_chain_length": len(self.checkpoint_chain)
            }
    
    def get_latest_checkpoint_id(self) -> Optional[uuid.UUID]:
        """Get the latest checkpoint ID"""
        with self._manager_lock:
            return self.checkpoint_chain[-1] if self.checkpoint_chain else None
    
    def get_checkpoint_baseline(self, checkpoint_id: uuid.UUID) -> Optional[Dict[str, float]]:
        """Get performance baseline for a checkpoint"""
        with self._manager_lock:
            return self.performance_baselines.get(checkpoint_id)
    
    def shutdown(self) -> None:
        """Shutdown checkpoint manager"""
        self._monitoring_active = False
        self._monitor_thread.join(timeout=5)
        
        # Final summary
        summary = self.get_checkpoint_summary()
        print(f"Checkpoint Manager shutdown. Summary: {summary}")
