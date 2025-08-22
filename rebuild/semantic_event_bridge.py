"""
Semantic Event Bridge - System-wide coordination for semantic operations
Bridges semantic system events with DjinnEventBus for kernel integration
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import asyncio

# Import kernel dependencies
from event_driven_coordination import DjinnEventBus, EventType
from violation_pressure_calculation import ViolationMonitor, ViolationClass
from temporal_isolation_safety import TemporalIsolationManager
from monitoring_observability import MonitoringObservability

# Import semantic components
from semantic_data_structures import (
    SemanticEvent, CharacterFormationEvent, WordFormationEvent,
    SentenceFormationEvent, DialogueFormationEvent, SemanticMilestoneEvent,
    SemanticViolationEvent, SemanticCheckpointEvent, SemanticRollbackEvent,
    SemanticEvolutionEvent, SemanticGovernanceEvent,
    EvolutionStage, RegressionSeverity, CheckpointType
)
from semantic_state_manager import SemanticStateManager

class SemanticEventType(Enum):
    """Semantic-specific event types"""
    # Formation events
    CHARACTER_FORMATION = "character_formation"
    WORD_FORMATION = "word_formation"
    SENTENCE_FORMATION = "sentence_formation"
    DIALOGUE_FORMATION = "dialogue_formation"
    
    # Evolution events
    EVOLUTION_STAGE_CHANGE = "evolution_stage_change"
    INDEPENDENCE_ACHIEVED = "independence_achieved"
    PATTERN_LEARNED = "pattern_learned"
    
    # Safety events
    FORMATION_FAILURE = "formation_failure"
    VP_THRESHOLD_EXCEEDED = "vp_threshold_exceeded"
    REGRESSION_DETECTED = "regression_detected"
    
    # Checkpoint events
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    
    # Milestone events
    VOCABULARY_MILESTONE = "vocabulary_milestone"
    ACCURACY_MILESTONE = "accuracy_milestone"
    TRANSCENDENCE_ACHIEVED = "transcendence_achieved"

class SemanticEventBridge:
    """
    Bridges semantic events with kernel's event-driven architecture
    Provides bidirectional event flow and coordination
    """
    
    def __init__(self,
                 event_bus: DjinnEventBus,
                 state_manager: SemanticStateManager,
                 violation_monitor: ViolationMonitor,
                 temporal_isolation: TemporalIsolationManager,
                 monitoring: Optional[MonitoringObservability] = None):
        """
        Initialize semantic event bridge with kernel integrations
        
        Args:
            event_bus: Core kernel event bus
            state_manager: Semantic state manager
            violation_monitor: VP monitoring system
            temporal_isolation: Safety isolation system
            monitoring: Optional monitoring system
        """
        # Kernel integrations
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.violation_monitor = violation_monitor
        self.temporal_isolation = temporal_isolation
        self.monitoring = monitoring
        
        # Event handling
        self.event_handlers: Dict[SemanticEventType, List[Callable]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=10000)
        self.event_metrics: Dict[str, int] = defaultdict(int)
        
        # Subscription tracking
        self.kernel_subscriptions: Set[str] = set()
        self.semantic_subscriptions: Set[SemanticEventType] = set()
        
        # Performance tracking
        self.event_latencies: Dict[str, List[float]] = defaultdict(list)
        self.event_success_rates: Dict[str, float] = {}
        
        # Thread safety
        self._bridge_lock = threading.RLock()
        
        # Async event loop for non-blocking operations
        self._event_loop = asyncio.new_event_loop()
        self._event_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._event_thread.start()
        
        # Initialize kernel event subscriptions
        self._subscribe_to_kernel_events()
        
    def _run_event_loop(self):
        """Run async event loop in separate thread"""
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_forever()
        
    def _subscribe_to_kernel_events(self):
        """Subscribe to relevant kernel events"""
        # Subscribe to violation pressure events
        self.event_bus.subscribe("VP_THRESHOLD_EXCEEDED", self._handle_kernel_vp_event)
        self.event_bus.subscribe("VP_STABILIZED", self._handle_kernel_stability_event)
        
        # Subscribe to isolation events
        self.event_bus.subscribe("ISOLATION_TRIGGERED", self._handle_isolation_event)
        self.event_bus.subscribe("ISOLATION_RELEASED", self._handle_isolation_release)
        
        # Subscribe to system health events
        self.event_bus.subscribe("SYSTEM_HEALTH_CRITICAL", self._handle_health_critical)
        self.event_bus.subscribe("SYSTEM_RECOVERY", self._handle_system_recovery)
        
        # Track subscriptions
        self.kernel_subscriptions.update([
            "VP_THRESHOLD_EXCEEDED", "VP_STABILIZED",
            "ISOLATION_TRIGGERED", "ISOLATION_RELEASED",
            "SYSTEM_HEALTH_CRITICAL", "SYSTEM_RECOVERY"
        ])
    
    def publish_semantic_event(self, event: SemanticEvent) -> None:
        """
        Publish semantic event to both semantic handlers and kernel bus
        
        Args:
            event: Semantic event to publish
        """
        with self._bridge_lock:
            start_time = datetime.utcnow()
            
            # Track event
            self.event_history.append(event)
            self.event_metrics[event.event_type] += 1
            
            # Determine event type category
            event_category = self._categorize_event(event)
            
            # Handle semantic-specific processing
            if event_category in self.event_handlers:
                for handler in self.event_handlers[event_category]:
                    try:
                        handler(event)
                    except Exception as e:
                        self._handle_event_error(event, handler, e)
            
            # Bridge to kernel event bus
            kernel_event = self._convert_to_kernel_event(event)
            self.event_bus.publish(kernel_event)
            
            # Track latency
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.event_latencies[event.event_type].append(latency)
            
            # Update monitoring if available
            if self.monitoring:
                self._update_monitoring(event, latency)
    
    def subscribe_semantic_event(self, 
                                event_type: SemanticEventType,
                                handler: Callable) -> None:
        """
        Subscribe to semantic event type
        
        Args:
            event_type: Type of semantic event
            handler: Callback function for event
        """
        with self._bridge_lock:
            self.event_handlers[event_type].append(handler)
            self.semantic_subscriptions.add(event_type)
    
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register handler for event type (compatibility method)
        
        Args:
            event_type: String name of event type
            handler: Callback function for event
        """
        # For compatibility with semantic components that expect register_handler
        # Store in a simple string-based handler registry
        if not hasattr(self, 'string_handlers'):
            self.string_handlers = {}
        
        if event_type not in self.string_handlers:
            self.string_handlers[event_type] = []
        self.string_handlers[event_type].append(handler)
        
        print(f"📡 Registered handler for {event_type} events")
    
    def publish_formation_event(self,
                               formation_type: str,
                               content: str,
                               success: bool,
                               vp: float,
                               consistency: float) -> None:
        """
        Publish formation event with automatic categorization
        
        Args:
            formation_type: Type of formation (character/word/sentence/dialogue)
            content: Formed content
            success: Whether formation succeeded
            vp: Violation pressure
            consistency: Mathematical consistency
        """
        # Create appropriate event based on type
        if formation_type == "character":
            event = CharacterFormationEvent(
                event_uuid=uuid.uuid4(),
                event_type="CHARACTER_FORMATION",
                timestamp=datetime.utcnow(),
                character=content,
                formation_success=success,
                violation_pressure=vp,
                mathematical_consistency=consistency
            )
        elif formation_type == "word":
            event = WordFormationEvent(
                event_uuid=uuid.uuid4(),
                event_type="WORD_FORMATION",
                timestamp=datetime.utcnow(),
                word=content,
                characters=list(content),
                formation_success=success,
                violation_pressure=vp,
                mathematical_consistency=consistency
            )
        elif formation_type == "sentence":
            event = SentenceFormationEvent(
                event_uuid=uuid.uuid4(),
                event_type="SENTENCE_FORMATION",
                timestamp=datetime.utcnow(),
                sentence=content,
                words=content.split(),
                formation_success=success,
                violation_pressure=vp,
                mathematical_consistency=consistency
            )
        elif formation_type == "dialogue":
            event = DialogueFormationEvent(
                event_uuid=uuid.uuid4(),
                event_type="DIALOGUE_FORMATION",
                timestamp=datetime.utcnow(),
                dialogue=content,
                sentences=content.split('. '),
                formation_success=success,
                violation_pressure=vp,
                mathematical_consistency=consistency
            )
        else:
            raise ValueError(f"Unknown formation type: {formation_type}")
        
        # Publish event
        self.publish_semantic_event(event)
        
        # Check for safety triggers
        if not success or vp > 0.7:
            self._trigger_safety_response(event, vp)
    
    def _categorize_event(self, event: SemanticEvent) -> Optional[SemanticEventType]:
        """Categorize semantic event for routing"""
        event_type_mapping = {
            "CHARACTER_FORMATION": SemanticEventType.CHARACTER_FORMATION,
            "WORD_FORMATION": SemanticEventType.WORD_FORMATION,
            "SENTENCE_FORMATION": SemanticEventType.SENTENCE_FORMATION,
            "DIALOGUE_FORMATION": SemanticEventType.DIALOGUE_FORMATION,
            "EVOLUTION_STAGE_CHANGE": SemanticEventType.EVOLUTION_STAGE_CHANGE,
            "CHECKPOINT_CREATED": SemanticEventType.CHECKPOINT_CREATED,
            "REGRESSION_DETECTED": SemanticEventType.REGRESSION_DETECTED,
        }
        return event_type_mapping.get(event.event_type)
    
    def _convert_to_kernel_event(self, semantic_event: SemanticEvent) -> Dict[str, Any]:
        """Convert semantic event to kernel event format"""
        kernel_event = {
            "event_type": f"SEMANTIC_{semantic_event.event_type}",
            "event_uuid": str(semantic_event.event_uuid),
            "timestamp": semantic_event.timestamp.isoformat(),
            "mathematical_validation": semantic_event.mathematical_validation,
            "payload": semantic_event.payload
        }
        
        # Add specific fields based on event type
        if isinstance(semantic_event, (CharacterFormationEvent, WordFormationEvent,
                                      SentenceFormationEvent, DialogueFormationEvent)):
            kernel_event["violation_pressure"] = semantic_event.violation_pressure
            kernel_event["mathematical_consistency"] = semantic_event.mathematical_consistency
            kernel_event["formation_success"] = semantic_event.formation_success
        
        return kernel_event
    
    def _trigger_safety_response(self, event: SemanticEvent, vp: float) -> None:
        """Trigger safety response for high VP or failures"""
        if vp > 0.9:
            # Critical VP - trigger immediate isolation
            self.temporal_isolation.isolate_operation(
                operation_id=str(event.event_uuid),
                reason=f"Critical semantic VP: {vp:.2f}",
                duration=timedelta(minutes=5)
            )
            
            # Publish critical event
            self.event_bus.publish({
                "event_type": "SEMANTIC_CRITICAL_VP",
                "event_uuid": str(event.event_uuid),
                "violation_pressure": vp,
                "action": "temporal_isolation",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif vp > 0.7:
            # High VP - monitor closely
            self.violation_monitor.add_violation(
                entity_id=str(event.event_uuid),
                violation_class=ViolationClass.SEMANTIC,
                severity=vp,
                description=f"High semantic VP in {event.event_type}"
            )
    
    def _handle_kernel_vp_event(self, event: Dict[str, Any]) -> None:
        """Handle VP events from kernel"""
        # Check if it affects semantic operations
        if event.get("entity_type") == "semantic":
            # Trigger checkpoint for safety
            self.state_manager.save_semantic_state(
                state_update={"last_vp_event": event},
                create_checkpoint=True
            )
            
            # Publish semantic-specific response
            self.publish_semantic_event(SemanticViolationEvent(
                event_uuid=uuid.uuid4(),
                event_type="VP_THRESHOLD_EXCEEDED",
                timestamp=datetime.utcnow(),
                violation=None,  # Would create full violation object
                automatic_response="checkpoint_created",
                safety_triggered=True
            ))
    
    def _handle_kernel_stability_event(self, event: Dict[str, Any]) -> None:
        """Handle stability events from kernel"""
        # Resume normal operations if previously restricted
        if event.get("entity_type") == "semantic":
            self.state_manager.save_semantic_state(
                state_update={"stability_restored": True}
            )
    
    def _handle_isolation_event(self, event: Dict[str, Any]) -> None:
        """Handle isolation events from kernel"""
        # Pause semantic operations during isolation
        if event.get("affects_semantic", False):
            self.publish_semantic_event(SemanticEvent(
                event_uuid=uuid.uuid4(),
                event_type="OPERATIONS_PAUSED",
                timestamp=datetime.utcnow(),
                payload={"reason": "temporal_isolation", "duration": event.get("duration")}
            ))
    
    def _handle_isolation_release(self, event: Dict[str, Any]) -> None:
        """Handle isolation release from kernel"""
        # Resume semantic operations
        if event.get("affects_semantic", False):
            self.publish_semantic_event(SemanticEvent(
                event_uuid=uuid.uuid4(),
                event_type="OPERATIONS_RESUMED",
                timestamp=datetime.utcnow(),
                payload={"reason": "isolation_released"}
            ))
    
    def _handle_health_critical(self, event: Dict[str, Any]) -> None:
        """Handle critical health events from kernel"""
        # Create emergency checkpoint
        self.state_manager.save_semantic_state(
            state_update={"health_critical": True},
            create_checkpoint=True
        )
        
        # Reduce semantic operations to essential only
        self.publish_semantic_event(SemanticEvent(
            event_uuid=uuid.uuid4(),
            event_type="EMERGENCY_MODE",
            timestamp=datetime.utcnow(),
            payload={"reason": "system_health_critical"}
        ))
    
    def _handle_system_recovery(self, event: Dict[str, Any]) -> None:
        """Handle system recovery events"""
        # Resume normal semantic operations
        self.state_manager.save_semantic_state(
            state_update={"health_critical": False}
        )
        
        self.publish_semantic_event(SemanticEvent(
            event_uuid=uuid.uuid4(),
            event_type="NORMAL_MODE",
            timestamp=datetime.utcnow(),
            payload={"reason": "system_recovered"}
        ))
    
    def _handle_event_error(self, event: SemanticEvent, handler: Callable, error: Exception) -> None:
        """Handle errors in event processing"""
        error_event = {
            "event_type": "SEMANTIC_EVENT_ERROR",
            "original_event": str(event.event_uuid),
            "handler": handler.__name__,
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log to kernel
        self.event_bus.publish(error_event)
        
        # Track error rate
        error_key = f"{event.event_type}_errors"
        self.event_metrics[error_key] += 1
    
    def _update_monitoring(self, event: SemanticEvent, latency: float) -> None:
        """Update monitoring system with semantic metrics"""
        if self.monitoring:
            # Update golden signals
            self.monitoring.golden_signals.record_request(
                service="semantic_system",
                endpoint=event.event_type,
                duration_ms=latency,
                status_code=200 if event.mathematical_validation else 500
            )
    
    def get_event_metrics(self) -> Dict[str, Any]:
        """Get comprehensive event metrics"""
        with self._bridge_lock:
            # Calculate success rates
            for event_type in self.event_metrics:
                if not event_type.endswith("_errors"):
                    error_key = f"{event_type}_errors"
                    total = self.event_metrics[event_type]
                    errors = self.event_metrics.get(error_key, 0)
                    if total > 0:
                        self.event_success_rates[event_type] = (total - errors) / total
            
            # Calculate average latencies
            avg_latencies = {}
            for event_type, latencies in self.event_latencies.items():
                if latencies:
                    avg_latencies[event_type] = sum(latencies) / len(latencies)
            
            return {
                "event_counts": dict(self.event_metrics),
                "success_rates": dict(self.event_success_rates),
                "average_latencies_ms": avg_latencies,
                "total_events": len(self.event_history),
                "kernel_subscriptions": list(self.kernel_subscriptions),
                "semantic_subscriptions": [s.value for s in self.semantic_subscriptions]
            }
    
    async def publish_async_event(self, event: SemanticEvent) -> None:
        """Publish event asynchronously for non-blocking operations"""
        await self._event_loop.run_in_executor(None, self.publish_semantic_event, event)
    
    def shutdown(self) -> None:
        """Shutdown event bridge cleanly"""
        # Unsubscribe from kernel events
        for event_type in self.kernel_subscriptions:
            # Would call event_bus.unsubscribe if method exists
            pass
        
        # Stop event loop
        self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        self._event_thread.join(timeout=5)
        
        # Final metrics
        metrics = self.get_event_metrics()
        self.event_bus.publish({
            "event_type": "SEMANTIC_BRIDGE_SHUTDOWN",
            "final_metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
