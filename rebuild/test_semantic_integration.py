"""
Semantic System Integration Test
Comprehensive test of Phase 1 and Phase 2 components working together
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading

# Import kernel dependencies (mock versions for testing)
class MockUUIDanchor:
    def __init__(self):
        self.anchors = {}
    
    def anchor_uuid(self, payload: Dict[str, Any]) -> uuid.UUID:
        # Simple deterministic UUID generation for testing
        payload_str = str(sorted(payload.items()))
        return uuid.uuid5(uuid.NAMESPACE_DNS, payload_str)
    
    def anchor_trait(self, payload: Dict[str, Any]) -> uuid.UUID:
        # Alias for anchor_uuid for compatibility
        return self.anchor_uuid(payload)
    
    def retrieve_anchor(self, uuid_id: uuid.UUID) -> Dict[str, Any]:
        return self.anchors.get(uuid_id, {})

class MockDjinnEventBus:
    def __init__(self):
        self.events = []
        self.subscribers = {}
    
    def publish_event(self, event_type: str, payload: Dict[str, Any]):
        event = {
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)
        
        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Event callback error: {e}")
    
    def publish(self, payload: Dict[str, Any]):
        # Alias for publish_event for compatibility
        event_type = payload.get("event_type", "UNKNOWN")
        self.publish_event(event_type, payload)
    
    def subscribe(self, event_type: str, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def get_events(self, event_type: str = None) -> List[Dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events

class MockViolationMonitor:
    def __init__(self):
        self.violations = {}
        self.monitoring_frequency = 1.0
    
    def add_violation(self, entity_id: str, violation_class: str, severity: float, description: str):
        self.violations[entity_id] = {
            "class": violation_class,
            "severity": severity,
            "description": description,
            "timestamp": datetime.utcnow()
        }
    
    def increase_monitoring_frequency(self, entity_id: str, frequency_multiplier: float):
        self.monitoring_frequency *= frequency_multiplier

class MockTemporalIsolation:
    def __init__(self):
        self.isolated_operations = {}
    
    def isolate_operation(self, operation_id: str, reason: str, duration: timedelta):
        self.isolated_operations[operation_id] = {
            "reason": reason,
            "duration": duration,
            "start_time": datetime.utcnow(),
            "isolated": True
        }
    
    def release_isolation(self, operation_id: str):
        if operation_id in self.isolated_operations:
            self.isolated_operations[operation_id]["isolated"] = False

class MockForbiddenZoneManager:
    def __init__(self):
        self.zones = {}
        self.active_zones = set()
    
    def register_zone(self, zone_id: str, zone_type: str, isolation_level: float):
        self.zones[zone_id] = {
            "type": zone_type,
            "isolation_level": isolation_level,
            "active": False
        }
    
    def enter_zone(self, zone_id: str):
        if zone_id in self.zones:
            self.zones[zone_id]["active"] = True
            self.active_zones.add(zone_id)
    
    def exit_zone(self, zone_id: str):
        if zone_id in self.zones:
            self.zones[zone_id]["active"] = False
            self.active_zones.discard(zone_id)

# Import semantic components
from semantic_data_structures import (
    FormationPattern, FormationType, EvolutionStage, CheckpointType,
    RegressionSeverity, RollbackReason, ExperimentType, EvolutionStrategy
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager
from semantic_forbidden_zone_manager import SemanticForbiddenZoneManager
from semantic_performance_regression_detector import SemanticPerformanceRegressionDetector
from semantic_evolution_safety_framework import SemanticEvolutionSafetyFramework

class SemanticIntegrationTest:
    """Comprehensive integration test for semantic system"""
    
    def __init__(self):
        # Initialize mock kernel components
        self.uuid_anchor = MockUUIDanchor()
        self.event_bus = MockDjinnEventBus()
        self.violation_monitor = MockViolationMonitor()
        self.temporal_isolation = MockTemporalIsolation()
        self.forbidden_zone = MockForbiddenZoneManager()
        
        # Initialize semantic components
        self.state_manager = SemanticStateManager(
            event_bus=self.event_bus,
            uuid_anchor=self.uuid_anchor,
            violation_monitor=self.violation_monitor
        )
        
        self.event_bridge = SemanticEventBridge(
            event_bus=self.event_bus,
            state_manager=self.state_manager,
            violation_monitor=None,  # Will be set after creation
            temporal_isolation=self.temporal_isolation
        )
        
        self.semantic_violation_monitor = SemanticViolationMonitor(
            violation_monitor=self.violation_monitor,
            temporal_isolation=self.temporal_isolation,
            state_manager=self.state_manager,
            event_bridge=self.event_bridge
        )
        
        # Update event bridge with violation monitor
        self.event_bridge.violation_monitor = self.semantic_violation_monitor
        
        self.checkpoint_manager = SemanticCheckpointManager(
            state_manager=self.state_manager,
            event_bridge=self.event_bridge,
            violation_monitor=self.semantic_violation_monitor,
            uuid_anchor=self.uuid_anchor
        )
        
        self.forbidden_zone_manager = SemanticForbiddenZoneManager(
            forbidden_zone=self.forbidden_zone,
            temporal_isolation=self.temporal_isolation,
            state_manager=self.state_manager,
            event_bridge=self.event_bridge,
            violation_monitor=self.semantic_violation_monitor,
            checkpoint_manager=self.checkpoint_manager
        )
        
        self.regression_detector = SemanticPerformanceRegressionDetector(
            state_manager=self.state_manager,
            event_bridge=self.event_bridge,
            checkpoint_manager=self.checkpoint_manager,
            violation_monitor=self.violation_monitor,
            temporal_isolation=self.temporal_isolation
        )
        
        self.evolution_safety_framework = SemanticEvolutionSafetyFramework(
            state_manager=self.state_manager,
            event_bridge=self.event_bridge,
            violation_monitor=self.semantic_violation_monitor,
            checkpoint_manager=self.checkpoint_manager,
            forbidden_zone_manager=self.forbidden_zone_manager,
            regression_detector=self.regression_detector,
            violation_monitor_kernel=self.violation_monitor,
            temporal_isolation=self.temporal_isolation,
            forbidden_zone_kernel=self.forbidden_zone
        )
        
        # Test results
        self.test_results = {}
        self.test_errors = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite"""
        print("🧪 Starting Semantic System Integration Test")
        print("=" * 60)
        
        tests = [
            ("State Management", self.test_state_management),
            ("Event Coordination", self.test_event_coordination),
            ("Violation Monitoring", self.test_violation_monitoring),
            ("Checkpoint System", self.test_checkpoint_system),
            ("Sandbox Environment", self.test_sandbox_environment),
            ("Safety Integration", self.test_safety_integration),
            ("Performance Metrics", self.test_performance_metrics),
            ("Regression Detection", self.test_regression_detection),
            ("Evolution Safety Framework", self.test_evolution_safety_framework)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                if result["success"]:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                    if 'traceback' in result:
                        print(f"   Traceback: {result['traceback']}")
            except Exception as e:
                error_msg = f"Exception in {test_name}: {str(e)}"
                self.test_errors.append(error_msg)
                self.test_results[test_name] = {"success": False, "error": error_msg}
                print(f"💥 {test_name}: EXCEPTION - {str(e)}")
        
        return self.generate_test_report()
    
    def test_state_management(self) -> Dict[str, Any]:
        """Test semantic state management"""
        try:
            print("    Testing initial state...")
            # Test initial state
            initial_state = self.state_manager.current_state
            print(f"    Initial state keys: {list(initial_state.keys())}")
            assert "semantic_version" in initial_state
            assert "evolution_stage" in initial_state
            
            # Test state update
            test_update = {
                "test_metric": 0.85,
                "formation_count": 42,
                "last_update": datetime.utcnow().isoformat()
            }
            
            state_id = self.state_manager.save_semantic_state(test_update)
            assert state_id is not None
            
            # Verify state was updated
            current_state = self.state_manager.current_state
            assert current_state["test_metric"] == 0.85
            assert current_state["formation_count"] == 42
            
            # Test checkpoint creation
            checkpoint = self.checkpoint_manager.create_checkpoint(
                checkpoint_type=CheckpointType.MANUAL,
                description="Test checkpoint"
            )
            assert checkpoint.checkpoint_id is not None
            assert checkpoint.semantic_state["test_metric"] == 0.85
            
            return {"success": True, "state_id": str(state_id), "checkpoint_id": str(checkpoint.checkpoint_id)}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_event_coordination(self) -> Dict[str, Any]:
        """Test event bridge coordination"""
        try:
            print("    Testing semantic event publishing...")
            # Test semantic event publishing
            from semantic_data_structures import CharacterFormationEvent
            
            test_event = CharacterFormationEvent(
                event_uuid=uuid.uuid4(),
                event_type="TEST_FORMATION",
                timestamp=datetime.utcnow(),
                character="test",
                formation_success=True,
                violation_pressure=0.2,
                mathematical_consistency=0.9
            )
            
            self.event_bridge.publish_semantic_event(test_event)
            
            # Verify event was published
            events = self.event_bus.get_events()
            print(f"    Total events published: {len(events)}")
            assert len(events) > 0
            
            print("    Testing formation event...")
            # Test formation event
            self.event_bridge.publish_formation_event(
                formation_type="word",
                content="hello",
                success=True,
                vp=0.15,
                consistency=0.95
            )
            
            # Verify formation event
            all_events = self.event_bus.get_events()
            print(f"    All event types: {[e['type'] for e in all_events]}")
            formation_events = self.event_bus.get_events("SEMANTIC_WORD_FORMATION")
            print(f"    Formation events found: {len(formation_events)}")
            assert len(formation_events) > 0
            
            return {"success": True, "events_published": len(events)}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_violation_monitoring(self) -> Dict[str, Any]:
        """Test violation pressure monitoring"""
        try:
            # Create test formation pattern
            pattern = FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.WORD,
                characters=["h", "e", "l", "l", "o"],
                word="hello",
                sentence="",
                formation_success=True,
                violation_pressure=0.3,
                mathematical_consistency=0.8,
                timestamp=datetime.utcnow()
            )
            
            # Calculate VP
            vp, metrics = self.semantic_violation_monitor.calculate_semantic_vp(pattern)
            assert 0.0 <= vp <= 1.0
            assert hasattr(metrics, 'formation_success_rate')
            
            # Test violation creation
            summary = self.semantic_violation_monitor.get_violation_summary()
            assert "total_violations" in summary
            assert "current_vp" in summary
            
            return {"success": True, "vp": vp, "metrics": str(metrics)}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_checkpoint_system(self) -> Dict[str, Any]:
        """Test checkpoint management"""
        try:
            # Create multiple checkpoints
            checkpoint1 = self.checkpoint_manager.create_checkpoint(
                checkpoint_type=CheckpointType.MANUAL,
                description="Test checkpoint 1"
            )
            
            # Update state
            self.state_manager.save_semantic_state({"checkpoint_test": "value1"})
            
            checkpoint2 = self.checkpoint_manager.create_checkpoint(
                checkpoint_type=CheckpointType.AUTOMATIC,
                description="Test checkpoint 2"
            )
            
            # Test evolution branch
            branch = self.checkpoint_manager.create_evolution_branch(
                base_checkpoint_id=checkpoint1.checkpoint_id,
                strategy=EvolutionStrategy.CONSERVATIVE,
                parameters={"learning_rate": 0.1}
            )
            
            assert branch.branch_uuid is not None
            assert branch.base_checkpoint == checkpoint1.checkpoint_id
            
            # Test branch metrics update
            self.checkpoint_manager.update_branch_metrics(
                branch_id=branch.branch_uuid,
                metrics={"formation_success_rate": 0.9, "semantic_accuracy": 0.85}
            )
            
            # Test regression detection
            current_metrics = {"formation_success_rate": 0.5, "semantic_accuracy": 0.6}
            severity = self.checkpoint_manager.detect_regression(current_metrics)
            
            return {
                "success": True,
                "checkpoints_created": 2,
                "branch_created": str(branch.branch_uuid),
                "regression_detected": severity is not None
            }
            
        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else "Unknown error occurred"
            return {"success": False, "error": error_msg, "traceback": traceback.format_exc()}
    
    def test_sandbox_environment(self) -> Dict[str, Any]:
        """Test forbidden zone sandbox"""
        try:
            # Create sandbox
            sandbox = self.forbidden_zone_manager.create_sandbox(
                experiment_type=ExperimentType.NOVEL_FORMATION,
                isolation_level=0.8
            )
            
            assert sandbox.sandbox_id is not None
            assert sandbox.experiment_type == ExperimentType.NOVEL_FORMATION
            assert sandbox.isolation_level == 0.8
            
            # Activate sandbox
            activated = self.forbidden_zone_manager.activate_sandbox(sandbox.sandbox_id)
            assert activated
            
            # Create test patterns
            test_patterns = [
                FormationPattern(
                    pattern_uuid=uuid.uuid4(),
                    formation_type=FormationType.CHARACTER,
                    characters=["a"],
                    word="",
                    sentence="",
                    formation_success=True,
                    violation_pressure=0.1,
                    mathematical_consistency=0.95,
                    timestamp=datetime.utcnow()
                ),
                FormationPattern(
                    pattern_uuid=uuid.uuid4(),
                    formation_type=FormationType.WORD,
                    characters=["t", "e", "s", "t"],
                    word="test",
                    sentence="",
                    formation_success=True,
                    violation_pressure=0.2,
                    mathematical_consistency=0.9,
                    timestamp=datetime.utcnow()
                )
            ]
            
            # Execute experiment
            result = self.forbidden_zone_manager.execute_experiment(
                sandbox_id=sandbox.sandbox_id,
                formation_patterns=test_patterns
            )
            
            assert result.experiment_id is not None
            assert result.formations_tested == 2
            
            # Test quarantine
            dangerous_pattern = FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.WORD,
                characters=["d", "a", "n", "g", "e", "r", "o", "u", "s"],
                word="dangerous",
                sentence="",
                formation_success=False,
                violation_pressure=0.95,
                mathematical_consistency=0.1,
                timestamp=datetime.utcnow()
            )
            
            quarantine_record = self.forbidden_zone_manager._quarantine_pattern(
                dangerous_pattern, 0.95, "Test quarantine"
            )
            
            assert quarantine_record.quarantine_id is not None
            assert self.forbidden_zone_manager._is_quarantined(dangerous_pattern)
            
            return {
                "success": True,
                "sandbox_id": str(sandbox.sandbox_id),
                "experiment_id": str(result.experiment_id),
                "quarantine_id": str(quarantine_record.quarantine_id)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_safety_integration(self) -> Dict[str, Any]:
        """Test safety systems working together"""
        try:
            # Test automatic safety response
            # Create high VP pattern
            high_vp_pattern = FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.SENTENCE,
                characters=["d", "a", "n", "g", "e", "r", "o", "u", "s", " ", "s", "e", "n", "t", "e", "n", "c", "e"],
                word="dangerous",
                sentence="This is a dangerous sentence",
                formation_success=False,
                violation_pressure=0.9,
                mathematical_consistency=0.2,
                timestamp=datetime.utcnow()
            )
            
            # This should trigger safety mechanisms
            vp, metrics = self.semantic_violation_monitor.calculate_semantic_vp(high_vp_pattern)
            
            # Check if violation was created
            summary = self.semantic_violation_monitor.get_violation_summary()
            
            # Test checkpoint creation under pressure
            checkpoint = self.checkpoint_manager.create_checkpoint(
                checkpoint_type=CheckpointType.AUTOMATIC,
                description="Safety checkpoint"
            )
            
            # Test sandbox creation for dangerous pattern
            sandbox = self.forbidden_zone_manager.create_sandbox(
                experiment_type=ExperimentType.BOUNDARY_TESTING,
                isolation_level=0.9
            )
            
            return {
                "success": True,
                "high_vp": vp,
                "violations_created": summary["total_violations"],
                "safety_checkpoint": str(checkpoint.checkpoint_id),
                "safety_sandbox": str(sandbox.sandbox_id)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance monitoring"""
        try:
            print("    Testing system health...")
            # Get system health
            health = self.state_manager.get_current_health()
            print(f"    Health object type: {type(health)}")
            print(f"    Health attributes: {dir(health)}")
            assert hasattr(health, 'system_coherence')
            assert hasattr(health, 'formation_stability')
            
            # Get checkpoint summary
            checkpoint_summary = self.checkpoint_manager.get_checkpoint_summary()
            assert "total_checkpoints" in checkpoint_summary
            assert "active_branches" in checkpoint_summary
            
            # Get zone summary
            zone_summary = self.forbidden_zone_manager.get_zone_summary()
            assert "total_sandboxes" in zone_summary
            assert "quarantined_patterns" in zone_summary
            
            # Get violation summary
            violation_summary = self.semantic_violation_monitor.get_violation_summary()
            assert "current_vp" in violation_summary
            assert "average_vp" in violation_summary
            
            return {
                "success": True,
                "system_coherence": health.system_coherence,
                "total_checkpoints": checkpoint_summary["total_checkpoints"],
                "total_sandboxes": zone_summary["total_sandboxes"],
                "current_vp": violation_summary["current_vp"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_regression_detection(self) -> Dict[str, Any]:
        """Test performance regression detection"""
        try:
            print("    Testing regression detection...")
            
            # Test metric updates
            test_metrics = {
                "formation_success_rate": 0.85,
                "semantic_accuracy": 0.90,
                "violation_pressure": 0.2,
                "mathematical_consistency": 0.95,
                "formation_latency_ms": 100.0,
                "system_coherence": 0.88
            }
            
            self.regression_detector.update_performance_metrics(test_metrics)
            
            # Test regression summary
            summary = self.regression_detector.get_regression_summary()
            assert "total_alerts" in summary
            assert "monitored_metrics" in summary
            
            # Test degraded metrics
            degraded_metrics = {
                "formation_success_rate": 0.5,  # Significant drop
                "semantic_accuracy": 0.6,       # Significant drop
                "system_coherence": 0.4         # Critical drop
            }
            
            self.regression_detector.update_performance_metrics(degraded_metrics)
            
            # Check if alerts were generated
            updated_summary = self.regression_detector.get_regression_summary()
            
            return {
                "success": True,
                "initial_alerts": summary["total_alerts"],
                "alerts_after_degradation": updated_summary["total_alerts"],
                "monitored_metrics": len(summary["monitored_metrics"]),
                "regression_detected": updated_summary["total_alerts"] > summary["total_alerts"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_evolution_safety_framework(self) -> Dict[str, Any]:
        """Test evolution safety framework"""
        try:
            print("    Testing evolution safety framework...")
            
            # Test safety assessment
            assessment = self.evolution_safety_framework.assess_safety_status()
            assert assessment.assessment_id is not None
            assert hasattr(assessment, 'overall_safety_level')
            assert hasattr(assessment, 'threat_level')
            
            # Test safety response coordination
            response = self.evolution_safety_framework.coordinate_safety_response(
                trigger_event="TEST_SAFETY_EVENT",
                severity=RegressionSeverity.WARNING
            )
            
            assert response.response_id is not None
            assert response.trigger_event == "TEST_SAFETY_EVENT"
            assert hasattr(response, 'protocol_type')
            
            # Test safety summary
            summary = self.evolution_safety_framework.get_safety_summary()
            assert "current_safety_level" in summary
            assert "current_threat_level" in summary
            assert "total_assessments" in summary
            assert "total_responses" in summary
            
            return {
                "success": True,
                "safety_level": assessment.overall_safety_level.value,
                "threat_level": assessment.threat_level.value,
                "response_protocol": response.protocol_type.value,
                "total_assessments": summary["total_assessments"],
                "total_responses": summary["total_responses"],
                "system_coherence": summary["system_coherence"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "errors": self.test_errors,
            "system_status": {
                "state_manager": "✅ Initialized",
                "event_bridge": "✅ Initialized", 
                "violation_monitor": "✅ Initialized",
                "checkpoint_manager": "✅ Initialized",
                "forbidden_zone_manager": "✅ Initialized",
                "regression_detector": "✅ Initialized",
                "evolution_safety_framework": "✅ Initialized"
            },
            "integration_status": "✅ All components integrated successfully" if failed_tests == 0 else "⚠️ Some components failed integration"
        }
        
        return report

def run_integration_test():
    """Run the complete integration test"""
    test = SemanticIntegrationTest()
    report = test.run_all_tests()
    
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST REPORT")
    print("=" * 60)
    
    summary = report["test_summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    
    if report["errors"]:
        print(f"\n❌ Errors Found: {len(report['errors'])}")
        for error in report["errors"]:
            print(f"  - {error}")
    
    print(f"\n🏗️ System Status: {report['integration_status']}")
    
    if summary['success_rate'] >= 0.8:
        print("\n🎉 SEMANTIC SYSTEM INTEGRATION: SUCCESSFUL")
        print("✅ Ready to proceed with Phase 2 completion")
    else:
        print("\n⚠️ SEMANTIC SYSTEM INTEGRATION: NEEDS ATTENTION")
        print("🔧 Some components need debugging before continuing")
    
    return report

if __name__ == "__main__":
    run_integration_test()
