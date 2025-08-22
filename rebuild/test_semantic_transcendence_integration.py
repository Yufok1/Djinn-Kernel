# rebuild/test_semantic_transcendence_integration.py
"""
Test Semantic Transcendence Integration
Verifies the Evolution Engine components work with the existing semantic system
"""

import unittest
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any

# Import mock dependencies (reuse from previous test)
class MockUUIDanchor:
    def __init__(self):
        self.base_uuid = uuid.uuid4()
    
    def anchor_uuid(self, payload: Any) -> uuid.UUID:
        if isinstance(payload, dict):
            payload_str = json.dumps(payload, sort_keys=True)
        else:
            payload_str = str(payload)
        return uuid.uuid5(self.base_uuid, payload_str)
    
    def anchor_trait(self, payload: Any) -> uuid.UUID:
        return self.anchor_uuid(payload)

class MockDjinnEventBus:
    def __init__(self):
        self.events = []
        self.handlers = {}
    
    def publish(self, event_type_or_payload, payload: Dict[str, Any] = None):
        if payload is None:
            # Single argument call - treat as payload
            self.events.append({'type': event_type_or_payload.get('event_type', 'unknown'), 'payload': event_type_or_payload})
        else:
            # Two argument call
            self.events.append({'type': event_type_or_payload, 'payload': payload})
    
    def register_handler(self, event_type: str, handler):
        self.handlers[event_type] = handler
    
    def subscribe(self, event_type: str, handler):
        self.register_handler(event_type, handler)

class MockViolationMonitor:
    def __init__(self):
        self.violations = []
    
    def calculate_violation_pressure(self, trait_data: Dict[str, Any]) -> float:
        return 0.5

class MockTraitConvergenceEngine:
    def __init__(self):
        self.convergences = []
    
    def calculate_convergence_stability(self, trait_data: Dict[str, Any]) -> float:
        return 0.7

class MockTemporalIsolationManager:
    def __init__(self):
        self.isolations = []
    
    def create_temporal_isolation(self, operation_id: uuid.UUID, duration: float):
        self.isolations.append({'operation_id': operation_id, 'duration': duration})

# Import semantic components
from semantic_data_structures import (
    SemanticTrait, MathematicalTrait, FormationPattern,
    SemanticComplexity, TraitCategory, FormationType
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager
from local_semantic_database import LocalSemanticDatabase
from mathematical_semantic_api import MathematicalSemanticAPI, QueryType, QueryResult
from semantic_transcendence import (
    SemanticTranscendence, TranscendenceLevel, LearningStrategy, EvolutionPhase
)

class TestSemanticTranscendenceIntegration(unittest.TestCase):
    """Test semantic transcendence integration with existing system"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock dependencies
        self.uuid_anchor = MockUUIDanchor()
        self.event_bus = MockDjinnEventBus()
        self.violation_monitor = MockViolationMonitor()
        self.trait_convergence = MockTraitConvergenceEngine()
        self.temporal_isolation = MockTemporalIsolationManager()
        
        # Create semantic components
        self.state_manager = SemanticStateManager(
            self.event_bus,
            self.uuid_anchor,
            self.violation_monitor
        )
        
        self.event_bridge = SemanticEventBridge(
            self.event_bus,
            self.state_manager,
            self.violation_monitor,
            self.temporal_isolation
        )
        
        self.semantic_violation_monitor = SemanticViolationMonitor(
            self.violation_monitor,
            self.temporal_isolation,
            self.state_manager,
            self.event_bridge
        )
        
        self.checkpoint_manager = SemanticCheckpointManager(
            self.state_manager,
            self.event_bridge,
            self.semantic_violation_monitor,
            self.uuid_anchor
        )
        
        # Create semantic foundation components
        self.semantic_database = LocalSemanticDatabase(
            self.state_manager,
            self.event_bridge,
            self.semantic_violation_monitor,
            self.checkpoint_manager,
            self.uuid_anchor,
            self.trait_convergence
        )
        
        self.semantic_api = MathematicalSemanticAPI(
            self.state_manager,
            self.event_bridge,
            self.semantic_violation_monitor,
            self.checkpoint_manager,
            self.uuid_anchor,
            self.trait_convergence,
            self.semantic_database
        )
        
        # Create semantic transcendence engine
        self.transcendence_engine = SemanticTranscendence(
            self.state_manager,
            self.event_bridge,
            self.semantic_violation_monitor,
            self.checkpoint_manager,
            self.uuid_anchor,
            self.trait_convergence,
            self.semantic_database,
            self.semantic_api
        )
    
    def test_transcendence_engine_initialization(self):
        """Test transcendence engine initialization"""
        # Verify engine was initialized
        self.assertIsNotNone(self.transcendence_engine)
        
        # Verify initial state
        evolution_state = self.transcendence_engine.evolution_state
        self.assertEqual(evolution_state.transcendence_level, TranscendenceLevel.FOUNDATION_DEPENDENT)
        self.assertEqual(evolution_state.evolution_phase, EvolutionPhase.INITIALIZATION)
        self.assertEqual(evolution_state.independence_score, 0.0)
        self.assertEqual(evolution_state.foundation_dependency, 1.0)
        
        # Verify initial strategies
        self.assertIn(LearningStrategy.IMITATION, self.transcendence_engine.active_strategies)
        
        # Verify metrics initialization
        metrics = self.transcendence_engine.evolution_metrics
        self.assertEqual(metrics.total_patterns_discovered, 0)
        self.assertEqual(metrics.autonomous_patterns_created, 0)
        self.assertEqual(metrics.synthesis_success_rate, 0.0)
    
    def test_transcendence_status_retrieval(self):
        """Test transcendence status retrieval"""
        status = self.transcendence_engine.get_transcendence_status()
        
        # Verify status structure
        self.assertIn('evolution_state', status)
        self.assertIn('evolution_metrics', status)
        self.assertIn('discovered_patterns_count', status)
        self.assertIn('autonomous_patterns_count', status)
        self.assertIn('active_strategies', status)
        self.assertIn('strategy_effectiveness', status)
        self.assertIn('transcendence_events_count', status)
        self.assertIn('last_updated', status)
        
        # Verify initial values
        self.assertEqual(status['discovered_patterns_count'], 0)
        self.assertEqual(status['autonomous_patterns_count'], 0)
        self.assertEqual(status['transcendence_events_count'], 0)
        self.assertIsInstance(status['active_strategies'], list)
        self.assertIn('imitation', status['active_strategies'])
    
    def test_semantic_understanding_evolution(self):
        """Test semantic understanding evolution"""
        # Create test formation context
        formation_context = {
            'formation_type': 'word',
            'target_concept': 'love',
            'complexity_level': 'intermediate',
            'semantic_constraints': {}
        }
        
        # Create test formation patterns
        test_patterns = [
            FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.CHARACTER,
                characters=['h', 'v', 'b'],
                word='hvb',
                semantic_traits={'emotional_intensity': 0.7, 'abstractness': 0.4},
                violation_pressure=0.3,
                formation_success=True,
                mathematical_consistency=0.8
            )
        ]
        
        # Evolve semantic understanding
        evolution_results = self.transcendence_engine.evolve_semantic_understanding(
            formation_context, test_patterns
        )
        
        # Verify evolution results structure
        self.assertIn('transcendence_level', evolution_results)
        self.assertIn('evolution_phase', evolution_results)
        self.assertIn('independence_score', evolution_results)
        self.assertIn('learning_velocity', evolution_results)
        self.assertIn('synthesis_capability', evolution_results)
        self.assertIn('foundation_dependency', evolution_results)
        self.assertIn('mathematical_semantic_fluency', evolution_results)
        self.assertIn('pattern_discoveries', evolution_results)
        self.assertIn('autonomous_synthesis', evolution_results)
        self.assertIn('transcendence_events', evolution_results)
        self.assertIn('evolution_guidance', evolution_results)
        self.assertIn('processing_time_ms', evolution_results)
        
        # Verify initial values are reasonable
        self.assertEqual(evolution_results['transcendence_level'], 'foundation_dependent')
        self.assertEqual(evolution_results['evolution_phase'], 'initialization')
        self.assertGreaterEqual(evolution_results['independence_score'], 0.0)
        self.assertLessEqual(evolution_results['independence_score'], 1.0)
        self.assertGreaterEqual(evolution_results['foundation_dependency'], 0.0)
        self.assertLessEqual(evolution_results['foundation_dependency'], 1.0)
        self.assertIsInstance(evolution_results['pattern_discoveries'], int)
        self.assertIsInstance(evolution_results['autonomous_synthesis'], dict)
        self.assertIsInstance(evolution_results['transcendence_events'], list)
        self.assertIsInstance(evolution_results['evolution_guidance'], dict)
        self.assertGreaterEqual(evolution_results['processing_time_ms'], 0.0)
    
    def test_transcendence_level_progression(self):
        """Test transcendence level progression logic"""
        # Test initial level
        self.assertEqual(
            self.transcendence_engine.evolution_state.transcendence_level,
            TranscendenceLevel.FOUNDATION_DEPENDENT
        )
        
        # Test level determination
        current_level = TranscendenceLevel.FOUNDATION_DEPENDENT
        
        # Low readiness score should keep same level
        next_level = self.transcendence_engine._determine_next_transcendence_level(0.1, current_level)
        self.assertEqual(next_level, TranscendenceLevel.FOUNDATION_DEPENDENT)
        
        # High readiness score should advance level
        next_level = self.transcendence_engine._determine_next_transcendence_level(0.4, current_level)
        self.assertEqual(next_level, TranscendenceLevel.GUIDED_LEARNING)
        
        # Very high readiness score from guided learning should advance further
        current_level = TranscendenceLevel.GUIDED_LEARNING
        next_level = self.transcendence_engine._determine_next_transcendence_level(0.6, current_level)
        self.assertEqual(next_level, TranscendenceLevel.AUTONOMOUS_SYNTHESIS)
    
    def test_learning_strategy_adaptation(self):
        """Test learning strategy adaptation based on transcendence level"""
        # Test foundation dependent level strategies
        self.transcendence_engine.evolution_state.transcendence_level = TranscendenceLevel.FOUNDATION_DEPENDENT
        self.transcendence_engine._update_active_strategies_for_level()
        self.assertEqual(self.transcendence_engine.active_strategies, {LearningStrategy.IMITATION})
        
        # Test guided learning level strategies
        self.transcendence_engine.evolution_state.transcendence_level = TranscendenceLevel.GUIDED_LEARNING
        self.transcendence_engine._update_active_strategies_for_level()
        expected_strategies = {LearningStrategy.IMITATION, LearningStrategy.INTERPOLATION}
        self.assertEqual(self.transcendence_engine.active_strategies, expected_strategies)
        
        # Test autonomous synthesis level strategies
        self.transcendence_engine.evolution_state.transcendence_level = TranscendenceLevel.AUTONOMOUS_SYNTHESIS
        self.transcendence_engine._update_active_strategies_for_level()
        expected_strategies = {LearningStrategy.INTERPOLATION, LearningStrategy.EXTRAPOLATION, LearningStrategy.SYNTHESIS}
        self.assertEqual(self.transcendence_engine.active_strategies, expected_strategies)
        
        # Test mathematical semantic level strategies
        self.transcendence_engine.evolution_state.transcendence_level = TranscendenceLevel.MATHEMATICAL_SEMANTIC
        self.transcendence_engine._update_active_strategies_for_level()
        expected_strategies = {LearningStrategy.SYNTHESIS, LearningStrategy.TRANSCENDENCE}
        self.assertEqual(self.transcendence_engine.active_strategies, expected_strategies)
        
        # Test transcendent level strategies
        self.transcendence_engine.evolution_state.transcendence_level = TranscendenceLevel.TRANSCENDENT
        self.transcendence_engine._update_active_strategies_for_level()
        self.assertEqual(self.transcendence_engine.active_strategies, {LearningStrategy.TRANSCENDENCE})
    
    def test_foundation_usage_calculation(self):
        """Test foundation usage calculation"""
        # Test with no queries
        usage = self.transcendence_engine._calculate_foundation_usage({})
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 1.0)
        
        # Test with formation context
        formation_context = {
            'recent_queries': 5,
            'semantic_lookups': 3,
            'formation_guidance_requests': 2
        }
        usage = self.transcendence_engine._calculate_foundation_usage(formation_context)
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 1.0)
    
    def test_autonomous_pattern_usage_calculation(self):
        """Test autonomous pattern usage calculation"""
        # Test with no patterns
        usage = self.transcendence_engine._calculate_autonomous_pattern_usage([])
        self.assertEqual(usage, 0.0)
        
        # Test with test patterns (none autonomous initially)
        test_patterns = [
            FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.CHARACTER,
                characters=['t', 'e', 's', 't'],
                word='test',
                semantic_traits={'abstractness': 0.4},
                violation_pressure=0.3,
                formation_success=True,
                mathematical_consistency=0.8
            )
        ]
        usage = self.transcendence_engine._calculate_autonomous_pattern_usage(test_patterns)
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 1.0)
    
    def test_mathematical_coherence_calculation(self):
        """Test mathematical coherence calculation"""
        # Test with no patterns
        coherence = self.transcendence_engine._calculate_mathematical_coherence([])
        self.assertEqual(coherence, 0.0)
        
        # Test with test patterns
        test_patterns = [
            FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.CHARACTER,
                characters=['l', 'o', 'v', 'e'],
                word='love',
                semantic_traits={'abstractness': 0.4, 'emotional_intensity': 0.8},
                violation_pressure=0.3,
                formation_success=True,
                mathematical_consistency=0.8
            ),
            FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.WORD,
                characters=['t', 'r', 'u', 't', 'h'],
                word='truth',
                semantic_traits={'abstractness': 0.5, 'complexity': 0.7},
                violation_pressure=0.4,
                formation_success=True,
                mathematical_consistency=0.7
            )
        ]
        coherence = self.transcendence_engine._calculate_mathematical_coherence(test_patterns)
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    def test_advancement_thresholds(self):
        """Test advancement thresholds for transcendence levels"""
        # Test all transcendence levels have thresholds
        for level in TranscendenceLevel:
            threshold = self.transcendence_engine._get_advancement_threshold(level)
            self.assertGreaterEqual(threshold, 0.0)
            self.assertLessEqual(threshold, 1.0)
        
        # Test threshold progression (should generally increase)
        foundation_threshold = self.transcendence_engine._get_advancement_threshold(TranscendenceLevel.FOUNDATION_DEPENDENT)
        guided_threshold = self.transcendence_engine._get_advancement_threshold(TranscendenceLevel.GUIDED_LEARNING)
        autonomous_threshold = self.transcendence_engine._get_advancement_threshold(TranscendenceLevel.AUTONOMOUS_SYNTHESIS)
        mathematical_threshold = self.transcendence_engine._get_advancement_threshold(TranscendenceLevel.MATHEMATICAL_SEMANTIC)
        transcendent_threshold = self.transcendence_engine._get_advancement_threshold(TranscendenceLevel.TRANSCENDENT)
        
        self.assertLessEqual(foundation_threshold, guided_threshold)
        self.assertLessEqual(guided_threshold, autonomous_threshold)
        self.assertLessEqual(autonomous_threshold, mathematical_threshold)
        self.assertLessEqual(mathematical_threshold, transcendent_threshold)
    
    def test_event_handler_registration(self):
        """Test event handler registration"""
        # Check that the transcendence engine was created without errors
        self.assertIsNotNone(self.transcendence_engine)
        
        # Verify that handlers are registered (they should exist in event bridge)
        if hasattr(self.event_bridge, 'string_handlers'):
            self.assertIn("FORMATION_PATTERN_DISCOVERED", self.event_bridge.string_handlers)
            self.assertIn("SEMANTIC_QUERY_COMPLETED", self.event_bridge.string_handlers)
            self.assertIn("MATHEMATICAL_BREAKTHROUGH", self.event_bridge.string_handlers)
        
        # Test event handling by calling the handler methods directly
        test_event = {
            'pattern_type': 'test_pattern',
            'discovery_method': 'synthesis',
            'confidence': 0.8
        }
        
        # This should not crash
        try:
            self.transcendence_engine._handle_pattern_discovery(test_event)
            self.transcendence_engine._handle_query_completion(test_event)
            self.transcendence_engine._handle_mathematical_breakthrough(test_event)
        except Exception as e:
            self.fail(f"Event handlers should not crash: {e}")
    
    def test_transcendence_integration_with_semantic_system(self):
        """Test transcendence engine integration with semantic system"""
        # Test that transcendence engine can access semantic database
        love_reference = self.semantic_database.get_semantic_data('love')
        self.assertIsNotNone(love_reference)
        
        # Test that transcendence engine can use semantic API
        query_metrics = self.semantic_api.get_query_metrics()
        self.assertIsInstance(query_metrics, dict)
        
        # Test evolution with real semantic context
        formation_context = {
            'formation_type': 'word',
            'target_concept': 'love',
            'semantic_reference': love_reference,
            'complexity_level': 'intermediate'
        }
        
        test_patterns = [
            FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.WORD,
                characters=['l', 'o', 'v', 'e'],
                word='love',
                sentence='I feel love.',
                semantic_traits={'valence': 0.8, 'arousal': 0.6, 'emotional_intensity': 0.8},
                violation_pressure=0.2,
                formation_success=True,
                mathematical_consistency=0.8
            )
        ]
        
        # This should integrate smoothly
        evolution_results = self.transcendence_engine.evolve_semantic_understanding(
            formation_context, test_patterns
        )
        
        self.assertIsInstance(evolution_results, dict)
        self.assertIn('transcendence_level', evolution_results)
        self.assertIn('independence_score', evolution_results)
    
    def test_pattern_discovery_framework(self):
        """Test pattern discovery framework"""
        # Create test context and patterns
        formation_context = {'formation_type': 'sentence', 'complexity': 'advanced'}
        current_patterns = [
            FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.SENTENCE,
                characters=['t', 'e', 's', 't'],
                word='test',
                sentence='This is a test sentence.',
                semantic_traits={'coherence': 0.8, 'complexity': 0.7},
                violation_pressure=0.3,
                formation_success=True,
                mathematical_consistency=0.8
            )
        ]
        
        # Test pattern discovery methods (they return empty lists by default)
        math_patterns = self.transcendence_engine._discover_mathematical_patterns(current_patterns)
        self.assertIsInstance(math_patterns, list)
        
        semantic_patterns = self.transcendence_engine._discover_semantic_structure_patterns(formation_context)
        self.assertIsInstance(semantic_patterns, list)
        
        recursive_patterns = self.transcendence_engine._discover_recursive_patterns(formation_context, current_patterns)
        self.assertIsInstance(recursive_patterns, list)
    
    def test_learning_strategy_application_framework(self):
        """Test learning strategy application framework"""
        formation_context = {'formation_type': 'dialogue', 'complexity': 'expert'}
        current_patterns = []
        
        # Test each learning strategy application
        imitation_results = self.transcendence_engine._apply_imitation_learning(formation_context, current_patterns)
        self.assertIsInstance(imitation_results, dict)
        self.assertIn('patterns_imitated', imitation_results)
        self.assertIn('imitation_success_rate', imitation_results)
        
        interpolation_results = self.transcendence_engine._apply_interpolation_learning(formation_context, current_patterns)
        self.assertIsInstance(interpolation_results, dict)
        self.assertIn('interpolation_pairs', interpolation_results)
        self.assertIn('patterns_created', interpolation_results)
        
        extrapolation_results = self.transcendence_engine._apply_extrapolation_learning(formation_context, current_patterns)
        self.assertIsInstance(extrapolation_results, dict)
        self.assertIn('extrapolation_opportunities', extrapolation_results)
        self.assertIn('patterns_created', extrapolation_results)
        
        synthesis_results = self.transcendence_engine._apply_synthesis_learning(formation_context, current_patterns)
        self.assertIsInstance(synthesis_results, dict)
        self.assertIn('synthesis_opportunities', synthesis_results)
        self.assertIn('patterns_created', synthesis_results)
        
        transcendence_results = self.transcendence_engine._apply_transcendence_learning(formation_context, current_patterns)
        self.assertIsInstance(transcendence_results, dict)
        self.assertIn('transcendence_opportunities', transcendence_results)
        self.assertIn('patterns_created', transcendence_results)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
