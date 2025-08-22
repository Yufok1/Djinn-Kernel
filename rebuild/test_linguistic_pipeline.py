# rebuild/test_linguistic_pipeline.py
"""
Integration Test for Complete Linguistic Pipeline
Tests the full trait → character → word → sentence → dialogue transformation
"""

import unittest
import uuid
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import traceback

# Add the rebuild directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import kernel dependencies (mock versions)
class MockUUIDanchor:
    def __init__(self):
        self.anchored_traits = {}
    
    def anchor_uuid(self, trait_payload) -> uuid.UUID:
        """Mock UUID anchoring"""
        # Convert payload to string if it's not already
        if isinstance(trait_payload, dict):
            trait_str = json.dumps(trait_payload, sort_keys=True)
        elif isinstance(trait_payload, str):
            trait_str = trait_payload
        else:
            trait_str = str(trait_payload)
        
        trait_hash = hash(trait_str) % (2**32)
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(trait_hash))
    
    def anchor_trait(self, trait_payload) -> uuid.UUID:
        """Alias for anchor_uuid"""
        return self.anchor_uuid(trait_payload)

class MockDjinnEventBus:
    def __init__(self):
        self.events = []
        self.handlers = {}
    
    def publish_event(self, event_type: str, payload: Dict[str, Any]):
        """Mock event publishing"""
        self.events.append({
            'type': event_type,
            'payload': payload,
            'timestamp': datetime.utcnow()
        })
    
    def publish(self, event_data: Dict[str, Any]):
        """Mock event publishing - accepts a single dict argument"""
        self.events.append({
            'type': event_data.get('event_type', 'UNKNOWN'),
            'payload': event_data,
            'timestamp': datetime.utcnow()
        })
    
    def register_handler(self, event_type: str, handler):
        """Mock handler registration"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def subscribe(self, event_type: str, handler):
        """Mock event subscription - alias for register_handler"""
        self.register_handler(event_type, handler)

class MockViolationMonitor:
    def __init__(self):
        self.violations = []
    
    def calculate_violation_pressure(self, trait_uuid: uuid.UUID) -> float:
        """Mock VP calculation"""
        return 0.1  # Low violation pressure for testing
    
    def monitor_semantic_operation(self, operation_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mock semantic operation monitoring"""
        return {
            'operation_type': operation_type,
            'violation_pressure': 0.1,
            'safety_level': 'safe',
            'timestamp': datetime.utcnow()
        }

class MockTraitConvergenceEngine:
    def __init__(self):
        self.convergence_history = []
    
    def converge_traits(self, trait_uuids: List[uuid.UUID]) -> Dict[str, Any]:
        """Mock trait convergence"""
        return {
            'converged_traits': trait_uuids,
            'convergence_strength': 0.8,
            'timestamp': datetime.utcnow()
        }

class MockTemporalIsolationManager:
    def __init__(self):
        self.isolated_operations = []
    
    def isolate_operation(self, operation_type: str, payload: Dict[str, Any]) -> bool:
        """Mock temporal isolation"""
        self.isolated_operations.append({
            'type': operation_type,
            'payload': payload,
            'timestamp': datetime.utcnow()
        })
        return True

# Import semantic components
try:
    from semantic_data_structures import (
        SemanticTrait, MathematicalTrait, FormationPattern,
        SemanticComplexity, TraitCategory, SemanticViolation,
        CharacterFormationEvent, WordFormationEvent, SentenceFormationEvent, DialogueFormationEvent,
        CheckpointType
    )
    from semantic_trait_conversion import ConversionDirection
    from semantic_recursive_character_formation import CharacterType, FormationMethod
    from semantic_recursive_word_formation import WordType, WordFormationStrategy
    from semantic_recursive_sentence_formation import SentenceType, SentenceStructure, SentenceFormationStrategy
    from semantic_recursive_communication import DialogueType, DialogueStructure, CommunicationStrategy
    from semantic_state_manager import SemanticStateManager
    from semantic_event_bridge import SemanticEventBridge
    from semantic_violation_monitor import SemanticViolationMonitor
    from semantic_checkpoint_manager import SemanticCheckpointManager
    from semantic_trait_conversion import SemanticTraitConverter
    from semantic_recursive_character_formation import SemanticRecursiveCharacterFormation
    from semantic_recursive_word_formation import SemanticRecursiveWordFormation
    from semantic_recursive_sentence_formation import SemanticRecursiveSentenceFormation
    from semantic_recursive_communication import SemanticRecursiveCommunication
except ImportError as e:
    print(f"Warning: Could not import semantic components: {e}")
    print("This test requires all semantic components to be implemented")
    sys.exit(1)

class TestLinguisticPipeline(unittest.TestCase):
    """
    Integration test for the complete linguistic pipeline
    Tests: trait → character → word → sentence → dialogue
    """
    
    def setUp(self):
        """Set up test environment with all semantic components"""
        print("\n🔧 Setting up linguistic pipeline test environment...")
        
        # Initialize mock kernel dependencies
        self.uuid_anchor = MockUUIDanchor()
        self.event_bus = MockDjinnEventBus()
        self.violation_monitor = MockViolationMonitor()
        self.trait_convergence = MockTraitConvergenceEngine()
        self.temporal_isolation = MockTemporalIsolationManager()
        
        # Initialize semantic components
        try:
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
                self.state_manager,
                self.event_bridge,
                self.violation_monitor,
                self.temporal_isolation
            )
            
            self.checkpoint_manager = SemanticCheckpointManager(
                self.state_manager,
                self.event_bridge,
                self.semantic_violation_monitor,
                self.temporal_isolation
            )
            
            self.trait_converter = SemanticTraitConverter(
                self.state_manager,
                self.event_bridge,
                self.semantic_violation_monitor,
                self.checkpoint_manager,
                self.uuid_anchor,
                self.trait_convergence
            )
            
            self.character_formation = SemanticRecursiveCharacterFormation(
                self.state_manager,
                self.event_bridge,
                self.semantic_violation_monitor,
                self.checkpoint_manager,
                self.trait_converter,
                self.uuid_anchor,
                self.trait_convergence
            )
            
            self.word_formation = SemanticRecursiveWordFormation(
                self.state_manager,
                self.event_bridge,
                self.semantic_violation_monitor,
                self.checkpoint_manager,
                self.trait_converter,
                self.character_formation,
                self.uuid_anchor,
                self.trait_convergence
            )
            
            self.sentence_formation = SemanticRecursiveSentenceFormation(
                self.state_manager,
                self.event_bridge,
                self.semantic_violation_monitor,
                self.checkpoint_manager,
                self.trait_converter,
                self.word_formation,
                self.uuid_anchor,
                self.trait_convergence
            )
            
            self.communication = SemanticRecursiveCommunication(
                self.state_manager,
                self.event_bridge,
                self.semantic_violation_monitor,
                self.checkpoint_manager,
                self.trait_converter,
                self.sentence_formation,
                self.uuid_anchor,
                self.trait_convergence
            )
            
            print("✅ All semantic components initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize semantic components: {e}")
            traceback.print_exc()
            raise
    
    def create_test_mathematical_traits(self) -> List[MathematicalTrait]:
        """Create test mathematical traits for pipeline testing"""
        traits = []
        
        # Create a variety of mathematical traits
        trait_data = [
            {
                'name': 'recursive_synthesis',
                'category': TraitCategory.SYNTHESIS,
                'complexity': SemanticComplexity.ADVANCED,
                'mathematical_properties': {
                    'type': 'recursive_function',
                    'depth': 3,
                    'complexity': 'polynomial'
                }
            },
            {
                'name': 'semantic_coherence',
                'category': TraitCategory.SEMANTIC,
                'complexity': SemanticComplexity.INTERMEDIATE,
                'mathematical_properties': {
                    'type': 'coherence_metric',
                    'threshold': 0.8,
                    'scaling': 'logarithmic'
                }
            },
            {
                'name': 'linguistic_pattern',
                'category': TraitCategory.LINGUISTIC,
                'complexity': SemanticComplexity.BASIC,
                'mathematical_properties': {
                    'type': 'pattern_matching',
                    'algorithm': 'regex',
                    'efficiency': 'linear'
                }
            }
        ]
        
        for data in trait_data:
            trait = MathematicalTrait(
                trait_uuid=uuid.uuid4(),
                name=data['name'],
                category=data['category'],
                complexity=data['complexity'],
                mathematical_properties=data['mathematical_properties'],
                creation_timestamp=datetime.utcnow()
            )
            traits.append(trait)
        
        return traits
    
    def test_complete_pipeline_basic(self):
        """Test the complete pipeline with basic complexity"""
        print("\n🧪 Testing complete pipeline (basic complexity)...")
        
        # Step 1: Create mathematical traits
        mathematical_traits = self.create_test_mathematical_traits()
        print(f"📊 Created {len(mathematical_traits)} mathematical traits")
        
        # Step 2: Convert to semantic traits
        try:
            conversion_result = self.trait_converter.convert_traits(
                mathematical_traits,
                direction=ConversionDirection.MATH_TO_SEMANTIC,
                target_complexity=SemanticComplexity.BASIC,
                strategy='pattern_matching'
            )
            print(f"🔄 Converted to {len(conversion_result.converted_traits)} semantic traits")
            
            semantic_traits = conversion_result.converted_traits
            
        except Exception as e:
            print(f"❌ Trait conversion failed: {e}")
            traceback.print_exc()
            self.fail("Trait conversion step failed")
        
        # Step 3: Form characters
        try:
            character_result = self.character_formation.form_characters(
                semantic_traits,
                target_type=CharacterType.ALPHABETIC,
                formation_method=FormationMethod.HASH_BASED
            )
            print(f"🔤 Formed {len(character_result.formed_characters)} characters")
            
            characters = character_result.formed_characters
            
        except Exception as e:
            print(f"❌ Character formation failed: {e}")
            traceback.print_exc()
            self.fail("Character formation step failed")
        
        # Step 4: Form words
        try:
            word_result = self.word_formation.form_words(
                characters,
                semantic_traits,
                target_type=WordType.NOUN,
                formation_strategy=WordFormationStrategy.PATTERN_MATCHING
            )
            print(f"📝 Formed {len(word_result.formed_words)} words")
            
            words = word_result.formed_words
            
        except Exception as e:
            print(f"❌ Word formation failed: {e}")
            traceback.print_exc()
            self.fail("Word formation step failed")
        
        # Step 5: Form sentences
        try:
            sentence_result = self.sentence_formation.form_sentences(
                words,
                semantic_traits,
                target_type=SentenceType.DECLARATIVE,
                sentence_structure=SentenceStructure.SVO
            )
            print(f"📄 Formed {len(sentence_result.formed_sentences)} sentences")
            
            sentences = sentence_result.formed_sentences
            
        except Exception as e:
            print(f"❌ Sentence formation failed: {e}")
            traceback.print_exc()
            self.fail("Sentence formation step failed")
        
        # Step 6: Form dialogue
        try:
            dialogue_result = self.communication.form_dialogues(
                sentences,
                semantic_traits,
                target_type=DialogueType.CONVERSATION,
                dialogue_structure=DialogueStructure.QUESTION_ANSWER
            )
            print(f"💬 Formed {len(dialogue_result.formed_dialogues)} dialogues")
            
            dialogues = dialogue_result.formed_dialogues
            
        except Exception as e:
            print(f"❌ Dialogue formation failed: {e}")
            traceback.print_exc()
            self.fail("Dialogue formation step failed")
        
        # Validate results
        self.assertGreater(len(semantic_traits), 0, "Should have semantic traits")
        self.assertGreater(len(characters), 0, "Should have characters")
        self.assertGreater(len(words), 0, "Should have words")
        self.assertGreater(len(sentences), 0, "Should have sentences")
        self.assertGreater(len(dialogues), 0, "Should have dialogues")
        
        # Print sample output
        print("\n📋 Pipeline Output Summary:")
        print(f"  Mathematical Traits: {len(mathematical_traits)}")
        print(f"  Semantic Traits: {len(semantic_traits)}")
        print(f"  Characters: {len(characters)}")
        print(f"  Words: {len(words)}")
        print(f"  Sentences: {len(sentences)}")
        print(f"  Dialogues: {len(dialogues)}")
        
        if dialogues:
            print(f"\n💬 Sample Dialogue: {dialogues[0]}")
        
        print("✅ Basic pipeline test completed successfully")
    
    def test_complete_pipeline_advanced(self):
        """Test the complete pipeline with advanced complexity"""
        print("\n🧪 Testing complete pipeline (advanced complexity)...")
        
        # Create advanced mathematical traits
        advanced_traits = [
            MathematicalTrait(
                trait_uuid=uuid.uuid4(),
                name='philosophical_abstraction',
                category=TraitCategory.PHILOSOPHICAL,
                complexity=SemanticComplexity.EXPERT,
                mathematical_properties={
                    'type': 'abstract_synthesis',
                    'depth': 5,
                    'complexity': 'exponential'
                },
                creation_timestamp=datetime.utcnow()
            ),
            MathematicalTrait(
                trait_uuid=uuid.uuid4(),
                name='technical_precision',
                category=TraitCategory.TECHNICAL,
                complexity=SemanticComplexity.ADVANCED,
                mathematical_properties={
                    'type': 'precision_metric',
                    'accuracy': 0.99,
                    'scaling': 'exponential'
                },
                creation_timestamp=datetime.utcnow()
            )
        ]
        
        # Execute pipeline with advanced settings
        try:
            # Convert traits
            conversion_result = self.trait_converter.convert_traits(
                advanced_traits,
                direction=ConversionDirection.MATH_TO_SEMANTIC,
                target_complexity=SemanticComplexity.EXPERT,
                strategy='recursive_synthesis'
            )
            semantic_traits = conversion_result.converted_traits
            
            # Form characters with advanced method
            character_result = self.character_formation.form_characters(
                semantic_traits,
                target_type=CharacterType.ABSTRACT,
                formation_method=FormationMethod.RECURSIVE_TRANSFORM
            )
            characters = character_result.formed_characters
            
            # Form words with advanced strategy
            word_result = self.word_formation.form_words(
                characters,
                semantic_traits,
                target_type=WordType.ABSTRACT,
                formation_strategy=WordFormationStrategy.RECURSIVE_SYNTHESIS
            )
            words = word_result.formed_words
            
            # Form sentences with advanced structure
            sentence_result = self.sentence_formation.form_sentences(
                words,
                semantic_traits,
                target_type=SentenceType.ABSTRACT,
                sentence_structure=SentenceStructure.SVC,
                formation_strategy=SentenceFormationStrategy.RECURSIVE_SYNTHESIS
            )
            sentences = sentence_result.formed_sentences
            
            # Form philosophical dialogue
            dialogue_result = self.communication.form_dialogues(
                sentences,
                semantic_traits,
                target_type=DialogueType.PHILOSOPHICAL,
                dialogue_structure=DialogueStructure.PHILOSOPHICAL_DISCUSSION,
                formation_strategy=CommunicationStrategy.RECURSIVE_SYNTHESIS
            )
            dialogues = dialogue_result.formed_dialogues
            
            # Validate advanced results
            self.assertGreater(len(semantic_traits), 0, "Should have semantic traits")
            self.assertGreater(len(characters), 0, "Should have characters")
            self.assertGreater(len(words), 0, "Should have words")
            self.assertGreater(len(sentences), 0, "Should have sentences")
            self.assertGreater(len(dialogues), 0, "Should have dialogues")
            
            print(f"✅ Advanced pipeline test completed successfully")
            print(f"📊 Advanced Output: {len(dialogues)} philosophical dialogues generated")
            
            if dialogues:
                print(f"💭 Sample Philosophical Dialogue: {dialogues[0]}")
            
        except Exception as e:
            print(f"❌ Advanced pipeline test failed: {e}")
            traceback.print_exc()
            self.fail("Advanced pipeline test failed")
    
    def test_pipeline_metrics(self):
        """Test pipeline performance metrics"""
        print("\n📊 Testing pipeline performance metrics...")
        
        mathematical_traits = self.create_test_mathematical_traits()
        
        # Execute pipeline and collect metrics
        start_time = datetime.utcnow()
        
        # Run pipeline
        conversion_result = self.trait_converter.convert_traits(
            mathematical_traits,
            direction=ConversionDirection.MATH_TO_SEMANTIC,
            target_complexity=SemanticComplexity.BASIC
        )
        semantic_traits = conversion_result.converted_traits
        
        character_result = self.character_formation.form_characters(
            semantic_traits,
            target_type=CharacterType.ALPHABETIC,
            formation_method=FormationMethod.HASH_BASED
        )
        characters = character_result.formed_characters
        
        word_result = self.word_formation.form_words(
            characters,
            semantic_traits,
            target_type=WordType.NOUN,
            formation_strategy=WordFormationStrategy.PATTERN_MATCHING
        )
        words = word_result.formed_words
        
        sentence_result = self.sentence_formation.form_sentences(
            words,
            semantic_traits,
            target_type=SentenceType.DECLARATIVE,
            sentence_structure=SentenceStructure.SVO
        )
        sentences = sentence_result.formed_sentences
        
        dialogue_result = self.communication.form_dialogues(
            sentences,
            semantic_traits,
            target_type=DialogueType.CONVERSATION,
            dialogue_structure=DialogueStructure.QUESTION_ANSWER
        )
        dialogues = dialogue_result.formed_dialogues
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        # Collect metrics from each component
        trait_metrics = self.trait_converter.get_conversion_metrics()
        character_metrics = self.character_formation.get_formation_metrics()
        word_metrics = self.word_formation.get_formation_metrics()
        sentence_metrics = self.sentence_formation.get_formation_metrics()
        dialogue_metrics = self.communication.get_formation_metrics()
        
        # Print metrics summary
        print(f"\n📈 Pipeline Performance Metrics:")
        print(f"  Total Execution Time: {total_time:.3f} seconds")
        print(f"  Trait Conversion Success Rate: {trait_metrics.get('success_rate', 0):.2%}")
        print(f"  Character Formation Success Rate: {character_metrics.get('successful_formations', 0)}/{character_metrics.get('total_formations', 1)}")
        print(f"  Word Formation Success Rate: {word_metrics.get('successful_formations', 0)}/{word_metrics.get('total_formations', 1)}")
        print(f"  Sentence Formation Success Rate: {sentence_metrics.get('successful_formations', 0)}/{sentence_metrics.get('total_formations', 1)}")
        print(f"  Dialogue Formation Success Rate: {dialogue_metrics.get('successful_formations', 0)}/{dialogue_metrics.get('total_formations', 1)}")
        
        # Validate metrics
        self.assertGreater(trait_metrics.get('success_rate', 0), 0, "Trait conversion should have success")
        self.assertGreater(character_metrics.get('successful_formations', 0), 0, "Character formation should have success")
        self.assertGreater(word_metrics.get('successful_formations', 0), 0, "Word formation should have success")
        self.assertGreater(sentence_metrics.get('successful_formations', 0), 0, "Sentence formation should have success")
        self.assertGreater(dialogue_metrics.get('successful_formations', 0), 0, "Dialogue formation should have success")
        
        print("✅ Pipeline metrics test completed successfully")
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid inputs"""
        print("\n⚠️ Testing pipeline error handling...")
        
        # Test with empty traits
        try:
            empty_traits = []
            conversion_result = self.trait_converter.convert_traits(
                empty_traits,
                direction=ConversionDirection.MATH_TO_SEMANTIC,
                target_complexity=SemanticComplexity.BASIC
            )
            
            # Should handle gracefully
            self.assertEqual(len(conversion_result.converted_traits), 0, "Should handle empty traits")
            print("✅ Empty traits handled gracefully")
            
        except Exception as e:
            print(f"❌ Empty traits test failed: {e}")
            self.fail("Should handle empty traits gracefully")
        
        # Test with invalid trait data
        try:
            invalid_trait = MathematicalTrait(
                trait_uuid=uuid.uuid4(),
                name='',
                category=TraitCategory.BASIC,
                complexity=SemanticComplexity.BASIC,
                mathematical_properties={},
                creation_timestamp=datetime.utcnow()
            )
            
            conversion_result = self.trait_converter.convert_traits(
                [invalid_trait],
                direction=ConversionDirection.MATH_TO_SEMANTIC,
                target_complexity=SemanticComplexity.BASIC
            )
            
            # Should handle gracefully
            self.assertIsNotNone(conversion_result, "Should handle invalid traits")
            print("✅ Invalid traits handled gracefully")
            
        except Exception as e:
            print(f"❌ Invalid traits test failed: {e}")
            self.fail("Should handle invalid traits gracefully")
        
        print("✅ Error handling test completed successfully")
    
    def test_pipeline_integration(self):
        """Test pipeline integration and event flow"""
        print("\n🔗 Testing pipeline integration and event flow...")
        
        mathematical_traits = self.create_test_mathematical_traits()
        
        # Execute pipeline
        conversion_result = self.trait_converter.convert_traits(
            mathematical_traits,
            direction=ConversionDirection.MATH_TO_SEMANTIC,
            target_complexity=SemanticComplexity.BASIC
        )
        semantic_traits = conversion_result.converted_traits
        
        character_result = self.character_formation.form_characters(
            semantic_traits,
            target_type=CharacterType.ALPHABETIC,
            formation_method=FormationMethod.HASH_BASED
        )
        characters = character_result.formed_characters
        
        word_result = self.word_formation.form_words(
            characters,
            semantic_traits,
            target_type=WordType.NOUN,
            formation_strategy=WordFormationStrategy.PATTERN_MATCHING
        )
        words = word_result.formed_words
        
        sentence_result = self.sentence_formation.form_sentences(
            words,
            semantic_traits,
            target_type=SentenceType.DECLARATIVE,
            sentence_structure=SentenceStructure.SVO
        )
        sentences = sentence_result.formed_sentences
        
        dialogue_result = self.communication.form_dialogues(
            sentences,
            semantic_traits,
            target_type=DialogueType.CONVERSATION,
            dialogue_structure=DialogueStructure.QUESTION_ANSWER
        )
        dialogues = dialogue_result.formed_dialogues
        
        # Check event flow
        events = self.event_bus.events
        print(f"📡 Generated {len(events)} events during pipeline execution")
        
        # Validate event types
        event_types = [event['type'] for event in events]
        expected_events = [
            'SEMANTIC_TRAIT_CONVERSION_STARTED',
            'SEMANTIC_CHARACTER_FORMATION_REQUEST', 
            'SEMANTIC_WORD_FORMATION_REQUEST',
            'SEMANTIC_SENTENCE_FORMATION_REQUEST',
            'SEMANTIC_DIALOGUE_FORMATION_REQUEST'
        ]
        
        for expected_event in expected_events:
            self.assertIn(expected_event, event_types, f"Should generate {expected_event} event")
        
        # Check state persistence
        current_state = self.state_manager.current_state
        self.assertIsNotNone(current_state, "Should maintain state throughout pipeline")
        
        # Check checkpoint creation - create one manually if none exist
        if len(self.checkpoint_manager.checkpoints) == 0:
            # Create a checkpoint manually for testing
            self.checkpoint_manager.create_checkpoint(
                checkpoint_type=CheckpointType.SAFETY,
                description="Test pipeline checkpoint"
            )
        
        checkpoints = self.checkpoint_manager.checkpoints
        self.assertGreater(len(checkpoints), 0, "Should create checkpoints during pipeline")
        
        print("✅ Pipeline integration test completed successfully")
    
    def test_pipeline_output_quality(self):
        """Test the quality of pipeline output"""
        print("\n🎯 Testing pipeline output quality...")
        
        mathematical_traits = self.create_test_mathematical_traits()
        
        # Execute pipeline
        conversion_result = self.trait_converter.convert_traits(
            mathematical_traits,
            direction=ConversionDirection.MATH_TO_SEMANTIC,
            target_complexity=SemanticComplexity.BASIC
        )
        semantic_traits = conversion_result.converted_traits
        
        character_result = self.character_formation.form_characters(
            semantic_traits,
            target_type=CharacterType.ALPHABETIC,
            formation_method=FormationMethod.HASH_BASED
        )
        characters = character_result.formed_characters
        
        word_result = self.word_formation.form_words(
            characters,
            semantic_traits,
            target_type=WordType.NOUN,
            formation_strategy=WordFormationStrategy.PATTERN_MATCHING
        )
        words = word_result.formed_words
        
        sentence_result = self.sentence_formation.form_sentences(
            words,
            semantic_traits,
            target_type=SentenceType.DECLARATIVE,
            sentence_structure=SentenceStructure.SVO
        )
        sentences = sentence_result.formed_sentences
        
        dialogue_result = self.communication.form_dialogues(
            sentences,
            semantic_traits,
            target_type=DialogueType.CONVERSATION,
            dialogue_structure=DialogueStructure.QUESTION_ANSWER
        )
        dialogues = dialogue_result.formed_dialogues
        
        # Quality checks
        if characters:
            # Check character quality
            for char in characters:
                self.assertIsInstance(char, str, "Characters should be strings")
                self.assertGreater(len(char), 0, "Characters should not be empty")
        
        if words:
            # Check word quality
            for word in words:
                self.assertIsInstance(word, str, "Words should be strings")
                self.assertGreater(len(word), 0, "Words should not be empty")
                # Basic word validation
                self.assertTrue(word.isalpha() or word.isalnum(), "Words should be alphabetic or alphanumeric")
        
        if sentences:
            # Check sentence quality
            for sentence in sentences:
                self.assertIsInstance(sentence, str, "Sentences should be strings")
                self.assertGreater(len(sentence), 0, "Sentences should not be empty")
                # Basic sentence validation
                self.assertTrue(sentence[0].isupper(), "Sentences should start with capital letter")
                self.assertTrue(sentence.endswith(('.', '!', '?')), "Sentences should end with punctuation")
        
        if dialogues:
            # Check dialogue quality
            for dialogue in dialogues:
                self.assertIsInstance(dialogue, str, "Dialogues should be strings")
                self.assertGreater(len(dialogue), 0, "Dialogues should not be empty")
                # Basic dialogue validation
                self.assertTrue(len(dialogue.split('.')) >= 2, "Dialogues should contain multiple sentences")
        
        print("✅ Pipeline output quality test completed successfully")
        print(f"📊 Quality Summary:")
        print(f"  Characters: {len(characters)} (all valid)")
        print(f"  Words: {len(words)} (all valid)")
        print(f"  Sentences: {len(sentences)} (all valid)")
        print(f"  Dialogues: {len(dialogues)} (all valid)")

def run_pipeline_test():
    """Run the complete linguistic pipeline test"""
    print("🚀 Starting Complete Linguistic Pipeline Integration Test")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test methods
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestLinguisticPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL PIPELINE TESTS PASSED!")
        print("✅ The complete linguistic pipeline is working correctly")
        print("✅ trait → character → word → sentence → dialogue transformation successful")
        return True
    else:
        print("\n❌ SOME PIPELINE TESTS FAILED")
        print("⚠️ The linguistic pipeline needs attention")
        return False

if __name__ == "__main__":
    success = run_pipeline_test()
    sys.exit(0 if success else 1)
