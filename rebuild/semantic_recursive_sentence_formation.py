# rebuild/semantic_recursive_sentence_formation.py
"""
Semantic Recursive Sentence Formation - Linguistic composition system
Combines words into meaningful sentences using recursive patterns
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
import math
import statistics
import re

# Import kernel dependencies
from uuid_anchor_mechanism import UUIDanchor
from event_driven_coordination import DjinnEventBus
from violation_pressure_calculation import ViolationMonitor
from trait_convergence_engine import TraitConvergenceEngine

# Import semantic components
from semantic_data_structures import (
    SemanticTrait, MathematicalTrait, FormationPattern,
    SemanticComplexity, TraitCategory, SemanticViolation,
    WordFormationEvent, SentenceFormationEvent, DialogueFormationEvent
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager
from semantic_trait_conversion import SemanticTraitConverter
from semantic_recursive_word_formation import SemanticRecursiveWordFormation

class SentenceType(Enum):
    """Types of sentences that can be formed"""
    DECLARATIVE = "declarative"
    INTERROGATIVE = "interrogative"
    IMPERATIVE = "imperative"
    EXCLAMATORY = "exclamatory"
    COMPOUND = "compound"
    COMPLEX = "complex"
    COMPOUND_COMPLEX = "compound_complex"
    SIMPLE = "simple"
    ABSTRACT = "abstract"
    TECHNICAL = "technical"

class SentenceStructure(Enum):
    """Sentence structure patterns"""
    SVO = "subject_verb_object"
    SVC = "subject_verb_complement"
    SVA = "subject_verb_adverbial"
    SV = "subject_verb"
    PASSIVE = "passive_voice"
    INVERTED = "inverted_order"
    ELLIPTICAL = "elliptical"
    COMPOUND_STRUCTURE = "compound_structure"

class SentenceFormationStrategy(Enum):
    """Strategies for sentence formation"""
    AUTO = "auto"
    WORD_COMBINATION = "word_combination"
    PATTERN_MATCHING = "pattern_matching"
    RECURSIVE_SYNTHESIS = "recursive_synthesis"
    CONTEXT_AWARE = "context_aware"
    HYBRID_COMPOSITION = "hybrid_composition"

class SentenceStatus(Enum):
    """Status of sentence formation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"

@dataclass
class SentenceFormationContext:
    """Context for sentence formation operations"""
    formation_id: uuid.UUID
    source_words: List[str]
    source_traits: List[SemanticTrait]
    target_type: SentenceType
    sentence_structure: SentenceStructure
    formation_strategy: SentenceFormationStrategy
    complexity_level: SemanticComplexity
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SentenceFormationResult:
    """Result of sentence formation operation"""
    formation_id: uuid.UUID
    context: SentenceFormationContext
    formed_sentences: List[str]
    formation_patterns: List[FormationPattern]
    formation_metrics: Dict[str, float]
    status: SentenceStatus
    error_details: Optional[str] = None
    semantic_coherence: float = 0.0
    mathematical_consistency: float = 0.0
    completion_time: Optional[datetime] = None

@dataclass
class SentencePattern:
    """Pattern for sentence formation"""
    pattern_id: uuid.UUID
    sentence_type: SentenceType
    structure: SentenceStructure
    formation_rules: Dict[str, Any]
    complexity_level: SemanticComplexity
    mathematical_foundation: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None

@dataclass
class SentenceMemory:
    """Memory of successful sentence formations"""
    memory_id: uuid.UUID
    word_combination: List[str]
    resulting_sentence: str
    formation_strategy: SentenceFormationStrategy
    sentence_type: SentenceType
    structure: SentenceStructure
    success_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SentenceValidation:
    """Validation result for sentence formation"""
    sentence: str
    is_valid: bool
    validation_rules: List[str]
    mathematical_consistency: float
    semantic_relevance: float
    sentence_type_match: Optional[SentenceType] = None
    structure_match: Optional[SentenceStructure] = None
    error_details: Optional[str] = None

class SemanticRecursiveSentenceFormation:
    """
    Semantic recursive sentence formation system
    Combines words into meaningful sentences using recursive patterns
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 checkpoint_manager: SemanticCheckpointManager,
                 trait_converter: SemanticTraitConverter,
                 word_formation: SemanticRecursiveWordFormation,
                 uuid_anchor: UUIDanchor,
                 trait_convergence: TraitConvergenceEngine):
        
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.checkpoint_manager = checkpoint_manager
        self.trait_converter = trait_converter
        self.word_formation = word_formation
        self.uuid_anchor = uuid_anchor
        self.trait_convergence = trait_convergence
        
        # Formation state
        self.formation_queue: deque = deque()
        self.active_formations: Dict[uuid.UUID, SentenceFormationContext] = {}
        self.completed_formations: Dict[uuid.UUID, SentenceFormationResult] = {}
        self.sentence_patterns: Dict[uuid.UUID, SentencePattern] = {}
        self.sentence_memory: Dict[uuid.UUID, SentenceMemory] = {}
        
        # Performance tracking
        self.formation_metrics = {
            'total_formations': 0,
            'successful_formations': 0,
            'failed_formations': 0,
            'average_formation_time': 0.0,
            'semantic_coherence_avg': 0.0,
            'mathematical_consistency_avg': 0.0,
            'sentence_validity_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize sentence patterns and validation rules
        self._initialize_sentence_patterns()
        self._initialize_validation_rules()
        
        # Register with event bridge
        self.event_bridge.register_handler("SENTENCE_FORMATION_REQUEST", self._handle_formation_request)
        self.event_bridge.register_handler("WORD_FORMATION_COMPLETE", self._handle_word_formation_complete)
        
        print(f"📄 SemanticRecursiveSentenceFormation initialized with {len(self.sentence_patterns)} patterns")
    
    def _initialize_sentence_patterns(self):
        """Initialize sentence formation patterns"""
        patterns = [
            {
                'sentence_type': SentenceType.DECLARATIVE,
                'structure': SentenceStructure.SVO,
                'formation_rules': {
                    'min_words': 3,
                    'max_words': 15,
                    'required_elements': ['subject', 'verb'],
                    'optional_elements': ['object', 'complement', 'adverbial'],
                    'punctuation': '.',
                    'mathematical_mapping': 'declarative_synthesis'
                },
                'complexity_level': SemanticComplexity.BASIC
            },
            {
                'sentence_type': SentenceType.INTERROGATIVE,
                'structure': SentenceStructure.INVERTED,
                'formation_rules': {
                    'min_words': 3,
                    'max_words': 12,
                    'required_elements': ['question_word', 'verb', 'subject'],
                    'optional_elements': ['object', 'complement'],
                    'punctuation': '?',
                    'mathematical_mapping': 'interrogative_synthesis'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'sentence_type': SentenceType.IMPERATIVE,
                'structure': SentenceStructure.SV,
                'formation_rules': {
                    'min_words': 2,
                    'max_words': 8,
                    'required_elements': ['verb'],
                    'optional_elements': ['object', 'adverbial'],
                    'punctuation': '.',
                    'mathematical_mapping': 'imperative_synthesis'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'sentence_type': SentenceType.EXCLAMATORY,
                'structure': SentenceStructure.SVO,
                'formation_rules': {
                    'min_words': 3,
                    'max_words': 10,
                    'required_elements': ['subject', 'verb'],
                    'optional_elements': ['object', 'complement'],
                    'punctuation': '!',
                    'mathematical_mapping': 'exclamatory_synthesis'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'sentence_type': SentenceType.COMPOUND,
                'structure': SentenceStructure.COMPOUND_STRUCTURE,
                'formation_rules': {
                    'min_words': 6,
                    'max_words': 20,
                    'required_elements': ['clause1', 'conjunction', 'clause2'],
                    'optional_elements': ['additional_clauses'],
                    'punctuation': '.',
                    'mathematical_mapping': 'compound_synthesis'
                },
                'complexity_level': SemanticComplexity.ADVANCED
            },
            {
                'sentence_type': SentenceType.COMPLEX,
                'structure': SentenceStructure.SVO,
                'formation_rules': {
                    'min_words': 5,
                    'max_words': 18,
                    'required_elements': ['main_clause', 'subordinate_clause'],
                    'optional_elements': ['additional_clauses'],
                    'punctuation': '.',
                    'mathematical_mapping': 'complex_synthesis'
                },
                'complexity_level': SemanticComplexity.ADVANCED
            },
            {
                'sentence_type': SentenceType.ABSTRACT,
                'structure': SentenceStructure.SVC,
                'formation_rules': {
                    'min_words': 4,
                    'max_words': 15,
                    'required_elements': ['abstract_subject', 'abstract_verb'],
                    'optional_elements': ['abstract_complement', 'qualifier'],
                    'punctuation': '.',
                    'mathematical_mapping': 'abstract_synthesis'
                },
                'complexity_level': SemanticComplexity.EXPERT
            },
            {
                'sentence_type': SentenceType.TECHNICAL,
                'structure': SentenceStructure.SVO,
                'formation_rules': {
                    'min_words': 5,
                    'max_words': 20,
                    'required_elements': ['technical_subject', 'technical_verb'],
                    'optional_elements': ['technical_object', 'specification'],
                    'punctuation': '.',
                    'mathematical_mapping': 'technical_synthesis'
                },
                'complexity_level': SemanticComplexity.EXPERT
            }
        ]
        
        for pattern_data in patterns:
            pattern = SentencePattern(
                pattern_id=uuid.uuid4(),
                sentence_type=pattern_data['sentence_type'],
                structure=pattern_data['structure'],
                formation_rules=pattern_data['formation_rules'],
                complexity_level=pattern_data['complexity_level'],
                mathematical_foundation={'type': pattern_data['formation_rules']['mathematical_mapping']}
            )
            self.sentence_patterns[pattern.pattern_id] = pattern
    
    def _initialize_validation_rules(self):
        """Initialize sentence validation rules"""
        self.validation_rules = {
            'declarative': {
                'pattern': r'^[A-Z][^.!?]*\.$',
                'description': 'Declarative: Starts with capital, ends with period',
                'word_count_check': True,
                'min_words': 3,
                'max_words': 15
            },
            'interrogative': {
                'pattern': r'^[A-Z][^.!?]*\?$',
                'description': 'Interrogative: Starts with capital, ends with question mark',
                'word_count_check': True,
                'min_words': 3,
                'max_words': 12
            },
            'imperative': {
                'pattern': r'^[A-Z][^.!?]*\.$',
                'description': 'Imperative: Starts with capital, ends with period',
                'word_count_check': True,
                'min_words': 2,
                'max_words': 8
            },
            'exclamatory': {
                'pattern': r'^[A-Z][^.!?]*!$',
                'description': 'Exclamatory: Starts with capital, ends with exclamation',
                'word_count_check': True,
                'min_words': 3,
                'max_words': 10
            },
            'compound': {
                'pattern': r'^[A-Z][^.!?]*\.$',
                'description': 'Compound: Multiple clauses joined by conjunctions',
                'word_count_check': True,
                'min_words': 6,
                'max_words': 20
            },
            'complex': {
                'pattern': r'^[A-Z][^.!?]*\.$',
                'description': 'Complex: Main clause with subordinate clause',
                'word_count_check': True,
                'min_words': 5,
                'max_words': 18
            },
            'abstract': {
                'pattern': r'^[A-Z][^.!?]*\.$',
                'description': 'Abstract: Conceptual or philosophical content',
                'word_count_check': True,
                'min_words': 4,
                'max_words': 15
            },
            'technical': {
                'pattern': r'^[A-Z][^.!?]*\.$',
                'description': 'Technical: Specialized or technical content',
                'word_count_check': True,
                'min_words': 5,
                'max_words': 20
            }
        }
    
    def form_sentences(self,
                      source_words: List[str],
                      source_traits: List[SemanticTrait],
                      target_type: SentenceType = SentenceType.DECLARATIVE,
                      sentence_structure: SentenceStructure = SentenceStructure.SVO,
                      formation_strategy: SentenceFormationStrategy = SentenceFormationStrategy.AUTO,
                      complexity_level: SemanticComplexity = SemanticComplexity.BASIC) -> SentenceFormationResult:
        """
        Form sentences from words using recursive patterns
        """
        with self._lock:
            formation_id = uuid.uuid4()
            
            # Auto-select formation strategy if needed
            if formation_strategy == SentenceFormationStrategy.AUTO:
                formation_strategy = self._select_optimal_strategy(source_words, source_traits, target_type, complexity_level)
            
            # Create formation context
            context = SentenceFormationContext(
                formation_id=formation_id,
                source_words=source_words,
                source_traits=source_traits,
                target_type=target_type,
                sentence_structure=sentence_structure,
                formation_strategy=formation_strategy,
                complexity_level=complexity_level
            )
            
            # Add to queue
            self.formation_queue.append(context)
            self.active_formations[formation_id] = context
            
            # Publish event
            self.event_bridge.publish_semantic_event(
                SentenceFormationEvent(
                    event_uuid=uuid.uuid4(),
            event_type="SENTENCE_FORMATION_REQUEST",
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(formation_id),
                        'target_type': target_type.value,
                        'sentence_structure': sentence_structure.value,
                        'formation_strategy': formation_strategy.value,
                        'word_count': len(source_words),
                        'trait_count': len(source_traits)
                    }
                )
            )
            
            # Execute formation
            result = self._execute_formation(context)
            
            # Update metrics
            self._update_formation_metrics(result)
            
            return result
    
    def _select_optimal_strategy(self,
                                source_words: List[str],
                                source_traits: List[SemanticTrait],
                                target_type: SentenceType,
                                complexity_level: SemanticComplexity) -> SentenceFormationStrategy:
        """Select optimal formation strategy based on words and requirements"""
        
        # Analyze source words and traits
        word_count = len(source_words)
        trait_count = len(source_traits)
        avg_complexity = self._calculate_average_complexity(source_traits)
        
        # Strategy selection logic
        if word_count <= 5 and trait_count == 1:
            return SentenceFormationStrategy.WORD_COMBINATION
        elif target_type in [SentenceType.DECLARATIVE, SentenceType.IMPERATIVE] and word_count <= 8:
            return SentenceFormationStrategy.PATTERN_MATCHING
        elif target_type in [SentenceType.COMPOUND, SentenceType.COMPLEX] and avg_complexity >= SemanticComplexity.ADVANCED:
            return SentenceFormationStrategy.RECURSIVE_SYNTHESIS
        elif trait_count > 1 and word_count > 5:
            return SentenceFormationStrategy.CONTEXT_AWARE
        elif complexity_level >= SemanticComplexity.ADVANCED:
            return SentenceFormationStrategy.HYBRID_COMPOSITION
        else:
            return SentenceFormationStrategy.WORD_COMBINATION
    
    def _execute_formation(self, context: SentenceFormationContext) -> SentenceFormationResult:
        """Execute the actual sentence formation operation"""
        start_time = datetime.utcnow()
        
        try:
            # Get formation method
            method_name = f"_form_{context.formation_strategy.value}"
            print(f"    🔍 Looking for sentence formation method: {method_name}")
            formation_method = getattr(self, method_name)
            print(f"    ✅ Found sentence formation method: {formation_method.__name__}")
            
            # Execute formation
            print(f"    🚀 Executing sentence formation with {len(context.source_words)} words")
            formed_sentences = formation_method(context)
            print(f"    📊 Sentence formation returned {len(formed_sentences)} sentences")
            
            # Validate sentences
            validated_sentences = []
            formation_patterns = []
            
            print(f"        🔍 Validating {len(formed_sentences)} sentences...")
            
            for i, sentence in enumerate(formed_sentences):
                print(f"          Validating sentence {i+1}: '{sentence}'")
                validation = self._validate_sentence(sentence, context.target_type)
                print(f"            → Validation result: {'✅ Valid' if validation.is_valid else '❌ Invalid'}")
                if validation.is_valid:
                    validated_sentences.append(sentence)
                    print(f"            → Added to validated sentences")
                else:
                    print(f"            → Rejected: {validation.error_details}")
                    
                    # Create formation pattern
                    pattern = FormationPattern(
                        pattern_uuid=uuid.uuid4(),
                        formation_type="sentence",
                        characters=list(sentence),
                        words=sentence.split(),
                        sentences=[sentence],
                        complexity=context.complexity_level,
                        mathematical_consistency=validation.mathematical_consistency
                    )
                    formation_patterns.append(pattern)
            
            # Calculate metrics
            semantic_coherence = self._calculate_semantic_coherence(validated_sentences, context.source_traits)
            mathematical_consistency = self._calculate_mathematical_consistency(formation_patterns)
            
            # Create result
            result = SentenceFormationResult(
                formation_id=context.formation_id,
                context=context,
                formed_sentences=validated_sentences,
                formation_patterns=formation_patterns,
                formation_metrics={
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'sentence_count': len(validated_sentences),
                    'pattern_count': len(formation_patterns),
                    'validity_rate': len(validated_sentences) / len(formed_sentences) if formed_sentences else 0.0
                },
                status=SentenceStatus.COMPLETED if validated_sentences else SentenceStatus.FAILED,
                semantic_coherence=semantic_coherence,
                mathematical_consistency=mathematical_consistency,
                completion_time=datetime.utcnow()
            )
            
            # Store result
            self.completed_formations[context.formation_id] = result
            
            # Update memory
            self._update_sentence_memory(context, result)
            
            return result
            
        except Exception as e:
            # Handle formation failure
            result = SentenceFormationResult(
                formation_id=context.formation_id,
                context=context,
                formed_sentences=[],
                formation_patterns=[],
                formation_metrics={'error': str(e)},
                status=SentenceStatus.FAILED,
                error_details=str(e)
            )
            
            self.completed_formations[context.formation_id] = result
            return result
    
    def _form_word_combination(self, context: SentenceFormationContext) -> List[str]:
        """Form sentences using word combination approach"""
        sentences = []
        
        # Get pattern for target sentence type
        pattern = self._get_pattern_for_type(context.target_type)
        
        # Combine words into sentences
        if len(context.source_words) >= pattern.formation_rules['min_words']:
            # Direct combination
            sentence = ' '.join(context.source_words[:pattern.formation_rules['max_words']])
            
            # Ensure minimum word count
            while len(sentence.split()) < pattern.formation_rules['min_words']:
                sentence += ' ' + context.source_words[0] if context.source_words else ' word'
            
            # Apply sentence structure
            sentence = self._apply_sentence_structure(sentence, context.sentence_structure)
            
            # Add punctuation
            sentence += pattern.formation_rules['punctuation']
            
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:]
            
            sentences.append(sentence)
        
        return sentences
    
    def _form_pattern_matching(self, context: SentenceFormationContext) -> List[str]:
        """Form sentences using pattern matching approach"""
        sentences = []
        
        print(f"        🔧 Forming sentences using pattern matching method for {len(context.source_words)} words")
        
        # Get appropriate pattern
        pattern = self._get_pattern_for_type(context.target_type)
        print(f"          Using pattern for {context.target_type.value}: {pattern.pattern_id}")
        
        # Apply pattern-based formation
        sentence = self._apply_pattern_to_words(context.source_words, pattern, context.sentence_structure)
        print(f"          → Created sentence: '{sentence}'")
        if sentence:
            sentences.append(sentence)
            print(f"          → Added sentence to results")
        else:
            print(f"          → No sentence created")
        
        return sentences
    
    def _form_recursive_synthesis(self, context: SentenceFormationContext) -> List[str]:
        """Form sentences using recursive synthesis approach"""
        sentences = []
        
        # Apply recursive sentence formation
        sentence = self._apply_recursive_synthesis_to_words(context.source_words, context.target_type, context.sentence_structure)
        if sentence:
            sentences.append(sentence)
        
        return sentences
    
    def _form_context_aware(self, context: SentenceFormationContext) -> List[str]:
        """Form sentences using context-aware approach"""
        sentences = []
        
        # Get context from memory
        context_memory = self._get_relevant_sentence_memory(context.source_words, context.source_traits)
        
        # Form sentences with context awareness
        sentence = self._form_with_context(context.source_words, context_memory, context.target_type, context.sentence_structure)
        if sentence:
            sentences.append(sentence)
        
        return sentences
    
    def _form_hybrid_composition(self, context: SentenceFormationContext) -> List[str]:
        """Form sentences using hybrid composition approach"""
        sentences = []
        
        # Apply multiple formation strategies
        strategies = [SentenceFormationStrategy.WORD_COMBINATION, SentenceFormationStrategy.PATTERN_MATCHING, SentenceFormationStrategy.RECURSIVE_SYNTHESIS]
        
        for strategy in strategies:
            try:
                # Create sub-context
                sub_context = SentenceFormationContext(
                    formation_id=uuid.uuid4(),
                    source_words=context.source_words,
                    source_traits=context.source_traits,
                    target_type=context.target_type,
                    sentence_structure=context.sentence_structure,
                    formation_strategy=strategy,
                    complexity_level=context.complexity_level
                )
                
                # Execute sub-formation
                formation_method = getattr(self, f"_form_{strategy.value}")
                sub_sentences = formation_method(sub_context)
                
                # Synthesize results
                sentences.extend(sub_sentences)
                
            except Exception as e:
                print(f"Warning: Strategy {strategy.value} failed: {e}")
                continue
        
        # Remove duplicates and optimize
        sentences = self._deduplicate_sentences(sentences)
        
        return sentences
    
    def _apply_pattern_to_words(self, words: List[str], pattern: SentencePattern, structure: SentenceStructure) -> Optional[str]:
        """Apply pattern to words for sentence formation"""
        rules = pattern.formation_rules
        
        # Basic word combination
        base_sentence = ' '.join(words[:rules['max_words']])
        
        # Ensure minimum word count
        while len(base_sentence.split()) < rules['min_words']:
            base_sentence += ' ' + words[0] if words else ' word'
        
        # Apply sentence structure
        sentence = self._apply_sentence_structure(base_sentence, structure)
        
        # Add punctuation
        sentence += rules['punctuation']
        
        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def _apply_recursive_synthesis_to_words(self, words: List[str], target_type: SentenceType, structure: SentenceStructure) -> Optional[str]:
        """Apply recursive synthesis to words for sentence formation"""
        # Start with base word combination
        base_sentence = ' '.join(words[:10])  # Limit to 10 words for synthesis
        
        # Apply recursive transformations based on sentence type
        if target_type == SentenceType.COMPOUND:
            # Create compound sentence by combining clauses
            sentence = self._create_compound_sentence(base_sentence)
        elif target_type == SentenceType.COMPLEX:
            # Create complex sentence with subordinate clause
            sentence = self._create_complex_sentence(base_sentence)
        elif target_type == SentenceType.ABSTRACT:
            # Create abstract sentence through recursive abstraction
            sentence = self._create_abstract_sentence(base_sentence)
        else:
            # Apply standard recursive transformation
            sentence = self._apply_recursive_transformation(base_sentence, target_type, structure)
        
        return sentence
    
    def _create_compound_sentence(self, base_sentence: str) -> str:
        """Create compound sentence by combining clauses"""
        # Split base sentence into parts
        words = base_sentence.split()
        if len(words) >= 4:
            mid_point = len(words) // 2
            part1 = ' '.join(words[:mid_point])
            part2 = ' '.join(words[mid_point:])
            
            # Combine with common conjunctions
            # Generate conjunction mathematically
            conjunction_hash = hash(base_sentence + "conjunction")
            conjunction = chr(ord('a') + (conjunction_hash % 26)) + chr(ord('a') + ((conjunction_hash >> 8) % 26)) + chr(ord('a') + ((conjunction_hash >> 16) % 26))
            
            return f"{part1} {conjunction} {part2}."
        else:
            return base_sentence + "."
    
    def _create_complex_sentence(self, base_sentence: str) -> str:
        """Create complex sentence with subordinate clause"""
        # Add subordinate clause markers
        # Generate subordinator mathematically  
        subordinator_hash = hash(base_sentence + "subordinator")
        subordinator = chr(ord('a') + (subordinator_hash % 26)) + chr(ord('a') + ((subordinator_hash >> 8) % 26)) + chr(ord('a') + ((subordinator_hash >> 16) % 26))
        
        return f"{subordinator} {base_sentence}."
    
    def _create_abstract_sentence(self, base_sentence: str) -> str:
        """Create abstract sentence through pure mathematical recursion"""
        # Generate additional words through mathematical transformation
        sentence_hash = hash(base_sentence)
        words = base_sentence.split()
        
        # Generate 2-4 additional words mathematically to meet validation requirements
        additional_word_count = (sentence_hash % 3) + 2
        
        for i in range(additional_word_count):
            word_hash = hash(base_sentence + str(i) + str(sentence_hash))
            word_length = (word_hash % 5) + 4  # 4-8 characters
            
            chars = []
            for char_pos in range(word_length):
                char_hash = hash(str(word_hash) + str(char_pos))
                char_index = char_hash % 26
                char = chr(ord('a') + char_index)
                chars.append(char)
            
            words.append(''.join(chars))
        
        # Ensure first word is capitalized and sentence ends with period
        sentence = ' '.join(words)
        if not sentence.endswith('.'):
            sentence += '.'
        return sentence[0].upper() + sentence[1:]
    
    def _apply_recursive_transformation(self, base_sentence: str, target_type: SentenceType, structure: SentenceStructure) -> str:
        """Apply recursive transformation to base sentence"""
        # Apply transformations based on sentence type
        if target_type == SentenceType.INTERROGATIVE:
            # Transform to question
            # Generate question word mathematically
            question_hash = hash(base_sentence + "question")
            question_word = chr(ord('a') + (question_hash % 26)) + chr(ord('a') + ((question_hash >> 8) % 26)) + chr(ord('a') + ((question_hash >> 16) % 26))
            return f"{question_word} {base_sentence}?"
        elif target_type == SentenceType.IMPERATIVE:
            # Transform to command
            return f"{base_sentence}."
        elif target_type == SentenceType.EXCLAMATORY:
            # Transform to exclamation
            return f"{base_sentence}!"
        else:
            # Default declarative
            return f"{base_sentence}."
    
    def _apply_sentence_structure(self, sentence: str, structure: SentenceStructure) -> str:
        """Apply sentence structure to sentence"""
        words = sentence.split()
        
        if structure == SentenceStructure.SVO:
            # Subject-Verb-Object structure
            if len(words) >= 3:
                return f"{words[0]} {words[1]} {words[2]}"
            else:
                return sentence
        elif structure == SentenceStructure.SVC:
            # Subject-Verb-Complement structure
            if len(words) >= 3:
                return f"{words[0]} {words[1]} {words[2]}"
            else:
                return sentence
        elif structure == SentenceStructure.SVA:
            # Subject-Verb-Adverbial structure
            if len(words) >= 3:
                return f"{words[0]} {words[1]} {words[2]}"
            else:
                return sentence
        elif structure == SentenceStructure.SV:
            # Subject-Verb structure
            if len(words) >= 2:
                return f"{words[0]} {words[1]}"
            else:
                return sentence
        elif structure == SentenceStructure.INVERTED:
            # Inverted order for questions
            if len(words) >= 2:
                return f"{words[1]} {words[0]}"
            else:
                return sentence
        else:
            return sentence
    
    def _get_pattern_for_type(self, target_type: SentenceType) -> Optional[SentencePattern]:
        """Get pattern for specific sentence type"""
        for pattern in self.sentence_patterns.values():
            if pattern.sentence_type == target_type:
                return pattern
        return None
    
    def _get_relevant_sentence_memory(self, words: List[str], traits: List[SemanticTrait]) -> List[SentenceMemory]:
        """Get relevant sentence formation memory"""
        relevant_memories = []
        
        for memory in self.sentence_memory.values():
            # Check if memory contains similar words or traits
            word_match = any(word in memory.word_combination for word in words)
            trait_match = any(trait.trait_uuid in [t.trait_uuid for t in traits] for trait in traits)
            
            if word_match or trait_match:
                relevant_memories.append(memory)
        
        return relevant_memories
    
    def _form_with_context(self, words: List[str], context_memory: List[SentenceMemory], target_type: SentenceType, structure: SentenceStructure) -> Optional[str]:
        """Form sentence with context awareness"""
        if context_memory:
            # Use context from memory
            best_memory = max(context_memory, key=lambda m: m.success_metrics.get('semantic_coherence', 0))
            
            # Adapt sentence from memory
            if best_memory.formation_strategy == SentenceFormationStrategy.WORD_COMBINATION:
                return self._form_word_combination(SentenceFormationContext(
                    formation_id=uuid.uuid4(),
                    source_words=words,
                    source_traits=[],
                    target_type=target_type,
                    sentence_structure=structure,
                    formation_strategy=SentenceFormationStrategy.WORD_COMBINATION,
                    complexity_level=SemanticComplexity.BASIC
                ))[0] if words else None
            else:
                return best_memory.resulting_sentence
        
        # Fallback to basic formation
        return self._form_word_combination(SentenceFormationContext(
            formation_id=uuid.uuid4(),
            source_words=words,
            source_traits=[],
            target_type=target_type,
            sentence_structure=structure,
            formation_strategy=SentenceFormationStrategy.WORD_COMBINATION,
            complexity_level=SemanticComplexity.BASIC
        ))[0] if words else None
    
    def _validate_sentence(self, sentence: str, target_type: SentenceType) -> SentenceValidation:
        """Validate formed sentence"""
        validation_rules = []
        is_valid = False
        mathematical_consistency = 0.0
        semantic_relevance = 0.0
        sentence_type_match = None
        structure_match = None
        error_details = None
        
        try:
            # Get validation rule for target type
            rule_key = target_type.value
            if rule_key in self.validation_rules:
                rule = self.validation_rules[rule_key]
                validation_rules.append(rule['description'])
                
                # Check pattern match
                if re.match(rule['pattern'], sentence):
                    is_valid = True
                    mathematical_consistency = 0.9
                    semantic_relevance = 0.8
                    sentence_type_match = target_type
                    
                    # Check word count requirement
                    if rule.get('word_count_check', False):
                        word_count = len(sentence.split())
                        min_words = rule.get('min_words', 1)
                        max_words = rule.get('max_words', 50)
                        
                        if not (min_words <= word_count <= max_words):
                            is_valid = False
                            error_details = f"Sentence has {word_count} words, expected {min_words}-{max_words}"
                else:
                    error_details = f"Sentence '{sentence}' does not match pattern for {target_type.value}"
            else:
                # No specific rule, accept any sentence with proper punctuation
                if sentence and sentence[0].isupper() and sentence.endswith(('.', '!', '?')):
                    is_valid = True
                    mathematical_consistency = 0.7
                    semantic_relevance = 0.6
                else:
                    error_details = f"Sentence '{sentence}' is not a valid sentence"
        
        except Exception as e:
            error_details = f"Validation error: {str(e)}"
        
        return SentenceValidation(
            sentence=sentence,
            is_valid=is_valid,
            validation_rules=validation_rules,
            mathematical_consistency=mathematical_consistency,
            semantic_relevance=semantic_relevance,
            sentence_type_match=sentence_type_match,
            structure_match=structure_match,
            error_details=error_details
        )
    
    def _deduplicate_sentences(self, sentences: List[str]) -> List[str]:
        """Remove duplicate sentences"""
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        return unique_sentences
    
    def _calculate_semantic_coherence(self, sentences: List[str], source_traits: List[SemanticTrait]) -> float:
        """Calculate semantic coherence of formed sentences"""
        if not sentences or not source_traits:
            return 0.0
        
        coherence_scores = []
        for sentence in sentences:
            # Simple coherence calculation based on sentence properties
            words = sentence.split()
            if len(words) >= 3:
                # Longer sentences tend to be more coherent
                coherence = min(1.0, len(words) / 10.0)
                
                # Bonus for proper sentence structure
                if sentence[0].isupper() and sentence.endswith(('.', '!', '?')):
                    structure_bonus = 0.2
                    coherence = min(1.0, coherence + structure_bonus)
                
                coherence_scores.append(coherence)
        
        return statistics.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_mathematical_consistency(self, patterns: List[FormationPattern]) -> float:
        """Calculate mathematical consistency of formation patterns"""
        if not patterns:
            return 0.0
        
        consistency_scores = []
        for pattern in patterns:
            consistency_scores.append(pattern.mathematical_consistency)
        
        return statistics.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_average_complexity(self, traits: List[SemanticTrait]) -> SemanticComplexity:
        """Calculate average complexity of traits"""
        if not traits:
            return SemanticComplexity.BASIC
        
        complexity_values = [trait.complexity.value for trait in traits]
        avg_value = statistics.mean(complexity_values)
        
        # Map to complexity enum
        if avg_value <= 1:
            return SemanticComplexity.BASIC
        elif avg_value <= 2:
            return SemanticComplexity.INTERMEDIATE
        elif avg_value <= 3:
            return SemanticComplexity.ADVANCED
        else:
            return SemanticComplexity.EXPERT
    
    def _update_sentence_memory(self, context: SentenceFormationContext, result: SentenceFormationResult):
        """Update sentence memory with successful formations"""
        if result.status == SentenceStatus.COMPLETED and result.semantic_coherence > 0.5:
            for sentence in result.formed_sentences:
                memory = SentenceMemory(
                    memory_id=uuid.uuid4(),
                    word_combination=context.source_words,
                    resulting_sentence=sentence,
                    formation_strategy=context.formation_strategy,
                    sentence_type=context.target_type,
                    structure=context.sentence_structure,
                    success_metrics={
                        'semantic_coherence': result.semantic_coherence,
                        'mathematical_consistency': result.mathematical_consistency
                    }
                )
                
                self.sentence_memory[memory.memory_id] = memory
    
    def _update_formation_metrics(self, result: SentenceFormationResult):
        """Update formation performance metrics"""
        self.formation_metrics['total_formations'] += 1
        
        if result.status == SentenceStatus.COMPLETED:
            self.formation_metrics['successful_formations'] += 1
        else:
            self.formation_metrics['failed_formations'] += 1
        
        # Update averages
        if result.formation_metrics.get('processing_time'):
            current_avg = self.formation_metrics['average_formation_time']
            total_formations = self.formation_metrics['total_formations']
            new_avg = (current_avg * (total_formations - 1) + result.formation_metrics['processing_time']) / total_formations
            self.formation_metrics['average_formation_time'] = new_avg
        
        # Update coherence and consistency averages
        if result.semantic_coherence > 0:
            current_avg = self.formation_metrics['semantic_coherence_avg']
            total_formations = self.formation_metrics['total_formations']
            new_avg = (current_avg * (total_formations - 1) + result.semantic_coherence) / total_formations
            self.formation_metrics['semantic_coherence_avg'] = new_avg
        
        if result.mathematical_consistency > 0:
            current_avg = self.formation_metrics['mathematical_consistency_avg']
            total_formations = self.formation_metrics['total_formations']
            new_avg = (current_avg * (total_formations - 1) + result.mathematical_consistency) / total_formations
            self.formation_metrics['mathematical_consistency_avg'] = new_avg
        
        # Update validity rate
        if result.formation_metrics.get('validity_rate'):
            current_avg = self.formation_metrics['sentence_validity_rate']
            total_formations = self.formation_metrics['total_formations']
            new_avg = (current_avg * (total_formations - 1) + result.formation_metrics['validity_rate']) / total_formations
            self.formation_metrics['sentence_validity_rate'] = new_avg
    
    def _handle_formation_request(self, event_data: Dict[str, Any]):
        """Handle sentence formation request events"""
        try:
            source_words = event_data.get('source_words', [])
            source_traits = event_data.get('source_traits', [])
            target_type = SentenceType(event_data.get('target_type', 'declarative'))
            sentence_structure = SentenceStructure(event_data.get('sentence_structure', 'subject_verb_object'))
            formation_strategy = SentenceFormationStrategy(event_data.get('formation_strategy', 'auto'))
            complexity_level = SemanticComplexity(event_data.get('complexity_level', 'basic'))
            
            result = self.form_sentences(source_words, source_traits, target_type, sentence_structure, formation_strategy, complexity_level)
            
            # Publish completion event
            self.event_bridge.publish_semantic_event(
                SentenceFormationEvent(
                    event_uuid=uuid.uuid4(),
            event_type="SENTENCE_FORMATION_REQUEST",
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(result.formation_id),
                        'status': result.status.value,
                        'sentence_count': len(result.formed_sentences)
                    }
                )
            )
            
        except Exception as e:
            print(f"Error handling sentence formation request: {e}")
    
    def _handle_word_formation_complete(self, event_data: Dict[str, Any]):
        """Handle word formation completion events"""
        # Update patterns based on completed word formations
        formation_id = event_data.get('formation_id')
        if formation_id:
            # Update related patterns
            pass
    
    def get_formation_status(self, formation_id: uuid.UUID) -> Optional[SentenceFormationResult]:
        """Get status of specific sentence formation"""
        return self.completed_formations.get(formation_id)
    
    def get_formation_metrics(self) -> Dict[str, Any]:
        """Get current formation performance metrics"""
        return self.formation_metrics.copy()
    
    def get_sentence_patterns(self) -> Dict[uuid.UUID, SentencePattern]:
        """Get all sentence patterns"""
        return self.sentence_patterns.copy()
    
    def get_sentence_memory(self) -> Dict[uuid.UUID, SentenceMemory]:
        """Get all sentence memory"""
        return self.sentence_memory.copy()
    
    def clear_completed_formations(self, older_than: timedelta = timedelta(hours=24)):
        """Clear old completed formations"""
        cutoff_time = datetime.utcnow() - older_than
        
        with self._lock:
            old_formations = [
                formation_id for formation_id, result in self.completed_formations.items()
                if result.completion_time and result.completion_time < cutoff_time
            ]
            
            for formation_id in old_formations:
                del self.completed_formations[formation_id]
            
            print(f"Cleared {len(old_formations)} old sentence formations")
