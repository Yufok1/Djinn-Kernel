# rebuild/semantic_recursive_word_formation.py
"""
Semantic Recursive Word Formation - Linguistic composition system
Combines individual characters into meaningful words using recursive patterns
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
    CharacterFormationEvent, WordFormationEvent, SentenceFormationEvent
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager
from semantic_trait_conversion import SemanticTraitConverter
from semantic_recursive_character_formation import SemanticRecursiveCharacterFormation

class WordType(Enum):
    """Types of words that can be formed"""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    INTERJECTION = "interjection"
    COMPOSITE = "composite"
    ABSTRACT = "abstract"

class WordFormationStrategy(Enum):
    """Strategies for word formation"""
    AUTO = "auto"
    CHARACTER_COMBINATION = "character_combination"
    PATTERN_MATCHING = "pattern_matching"
    RECURSIVE_SYNTHESIS = "recursive_synthesis"
    CONTEXT_AWARE = "context_aware"
    HYBRID_COMPOSITION = "hybrid_composition"

class WordStatus(Enum):
    """Status of word formation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"

@dataclass
class WordFormationContext:
    """Context for word formation operations"""
    formation_id: uuid.UUID
    source_characters: List[str]
    source_traits: List[SemanticTrait]
    target_type: WordType
    formation_strategy: WordFormationStrategy
    complexity_level: SemanticComplexity
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WordFormationResult:
    """Result of word formation operation"""
    formation_id: uuid.UUID
    context: WordFormationContext
    formed_words: List[str]
    formation_patterns: List[FormationPattern]
    formation_metrics: Dict[str, float]
    status: WordStatus
    error_details: Optional[str] = None
    semantic_coherence: float = 0.0
    mathematical_consistency: float = 0.0
    completion_time: Optional[datetime] = None

@dataclass
class WordPattern:
    """Pattern for word formation"""
    pattern_id: uuid.UUID
    word_type: WordType
    formation_rules: Dict[str, Any]
    complexity_level: SemanticComplexity
    mathematical_foundation: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None

@dataclass
class WordMemory:
    """Memory of successful word formations"""
    memory_id: uuid.UUID
    character_combination: List[str]
    resulting_word: str
    formation_strategy: WordFormationStrategy
    word_type: WordType
    success_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WordValidation:
    """Validation result for word formation"""
    word: str
    is_valid: bool
    validation_rules: List[str]
    mathematical_consistency: float
    semantic_relevance: float
    word_type_match: Optional[WordType] = None
    error_details: Optional[str] = None

class SemanticRecursiveWordFormation:
    """
    Semantic recursive word formation system
    Combines individual characters into meaningful words using recursive patterns
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 checkpoint_manager: SemanticCheckpointManager,
                 trait_converter: SemanticTraitConverter,
                 character_formation: SemanticRecursiveCharacterFormation,
                 uuid_anchor: UUIDanchor,
                 trait_convergence: TraitConvergenceEngine):
        
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.checkpoint_manager = checkpoint_manager
        self.trait_converter = trait_converter
        self.character_formation = character_formation
        self.uuid_anchor = uuid_anchor
        self.trait_convergence = trait_convergence
        
        # Formation state
        self.formation_queue: deque = deque()
        self.active_formations: Dict[uuid.UUID, WordFormationContext] = {}
        self.completed_formations: Dict[uuid.UUID, WordFormationResult] = {}
        self.word_patterns: Dict[uuid.UUID, WordPattern] = {}
        self.word_memory: Dict[uuid.UUID, WordMemory] = {}
        
        # Performance tracking
        self.formation_metrics = {
            'total_formations': 0,
            'successful_formations': 0,
            'failed_formations': 0,
            'average_formation_time': 0.0,
            'semantic_coherence_avg': 0.0,
            'mathematical_consistency_avg': 0.0,
            'word_validity_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize word patterns and validation rules
        self._initialize_word_patterns()
        self._initialize_validation_rules()
        
        # Register with event bridge
        self.event_bridge.register_handler("WORD_FORMATION_REQUEST", self._handle_formation_request)
        self.event_bridge.register_handler("CHARACTER_FORMATION_COMPLETE", self._handle_character_formation_complete)
        
        print(f"📝 SemanticRecursiveWordFormation initialized with {len(self.word_patterns)} patterns")
    
    def _initialize_word_patterns(self):
        """Initialize word formation patterns"""
        patterns = [
            {
                'word_type': WordType.NOUN,
                'formation_rules': {
                    'min_length': 3,
                    'max_length': 12,
                    'vowel_requirement': True,
                    'consonant_patterns': ['CVC', 'CVCC', 'CCVC'],
                    'suffix_patterns': ['er', 'ing', 'tion', 'ness'],
                    'mathematical_mapping': 'trait_combination'
                },
                'complexity_level': SemanticComplexity.BASIC
            },
            {
                'word_type': WordType.VERB,
                'formation_rules': {
                    'min_length': 3,
                    'max_length': 10,
                    'vowel_requirement': True,
                    'consonant_patterns': ['CVC', 'CVCC'],
                    'suffix_patterns': ['ing', 'ed', 'er', 'ate'],
                    'mathematical_mapping': 'action_synthesis'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'word_type': WordType.ADJECTIVE,
                'formation_rules': {
                    'min_length': 4,
                    'max_length': 12,
                    'vowel_requirement': True,
                    'consonant_patterns': ['CVC', 'CVCC', 'CCVC'],
                    'suffix_patterns': ['al', 'ous', 'ful', 'less', 'able'],
                    'mathematical_mapping': 'property_synthesis'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'word_type': WordType.ADVERB,
                'formation_rules': {
                    'min_length': 4,
                    'max_length': 10,
                    'vowel_requirement': True,
                    'consonant_patterns': ['CVC', 'CVCC'],
                    'suffix_patterns': ['ly', 'ward', 'wise'],
                    'mathematical_mapping': 'modifier_synthesis'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'word_type': WordType.COMPOSITE,
                'formation_rules': {
                    'min_length': 6,
                    'max_length': 15,
                    'vowel_requirement': True,
                    'consonant_patterns': ['CVC', 'CVCC', 'CCVC'],
                    'compound_patterns': ['noun+noun', 'adj+noun', 'verb+noun'],
                    'mathematical_mapping': 'composite_synthesis'
                },
                'complexity_level': SemanticComplexity.ADVANCED
            },
            {
                'word_type': WordType.ABSTRACT,
                'formation_rules': {
                    'min_length': 5,
                    'max_length': 12,
                    'vowel_requirement': True,
                    'consonant_patterns': ['CVC', 'CVCC', 'CCVC'],
                    'suffix_patterns': ['ism', 'ity', 'tion', 'ness'],
                    'mathematical_mapping': 'abstract_synthesis'
                },
                'complexity_level': SemanticComplexity.EXPERT
            }
        ]
        
        for pattern_data in patterns:
            pattern = WordPattern(
                pattern_id=uuid.uuid4(),
                word_type=pattern_data['word_type'],
                formation_rules=pattern_data['formation_rules'],
                complexity_level=pattern_data['complexity_level'],
                mathematical_foundation={'type': pattern_data['formation_rules']['mathematical_mapping']}
            )
            self.word_patterns[pattern.pattern_id] = pattern
    
    def _initialize_validation_rules(self):
        """Initialize word validation rules"""
        self.validation_rules = {
            'noun': {
                'pattern': r'^[a-zA-Z]{3,12}$',
                'description': 'Noun: 3-12 letters, alphabetic only',
                'vowel_check': True
            },
            'verb': {
                'pattern': r'^[a-zA-Z]{3,10}$',
                'description': 'Verb: 3-10 letters, alphabetic only',
                'vowel_check': True
            },
            'adjective': {
                'pattern': r'^[a-zA-Z]{4,12}$',
                'description': 'Adjective: 4-12 letters, alphabetic only',
                'vowel_check': True
            },
            'adverb': {
                'pattern': r'^[a-zA-Z]{4,10}$',
                'description': 'Adverb: 4-10 letters, alphabetic only',
                'vowel_check': True
            },
            'pronoun': {
                'pattern': r'^[a-zA-Z]{2,6}$',
                'description': 'Pronoun: 2-6 letters, alphabetic only',
                'vowel_check': True
            },
            'preposition': {
                'pattern': r'^[a-zA-Z]{2,5}$',
                'description': 'Preposition: 2-5 letters, alphabetic only',
                'vowel_check': True
            },
            'conjunction': {
                'pattern': r'^[a-zA-Z]{2,7}$',
                'description': 'Conjunction: 2-7 letters, alphabetic only',
                'vowel_check': True
            },
            'interjection': {
                'pattern': r'^[a-zA-Z]{2,8}$',
                'description': 'Interjection: 2-8 letters, alphabetic only',
                'vowel_check': True
            },
            'composite': {
                'pattern': r'^[a-zA-Z]{6,15}$',
                'description': 'Composite: 6-15 letters, alphabetic only',
                'vowel_check': True
            },
            'abstract': {
                'pattern': r'^[a-zA-Zαβγδεζηθικλμνξοπρστυφχψω]{5,12}$',
                'description': 'Abstract: 5-12 letters, alphabetic and Greek characters',
                'vowel_check': True
            }
        }
    
    def form_words(self,
                  source_characters: List[str],
                  source_traits: List[SemanticTrait],
                  target_type: WordType = WordType.NOUN,
                  formation_strategy: WordFormationStrategy = WordFormationStrategy.AUTO,
                  complexity_level: SemanticComplexity = SemanticComplexity.BASIC) -> WordFormationResult:
        """
        Form words from characters using recursive patterns
        """
        with self._lock:
            formation_id = uuid.uuid4()
            
            # Auto-select formation strategy if needed
            if formation_strategy == WordFormationStrategy.AUTO:
                formation_strategy = self._select_optimal_strategy(source_characters, source_traits, target_type, complexity_level)
            
            # Create formation context
            context = WordFormationContext(
                formation_id=formation_id,
                source_characters=source_characters,
                source_traits=source_traits,
                target_type=target_type,
                formation_strategy=formation_strategy,
                complexity_level=complexity_level
            )
            
            # Add to queue
            self.formation_queue.append(context)
            self.active_formations[formation_id] = context
            
            # Publish event
            self.event_bridge.publish_semantic_event(
                WordFormationEvent(
                    event_uuid=uuid.uuid4(),
            event_type="WORD_FORMATION_REQUEST",
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(formation_id),
                        'target_type': target_type.value,
                        'formation_strategy': formation_strategy.value,
                        'character_count': len(source_characters),
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
                                source_characters: List[str],
                                source_traits: List[SemanticTrait],
                                target_type: WordType,
                                complexity_level: SemanticComplexity) -> WordFormationStrategy:
        """Select optimal formation strategy based on characters and requirements"""
        
        # Analyze source characters and traits
        char_count = len(source_characters)
        trait_count = len(source_traits)
        avg_complexity = self._calculate_average_complexity(source_traits)
        
        # Strategy selection logic
        if char_count <= 5 and trait_count == 1:
            return WordFormationStrategy.CHARACTER_COMBINATION
        elif target_type in [WordType.NOUN, WordType.VERB] and char_count <= 8:
            return WordFormationStrategy.PATTERN_MATCHING
        elif target_type in [WordType.COMPOSITE, WordType.ABSTRACT] and avg_complexity >= SemanticComplexity.ADVANCED:
            return WordFormationStrategy.RECURSIVE_SYNTHESIS
        elif trait_count > 1 and char_count > 5:
            return WordFormationStrategy.CONTEXT_AWARE
        elif complexity_level >= SemanticComplexity.ADVANCED:
            return WordFormationStrategy.HYBRID_COMPOSITION
        else:
            return WordFormationStrategy.CHARACTER_COMBINATION
    
    def _execute_formation(self, context: WordFormationContext) -> WordFormationResult:
        """Execute the actual word formation operation"""
        start_time = datetime.utcnow()
        
        try:
            # Get formation method
            method_name = f"_form_{context.formation_strategy.value}"
            print(f"    🔍 Looking for word formation method: {method_name}")
            formation_method = getattr(self, method_name)
            print(f"    ✅ Found word formation method: {formation_method.__name__}")
            
            # Execute formation
            print(f"    🚀 Executing word formation with {len(context.source_characters)} characters")
            formed_words = formation_method(context)
            print(f"    📊 Word formation returned {len(formed_words)} words")
            
            # Validate words
            validated_words = []
            formation_patterns = []
            
            print(f"        🔍 Validating {len(formed_words)} words...")
            
            for i, word in enumerate(formed_words):
                print(f"          Validating word {i+1}: '{word}'")
                validation = self._validate_word(word, context.target_type)
                print(f"            → Validation result: {'✅ Valid' if validation.is_valid else '❌ Invalid'}")
                if validation.is_valid:
                    validated_words.append(word)
                    print(f"            → Added to validated words")
                else:
                    print(f"            → Rejected: {validation.error_details}")
                    
                    # Create formation pattern
                    pattern = FormationPattern(
                        pattern_uuid=uuid.uuid4(),
                        formation_type="word",
                        characters=list(word),
                        words=[word],
                        sentences=[],
                        complexity=context.complexity_level,
                        mathematical_consistency=validation.mathematical_consistency
                    )
                    formation_patterns.append(pattern)
            
            # Calculate metrics
            semantic_coherence = self._calculate_semantic_coherence(validated_words, context.source_traits)
            mathematical_consistency = self._calculate_mathematical_consistency(formation_patterns)
            
            # Create result
            result = WordFormationResult(
                formation_id=context.formation_id,
                context=context,
                formed_words=validated_words,
                formation_patterns=formation_patterns,
                formation_metrics={
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'word_count': len(validated_words),
                    'pattern_count': len(formation_patterns),
                    'validity_rate': len(validated_words) / len(formed_words) if formed_words else 0.0
                },
                status=WordStatus.COMPLETED if validated_words else WordStatus.FAILED,
                semantic_coherence=semantic_coherence,
                mathematical_consistency=mathematical_consistency,
                completion_time=datetime.utcnow()
            )
            
            # Store result
            self.completed_formations[context.formation_id] = result
            
            # Update memory
            self._update_word_memory(context, result)
            
            return result
            
        except Exception as e:
            # Handle formation failure
            result = WordFormationResult(
                formation_id=context.formation_id,
                context=context,
                formed_words=[],
                formation_patterns=[],
                formation_metrics={'error': str(e)},
                status=WordStatus.FAILED,
                error_details=str(e)
            )
            
            self.completed_formations[context.formation_id] = result
            return result
    
    def _form_character_combination(self, context: WordFormationContext) -> List[str]:
        """Form words using character combination approach"""
        words = []
        
        # Get pattern for target word type
        pattern = self._get_pattern_for_type(context.target_type)
        
        # Combine characters into words
        if len(context.source_characters) >= pattern.formation_rules['min_length']:
            # Direct combination
            word = ''.join(context.source_characters[:pattern.formation_rules['max_length']])
            
            # Ensure minimum length
            while len(word) < pattern.formation_rules['min_length']:
                word += 'a'
            
            # Apply vowel requirement
            if pattern.formation_rules['vowel_requirement'] and not any(vowel in word for vowel in 'aeiou'):
                word = word[:2] + 'a' + word[2:]
            
            words.append(word)
        
        return words
    
    def _form_pattern_matching(self, context: WordFormationContext) -> List[str]:
        """Form words using pattern matching approach"""
        words = []
        
        print(f"        🔧 Forming words using pattern matching method for {len(context.source_characters)} characters")
        
        # Get appropriate pattern
        pattern = self._get_pattern_for_type(context.target_type)
        print(f"          Using pattern for {context.target_type.value}: {pattern.pattern_id}")
        
        # Apply pattern-based formation
        word = self._apply_pattern_to_characters(context.source_characters, pattern)
        print(f"          → Created word: '{word}'")
        if word:
            words.append(word)
            print(f"          → Added word to results")
        else:
            print(f"          → No word created")
        
        return words
    
    def _form_recursive_synthesis(self, context: WordFormationContext) -> List[str]:
        """Form words using recursive synthesis approach"""
        words = []
        
        # Apply recursive word formation
        word = self._apply_recursive_synthesis_to_characters(context.source_characters, context.target_type)
        if word:
            words.append(word)
        
        return words
    
    def _form_context_aware(self, context: WordFormationContext) -> List[str]:
        """Form words using context-aware approach"""
        words = []
        
        # Get context from memory
        context_memory = self._get_relevant_word_memory(context.source_characters, context.source_traits)
        
        # Form words with context awareness
        word = self._form_with_context(context.source_characters, context_memory, context.target_type)
        if word:
            words.append(word)
        
        return words
    
    def _form_hybrid_composition(self, context: WordFormationContext) -> List[str]:
        """Form words using hybrid composition approach"""
        words = []
        
        # Apply multiple formation strategies
        strategies = [WordFormationStrategy.CHARACTER_COMBINATION, WordFormationStrategy.PATTERN_MATCHING, WordFormationStrategy.RECURSIVE_SYNTHESIS]
        
        for strategy in strategies:
            try:
                # Create sub-context
                sub_context = WordFormationContext(
                    formation_id=uuid.uuid4(),
                    source_characters=context.source_characters,
                    source_traits=context.source_traits,
                    target_type=context.target_type,
                    formation_strategy=strategy,
                    complexity_level=context.complexity_level
                )
                
                # Execute sub-formation
                formation_method = getattr(self, f"_form_{strategy.value}")
                sub_words = formation_method(sub_context)
                
                # Synthesize results
                words.extend(sub_words)
                
            except Exception as e:
                print(f"Warning: Strategy {strategy.value} failed: {e}")
                continue
        
        # Remove duplicates and optimize
        words = self._deduplicate_words(words)
        
        return words
    
    def _apply_pattern_to_characters(self, characters: List[str], pattern: WordPattern) -> Optional[str]:
        """Apply pattern to characters for word formation"""
        rules = pattern.formation_rules
        
        # Basic character combination
        base_word = ''.join(characters[:rules['max_length']])
        
        # Ensure minimum length
        while len(base_word) < rules['min_length']:
            base_word += 'a'
        
        # Apply vowel requirement
        if rules['vowel_requirement'] and not any(vowel in base_word for vowel in 'aeiou'):
            base_word = base_word[:2] + 'a' + base_word[2:]
        
        # Apply suffix patterns if available
        if 'suffix_patterns' in rules and rules['suffix_patterns']:
            # Select suffix based on word type
            suffix = self._select_appropriate_suffix(pattern.word_type, rules['suffix_patterns'])
            if suffix and len(base_word) + len(suffix) <= rules['max_length']:
                base_word += suffix
        
        return base_word[:rules['max_length']]
    
    def _apply_recursive_synthesis_to_characters(self, characters: List[str], target_type: WordType) -> Optional[str]:
        """Apply recursive synthesis to characters for word formation"""
        # Start with base character combination
        base_word = ''.join(characters[:8])  # Limit to 8 characters for synthesis
        
        # Apply recursive transformations based on word type
        if target_type == WordType.COMPOSITE:
            # Create composite word by combining patterns
            word = self._create_composite_word(base_word)
        elif target_type == WordType.ABSTRACT:
            # Create abstract word through recursive abstraction
            word = self._create_abstract_word(base_word)
        else:
            # Apply standard recursive transformation
            word = self._apply_recursive_transformation(base_word, target_type)
        
        return word
    
    def _create_composite_word(self, base_word: str) -> str:
        """Create composite word through pattern combination"""
        # Split base word into components
        if len(base_word) >= 6:
            mid_point = len(base_word) // 2
            part1 = base_word[:mid_point]
            part2 = base_word[mid_point:]
            
            # Combine with common composite patterns
            composite_patterns = ['er', 'ing', 'able', 'ful']
            suffix = composite_patterns[hash(part1) % len(composite_patterns)]
            
            return part1 + part2 + suffix
        else:
            return base_word + 'er'
    
    def _create_abstract_word(self, base_word: str) -> str:
        """Create abstract word through recursive abstraction"""
        # Apply abstract suffixes
        abstract_suffixes = ['ism', 'ity', 'tion', 'ness', 'ment']
        suffix = abstract_suffixes[hash(base_word) % len(abstract_suffixes)]
        
        return base_word + suffix
    
    def _apply_recursive_transformation(self, base_word: str, target_type: WordType) -> str:
        """Apply recursive transformation to base word"""
        # Apply transformations based on word type
        if target_type == WordType.NOUN:
            suffixes = ['er', 'ing', 'tion', 'ness']
        elif target_type == WordType.VERB:
            suffixes = ['ing', 'ed', 'er', 'ate']
        elif target_type == WordType.ADJECTIVE:
            suffixes = ['al', 'ous', 'ful', 'less', 'able']
        elif target_type == WordType.ADVERB:
            suffixes = ['ly', 'ward', 'wise']
        else:
            suffixes = ['er', 'ing', 'able']
        
        # Select suffix based on word hash
        suffix = suffixes[hash(base_word) % len(suffixes)]
        
        return base_word + suffix
    
    def _select_appropriate_suffix(self, word_type: WordType, available_suffixes: List[str]) -> Optional[str]:
        """Select appropriate suffix for word type"""
        # Map word types to preferred suffixes
        suffix_preferences = {
            WordType.NOUN: ['er', 'ing', 'tion', 'ness'],
            WordType.VERB: ['ing', 'ed', 'er', 'ate'],
            WordType.ADJECTIVE: ['al', 'ous', 'ful', 'less', 'able'],
            WordType.ADVERB: ['ly', 'ward', 'wise'],
            WordType.COMPOSITE: ['er', 'ing', 'able', 'ful'],
            WordType.ABSTRACT: ['ism', 'ity', 'tion', 'ness']
        }
        
        preferred = suffix_preferences.get(word_type, [])
        
        # Find intersection of preferred and available suffixes
        common_suffixes = [s for s in preferred if s in available_suffixes]
        
        if common_suffixes:
            return common_suffixes[0]
        elif available_suffixes:
            return available_suffixes[0]
        else:
            return None
    
    def _get_pattern_for_type(self, target_type: WordType) -> Optional[WordPattern]:
        """Get pattern for specific word type"""
        for pattern in self.word_patterns.values():
            if pattern.word_type == target_type:
                return pattern
        return None
    
    def _get_relevant_word_memory(self, characters: List[str], traits: List[SemanticTrait]) -> List[WordMemory]:
        """Get relevant word formation memory"""
        relevant_memories = []
        
        for memory in self.word_memory.values():
            # Check if memory contains similar characters or traits
            char_match = any(char in memory.character_combination for char in characters)
            trait_match = any(trait.trait_uuid in [t.trait_uuid for t in traits] for trait in traits)
            
            if char_match or trait_match:
                relevant_memories.append(memory)
        
        return relevant_memories
    
    def _form_with_context(self, characters: List[str], context_memory: List[WordMemory], target_type: WordType) -> Optional[str]:
        """Form word with context awareness"""
        if context_memory:
            # Use context from memory
            best_memory = max(context_memory, key=lambda m: m.success_metrics.get('semantic_coherence', 0))
            
            # Adapt word from memory
            if best_memory.formation_strategy == WordFormationStrategy.CHARACTER_COMBINATION:
                return self._form_character_combination(WordFormationContext(
                    formation_id=uuid.uuid4(),
                    source_characters=characters,
                    source_traits=[],
                    target_type=target_type,
                    formation_strategy=WordFormationStrategy.CHARACTER_COMBINATION,
                    complexity_level=SemanticComplexity.BASIC
                ))[0] if characters else None
            else:
                return best_memory.resulting_word
        
        # Fallback to basic formation
        return self._form_character_combination(WordFormationContext(
            formation_id=uuid.uuid4(),
            source_characters=characters,
            source_traits=[],
            target_type=target_type,
            formation_strategy=WordFormationStrategy.CHARACTER_COMBINATION,
            complexity_level=SemanticComplexity.BASIC
        ))[0] if characters else None
    
    def _validate_word(self, word: str, target_type: WordType) -> WordValidation:
        """Validate formed word"""
        validation_rules = []
        is_valid = False
        mathematical_consistency = 0.0
        semantic_relevance = 0.0
        word_type_match = None
        error_details = None
        
        try:
            # Get validation rule for target type
            rule_key = target_type.value
            if rule_key in self.validation_rules:
                rule = self.validation_rules[rule_key]
                validation_rules.append(rule['description'])
                
                # Check pattern match
                if re.match(rule['pattern'], word):
                    is_valid = True
                    mathematical_consistency = 0.9
                    semantic_relevance = 0.8
                    word_type_match = target_type
                    
                    # Check vowel requirement
                    if rule.get('vowel_check', False) and not any(vowel in word.lower() for vowel in 'aeiou'):
                        is_valid = False
                        error_details = f"Word '{word}' lacks required vowels"
                else:
                    error_details = f"Word '{word}' does not match pattern for {target_type.value}"
            else:
                # No specific rule, accept any alphabetic word
                if word.isalpha() and len(word) >= 2:
                    is_valid = True
                    mathematical_consistency = 0.7
                    semantic_relevance = 0.6
                else:
                    error_details = f"Word '{word}' is not a valid alphabetic word"
        
        except Exception as e:
            error_details = f"Validation error: {str(e)}"
        
        return WordValidation(
            word=word,
            is_valid=is_valid,
            validation_rules=validation_rules,
            mathematical_consistency=mathematical_consistency,
            semantic_relevance=semantic_relevance,
            word_type_match=word_type_match,
            error_details=error_details
        )
    
    def _deduplicate_words(self, words: List[str]) -> List[str]:
        """Remove duplicate words"""
        seen = set()
        unique_words = []
        
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        return unique_words
    
    def _calculate_semantic_coherence(self, words: List[str], source_traits: List[SemanticTrait]) -> float:
        """Calculate semantic coherence of formed words"""
        if not words or not source_traits:
            return 0.0
        
        coherence_scores = []
        for word in words:
            # Simple coherence calculation based on word properties
            if len(word) >= 3:
                # Longer words tend to be more coherent
                coherence = min(1.0, len(word) / 8.0)
                
                # Bonus for vowel-consonant balance
                vowels = sum(1 for char in word.lower() if char in 'aeiou')
                consonants = len(word) - vowels
                if vowels > 0 and consonants > 0:
                    balance_score = min(1.0, min(vowels, consonants) / max(vowels, consonants))
                    coherence = (coherence + balance_score) / 2
                
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
    
    def _update_word_memory(self, context: WordFormationContext, result: WordFormationResult):
        """Update word memory with successful formations"""
        if result.status == WordStatus.COMPLETED and result.semantic_coherence > 0.5:
            for word in result.formed_words:
                memory = WordMemory(
                    memory_id=uuid.uuid4(),
                    character_combination=context.source_characters,
                    resulting_word=word,
                    formation_strategy=context.formation_strategy,
                    word_type=context.target_type,
                    success_metrics={
                        'semantic_coherence': result.semantic_coherence,
                        'mathematical_consistency': result.mathematical_consistency
                    }
                )
                
                self.word_memory[memory.memory_id] = memory
    
    def _update_formation_metrics(self, result: WordFormationResult):
        """Update formation performance metrics"""
        self.formation_metrics['total_formations'] += 1
        
        if result.status == WordStatus.COMPLETED:
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
            current_avg = self.formation_metrics['word_validity_rate']
            total_formations = self.formation_metrics['total_formations']
            new_avg = (current_avg * (total_formations - 1) + result.formation_metrics['validity_rate']) / total_formations
            self.formation_metrics['word_validity_rate'] = new_avg
    
    def _handle_formation_request(self, event_data: Dict[str, Any]):
        """Handle word formation request events"""
        try:
            source_characters = event_data.get('source_characters', [])
            source_traits = event_data.get('source_traits', [])
            target_type = WordType(event_data.get('target_type', 'noun'))
            formation_strategy = WordFormationStrategy(event_data.get('formation_strategy', 'auto'))
            complexity_level = SemanticComplexity(event_data.get('complexity_level', 'basic'))
            
            result = self.form_words(source_characters, source_traits, target_type, formation_strategy, complexity_level)
            
            # Publish completion event
            self.event_bridge.publish_semantic_event(
                WordFormationEvent(
                    event_uuid=uuid.uuid4(),
            event_type="WORD_FORMATION_REQUEST",
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(result.formation_id),
                        'status': result.status.value,
                        'word_count': len(result.formed_words)
                    }
                )
            )
            
        except Exception as e:
            print(f"Error handling word formation request: {e}")
    
    def _handle_character_formation_complete(self, event_data: Dict[str, Any]):
        """Handle character formation completion events"""
        # Update patterns based on completed character formations
        formation_id = event_data.get('formation_id')
        if formation_id:
            # Update related patterns
            pass
    
    def get_formation_status(self, formation_id: uuid.UUID) -> Optional[WordFormationResult]:
        """Get status of specific word formation"""
        return self.completed_formations.get(formation_id)
    
    def get_formation_metrics(self) -> Dict[str, Any]:
        """Get current formation performance metrics"""
        return self.formation_metrics.copy()
    
    def get_word_patterns(self) -> Dict[uuid.UUID, WordPattern]:
        """Get all word patterns"""
        return self.word_patterns.copy()
    
    def get_word_memory(self) -> Dict[uuid.UUID, WordMemory]:
        """Get all word memory"""
        return self.word_memory.copy()
    
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
            
            print(f"Cleared {len(old_formations)} old word formations")
