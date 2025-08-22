# rebuild/enhanced_mathematical_typewriter.py
"""
Enhanced Mathematical Typewriter - Recursive linguistic formation system
Builds complex linguistic structures from semantic traits using mathematical patterns
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
    CharacterFormationEvent, WordFormationEvent, SentenceFormationEvent,
    DialogueFormationEvent, SemanticMilestoneEvent
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager
from semantic_trait_conversion import SemanticTraitConverter

class FormationLevel(Enum):
    """Levels of linguistic formation"""
    CHARACTER = "character"
    WORD = "word"
    PHRASE = "phrase"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    DIALOGUE = "dialogue"
    NARRATIVE = "narrative"

class FormationStrategy(Enum):
    """Strategies for linguistic formation"""
    LINEAR = "linear"
    RECURSIVE = "recursive"
    PATTERN_BASED = "pattern_based"
    CONTEXT_AWARE = "context_aware"
    HYBRID = "hybrid"

class FormationStatus(Enum):
    """Status of formation operation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class FormationContext:
    """Context for linguistic formation operations"""
    formation_id: uuid.UUID
    level: FormationLevel
    strategy: FormationStrategy
    source_traits: List[SemanticTrait]
    target_complexity: SemanticComplexity
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FormationResult:
    """Result of linguistic formation operation"""
    formation_id: uuid.UUID
    context: FormationContext
    formed_structures: List[str]
    formation_patterns: List[FormationPattern]
    formation_metrics: Dict[str, float]
    status: FormationStatus
    error_details: Optional[str] = None
    semantic_coherence: float = 0.0
    mathematical_consistency: float = 0.0
    completion_time: Optional[datetime] = None

@dataclass
class LinguisticPattern:
    """Pattern for linguistic formation"""
    pattern_id: uuid.UUID
    pattern_type: str
    formation_rules: Dict[str, Any]
    complexity_level: SemanticComplexity
    mathematical_foundation: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None

@dataclass
class FormationMemory:
    """Memory of successful formations"""
    memory_id: uuid.UUID
    pattern_sequence: List[str]
    input_traits: List[uuid.UUID]
    output_structures: List[str]
    success_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)

class EnhancedMathematicalTypewriter:
    """
    Enhanced mathematical typewriter system
    Builds complex linguistic structures from semantic traits
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 checkpoint_manager: SemanticCheckpointManager,
                 trait_converter: SemanticTraitConverter,
                 uuid_anchor: UUIDanchor,
                 trait_convergence: TraitConvergenceEngine):
        
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.checkpoint_manager = checkpoint_manager
        self.trait_converter = trait_converter
        self.uuid_anchor = uuid_anchor
        self.trait_convergence = trait_convergence
        
        # Formation state
        self.formation_queue: deque = deque()
        self.active_formations: Dict[uuid.UUID, FormationContext] = {}
        self.completed_formations: Dict[uuid.UUID, FormationResult] = {}
        self.linguistic_patterns: Dict[uuid.UUID, LinguisticPattern] = {}
        self.formation_memory: Dict[uuid.UUID, FormationMemory] = {}
        
        # Performance tracking
        self.formation_metrics = {
            'total_formations': 0,
            'successful_formations': 0,
            'failed_formations': 0,
            'average_formation_time': 0.0,
            'semantic_coherence_avg': 0.0,
            'mathematical_consistency_avg': 0.0,
            'pattern_success_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize formation strategies and patterns
        self._initialize_formation_strategies()
        self._initialize_linguistic_patterns()
        
        # Register with event bridge
        self.event_bridge.register_handler("LINGUISTIC_FORMATION_REQUEST", self._handle_formation_request)
        self.event_bridge.register_handler("TRAIT_CONVERSION_COMPLETE", self._handle_conversion_complete)
        
        print(f"⌨️ EnhancedMathematicalTypewriter initialized with {len(self.formation_strategies)} strategies and {len(self.linguistic_patterns)} patterns")
    
    def _initialize_formation_strategies(self):
        """Initialize available formation strategies"""
        self.formation_strategies = {
            'linear': {
                'description': 'Sequential formation from traits to structures',
                'complexity_range': [SemanticComplexity.BASIC, SemanticComplexity.INTERMEDIATE],
                'supports_levels': [FormationLevel.CHARACTER, FormationLevel.WORD, FormationLevel.SENTENCE]
            },
            'recursive': {
                'description': 'Recursive typewriter approach with self-reference',
                'complexity_range': [SemanticComplexity.ADVANCED, SemanticComplexity.EXPERT],
                'supports_levels': [FormationLevel.CHARACTER, FormationLevel.WORD, FormationLevel.SENTENCE, FormationLevel.DIALOGUE]
            },
            'pattern_based': {
                'description': 'Formation based on mathematical pattern matching',
                'complexity_range': [SemanticComplexity.INTERMEDIATE, SemanticComplexity.ADVANCED],
                'supports_levels': [FormationLevel.WORD, FormationLevel.PHRASE, FormationLevel.SENTENCE]
            },
            'context_aware': {
                'description': 'Context-sensitive formation with memory',
                'complexity_range': [SemanticComplexity.ADVANCED, SemanticComplexity.EXPERT],
                'supports_levels': [FormationLevel.SENTENCE, FormationLevel.PARAGRAPH, FormationLevel.DIALOGUE]
            },
            'hybrid': {
                'description': 'Combination of multiple formation approaches',
                'complexity_range': [SemanticComplexity.INTERMEDIATE, SemanticComplexity.EXPERT],
                'supports_levels': [FormationLevel.WORD, FormationLevel.SENTENCE, FormationLevel.DIALOGUE, FormationLevel.NARRATIVE]
            }
        }
    
    def _initialize_linguistic_patterns(self):
        """Initialize linguistic formation patterns"""
        patterns = [
            {
                'pattern_type': 'character_formation',
                'formation_rules': {
                    'min_length': 1,
                    'max_length': 1,
                    'allowed_chars': 'abcdefghijklmnopqrstuvwxyz',
                    'mathematical_mapping': 'hash_to_ascii'
                },
                'complexity_level': SemanticComplexity.BASIC
            },
            {
                'pattern_type': 'word_formation',
                'formation_rules': {
                    'min_length': 3,
                    'max_length': 12,
                    'vowel_requirement': True,
                    'consonant_patterns': ['CVC', 'CVCC', 'CCVC'],
                    'mathematical_mapping': 'trait_combination'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'pattern_type': 'sentence_formation',
                'formation_rules': {
                    'min_words': 3,
                    'max_words': 20,
                    'structure_patterns': ['SVO', 'SVC', 'SVA'],
                    'punctuation_rules': True,
                    'mathematical_mapping': 'recursive_synthesis'
                },
                'complexity_level': SemanticComplexity.ADVANCED
            },
            {
                'pattern_type': 'dialogue_formation',
                'formation_rules': {
                    'min_sentences': 2,
                    'max_sentences': 10,
                    'speaker_patterns': ['A', 'B', 'A'],
                    'context_continuity': True,
                    'mathematical_mapping': 'context_aware_synthesis'
                },
                'complexity_level': SemanticComplexity.EXPERT
            }
        ]
        
        for pattern_data in patterns:
            pattern = LinguisticPattern(
                pattern_id=uuid.uuid4(),
                pattern_type=pattern_data['pattern_type'],
                formation_rules=pattern_data['formation_rules'],
                complexity_level=pattern_data['complexity_level'],
                mathematical_foundation={'type': pattern_data['formation_rules']['mathematical_mapping']}
            )
            self.linguistic_patterns[pattern.pattern_id] = pattern
    
    def form_linguistic_structures(self,
                                  source_traits: List[SemanticTrait],
                                  level: FormationLevel,
                                  strategy: FormationStrategy = FormationStrategy.AUTO,
                                  target_complexity: SemanticComplexity = SemanticComplexity.INTERMEDIATE) -> FormationResult:
        """
        Form linguistic structures from semantic traits
        """
        with self._lock:
            formation_id = uuid.uuid4()
            
            # Auto-select strategy if needed
            if strategy == FormationStrategy.AUTO:
                strategy = self._select_optimal_strategy(source_traits, level, target_complexity)
            
            # Create formation context
            context = FormationContext(
                formation_id=formation_id,
                level=level,
                strategy=strategy,
                source_traits=source_traits,
                target_complexity=target_complexity
            )
            
            # Add to queue
            self.formation_queue.append(context)
            self.active_formations[formation_id] = context
            
            # Publish event
            self.event_bridge.publish_semantic_event(
                CharacterFormationEvent(
                    event_id=uuid.uuid4(),
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(formation_id),
                        'level': level.value,
                        'strategy': strategy.value,
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
                                source_traits: List[SemanticTrait],
                                level: FormationLevel,
                                target_complexity: SemanticComplexity) -> FormationStrategy:
        """Select optimal formation strategy based on traits and requirements"""
        
        # Analyze source traits
        trait_count = len(source_traits)
        avg_complexity = self._calculate_average_complexity(source_traits)
        
        # Strategy selection logic
        if level == FormationLevel.CHARACTER:
            return FormationStrategy.LINEAR
        elif level == FormationLevel.WORD and trait_count <= 5:
            return FormationStrategy.PATTERN_BASED
        elif level == FormationLevel.SENTENCE and avg_complexity >= SemanticComplexity.ADVANCED:
            return FormationStrategy.CONTEXT_AWARE
        elif level in [FormationLevel.DIALOGUE, FormationLevel.NARRATIVE]:
            return FormationStrategy.HYBRID
        else:
            return FormationStrategy.RECURSIVE
    
    def _execute_formation(self, context: FormationContext) -> FormationResult:
        """Execute the actual formation operation"""
        start_time = datetime.utcnow()
        
        try:
            # Get formation method
            formation_method = getattr(self, f"_form_{context.strategy.value}")
            
            # Execute formation
            formed_structures, formation_patterns = formation_method(context)
            
            # Calculate metrics
            semantic_coherence = self._calculate_semantic_coherence(formed_structures)
            mathematical_consistency = self._calculate_mathematical_consistency(formation_patterns)
            
            # Create result
            result = FormationResult(
                formation_id=context.formation_id,
                context=context,
                formed_structures=formed_structures,
                formation_patterns=formation_patterns,
                formation_metrics={
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'structure_count': len(formed_structures),
                    'pattern_count': len(formation_patterns),
                    'complexity_achieved': self._calculate_average_complexity(context.source_traits).value
                },
                status=FormationStatus.COMPLETED,
                semantic_coherence=semantic_coherence,
                mathematical_consistency=mathematical_consistency,
                completion_time=datetime.utcnow()
            )
            
            # Store result
            self.completed_formations[context.formation_id] = result
            
            # Update memory
            self._update_formation_memory(context, result)
            
            return result
            
        except Exception as e:
            # Handle formation failure
            result = FormationResult(
                formation_id=context.formation_id,
                context=context,
                formed_structures=[],
                formation_patterns=[],
                formation_metrics={'error': str(e)},
                status=FormationStatus.FAILED,
                error_details=str(e)
            )
            
            self.completed_formations[context.formation_id] = result
            return result
    
    def _form_linear(self, context: FormationContext) -> Tuple[List[str], List[FormationPattern]]:
        """Form structures using linear approach"""
        formed_structures = []
        formation_patterns = []
        
        for trait in context.source_traits:
            if context.level == FormationLevel.CHARACTER:
                structure = self._form_character_linear(trait)
            elif context.level == FormationLevel.WORD:
                structure = self._form_word_linear(trait)
            elif context.level == FormationLevel.SENTENCE:
                structure = self._form_sentence_linear(trait)
            else:
                structure = self._form_default_linear(trait)
            
            formed_structures.append(structure)
            
            # Create formation pattern
            pattern = FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type="linear",
                characters=[structure[0]] if structure else [],
                words=[structure] if context.level == FormationLevel.WORD else [],
                sentences=[structure] if context.level == FormationLevel.SENTENCE else [],
                complexity=context.target_complexity,
                mathematical_consistency=trait.mathematical_consistency
            )
            formation_patterns.append(pattern)
        
        return formed_structures, formation_patterns
    
    def _form_recursive(self, context: FormationContext) -> Tuple[List[str], List[FormationPattern]]:
        """Form structures using recursive typewriter approach"""
        formed_structures = []
        formation_patterns = []
        
        # Extract characters from traits
        all_characters = []
        for trait in context.source_traits:
            chars = self._extract_characters_from_trait(trait)
            all_characters.extend(chars)
        
        # Recursive formation process
        if context.level == FormationLevel.CHARACTER:
            structures = self._recursive_character_formation(all_characters)
        elif context.level == FormationLevel.WORD:
            structures = self._recursive_word_formation(all_characters)
        elif context.level == FormationLevel.SENTENCE:
            structures = self._recursive_sentence_formation(all_characters)
        elif context.level == FormationLevel.DIALOGUE:
            structures = self._recursive_dialogue_formation(all_characters)
        else:
            structures = self._recursive_default_formation(all_characters)
        
        formed_structures.extend(structures)
        
        # Create formation patterns
        for structure in structures:
            pattern = FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type="recursive",
                characters=list(structure) if context.level == FormationLevel.CHARACTER else [],
                words=[structure] if context.level == FormationLevel.WORD else [],
                sentences=[structure] if context.level == FormationLevel.SENTENCE else [],
                complexity=context.target_complexity,
                mathematical_consistency=0.9  # High consistency for recursive patterns
            )
            formation_patterns.append(pattern)
        
        return formed_structures, formation_patterns
    
    def _form_pattern_based(self, context: FormationContext) -> Tuple[List[str], List[FormationPattern]]:
        """Form structures using pattern-based approach"""
        formed_structures = []
        formation_patterns = []
        
        # Get appropriate patterns
        patterns = self._get_patterns_for_level(context.level)
        
        for trait in context.source_traits:
            # Match trait to pattern
            best_pattern = self._find_best_pattern(trait, patterns)
            
            if best_pattern:
                structure = self._apply_pattern(trait, best_pattern)
                formed_structures.append(structure)
                
                # Create formation pattern
                pattern = FormationPattern(
                    pattern_uuid=uuid.uuid4(),
                    formation_type="pattern_based",
                    characters=list(structure) if context.level == FormationLevel.CHARACTER else [],
                    words=[structure] if context.level == FormationLevel.WORD else [],
                    sentences=[structure] if context.level == FormationLevel.SENTENCE else [],
                    complexity=context.target_complexity,
                    mathematical_consistency=trait.mathematical_consistency
                )
                formation_patterns.append(pattern)
        
        return formed_structures, formation_patterns
    
    def _form_context_aware(self, context: FormationContext) -> Tuple[List[str], List[FormationPattern]]:
        """Form structures using context-aware approach"""
        formed_structures = []
        formation_patterns = []
        
        # Get context from memory
        context_memory = self._get_relevant_memory(context.source_traits)
        
        # Form structures with context awareness
        for trait in context.source_traits:
            structure = self._form_with_context(trait, context_memory, context.level)
            formed_structures.append(structure)
            
            # Create formation pattern
            pattern = FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type="context_aware",
                characters=list(structure) if context.level == FormationLevel.CHARACTER else [],
                words=[structure] if context.level == FormationLevel.WORD else [],
                sentences=[structure] if context.level == FormationLevel.SENTENCE else [],
                complexity=context.target_complexity,
                mathematical_consistency=trait.mathematical_consistency
            )
            formation_patterns.append(pattern)
        
        return formed_structures, formation_patterns
    
    def _form_hybrid(self, context: FormationContext) -> Tuple[List[str], List[FormationPattern]]:
        """Form structures using hybrid approach"""
        formed_structures = []
        formation_patterns = []
        
        # Apply multiple formation strategies
        strategies = [FormationStrategy.LINEAR, FormationStrategy.PATTERN_BASED, FormationStrategy.RECURSIVE]
        
        for strategy in strategies:
            try:
                # Create sub-context
                sub_context = FormationContext(
                    formation_id=uuid.uuid4(),
                    level=context.level,
                    strategy=strategy,
                    source_traits=context.source_traits,
                    target_complexity=context.target_complexity
                )
                
                # Execute sub-formation
                formation_method = getattr(self, f"_form_{strategy.value}")
                sub_structures, sub_patterns = formation_method(sub_context)
                
                # Synthesize results
                formed_structures.extend(sub_structures)
                formation_patterns.extend(sub_patterns)
                
            except Exception as e:
                print(f"Warning: Strategy {strategy.value} failed: {e}")
                continue
        
        # Remove duplicates and optimize
        formed_structures = self._deduplicate_structures(formed_structures)
        formation_patterns = self._optimize_patterns(formation_patterns)
        
        return formed_structures, formation_patterns
    
    def _form_character_linear(self, trait: SemanticTrait) -> str:
        """Form character using linear approach"""
        if trait.semantic_meaning:
            return trait.semantic_meaning[0]
        else:
            # Generate character from trait properties
            trait_hash = hashlib.md5(trait.trait_name.encode()).hexdigest()
            char_index = int(trait_hash[:8], 16) % 26
            return chr(ord('a') + char_index)
    
    def _form_word_linear(self, trait: SemanticTrait) -> str:
        """Form word using linear approach"""
        if trait.semantic_meaning and len(trait.semantic_meaning) >= 3:
            return trait.semantic_meaning
        else:
            # Generate word from trait properties
            base_chars = self._extract_characters_from_trait(trait)
            word = ''.join(base_chars)
            
            # Ensure minimum length
            while len(word) < 3:
                word += 'a'
            
            return word[:12]  # Max length
    
    def _form_sentence_linear(self, trait: SemanticTrait) -> str:
        """Form sentence using linear approach"""
        if trait.semantic_meaning:
            return trait.semantic_meaning + "."
        else:
            # Generate sentence from trait
            word = self._form_word_linear(trait)
            return f"The {word} is present."
    
    def _form_default_linear(self, trait: SemanticTrait) -> str:
        """Form default structure using linear approach"""
        return trait.semantic_meaning or trait.trait_name
    
    def _recursive_character_formation(self, characters: List[str]) -> List[str]:
        """Form characters using recursive approach"""
        formed_characters = []
        
        for char in characters:
            # Apply recursive transformation
            transformed_char = self._apply_recursive_transformation(char)
            formed_characters.append(transformed_char)
        
        return formed_characters
    
    def _recursive_word_formation(self, characters: List[str]) -> List[str]:
        """Form words using recursive approach"""
        words = []
        current_word = ""
        
        for char in characters:
            current_word += char
            if len(current_word) >= 3:
                # Apply recursive word formation
                formed_word = self._apply_recursive_word_formation(current_word)
                words.append(formed_word)
                current_word = ""
        
        if current_word:
            formed_word = self._apply_recursive_word_formation(current_word)
            words.append(formed_word)
        
        return words if words else ['default']
    
    def _recursive_sentence_formation(self, characters: List[str]) -> List[str]:
        """Form sentences using recursive approach"""
        words = self._recursive_word_formation(characters)
        
        sentences = []
        current_sentence = []
        
        for word in words:
            current_sentence.append(word)
            if len(current_sentence) >= 3:
                # Apply recursive sentence formation
                formed_sentence = self._apply_recursive_sentence_formation(current_sentence)
                sentences.append(formed_sentence)
                current_sentence = []
        
        if current_sentence:
            formed_sentence = self._apply_recursive_sentence_formation(current_sentence)
            sentences.append(formed_sentence)
        
        return sentences if sentences else ["Default sentence."]
    
    def _recursive_dialogue_formation(self, characters: List[str]) -> List[str]:
        """Form dialogue using recursive approach"""
        sentences = self._recursive_sentence_formation(characters)
        
        dialogues = []
        current_dialogue = []
        
        for sentence in sentences:
            current_dialogue.append(sentence)
            if len(current_dialogue) >= 2:
                # Apply recursive dialogue formation
                formed_dialogue = self._apply_recursive_dialogue_formation(current_dialogue)
                dialogues.append(formed_dialogue)
                current_dialogue = []
        
        if current_dialogue:
            formed_dialogue = self._apply_recursive_dialogue_formation(current_dialogue)
            dialogues.append(formed_dialogue)
        
        return dialogues if dialogues else ["A: Hello. B: Hello."]
    
    def _recursive_default_formation(self, characters: List[str]) -> List[str]:
        """Form default structures using recursive approach"""
        return self._recursive_word_formation(characters)
    
    def _apply_recursive_transformation(self, char: str) -> str:
        """Apply recursive transformation to character"""
        # Simple recursive transformation
        char_code = ord(char.lower())
        transformed_code = (char_code - ord('a') + 1) % 26 + ord('a')
        return chr(transformed_code)
    
    def _apply_recursive_word_formation(self, word: str) -> str:
        """Apply recursive word formation"""
        # Ensure word follows basic rules
        if len(word) < 3:
            word += 'a' * (3 - len(word))
        
        # Apply vowel insertion if needed
        if not any(vowel in word for vowel in 'aeiou'):
            word = word[:2] + 'a' + word[2:]
        
        return word[:12]  # Max length
    
    def _apply_recursive_sentence_formation(self, words: List[str]) -> str:
        """Apply recursive sentence formation"""
        if not words:
            return "Default sentence."
        
        # Basic sentence structure
        sentence = " ".join(words)
        
        # Add punctuation
        if not sentence.endswith(('.', '!', '?')):
            sentence += "."
        
        return sentence
    
    def _apply_recursive_dialogue_formation(self, sentences: List[str]) -> str:
        """Apply recursive dialogue formation"""
        if len(sentences) < 2:
            return "A: Hello. B: Hello."
        
        dialogue_parts = []
        for i, sentence in enumerate(sentences):
            speaker = "A" if i % 2 == 0 else "B"
            dialogue_parts.append(f"{speaker}: {sentence}")
        
        return " ".join(dialogue_parts)
    
    def _get_patterns_for_level(self, level: FormationLevel) -> List[LinguisticPattern]:
        """Get patterns appropriate for formation level"""
        patterns = []
        
        for pattern in self.linguistic_patterns.values():
            if pattern.pattern_type == f"{level.value}_formation":
                patterns.append(pattern)
        
        return patterns
    
    def _find_best_pattern(self, trait: SemanticTrait, patterns: List[LinguisticPattern]) -> Optional[LinguisticPattern]:
        """Find best pattern for trait"""
        if not patterns:
            return None
        
        # Simple pattern selection based on complexity match
        for pattern in patterns:
            if pattern.complexity_level == trait.complexity:
                return pattern
        
        # Fallback to first pattern
        return patterns[0]
    
    def _apply_pattern(self, trait: SemanticTrait, pattern: LinguisticPattern) -> str:
        """Apply pattern to trait"""
        rules = pattern.formation_rules
        
        if pattern.pattern_type == 'character_formation':
            return self._apply_character_pattern(trait, rules)
        elif pattern.pattern_type == 'word_formation':
            return self._apply_word_pattern(trait, rules)
        elif pattern.pattern_type == 'sentence_formation':
            return self._apply_sentence_pattern(trait, rules)
        elif pattern.pattern_type == 'dialogue_formation':
            return self._apply_dialogue_pattern(trait, rules)
        else:
            return trait.semantic_meaning or trait.trait_name
    
    def _apply_character_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply character formation pattern"""
        if trait.semantic_meaning:
            char = trait.semantic_meaning[0].lower()
            if char in rules['allowed_chars']:
                return char
        
        # Generate character from trait
        trait_hash = hashlib.md5(trait.trait_name.encode()).hexdigest()
        char_index = int(trait_hash[:8], 16) % 26
        return chr(ord('a') + char_index)
    
    def _apply_word_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply word formation pattern"""
        if trait.semantic_meaning and len(trait.semantic_meaning) >= rules['min_length']:
            word = trait.semantic_meaning[:rules['max_length']]
            
            # Check vowel requirement
            if rules['vowel_requirement'] and not any(vowel in word for vowel in 'aeiou'):
                word = word[:2] + 'a' + word[2:]
            
            return word
        
        # Generate word from trait
        base_chars = self._extract_characters_from_trait(trait)
        word = ''.join(base_chars)
        
        # Ensure minimum length
        while len(word) < rules['min_length']:
            word += 'a'
        
        return word[:rules['max_length']]
    
    def _apply_sentence_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply sentence formation pattern"""
        if trait.semantic_meaning:
            return trait.semantic_meaning + "."
        
        # Generate sentence from trait
        word = self._apply_word_pattern(trait, {'min_length': 3, 'max_length': 12, 'vowel_requirement': True})
        return f"The {word} is present."
    
    def _apply_dialogue_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply dialogue formation pattern"""
        sentence = self._apply_sentence_pattern(trait, {'min_words': 3, 'max_words': 20})
        return f"A: {sentence} B: I understand."
    
    def _get_relevant_memory(self, traits: List[SemanticTrait]) -> List[FormationMemory]:
        """Get relevant formation memory"""
        relevant_memories = []
        
        for memory in self.formation_memory.values():
            # Check if memory contains similar traits
            for trait in traits:
                if trait.trait_uuid in memory.input_traits:
                    relevant_memories.append(memory)
                    break
        
        return relevant_memories
    
    def _form_with_context(self, trait: SemanticTrait, context_memory: List[FormationMemory], level: FormationLevel) -> str:
        """Form structure with context awareness"""
        if context_memory:
            # Use context from memory
            best_memory = max(context_memory, key=lambda m: m.success_metrics.get('semantic_coherence', 0))
            
            # Adapt pattern from memory
            if best_memory.pattern_sequence:
                pattern_type = best_memory.pattern_sequence[0]
                return self._adapt_pattern_to_trait(trait, pattern_type)
        
        # Fallback to basic formation
        if level == FormationLevel.CHARACTER:
            return self._form_character_linear(trait)
        elif level == FormationLevel.WORD:
            return self._form_word_linear(trait)
        elif level == FormationLevel.SENTENCE:
            return self._form_sentence_linear(trait)
        else:
            return trait.semantic_meaning or trait.trait_name
    
    def _adapt_pattern_to_trait(self, trait: SemanticTrait, pattern_type: str) -> str:
        """Adapt pattern to specific trait"""
        if pattern_type == 'character_formation':
            return self._form_character_linear(trait)
        elif pattern_type == 'word_formation':
            return self._form_word_linear(trait)
        elif pattern_type == 'sentence_formation':
            return self._form_sentence_linear(trait)
        else:
            return trait.semantic_meaning or trait.trait_name
    
    def _extract_characters_from_trait(self, trait: SemanticTrait) -> List[str]:
        """Extract characters from semantic trait"""
        if trait.semantic_meaning:
            return list(trait.semantic_meaning.lower())
        else:
            # Generate characters from trait name
            return list(trait.trait_name.lower())
    
    def _deduplicate_structures(self, structures: List[str]) -> List[str]:
        """Remove duplicate structures"""
        seen = set()
        unique_structures = []
        
        for structure in structures:
            if structure not in seen:
                seen.add(structure)
                unique_structures.append(structure)
        
        return unique_structures
    
    def _optimize_patterns(self, patterns: List[FormationPattern]) -> List[FormationPattern]:
        """Optimize formation patterns"""
        # Remove duplicate patterns
        seen_uuids = set()
        unique_patterns = []
        
        for pattern in patterns:
            if pattern.pattern_uuid not in seen_uuids:
                seen_uuids.add(pattern.pattern_uuid)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _calculate_semantic_coherence(self, structures: List[str]) -> float:
        """Calculate semantic coherence of formed structures"""
        if not structures:
            return 0.0
        
        coherence_scores = []
        for structure in structures:
            # Simple coherence calculation based on structure properties
            if len(structure) > 0:
                # Basic coherence: longer structures tend to be more coherent
                coherence = min(1.0, len(structure) / 10.0)
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
    
    def _update_formation_memory(self, context: FormationContext, result: FormationResult):
        """Update formation memory with successful patterns"""
        if result.status == FormationStatus.COMPLETED and result.semantic_coherence > 0.5:
            memory = FormationMemory(
                memory_id=uuid.uuid4(),
                pattern_sequence=[context.strategy.value],
                input_traits=[trait.trait_uuid for trait in context.source_traits],
                output_structures=result.formed_structures,
                success_metrics={
                    'semantic_coherence': result.semantic_coherence,
                    'mathematical_consistency': result.mathematical_consistency
                }
            )
            
            self.formation_memory[memory.memory_id] = memory
    
    def _update_formation_metrics(self, result: FormationResult):
        """Update formation performance metrics"""
        self.formation_metrics['total_formations'] += 1
        
        if result.status == FormationStatus.COMPLETED:
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
    
    def _handle_formation_request(self, event_data: Dict[str, Any]):
        """Handle formation request events"""
        try:
            source_traits = event_data.get('source_traits', [])
            level = FormationLevel(event_data.get('level', 'word'))
            strategy = FormationStrategy(event_data.get('strategy', 'auto'))
            target_complexity = SemanticComplexity(event_data.get('target_complexity', 'intermediate'))
            
            result = self.form_linguistic_structures(source_traits, level, strategy, target_complexity)
            
            # Publish completion event
            self.event_bridge.publish_semantic_event(
                WordFormationEvent(
                    event_id=uuid.uuid4(),
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(result.formation_id),
                        'status': result.status.value,
                        'structure_count': len(result.formed_structures)
                    }
                )
            )
            
        except Exception as e:
            print(f"Error handling formation request: {e}")
    
    def _handle_conversion_complete(self, event_data: Dict[str, Any]):
        """Handle conversion completion events"""
        # Update patterns based on completed conversions
        conversion_id = event_data.get('conversion_id')
        if conversion_id:
            # Update related patterns
            pass
    
    def get_formation_status(self, formation_id: uuid.UUID) -> Optional[FormationResult]:
        """Get status of specific formation"""
        return self.completed_formations.get(formation_id)
    
    def get_formation_metrics(self) -> Dict[str, Any]:
        """Get current formation performance metrics"""
        return self.formation_metrics.copy()
    
    def get_linguistic_patterns(self) -> Dict[uuid.UUID, LinguisticPattern]:
        """Get all linguistic patterns"""
        return self.linguistic_patterns.copy()
    
    def get_formation_memory(self) -> Dict[uuid.UUID, FormationMemory]:
        """Get all formation memory"""
        return self.formation_memory.copy()
    
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
            
            print(f"Cleared {len(old_formations)} old formations")
