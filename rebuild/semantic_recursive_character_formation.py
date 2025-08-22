# rebuild/semantic_recursive_character_formation.py
"""
Semantic Recursive Character Formation - Foundation linguistic system
Creates individual characters from mathematical traits using recursive patterns
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

class CharacterType(Enum):
    """Types of characters that can be formed"""
    ALPHABETIC = "alphabetic"
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    COMPOSITE = "composite"
    ABSTRACT = "abstract"

class FormationMethod(Enum):
    """Methods for character formation"""
    AUTO = "auto"
    HASH_BASED = "hash_based"
    PATTERN_MATCHING = "pattern_matching"
    RECURSIVE_TRANSFORM = "recursive_transform"
    CONTEXT_SENSITIVE = "context_sensitive"
    HYBRID_SYNTHESIS = "hybrid_synthesis"

class CharacterStatus(Enum):
    """Status of character formation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"

@dataclass
class CharacterFormationContext:
    """Context for character formation operations"""
    formation_id: uuid.UUID
    source_traits: List[SemanticTrait]
    target_type: CharacterType
    formation_method: FormationMethod
    complexity_level: SemanticComplexity
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CharacterFormationResult:
    """Result of character formation operation"""
    formation_id: uuid.UUID
    context: CharacterFormationContext
    formed_characters: List[str]
    formation_patterns: List[FormationPattern]
    formation_metrics: Dict[str, float]
    status: CharacterStatus
    error_details: Optional[str] = None
    semantic_coherence: float = 0.0
    mathematical_consistency: float = 0.0
    completion_time: Optional[datetime] = None

@dataclass
class CharacterPattern:
    """Pattern for character formation"""
    pattern_id: uuid.UUID
    character_type: CharacterType
    formation_rules: Dict[str, Any]
    complexity_level: SemanticComplexity
    mathematical_foundation: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None

@dataclass
class CharacterMemory:
    """Memory of successful character formations"""
    memory_id: uuid.UUID
    trait_combination: List[uuid.UUID]
    resulting_character: str
    formation_method: FormationMethod
    success_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CharacterValidation:
    """Validation result for character formation"""
    character: str
    is_valid: bool
    validation_rules: List[str]
    mathematical_consistency: float
    semantic_relevance: float
    error_details: Optional[str] = None

class SemanticRecursiveCharacterFormation:
    """
    Semantic recursive character formation system
    Creates individual characters from mathematical traits using recursive patterns
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
        self.active_formations: Dict[uuid.UUID, CharacterFormationContext] = {}
        self.completed_formations: Dict[uuid.UUID, CharacterFormationResult] = {}
        self.character_patterns: Dict[uuid.UUID, CharacterPattern] = {}
        self.character_memory: Dict[uuid.UUID, CharacterMemory] = {}
        
        # Performance tracking
        self.formation_metrics = {
            'total_formations': 0,
            'successful_formations': 0,
            'failed_formations': 0,
            'average_formation_time': 0.0,
            'semantic_coherence_avg': 0.0,
            'mathematical_consistency_avg': 0.0,
            'character_validity_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize character patterns and validation rules
        self._initialize_character_patterns()
        self._initialize_validation_rules()
        
        # Register with event bridge
        self.event_bridge.register_handler("CHARACTER_FORMATION_REQUEST", self._handle_formation_request)
        self.event_bridge.register_handler("TRAIT_CONVERSION_COMPLETE", self._handle_conversion_complete)
        
        print(f"🔤 SemanticRecursiveCharacterFormation initialized with {len(self.character_patterns)} patterns")
    
    def _initialize_character_patterns(self):
        """Initialize character formation patterns"""
        patterns = [
            {
                'character_type': CharacterType.ALPHABETIC,
                'formation_rules': {
                    'allowed_chars': 'abcdefghijklmnopqrstuvwxyz',
                    'case_sensitive': False,
                    'min_length': 1,
                    'max_length': 1,
                    'mathematical_mapping': 'hash_to_ascii'
                },
                'complexity_level': SemanticComplexity.BASIC
            },
            {
                'character_type': CharacterType.NUMERIC,
                'formation_rules': {
                    'allowed_chars': '0123456789',
                    'min_length': 1,
                    'max_length': 1,
                    'mathematical_mapping': 'trait_value_modulo'
                },
                'complexity_level': SemanticComplexity.BASIC
            },
            {
                'character_type': CharacterType.SYMBOLIC,
                'formation_rules': {
                    'allowed_chars': '!@#$%^&*()_+-=[]{}|;:,.<>?',
                    'min_length': 1,
                    'max_length': 1,
                    'mathematical_mapping': 'hash_to_symbol'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'character_type': CharacterType.COMPOSITE,
                'formation_rules': {
                    'allowed_chars': 'abcdefghijklmnopqrstuvwxyz0123456789',
                    'min_length': 1,
                    'max_length': 2,
                    'mathematical_mapping': 'trait_combination'
                },
                'complexity_level': SemanticComplexity.ADVANCED
            },
            {
                'character_type': CharacterType.ABSTRACT,
                'formation_rules': {
                    'allowed_chars': 'αβγδεζηθικλμνξοπρστυφχψω',
                    'min_length': 1,
                    'max_length': 1,
                    'mathematical_mapping': 'recursive_abstraction'
                },
                'complexity_level': SemanticComplexity.EXPERT
            }
        ]
        
        for pattern_data in patterns:
            pattern = CharacterPattern(
                pattern_id=uuid.uuid4(),
                character_type=pattern_data['character_type'],
                formation_rules=pattern_data['formation_rules'],
                complexity_level=pattern_data['complexity_level'],
                mathematical_foundation={'type': pattern_data['formation_rules']['mathematical_mapping']}
            )
            self.character_patterns[pattern.pattern_id] = pattern
    
    def _initialize_validation_rules(self):
        """Initialize character validation rules"""
        self.validation_rules = {
            'alphabetic': {
                'pattern': r'^[a-zA-Z]$',
                'description': 'Single alphabetic character'
            },
            'numeric': {
                'pattern': r'^[0-9]$',
                'description': 'Single numeric character'
            },
            'symbolic': {
                'pattern': r'^[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]$',
                'description': 'Single symbolic character'
            },
            'composite': {
                'pattern': r'^[a-zA-Z0-9]{1,2}$',
                'description': 'One or two alphanumeric characters'
            },
            'abstract': {
                'pattern': r'^[αβγδεζηθικλμνξοπρστυφχψω]$',
                'description': 'Single Greek character'
            }
        }
    
    def form_characters(self,
                       source_traits: List[SemanticTrait],
                       target_type: CharacterType = CharacterType.ALPHABETIC,
                       formation_method: FormationMethod = FormationMethod.AUTO,
                       complexity_level: SemanticComplexity = SemanticComplexity.BASIC) -> CharacterFormationResult:
        """
        Form characters from semantic traits using recursive patterns
        """
        with self._lock:
            formation_id = uuid.uuid4()
            
            # Auto-select formation method if needed
            if formation_method == FormationMethod.AUTO:
                formation_method = self._select_optimal_method(source_traits, target_type, complexity_level)
            
            # Create formation context
            context = CharacterFormationContext(
                formation_id=formation_id,
                source_traits=source_traits,
                target_type=target_type,
                formation_method=formation_method,
                complexity_level=complexity_level
            )
            
            # Add to queue
            self.formation_queue.append(context)
            self.active_formations[formation_id] = context
            
            # Publish event
            self.event_bridge.publish_semantic_event(
                CharacterFormationEvent(
                    event_uuid=uuid.uuid4(),
            event_type="CHARACTER_FORMATION_REQUEST",
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(formation_id),
                        'target_type': target_type.value,
                        'formation_method': formation_method.value,
                        'trait_count': len(source_traits)
                    }
                )
            )
            
            # Execute formation
            result = self._execute_formation(context)
            
            # Update metrics
            self._update_formation_metrics(result)
            
            return result
    
    def _select_optimal_method(self,
                              source_traits: List[SemanticTrait],
                              target_type: CharacterType,
                              complexity_level: SemanticComplexity) -> FormationMethod:
        """Select optimal formation method based on traits and requirements"""
        
        # Analyze source traits
        trait_count = len(source_traits)
        avg_complexity = self._calculate_average_complexity(source_traits)
        
        # Method selection logic
        if target_type == CharacterType.ALPHABETIC and trait_count == 1:
            return FormationMethod.HASH_BASED
        elif target_type == CharacterType.NUMERIC:
            return FormationMethod.PATTERN_MATCHING
        elif target_type == CharacterType.SYMBOLIC:
            return FormationMethod.RECURSIVE_TRANSFORM
        elif target_type == CharacterType.COMPOSITE and trait_count > 1:
            return FormationMethod.CONTEXT_SENSITIVE
        elif target_type == CharacterType.ABSTRACT and avg_complexity >= SemanticComplexity.ADVANCED:
            return FormationMethod.HYBRID_SYNTHESIS
        else:
            return FormationMethod.HASH_BASED
    
    def _execute_formation(self, context: CharacterFormationContext) -> CharacterFormationResult:
        """Execute the actual character formation operation"""
        start_time = datetime.utcnow()
        
        try:
            # Get formation method
            method_name = f"_form_{context.formation_method.value}"
            print(f"    🔍 Looking for method: {method_name}")
            formation_method = getattr(self, method_name)
            print(f"    ✅ Found method: {formation_method.__name__}")
            
            # Execute formation
            print(f"    🚀 Executing formation with {len(context.source_traits)} traits")
            formed_characters = formation_method(context)
            print(f"    📊 Formation returned {len(formed_characters)} characters")
            
            # Validate characters
            validated_characters = []
            formation_patterns = []
            
            print(f"        🔍 Validating {len(formed_characters)} characters...")
            
            for i, character in enumerate(formed_characters):
                print(f"          Validating character {i+1}: '{character}'")
                validation = self._validate_character(character, context.target_type)
                print(f"            → Validation result: {'✅ Valid' if validation.is_valid else '❌ Invalid'}")
                if validation.is_valid:
                    validated_characters.append(character)
                    print(f"            → Added to validated characters")
                else:
                    print(f"            → Rejected: {validation.error_details}")
                    
                    # Create formation pattern
                    pattern = FormationPattern(
                        pattern_uuid=uuid.uuid4(),
                        formation_type="character",
                        characters=[character],
                        words=[],
                        sentences=[],
                        complexity=context.complexity_level,
                        mathematical_consistency=validation.mathematical_consistency
                    )
                    formation_patterns.append(pattern)
            
            # Calculate metrics
            semantic_coherence = self._calculate_semantic_coherence(validated_characters, context.source_traits)
            mathematical_consistency = self._calculate_mathematical_consistency(formation_patterns)
            
            # Create result
            result = CharacterFormationResult(
                formation_id=context.formation_id,
                context=context,
                formed_characters=validated_characters,
                formation_patterns=formation_patterns,
                formation_metrics={
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'character_count': len(validated_characters),
                    'pattern_count': len(formation_patterns),
                    'validity_rate': len(validated_characters) / len(formed_characters) if formed_characters else 0.0
                },
                status=CharacterStatus.COMPLETED if validated_characters else CharacterStatus.FAILED,
                semantic_coherence=semantic_coherence,
                mathematical_consistency=mathematical_consistency,
                completion_time=datetime.utcnow()
            )
            
            # Store result
            self.completed_formations[context.formation_id] = result
            
            # Update memory
            self._update_character_memory(context, result)
            
            return result
            
        except Exception as e:
            # Handle formation failure
            result = CharacterFormationResult(
                formation_id=context.formation_id,
                context=context,
                formed_characters=[],
                formation_patterns=[],
                formation_metrics={'error': str(e)},
                status=CharacterStatus.FAILED,
                error_details=str(e)
            )
            
            self.completed_formations[context.formation_id] = result
            return result
    
    def _form_hash_based(self, context: CharacterFormationContext) -> List[str]:
        """Form characters using hash-based approach"""
        characters = []
        
        print(f"        🔧 Forming characters using hash-based method for {len(context.source_traits)} traits")
        
        for i, trait in enumerate(context.source_traits):
            print(f"          Processing trait {i+1}: {trait.name}")
            # Generate hash from trait properties
            trait_hash = hashlib.md5(trait.name.encode()).hexdigest()
            
            if context.target_type == CharacterType.ALPHABETIC:
                char_index = int(trait_hash[:8], 16) % 26
                character = chr(ord('a') + char_index)
            elif context.target_type == CharacterType.NUMERIC:
                char_index = int(trait_hash[:8], 16) % 10
                character = str(char_index)
            elif context.target_type == CharacterType.SYMBOLIC:
                symbols = list('!@#$%^&*()_+-=[]{}|;:,.<>?')
                char_index = int(trait_hash[:8], 16) % len(symbols)
                character = symbols[char_index]
            elif context.target_type == CharacterType.ABSTRACT:
                # Use Greek characters for abstract types
                greek_chars = list('αβγδεζηθικλμνξοπρστυφχψω')
                char_index = int(trait_hash[:8], 16) % len(greek_chars)
                character = greek_chars[char_index]
            elif context.target_type == CharacterType.COMPOSITE:
                # Create composite characters (up to 2 chars)
                char_index1 = int(trait_hash[:8], 16) % 26
                char_index2 = int(trait_hash[8:16], 16) % 10
                character = chr(ord('a') + char_index1) + str(char_index2)
            else:
                # Default to alphabetic
                char_index = int(trait_hash[:8], 16) % 26
                character = chr(ord('a') + char_index)
            
            print(f"            → Created character: '{character}'")
            characters.append(character)
        
        return characters
    
    def _form_pattern_matching(self, context: CharacterFormationContext) -> List[str]:
        """Form characters using pattern matching approach"""
        characters = []
        
        # Get appropriate pattern
        pattern = self._get_pattern_for_type(context.target_type)
        
        for trait in context.source_traits:
            character = self._apply_pattern_to_trait(trait, pattern)
            characters.append(character)
        
        return characters
    
    def _form_recursive_transform(self, context: CharacterFormationContext) -> List[str]:
        """Form characters using recursive transformation approach"""
        characters = []
        
        print(f"        🔧 Forming characters using recursive transform method for {len(context.source_traits)} traits")
        
        for i, trait in enumerate(context.source_traits):
            print(f"          Processing trait {i+1}: {trait.name}")
            try:
                # Apply recursive transformation
                character = self._apply_recursive_transformation(trait, context.target_type)
                print(f"            → Created character: '{character}'")
                characters.append(character)
            except Exception as e:
                print(f"            ❌ Error processing trait {i+1}: {e}")
                continue
        
        print(f"        📊 Recursive transform returned {len(characters)} characters")
        print(f"        🔍 Characters created: {characters}")
        return characters
    
    def _form_context_sensitive(self, context: CharacterFormationContext) -> List[str]:
        """Form characters using context-sensitive approach"""
        characters = []
        
        # Get context from memory
        context_memory = self._get_relevant_character_memory(context.source_traits)
        
        for trait in context.source_traits:
            character = self._form_with_context(trait, context_memory, context.target_type)
            characters.append(character)
        
        return characters
    
    def _form_hybrid_synthesis(self, context: CharacterFormationContext) -> List[str]:
        """Form characters using hybrid synthesis approach"""
        characters = []
        
        # Apply multiple formation methods
        methods = [FormationMethod.HASH_BASED, FormationMethod.PATTERN_MATCHING, FormationMethod.RECURSIVE_TRANSFORM]
        
        for method in methods:
            try:
                # Create sub-context
                sub_context = CharacterFormationContext(
                    formation_id=uuid.uuid4(),
                    source_traits=context.source_traits,
                    target_type=context.target_type,
                    formation_method=method,
                    complexity_level=context.complexity_level
                )
                
                # Execute sub-formation
                formation_method = getattr(self, f"_form_{method.value}")
                sub_characters = formation_method(sub_context)
                
                # Synthesize results
                characters.extend(sub_characters)
                
            except Exception as e:
                print(f"Warning: Method {method.value} failed: {e}")
                continue
        
        # Remove duplicates and optimize
        characters = self._deduplicate_characters(characters)
        
        return characters
    
    def _apply_recursive_transformation(self, trait: SemanticTrait, target_type: CharacterType) -> str:
        """Apply recursive transformation to trait"""
        # Get base character
        base_char = self._form_hash_based(CharacterFormationContext(
            formation_id=uuid.uuid4(),
            source_traits=[trait],
            target_type=target_type,
            formation_method=FormationMethod.HASH_BASED,
            complexity_level=SemanticComplexity.BASIC
        ))[0]
        
        # Apply recursive transformation
        if target_type == CharacterType.ALPHABETIC:
            # Shift character by trait complexity
            shift = trait.complexity.value
            char_code = ord(base_char.lower())
            transformed_code = (char_code - ord('a') + shift) % 26 + ord('a')
            return chr(transformed_code)
        elif target_type == CharacterType.NUMERIC:
            # Transform to numeric based on trait value
            if hasattr(trait, 'mathematical_consistency'):
                return str(int(trait.mathematical_consistency * 10) % 10)
            else:
                return base_char
        elif target_type == CharacterType.SYMBOLIC:
            # Transform to symbolic based on trait properties
            symbols = list('!@#$%^&*()_+-=[]{}|;:,.<>?')
            symbol_index = hash(trait.name) % len(symbols)
            return symbols[symbol_index]
        elif target_type == CharacterType.ABSTRACT:
            # Transform to Greek characters for abstract types
            greek_chars = list('αβγδεζηθικλμνξοπρστυφχψω')
            char_index = hash(trait.name) % len(greek_chars)
            return greek_chars[char_index]
        elif target_type == CharacterType.COMPOSITE:
            # Create composite character 
            char_index1 = hash(trait.name) % 26
            char_index2 = hash(trait.name + "2") % 10
            return chr(ord('a') + char_index1) + str(char_index2)
        else:
            return base_char
    
    def _get_pattern_for_type(self, target_type: CharacterType) -> Optional[CharacterPattern]:
        """Get pattern for specific character type"""
        for pattern in self.character_patterns.values():
            if pattern.character_type == target_type:
                return pattern
        return None
    
    def _apply_pattern_to_trait(self, trait: SemanticTrait, pattern: CharacterPattern) -> str:
        """Apply pattern to trait"""
        rules = pattern.formation_rules
        
        if pattern.character_type == CharacterType.ALPHABETIC:
            return self._apply_alphabetic_pattern(trait, rules)
        elif pattern.character_type == CharacterType.NUMERIC:
            return self._apply_numeric_pattern(trait, rules)
        elif pattern.character_type == CharacterType.SYMBOLIC:
            return self._apply_symbolic_pattern(trait, rules)
        elif pattern.character_type == CharacterType.COMPOSITE:
            return self._apply_composite_pattern(trait, rules)
        elif pattern.character_type == CharacterType.ABSTRACT:
            return self._apply_abstract_pattern(trait, rules)
        else:
            return 'a'  # Default fallback
    
    def _apply_alphabetic_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply alphabetic pattern to trait"""
        if trait.semantic_meaning:
            char = trait.semantic_meaning[0].lower()
            if char in rules['allowed_chars']:
                return char
        
        # Generate character from trait
        trait_hash = hashlib.md5(trait.trait_name.encode()).hexdigest()
        char_index = int(trait_hash[:8], 16) % 26
        return chr(ord('a') + char_index)
    
    def _apply_numeric_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply numeric pattern to trait"""
        if hasattr(trait, 'mathematical_consistency'):
            # Use mathematical consistency as base
            value = int(trait.mathematical_consistency * 10) % 10
            return str(value)
        else:
            # Generate from trait name
            trait_hash = hashlib.md5(trait.trait_name.encode()).hexdigest()
            value = int(trait_hash[:8], 16) % 10
            return str(value)
    
    def _apply_symbolic_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply symbolic pattern to trait"""
        symbols = list(rules['allowed_chars'])
        trait_hash = hashlib.md5(trait.trait_name.encode()).hexdigest()
        symbol_index = int(trait_hash[:8], 16) % len(symbols)
        return symbols[symbol_index]
    
    def _apply_composite_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply composite pattern to trait"""
        # Create composite character from trait properties
        base_chars = list(trait.trait_name.lower())
        composite = ''.join(base_chars[:rules['max_length']])
        
        # Ensure minimum length
        while len(composite) < rules['min_length']:
            composite += 'a'
        
        return composite[:rules['max_length']]
    
    def _apply_abstract_pattern(self, trait: SemanticTrait, rules: Dict[str, Any]) -> str:
        """Apply abstract pattern to trait"""
        greek_chars = list(rules['allowed_chars'])
        trait_hash = hashlib.md5(trait.trait_name.encode()).hexdigest()
        char_index = int(trait_hash[:8], 16) % len(greek_chars)
        return greek_chars[char_index]
    
    def _get_relevant_character_memory(self, traits: List[SemanticTrait]) -> List[CharacterMemory]:
        """Get relevant character formation memory"""
        relevant_memories = []
        
        for memory in self.character_memory.values():
            # Check if memory contains similar traits
            for trait in traits:
                if trait.trait_uuid in memory.trait_combination:
                    relevant_memories.append(memory)
                    break
        
        return relevant_memories
    
    def _form_with_context(self, trait: SemanticTrait, context_memory: List[CharacterMemory], target_type: CharacterType) -> str:
        """Form character with context awareness"""
        if context_memory:
            # Use context from memory
            best_memory = max(context_memory, key=lambda m: m.success_metrics.get('semantic_coherence', 0))
            
            # Adapt character from memory
            if best_memory.formation_method == FormationMethod.HASH_BASED:
                return self._form_hash_based(CharacterFormationContext(
                    formation_id=uuid.uuid4(),
                    source_traits=[trait],
                    target_type=target_type,
                    formation_method=FormationMethod.HASH_BASED,
                    complexity_level=SemanticComplexity.BASIC
                ))[0]
            else:
                return best_memory.resulting_character
        
        # Fallback to basic formation
        return self._form_hash_based(CharacterFormationContext(
            formation_id=uuid.uuid4(),
            source_traits=[trait],
            target_type=target_type,
            formation_method=FormationMethod.HASH_BASED,
            complexity_level=SemanticComplexity.BASIC
        ))[0]
    
    def _validate_character(self, character: str, target_type: CharacterType) -> CharacterValidation:
        """Validate formed character"""
        validation_rules = []
        is_valid = False
        mathematical_consistency = 0.0
        semantic_relevance = 0.0
        error_details = None
        
        try:
            # Get validation rule for target type
            rule_key = target_type.value
            if rule_key in self.validation_rules:
                rule = self.validation_rules[rule_key]
                validation_rules.append(rule['description'])
                
                # Check pattern match
                if re.match(rule['pattern'], character):
                    is_valid = True
                    mathematical_consistency = 0.9
                    semantic_relevance = 0.8
                else:
                    error_details = f"Character '{character}' does not match pattern '{rule['pattern']}' for {target_type.value}"
            else:
                # No specific rule, accept any single character
                if len(character) == 1:
                    is_valid = True
                    mathematical_consistency = 0.7
                    semantic_relevance = 0.6
                else:
                    error_details = f"Character '{character}' is not a single character"
        
        except Exception as e:
            error_details = f"Validation error: {str(e)}"
        
        return CharacterValidation(
            character=character,
            is_valid=is_valid,
            validation_rules=validation_rules,
            mathematical_consistency=mathematical_consistency,
            semantic_relevance=semantic_relevance,
            error_details=error_details
        )
    
    def _deduplicate_characters(self, characters: List[str]) -> List[str]:
        """Remove duplicate characters"""
        seen = set()
        unique_characters = []
        
        for character in characters:
            if character not in seen:
                seen.add(character)
                unique_characters.append(character)
        
        return unique_characters
    
    def _calculate_semantic_coherence(self, characters: List[str], source_traits: List[SemanticTrait]) -> float:
        """Calculate semantic coherence of formed characters"""
        if not characters or not source_traits:
            return 0.0
        
        coherence_scores = []
        for character in characters:
            # Simple coherence calculation based on character properties
            if character.isalpha():
                coherence = 0.8  # Alphabetic characters have high coherence
            elif character.isdigit():
                coherence = 0.7  # Numeric characters have medium coherence
            else:
                coherence = 0.5  # Symbolic characters have lower coherence
            
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
    
    def _update_character_memory(self, context: CharacterFormationContext, result: CharacterFormationResult):
        """Update character memory with successful formations"""
        if result.status == CharacterStatus.COMPLETED and result.semantic_coherence > 0.5:
            for character in result.formed_characters:
                memory = CharacterMemory(
                    memory_id=uuid.uuid4(),
                    trait_combination=[trait.trait_uuid for trait in context.source_traits],
                    resulting_character=character,
                    formation_method=context.formation_method,
                    success_metrics={
                        'semantic_coherence': result.semantic_coherence,
                        'mathematical_consistency': result.mathematical_consistency
                    }
                )
                
                self.character_memory[memory.memory_id] = memory
    
    def _update_formation_metrics(self, result: CharacterFormationResult):
        """Update formation performance metrics"""
        self.formation_metrics['total_formations'] += 1
        
        if result.status == CharacterStatus.COMPLETED:
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
            current_avg = self.formation_metrics['character_validity_rate']
            total_formations = self.formation_metrics['total_formations']
            new_avg = (current_avg * (total_formations - 1) + result.formation_metrics['validity_rate']) / total_formations
            self.formation_metrics['character_validity_rate'] = new_avg
    
    def _handle_formation_request(self, event_data: Dict[str, Any]):
        """Handle character formation request events"""
        try:
            source_traits = event_data.get('source_traits', [])
            target_type = CharacterType(event_data.get('target_type', 'alphabetic'))
            formation_method = FormationMethod(event_data.get('formation_method', 'auto'))
            complexity_level = SemanticComplexity(event_data.get('complexity_level', 'basic'))
            
            result = self.form_characters(source_traits, target_type, formation_method, complexity_level)
            
            # Publish completion event
            self.event_bridge.publish_semantic_event(
                CharacterFormationEvent(
                    event_uuid=uuid.uuid4(),
            event_type="CHARACTER_FORMATION_REQUEST",
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(result.formation_id),
                        'status': result.status.value,
                        'character_count': len(result.formed_characters)
                    }
                )
            )
            
        except Exception as e:
            print(f"Error handling character formation request: {e}")
    
    def _handle_conversion_complete(self, event_data: Dict[str, Any]):
        """Handle conversion completion events"""
        # Update patterns based on completed conversions
        conversion_id = event_data.get('conversion_id')
        if conversion_id:
            # Update related patterns
            pass
    
    def get_formation_status(self, formation_id: uuid.UUID) -> Optional[CharacterFormationResult]:
        """Get status of specific character formation"""
        return self.completed_formations.get(formation_id)
    
    def get_formation_metrics(self) -> Dict[str, Any]:
        """Get current formation performance metrics"""
        return self.formation_metrics.copy()
    
    def get_character_patterns(self) -> Dict[uuid.UUID, CharacterPattern]:
        """Get all character patterns"""
        return self.character_patterns.copy()
    
    def get_character_memory(self) -> Dict[uuid.UUID, CharacterMemory]:
        """Get all character memory"""
        return self.character_memory.copy()
    
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
            
            print(f"Cleared {len(old_formations)} old character formations")
