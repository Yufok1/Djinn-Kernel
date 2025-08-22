# rebuild/semantic_recursive_communication.py
"""
Semantic Recursive Communication - Dialogue composition system
Combines sentences into meaningful dialogues using recursive patterns
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
    SentenceFormationEvent, DialogueFormationEvent, CommunicationEvent
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager
from semantic_trait_conversion import SemanticTraitConverter
from semantic_recursive_sentence_formation import SemanticRecursiveSentenceFormation

class DialogueType(Enum):
    """Types of dialogues that can be formed"""
    CONVERSATION = "conversation"
    INTERROGATION = "interrogation"
    NARRATIVE = "narrative"
    TECHNICAL = "technical"
    PHILOSOPHICAL = "philosophical"
    INSTRUCTIONAL = "instructional"
    DEBATE = "debate"
    COLLABORATIVE = "collaborative"
    ABSTRACT = "abstract"
    METACOGNITIVE = "metacognitive"

class DialogueStructure(Enum):
    """Dialogue structure patterns"""
    QUESTION_ANSWER = "question_answer"
    STATEMENT_RESPONSE = "statement_response"
    NARRATIVE_FLOW = "narrative_flow"
    TECHNICAL_EXCHANGE = "technical_exchange"
    PHILOSOPHICAL_DISCUSSION = "philosophical_discussion"
    INSTRUCTION_FOLLOWUP = "instruction_followup"
    DEBATE_FORMAT = "debate_format"
    COLLABORATIVE_BUILD = "collaborative_build"

class CommunicationStrategy(Enum):
    """Strategies for dialogue formation"""
    AUTO = "auto"
    SENTENCE_COMBINATION = "sentence_combination"
    PATTERN_MATCHING = "pattern_matching"
    RECURSIVE_SYNTHESIS = "recursive_synthesis"
    CONTEXT_AWARE = "context_aware"
    HYBRID_COMPOSITION = "hybrid_composition"

class CommunicationStatus(Enum):
    """Status of dialogue formation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"

@dataclass
class CommunicationContext:
    """Context for dialogue formation operations"""
    formation_id: uuid.UUID
    source_sentences: List[str]
    source_traits: List[SemanticTrait]
    target_type: DialogueType
    dialogue_structure: DialogueStructure
    formation_strategy: CommunicationStrategy
    complexity_level: SemanticComplexity
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CommunicationResult:
    """Result of dialogue formation operation"""
    formation_id: uuid.UUID
    context: CommunicationContext
    formed_dialogues: List[str]
    formation_patterns: List[FormationPattern]
    formation_metrics: Dict[str, float]
    status: CommunicationStatus
    error_details: Optional[str] = None
    semantic_coherence: float = 0.0
    mathematical_consistency: float = 0.0
    completion_time: Optional[datetime] = None

@dataclass
class DialoguePattern:
    """Pattern for dialogue formation"""
    pattern_id: uuid.UUID
    dialogue_type: DialogueType
    structure: DialogueStructure
    formation_rules: Dict[str, Any]
    complexity_level: SemanticComplexity
    mathematical_foundation: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None

@dataclass
class DialogueMemory:
    """Memory of successful dialogue formations"""
    memory_id: uuid.UUID
    sentence_combination: List[str]
    resulting_dialogue: str
    formation_strategy: CommunicationStrategy
    dialogue_type: DialogueType
    structure: DialogueStructure
    success_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DialogueValidation:
    """Validation result for dialogue formation"""
    dialogue: str
    is_valid: bool
    validation_rules: List[str]
    mathematical_consistency: float
    semantic_relevance: float
    dialogue_type_match: Optional[DialogueType] = None
    structure_match: Optional[DialogueStructure] = None
    error_details: Optional[str] = None

class SemanticRecursiveCommunication:
    """
    Semantic recursive communication system
    Combines sentences into meaningful dialogues using recursive patterns
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 checkpoint_manager: SemanticCheckpointManager,
                 trait_converter: SemanticTraitConverter,
                 sentence_formation: SemanticRecursiveSentenceFormation,
                 uuid_anchor: UUIDanchor,
                 trait_convergence: TraitConvergenceEngine):
        
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.checkpoint_manager = checkpoint_manager
        self.trait_converter = trait_converter
        self.sentence_formation = sentence_formation
        self.uuid_anchor = uuid_anchor
        self.trait_convergence = trait_convergence
        
        # Formation state
        self.formation_queue: deque = deque()
        self.active_formations: Dict[uuid.UUID, CommunicationContext] = {}
        self.completed_formations: Dict[uuid.UUID, CommunicationResult] = {}
        self.dialogue_patterns: Dict[uuid.UUID, DialoguePattern] = {}
        self.dialogue_memory: Dict[uuid.UUID, DialogueMemory] = {}
        
        # Performance tracking
        self.formation_metrics = {
            'total_formations': 0,
            'successful_formations': 0,
            'failed_formations': 0,
            'average_formation_time': 0.0,
            'semantic_coherence_avg': 0.0,
            'mathematical_consistency_avg': 0.0,
            'dialogue_validity_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize dialogue patterns and validation rules
        self._initialize_dialogue_patterns()
        self._initialize_validation_rules()
        
        # Register with event bridge
        self.event_bridge.register_handler("DIALOGUE_FORMATION_REQUEST", self._handle_formation_request)
        self.event_bridge.register_handler("SENTENCE_FORMATION_COMPLETE", self._handle_sentence_formation_complete)
        
        print(f"💬 SemanticRecursiveCommunication initialized with {len(self.dialogue_patterns)} patterns")
    
    def _initialize_dialogue_patterns(self):
        """Initialize dialogue formation patterns"""
        patterns = [
            {
                'dialogue_type': DialogueType.CONVERSATION,
                'structure': DialogueStructure.QUESTION_ANSWER,
                'formation_rules': {
                    'min_sentences': 2,
                    'max_sentences': 10,
                    'required_elements': ['question', 'answer'],
                    'optional_elements': ['followup', 'clarification'],
                    'mathematical_mapping': 'conversation_synthesis'
                },
                'complexity_level': SemanticComplexity.BASIC
            },
            {
                'dialogue_type': DialogueType.TECHNICAL,
                'structure': DialogueStructure.TECHNICAL_EXCHANGE,
                'formation_rules': {
                    'min_sentences': 3,
                    'max_sentences': 15,
                    'required_elements': ['problem', 'solution', 'verification'],
                    'optional_elements': ['optimization', 'documentation'],
                    'mathematical_mapping': 'technical_synthesis'
                },
                'complexity_level': SemanticComplexity.ADVANCED
            },
            {
                'dialogue_type': DialogueType.PHILOSOPHICAL,
                'structure': DialogueStructure.PHILOSOPHICAL_DISCUSSION,
                'formation_rules': {
                    'min_sentences': 4,
                    'max_sentences': 20,
                    'required_elements': ['premise', 'analysis', 'conclusion'],
                    'optional_elements': ['counterpoint', 'synthesis'],
                    'mathematical_mapping': 'philosophical_synthesis'
                },
                'complexity_level': SemanticComplexity.EXPERT
            },
            {
                'dialogue_type': DialogueType.NARRATIVE,
                'structure': DialogueStructure.NARRATIVE_FLOW,
                'formation_rules': {
                    'min_sentences': 3,
                    'max_sentences': 12,
                    'required_elements': ['setup', 'development', 'resolution'],
                    'optional_elements': ['twist', 'reflection'],
                    'mathematical_mapping': 'narrative_synthesis'
                },
                'complexity_level': SemanticComplexity.INTERMEDIATE
            },
            {
                'dialogue_type': DialogueType.ABSTRACT,
                'structure': DialogueStructure.PHILOSOPHICAL_DISCUSSION,
                'formation_rules': {
                    'min_sentences': 3,
                    'max_sentences': 15,
                    'required_elements': ['concept', 'exploration', 'abstraction'],
                    'optional_elements': ['metaphor', 'generalization'],
                    'mathematical_mapping': 'abstract_synthesis'
                },
                'complexity_level': SemanticComplexity.EXPERT
            }
        ]
        
        for pattern_data in patterns:
            pattern = DialoguePattern(
                pattern_id=uuid.uuid4(),
                dialogue_type=pattern_data['dialogue_type'],
                structure=pattern_data['structure'],
                formation_rules=pattern_data['formation_rules'],
                complexity_level=pattern_data['complexity_level'],
                mathematical_foundation={'type': pattern_data['formation_rules']['mathematical_mapping']}
            )
            self.dialogue_patterns[pattern.pattern_id] = pattern
    
    def _initialize_validation_rules(self):
        """Initialize dialogue validation rules"""
        self.validation_rules = {
            'conversation': {
                'pattern': r'^[A-Z][^.!?]*[.!?]\s+[A-Z][^.!?]*[.!?]',
                'description': 'Conversation: Multiple sentences with proper punctuation',
                'sentence_count_check': True,
                'min_sentences': 2,
                'max_sentences': 10
            },
            'technical': {
                'pattern': r'^[A-Z][^.!?]*[.!?](\s+[A-Z][^.!?]*[.!?])+',
                'description': 'Technical: Structured technical exchange',
                'sentence_count_check': True,
                'min_sentences': 3,
                'max_sentences': 15
            },
            'philosophical': {
                'pattern': r'^[A-Z][^.!?]*[.!?](\s+[A-Z][^.!?]*[.!?])+',
                'description': 'Philosophical: Deep conceptual discussion',
                'sentence_count_check': True,
                'min_sentences': 4,
                'max_sentences': 20
            },
            'narrative': {
                'pattern': r'^[A-Z][^.!?]*[.!?](\s+[A-Z][^.!?]*[.!?])+',
                'description': 'Narrative: Story-like flow',
                'sentence_count_check': True,
                'min_sentences': 3,
                'max_sentences': 12
            },
            'abstract': {
                'pattern': r'^[A-Z][^.!?]*[.!?](\s+[A-Z][^.!?]*[.!?])+',
                'description': 'Abstract: Conceptual abstraction',
                'sentence_count_check': True,
                'min_sentences': 3,
                'max_sentences': 15
            }
        }
    
    def form_dialogues(self,
                      source_sentences: List[str],
                      source_traits: List[SemanticTrait],
                      target_type: DialogueType = DialogueType.CONVERSATION,
                      dialogue_structure: DialogueStructure = DialogueStructure.QUESTION_ANSWER,
                      formation_strategy: CommunicationStrategy = CommunicationStrategy.AUTO,
                      complexity_level: SemanticComplexity = SemanticComplexity.BASIC) -> CommunicationResult:
        """
        Form dialogues from sentences using recursive patterns
        """
        with self._lock:
            formation_id = uuid.uuid4()
            
            # Auto-select formation strategy if needed
            if formation_strategy == CommunicationStrategy.AUTO:
                formation_strategy = self._select_optimal_strategy(source_sentences, source_traits, target_type, complexity_level)
            
            # Create formation context
            context = CommunicationContext(
                formation_id=formation_id,
                source_sentences=source_sentences,
                source_traits=source_traits,
                target_type=target_type,
                dialogue_structure=dialogue_structure,
                formation_strategy=formation_strategy,
                complexity_level=complexity_level
            )
            
            # Add to queue
            self.formation_queue.append(context)
            self.active_formations[formation_id] = context
            
            # Publish event
            self.event_bridge.publish_semantic_event(
                DialogueFormationEvent(
                    event_uuid=uuid.uuid4(),
            event_type="DIALOGUE_FORMATION_REQUEST",
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(formation_id),
                        'target_type': target_type.value,
                        'dialogue_structure': dialogue_structure.value,
                        'formation_strategy': formation_strategy.value,
                        'sentence_count': len(source_sentences),
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
                                source_sentences: List[str],
                                source_traits: List[SemanticTrait],
                                target_type: DialogueType,
                                complexity_level: SemanticComplexity) -> CommunicationStrategy:
        """Select optimal formation strategy based on sentences and requirements"""
        
        # Analyze source sentences and traits
        sentence_count = len(source_sentences)
        trait_count = len(source_traits)
        avg_complexity = self._calculate_average_complexity(source_traits)
        
        # Strategy selection logic
        if sentence_count <= 3 and trait_count == 1:
            return CommunicationStrategy.SENTENCE_COMBINATION
        elif target_type in [DialogueType.CONVERSATION, DialogueType.NARRATIVE] and sentence_count <= 5:
            return CommunicationStrategy.PATTERN_MATCHING
        elif target_type in [DialogueType.PHILOSOPHICAL, DialogueType.ABSTRACT] and avg_complexity >= SemanticComplexity.ADVANCED:
            return CommunicationStrategy.RECURSIVE_SYNTHESIS
        elif trait_count > 1 and sentence_count > 3:
            return CommunicationStrategy.CONTEXT_AWARE
        elif complexity_level >= SemanticComplexity.ADVANCED:
            return CommunicationStrategy.HYBRID_COMPOSITION
        else:
            return CommunicationStrategy.SENTENCE_COMBINATION
    
    def _execute_formation(self, context: CommunicationContext) -> CommunicationResult:
        """Execute the actual dialogue formation operation"""
        start_time = datetime.utcnow()
        
        try:
            # Get formation method
            method_name = f"_form_{context.formation_strategy.value}"
            print(f"    🔍 Looking for dialogue formation method: {method_name}")
            formation_method = getattr(self, method_name)
            print(f"    ✅ Found dialogue formation method: {formation_method.__name__}")
            
            # Execute formation
            print(f"    🚀 Executing dialogue formation with {len(context.source_sentences)} sentences")
            formed_dialogues = formation_method(context)
            print(f"    📊 Dialogue formation returned {len(formed_dialogues)} dialogues")
            
            # Validate dialogues
            validated_dialogues = []
            formation_patterns = []
            
            print(f"        🔍 Validating {len(formed_dialogues)} dialogues...")
            
            for i, dialogue in enumerate(formed_dialogues):
                print(f"          Validating dialogue {i+1}: '{dialogue}'")
                validation = self._validate_dialogue(dialogue, context.target_type)
                print(f"            → Validation result: {'✅ Valid' if validation.is_valid else '❌ Invalid'}")
                if validation.is_valid:
                    validated_dialogues.append(dialogue)
                    print(f"            → Added to validated dialogues")
                else:
                    print(f"            → Rejected: {validation.error_details}")
                    
                    # Create formation pattern
                    pattern = FormationPattern(
                        pattern_uuid=uuid.uuid4(),
                        formation_type="dialogue",
                        characters=list(dialogue),
                        words=dialogue.split(),
                        sentences=context.source_sentences,
                        complexity=context.complexity_level,
                        mathematical_consistency=validation.mathematical_consistency
                    )
                    formation_patterns.append(pattern)
            
            # Calculate metrics
            semantic_coherence = self._calculate_semantic_coherence(validated_dialogues, context.source_traits)
            mathematical_consistency = self._calculate_mathematical_consistency(formation_patterns)
            
            # Create result
            result = CommunicationResult(
                formation_id=context.formation_id,
                context=context,
                formed_dialogues=validated_dialogues,
                formation_patterns=formation_patterns,
                formation_metrics={
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'dialogue_count': len(validated_dialogues),
                    'pattern_count': len(formation_patterns),
                    'validity_rate': len(validated_dialogues) / len(formed_dialogues) if formed_dialogues else 0.0
                },
                status=CommunicationStatus.COMPLETED if validated_dialogues else CommunicationStatus.FAILED,
                semantic_coherence=semantic_coherence,
                mathematical_consistency=mathematical_consistency,
                completion_time=datetime.utcnow()
            )
            
            # Store result
            self.completed_formations[context.formation_id] = result
            
            # Update memory
            self._update_dialogue_memory(context, result)
            
            return result
            
        except Exception as e:
            # Handle formation failure
            result = CommunicationResult(
                formation_id=context.formation_id,
                context=context,
                formed_dialogues=[],
                formation_patterns=[],
                formation_metrics={'error': str(e)},
                status=CommunicationStatus.FAILED,
                error_details=str(e)
            )
            
            self.completed_formations[context.formation_id] = result
            return result
    
    def _form_sentence_combination(self, context: CommunicationContext) -> List[str]:
        """Form dialogues using sentence combination approach"""
        dialogues = []
        
        # Get pattern for target dialogue type
        pattern = self._get_pattern_for_type(context.target_type)
        
        # Combine sentences into dialogues
        if len(context.source_sentences) >= pattern.formation_rules['min_sentences']:
            # Direct combination
            dialogue = ' '.join(context.source_sentences[:pattern.formation_rules['max_sentences']])
            
            # Ensure minimum sentence count
            while len(dialogue.split('.')) < pattern.formation_rules['min_sentences']:
                dialogue += ' ' + context.source_sentences[0] if context.source_sentences else ' Sentence.'
            
            # Apply dialogue structure
            dialogue = self._apply_dialogue_structure(dialogue, context.dialogue_structure)
            
            dialogues.append(dialogue)
        
        return dialogues
    
    def _form_pattern_matching(self, context: CommunicationContext) -> List[str]:
        """Form dialogues using pattern matching approach"""
        dialogues = []
        
        print(f"        🔧 Forming dialogues using pattern matching method for {len(context.source_sentences)} sentences")
        
        # Get appropriate pattern
        pattern = self._get_pattern_for_type(context.target_type)
        print(f"          Using pattern for {context.target_type.value}: {pattern.pattern_id if pattern else 'None'}")
        
        # Apply pattern-based formation
        dialogue = self._apply_pattern_to_sentences(context.source_sentences, pattern, context.dialogue_structure)
        print(f"          → Created dialogue: '{dialogue}'")
        if dialogue:
            dialogues.append(dialogue)
            print(f"          → Added dialogue to results")
        else:
            print(f"          → No dialogue created")
        
        return dialogues
    
    def _form_recursive_synthesis(self, context: CommunicationContext) -> List[str]:
        """Form dialogues using recursive synthesis approach"""
        dialogues = []
        
        # Apply recursive dialogue formation
        dialogue = self._apply_recursive_synthesis_to_sentences(context.source_sentences, context.target_type, context.dialogue_structure)
        if dialogue:
            dialogues.append(dialogue)
        
        return dialogues
    
    def _form_context_aware(self, context: CommunicationContext) -> List[str]:
        """Form dialogues using context-aware approach"""
        dialogues = []
        
        # Get context from memory
        context_memory = self._get_relevant_dialogue_memory(context.source_sentences, context.source_traits)
        
        # Form dialogues with context awareness
        dialogue = self._form_with_context(context.source_sentences, context_memory, context.target_type, context.dialogue_structure)
        if dialogue:
            dialogues.append(dialogue)
        
        return dialogues
    
    def _form_hybrid_composition(self, context: CommunicationContext) -> List[str]:
        """Form dialogues using hybrid composition approach"""
        dialogues = []
        
        # Apply multiple formation strategies
        strategies = [CommunicationStrategy.SENTENCE_COMBINATION, CommunicationStrategy.PATTERN_MATCHING, CommunicationStrategy.RECURSIVE_SYNTHESIS]
        
        for strategy in strategies:
            try:
                # Create sub-context
                sub_context = CommunicationContext(
                    formation_id=uuid.uuid4(),
                    source_sentences=context.source_sentences,
                    source_traits=context.source_traits,
                    target_type=context.target_type,
                    dialogue_structure=context.dialogue_structure,
                    formation_strategy=strategy,
                    complexity_level=context.complexity_level
                )
                
                # Execute sub-formation
                formation_method = getattr(self, f"_form_{strategy.value}")
                sub_dialogues = formation_method(sub_context)
                
                # Synthesize results
                dialogues.extend(sub_dialogues)
                
            except Exception as e:
                print(f"Warning: Strategy {strategy.value} failed: {e}")
                continue
        
        # Remove duplicates and optimize
        dialogues = self._deduplicate_dialogues(dialogues)
        
        return dialogues
    
    def _apply_pattern_to_sentences(self, sentences: List[str], pattern: DialoguePattern, structure: DialogueStructure) -> Optional[str]:
        """Apply pattern to sentences for dialogue formation"""
        rules = pattern.formation_rules
        
        # Basic sentence combination
        base_dialogue = ' '.join(sentences[:rules['max_sentences']])
        
        # Ensure minimum sentence count
        while len(base_dialogue.split('.')) < rules['min_sentences']:
            base_dialogue += ' ' + sentences[0] if sentences else ' Sentence.'
        
        # Apply dialogue structure
        dialogue = self._apply_dialogue_structure(base_dialogue, structure)
        
        return dialogue
    
    def _apply_recursive_synthesis_to_sentences(self, sentences: List[str], target_type: DialogueType, structure: DialogueStructure) -> Optional[str]:
        """Apply recursive synthesis to sentences for dialogue formation"""
        # Start with base sentence combination
        base_dialogue = ' '.join(sentences[:8])  # Limit to 8 sentences for synthesis
        
        # Apply recursive transformations based on dialogue type
        if target_type == DialogueType.PHILOSOPHICAL:
            # Create philosophical dialogue through recursive abstraction
            dialogue = self._create_philosophical_dialogue(base_dialogue)
        elif target_type == DialogueType.ABSTRACT:
            # Create abstract dialogue through recursive abstraction
            dialogue = self._create_abstract_dialogue(base_dialogue)
        elif target_type == DialogueType.TECHNICAL:
            # Create technical dialogue through recursive synthesis
            dialogue = self._create_technical_dialogue(base_dialogue)
        else:
            # Apply standard recursive transformation
            dialogue = self._apply_recursive_transformation(base_dialogue, target_type, structure)
        
        return dialogue
    
    def _create_philosophical_dialogue(self, base_dialogue: str) -> str:
        """Create philosophical dialogue through pure mathematical recursion"""
        # Generate multiple sentences through mathematical transformation
        dialogue_hash = hash(base_dialogue)
        sentences = [base_dialogue]
        
        # Generate 3 additional sentences through hash-based recursion
        for i in range(3):
            # Create new hash from existing content + iteration
            iteration_hash = hash(base_dialogue + str(i) + str(dialogue_hash))
            
            # Generate new sentence through mathematical word synthesis
            sentence_length = (iteration_hash % 8) + 4  # 4-11 words
            words = []
            
            for word_pos in range(sentence_length):
                word_hash = hash(str(iteration_hash) + str(word_pos))
                # Generate word through mathematical character synthesis
                word_length = (word_hash % 6) + 3  # 3-8 characters
                chars = []
                
                for char_pos in range(word_length):
                    char_hash = hash(str(word_hash) + str(char_pos))
                    # Use mathematical transformation to generate characters
                    char_index = char_hash % 26
                    char = chr(ord('a') + char_index)
                    chars.append(char)
                
                words.append(''.join(chars))
            
            # Capitalize first word and add period
            words[0] = words[0].capitalize()
            new_sentence = ' '.join(words) + '.'
            sentences.append(new_sentence)
        
        return ' '.join(sentences)
    
    def _create_abstract_dialogue(self, base_dialogue: str) -> str:
        """Create abstract dialogue through pure mathematical recursion"""
        return self._create_philosophical_dialogue(base_dialogue)  # Use same mathematical approach
    
    def _create_technical_dialogue(self, base_dialogue: str) -> str:
        """Create technical dialogue through pure mathematical recursion"""
        return self._create_philosophical_dialogue(base_dialogue)  # Use same mathematical approach
    
    def _apply_recursive_transformation(self, base_dialogue: str, target_type: DialogueType, structure: DialogueStructure) -> str:
        """Apply recursive transformation to base dialogue"""
        # Apply transformations based on dialogue type
        if target_type == DialogueType.CONVERSATION:
            # Transform to conversation format
            return f"Q: {base_dialogue} A: Response."
        elif target_type == DialogueType.NARRATIVE:
            # Transform to narrative format
            return f"Once, {base_dialogue}"
        else:
            # Default dialogue format
            return base_dialogue
    
    def _apply_dialogue_structure(self, dialogue: str, structure: DialogueStructure) -> str:
        """Apply dialogue structure to dialogue"""
        sentences = dialogue.split('.')
        
        if structure == DialogueStructure.QUESTION_ANSWER:
            # Question-Answer structure
            if len(sentences) >= 2:
                return f"Q: {sentences[0]}. A: {sentences[1]}."
            else:
                return dialogue
        elif structure == DialogueStructure.STATEMENT_RESPONSE:
            # Statement-Response structure
            if len(sentences) >= 2:
                return f"Statement: {sentences[0]}. Response: {sentences[1]}."
            else:
                return dialogue
        elif structure == DialogueStructure.NARRATIVE_FLOW:
            # Narrative flow structure
            return f"Narrative: {dialogue}"
        elif structure == DialogueStructure.TECHNICAL_EXCHANGE:
            # Technical exchange structure
            return f"Technical: {dialogue}"
        elif structure == DialogueStructure.PHILOSOPHICAL_DISCUSSION:
            # Philosophical discussion structure
            return f"Philosophical: {dialogue}"
        else:
            return dialogue
    
    def _get_pattern_for_type(self, target_type: DialogueType) -> Optional[DialoguePattern]:
        """Get pattern for specific dialogue type"""
        for pattern in self.dialogue_patterns.values():
            if pattern.dialogue_type == target_type:
                return pattern
        return None
    
    def _get_relevant_dialogue_memory(self, sentences: List[str], traits: List[SemanticTrait]) -> List[DialogueMemory]:
        """Get relevant dialogue formation memory"""
        relevant_memories = []
        
        for memory in self.dialogue_memory.values():
            # Check if memory contains similar sentences or traits
            sentence_match = any(sentence in memory.sentence_combination for sentence in sentences)
            trait_match = any(trait.trait_uuid in [t.trait_uuid for t in traits] for trait in traits)
            
            if sentence_match or trait_match:
                relevant_memories.append(memory)
        
        return relevant_memories
    
    def _form_with_context(self, sentences: List[str], context_memory: List[DialogueMemory], target_type: DialogueType, structure: DialogueStructure) -> Optional[str]:
        """Form dialogue with context awareness"""
        if context_memory:
            # Use context from memory
            best_memory = max(context_memory, key=lambda m: m.success_metrics.get('semantic_coherence', 0))
            
            # Adapt dialogue from memory
            if best_memory.formation_strategy == CommunicationStrategy.SENTENCE_COMBINATION:
                return self._form_sentence_combination(CommunicationContext(
                    formation_id=uuid.uuid4(),
                    source_sentences=sentences,
                    source_traits=[],
                    target_type=target_type,
                    dialogue_structure=structure,
                    formation_strategy=CommunicationStrategy.SENTENCE_COMBINATION,
                    complexity_level=SemanticComplexity.BASIC
                ))[0] if sentences else None
            else:
                return best_memory.resulting_dialogue
        
        # Fallback to basic formation
        return self._form_sentence_combination(CommunicationContext(
            formation_id=uuid.uuid4(),
            source_sentences=sentences,
            source_traits=[],
            target_type=target_type,
            dialogue_structure=structure,
            formation_strategy=CommunicationStrategy.SENTENCE_COMBINATION,
            complexity_level=SemanticComplexity.BASIC
        ))[0] if sentences else None
    
    def _validate_dialogue(self, dialogue: str, target_type: DialogueType) -> DialogueValidation:
        """Validate formed dialogue"""
        validation_rules = []
        is_valid = False
        mathematical_consistency = 0.0
        semantic_relevance = 0.0
        dialogue_type_match = None
        structure_match = None
        error_details = None
        
        try:
            # Get validation rule for target type
            rule_key = target_type.value
            if rule_key in self.validation_rules:
                rule = self.validation_rules[rule_key]
                validation_rules.append(rule['description'])
                
                # Check pattern match
                if re.match(rule['pattern'], dialogue):
                    is_valid = True
                    mathematical_consistency = 0.9
                    semantic_relevance = 0.8
                    dialogue_type_match = target_type
                    
                    # Check sentence count requirement
                    if rule.get('sentence_count_check', False):
                        sentence_count = len(dialogue.split('.'))
                        min_sentences = rule.get('min_sentences', 1)
                        max_sentences = rule.get('max_sentences', 50)
                        
                        if not (min_sentences <= sentence_count <= max_sentences):
                            is_valid = False
                            error_details = f"Dialogue has {sentence_count} sentences, expected {min_sentences}-{max_sentences}"
                else:
                    error_details = f"Dialogue '{dialogue}' does not match pattern for {target_type.value}"
            else:
                # No specific rule, accept any dialogue with proper structure
                if dialogue and len(dialogue.split('.')) >= 2:
                    is_valid = True
                    mathematical_consistency = 0.7
                    semantic_relevance = 0.6
                else:
                    error_details = f"Dialogue '{dialogue}' is not a valid dialogue"
        
        except Exception as e:
            error_details = f"Validation error: {str(e)}"
        
        return DialogueValidation(
            dialogue=dialogue,
            is_valid=is_valid,
            validation_rules=validation_rules,
            mathematical_consistency=mathematical_consistency,
            semantic_relevance=semantic_relevance,
            dialogue_type_match=dialogue_type_match,
            structure_match=structure_match,
            error_details=error_details
        )
    
    def _deduplicate_dialogues(self, dialogues: List[str]) -> List[str]:
        """Remove duplicate dialogues"""
        seen = set()
        unique_dialogues = []
        
        for dialogue in dialogues:
            if dialogue not in seen:
                seen.add(dialogue)
                unique_dialogues.append(dialogue)
        
        return unique_dialogues
    
    def _calculate_semantic_coherence(self, dialogues: List[str], source_traits: List[SemanticTrait]) -> float:
        """Calculate semantic coherence of formed dialogues"""
        if not dialogues or not source_traits:
            return 0.0
        
        coherence_scores = []
        for dialogue in dialogues:
            # Simple coherence calculation based on dialogue properties
            sentences = dialogue.split('.')
            if len(sentences) >= 2:
                # Longer dialogues tend to be more coherent
                coherence = min(1.0, len(sentences) / 8.0)
                
                # Bonus for proper dialogue structure
                if dialogue[0].isupper() and dialogue.endswith(('.', '!', '?')):
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
    
    def _update_dialogue_memory(self, context: CommunicationContext, result: CommunicationResult):
        """Update dialogue memory with successful formations"""
        if result.status == CommunicationStatus.COMPLETED and result.semantic_coherence > 0.5:
            for dialogue in result.formed_dialogues:
                memory = DialogueMemory(
                    memory_id=uuid.uuid4(),
                    sentence_combination=context.source_sentences,
                    resulting_dialogue=dialogue,
                    formation_strategy=context.formation_strategy,
                    dialogue_type=context.target_type,
                    structure=context.dialogue_structure,
                    success_metrics={
                        'semantic_coherence': result.semantic_coherence,
                        'mathematical_consistency': result.mathematical_consistency
                    }
                )
                
                self.dialogue_memory[memory.memory_id] = memory
    
    def _update_formation_metrics(self, result: CommunicationResult):
        """Update formation performance metrics"""
        self.formation_metrics['total_formations'] += 1
        
        if result.status == CommunicationStatus.COMPLETED:
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
            current_avg = self.formation_metrics['dialogue_validity_rate']
            total_formations = self.formation_metrics['total_formations']
            new_avg = (current_avg * (total_formations - 1) + result.formation_metrics['validity_rate']) / total_formations
            self.formation_metrics['dialogue_validity_rate'] = new_avg
    
    def _handle_formation_request(self, event_data: Dict[str, Any]):
        """Handle dialogue formation request events"""
        try:
            source_sentences = event_data.get('source_sentences', [])
            source_traits = event_data.get('source_traits', [])
            target_type = DialogueType(event_data.get('target_type', 'conversation'))
            dialogue_structure = DialogueStructure(event_data.get('dialogue_structure', 'question_answer'))
            formation_strategy = CommunicationStrategy(event_data.get('formation_strategy', 'auto'))
            complexity_level = SemanticComplexity(event_data.get('complexity_level', 'basic'))
            
            result = self.form_dialogues(source_sentences, source_traits, target_type, dialogue_structure, formation_strategy, complexity_level)
            
            # Publish completion event
            self.event_bridge.publish_semantic_event(
                DialogueFormationEvent(
                    event_uuid=uuid.uuid4(),
            event_type="DIALOGUE_FORMATION_REQUEST",
                    timestamp=datetime.utcnow(),
                    payload={
                        'formation_id': str(result.formation_id),
                        'status': result.status.value,
                        'dialogue_count': len(result.formed_dialogues)
                    }
                )
            )
            
        except Exception as e:
            print(f"Error handling dialogue formation request: {e}")
    
    def _handle_sentence_formation_complete(self, event_data: Dict[str, Any]):
        """Handle sentence formation completion events"""
        # Update patterns based on completed sentence formations
        formation_id = event_data.get('formation_id')
        if formation_id:
            # Update related patterns
            pass
    
    def get_formation_status(self, formation_id: uuid.UUID) -> Optional[CommunicationResult]:
        """Get status of specific dialogue formation"""
        return self.completed_formations.get(formation_id)
    
    def get_formation_metrics(self) -> Dict[str, Any]:
        """Get current formation performance metrics"""
        return self.formation_metrics.copy()
    
    def get_dialogue_patterns(self) -> Dict[uuid.UUID, DialoguePattern]:
        """Get all dialogue patterns"""
        return self.dialogue_patterns.copy()
    
    def get_dialogue_memory(self) -> Dict[uuid.UUID, DialogueMemory]:
        """Get all dialogue memory"""
        return self.dialogue_memory.copy()
    
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
            
            print(f"Cleared {len(old_formations)} old dialogue formations")
