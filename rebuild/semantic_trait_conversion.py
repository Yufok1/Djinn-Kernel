# rebuild/semantic_trait_conversion.py
"""
Semantic Trait Conversion - Core semantic understanding bridge
Converts between mathematical traits and semantic understanding
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

# Import kernel dependencies
from uuid_anchor_mechanism import UUIDanchor
from event_driven_coordination import DjinnEventBus
from violation_pressure_calculation import ViolationMonitor
from trait_convergence_engine import TraitConvergenceEngine

# Import semantic components
from semantic_data_structures import (
    SemanticTrait, MathematicalTrait, TraitConversion,
    ConversionType, SemanticComplexity, TraitCategory,
    FormationPattern, FormationType, SemanticViolation, SafetyNet,
    CharacterFormationEvent, WordFormationEvent, SentenceFormationEvent
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager

class ConversionDirection(Enum):
    """Direction of trait conversion"""
    MATH_TO_SEMANTIC = "math_to_semantic"
    SEMANTIC_TO_MATH = "semantic_to_math"
    BIDIRECTIONAL = "bidirectional"

class ConversionStatus(Enum):
    """Status of conversion operation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class ConversionContext:
    """Context for trait conversion operations"""
    conversion_id: uuid.UUID
    direction: ConversionDirection
    source_traits: List[Union[MathematicalTrait, SemanticTrait]]
    target_complexity: SemanticComplexity
    conversion_strategy: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConversionResult:
    """Result of trait conversion operation"""
    conversion_id: uuid.UUID
    context: ConversionContext
    converted_traits: List[Union[MathematicalTrait, SemanticTrait]]
    conversion_metrics: Dict[str, float]
    status: ConversionStatus
    error_details: Optional[str] = None
    semantic_coherence: float = 0.0
    mathematical_consistency: float = 0.0
    completion_time: Optional[datetime] = None

@dataclass
class TraitMapping:
    """Mapping between mathematical and semantic traits"""
    mapping_id: uuid.UUID
    mathematical_trait: MathematicalTrait
    semantic_trait: SemanticTrait
    confidence: float
    bidirectional: bool
    conversion_history: List[uuid.UUID] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

class SemanticTraitConverter:
    """
    Core semantic trait conversion system
    Bridges mathematical traits with semantic understanding
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 checkpoint_manager: SemanticCheckpointManager,
                 uuid_anchor: UUIDanchor,
                 trait_convergence: TraitConvergenceEngine):
        
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.checkpoint_manager = checkpoint_manager
        self.uuid_anchor = uuid_anchor
        self.trait_convergence = trait_convergence
        
        # Conversion state
        self.conversion_queue: deque = deque()
        self.active_conversions: Dict[uuid.UUID, ConversionContext] = {}
        self.completed_conversions: Dict[uuid.UUID, ConversionResult] = {}
        self.trait_mappings: Dict[uuid.UUID, TraitMapping] = {}
        
        # Performance tracking
        self.conversion_metrics = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'average_conversion_time': 0.0,
            'semantic_coherence_avg': 0.0,
            'mathematical_consistency_avg': 0.0,
            'success_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize conversion strategies
        self._initialize_conversion_strategies()
        
        # Register with event bridge
        self.event_bridge.register_handler("TRAIT_CONVERSION_REQUEST", self._handle_conversion_request)
        self.event_bridge.register_handler("SEMANTIC_FORMATION_COMPLETE", self._handle_formation_complete)
        
        print(f"🔄 SemanticTraitConverter initialized with {len(self.conversion_strategies)} strategies")
    
    def _initialize_conversion_strategies(self):
        """Initialize available conversion strategies"""
        self.conversion_strategies = {
            'character_based': {
                'description': 'Convert traits to character formations',
                'complexity_range': [SemanticComplexity.BASIC, SemanticComplexity.INTERMEDIATE],
                'bidirectional': True
            },
            'pattern_matching': {
                'description': 'Match mathematical patterns to semantic patterns',
                'complexity_range': [SemanticComplexity.INTERMEDIATE, SemanticComplexity.ADVANCED],
                'bidirectional': True
            },
            'recursive_formation': {
                'description': 'Use recursive typewriter approach',
                'complexity_range': [SemanticComplexity.ADVANCED, SemanticComplexity.EXPERT],
                'bidirectional': False
            },
            'hybrid_synthesis': {
                'description': 'Combine multiple conversion approaches',
                'complexity_range': [SemanticComplexity.INTERMEDIATE, SemanticComplexity.EXPERT],
                'bidirectional': True
            }
        }
    
    def convert_traits(self,
                      source_traits: List[Union[MathematicalTrait, SemanticTrait]],
                      direction: ConversionDirection,
                      target_complexity: SemanticComplexity,
                      strategy: str = 'auto') -> ConversionResult:
        """
        Convert traits between mathematical and semantic representations
        """
        with self._lock:
            conversion_id = uuid.uuid4()
            
            # Auto-select strategy if needed
            if strategy == 'auto':
                strategy = self._select_optimal_strategy(source_traits, direction, target_complexity)
            
            # Create conversion context
            context = ConversionContext(
                conversion_id=conversion_id,
                direction=direction,
                source_traits=source_traits,
                target_complexity=target_complexity,
                conversion_strategy=strategy
            )
            
            # Add to queue
            self.conversion_queue.append(context)
            self.active_conversions[conversion_id] = context
            
            # Publish event
            self.event_bridge.publish_semantic_event(
                CharacterFormationEvent(
                    event_uuid=uuid.uuid4(),
                    event_type="TRAIT_CONVERSION_STARTED",
                    timestamp=datetime.utcnow(),
                    payload={
                        'conversion_id': str(conversion_id),
                        'direction': direction.value,
                        'strategy': strategy,
                        'trait_count': len(source_traits)
                    }
                )
            )
            
            # Execute conversion
            print(f"🚀 Executing conversion with strategy: {strategy}")
            result = self._execute_conversion(context)
            print(f"📊 Conversion result: {len(result.converted_traits)} traits converted")
            
            # Update metrics
            self._update_conversion_metrics(result)
            
            return result
    
    def _select_optimal_strategy(self,
                                source_traits: List[Union[MathematicalTrait, SemanticTrait]],
                                direction: ConversionDirection,
                                target_complexity: SemanticComplexity) -> str:
        """Select optimal conversion strategy based on traits and requirements"""
        
        # Analyze source traits
        trait_count = len(source_traits)
        avg_complexity = self._calculate_average_complexity(source_traits)
        
        # Strategy selection logic
        if trait_count <= 3 and avg_complexity <= SemanticComplexity.BASIC:
            return 'character_based'
        elif trait_count <= 10 and avg_complexity <= SemanticComplexity.INTERMEDIATE:
            return 'pattern_matching'
        elif direction == ConversionDirection.MATH_TO_SEMANTIC:
            return 'recursive_formation'
        else:
            return 'hybrid_synthesis'
    
    def _execute_conversion(self, context: ConversionContext) -> ConversionResult:
        """Execute the actual conversion operation"""
        start_time = datetime.utcnow()
        
        try:
            print(f"    🔍 Starting _execute_conversion for {len(context.source_traits)} traits")
            
            # Get conversion method
            conversion_method = getattr(self, f"_convert_{context.conversion_strategy}")
            print(f"    📋 Got conversion method: {conversion_method.__name__}")
            
            # Execute conversion
            converted_traits = conversion_method(context)
            print(f"    📊 Conversion method returned {len(converted_traits)} traits")
            
            # Calculate metrics
            print(f"    📈 Calculating semantic coherence...")
            semantic_coherence = self._calculate_semantic_coherence(converted_traits)
            print(f"    📈 Calculating mathematical consistency...")
            mathematical_consistency = self._calculate_mathematical_consistency(converted_traits)
            
            # Create result
            print(f"    🏗️ Creating ConversionResult with {len(converted_traits)} traits")
            result = ConversionResult(
                conversion_id=context.conversion_id,
                context=context,
                converted_traits=converted_traits,
                conversion_metrics={
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'trait_count': len(converted_traits),
                    'complexity_achieved': self._calculate_average_complexity(converted_traits).value
                },
                status=ConversionStatus.COMPLETED,
                semantic_coherence=semantic_coherence,
                mathematical_consistency=mathematical_consistency,
                completion_time=datetime.utcnow()
            )
            print(f"    ✅ Created ConversionResult with {len(result.converted_traits)} traits")
            
            # Store result
            self.completed_conversions[context.conversion_id] = result
            
            # Update mappings
            self._update_trait_mappings(context, converted_traits)
            
            return result
            
        except Exception as e:
            print(f"    ❌ Exception in _execute_conversion: {e}")
            import traceback
            traceback.print_exc()
            
            # Handle conversion failure
            result = ConversionResult(
                conversion_id=context.conversion_id,
                context=context,
                converted_traits=[],
                conversion_metrics={'error': str(e)},
                status=ConversionStatus.FAILED,
                error_details=str(e)
            )
            
            self.completed_conversions[context.conversion_id] = result
            return result
    
    def _convert_character_based(self, context: ConversionContext) -> List[Union[MathematicalTrait, SemanticTrait]]:
        """Convert traits using character-based approach"""
        converted_traits = []
        
        print(f"🔍 Converting {len(context.source_traits)} traits using character-based method")
        
        for i, trait in enumerate(context.source_traits):
            print(f"  Processing trait {i+1}: {trait.name} ({trait.category.value})")
            
            if context.direction == ConversionDirection.MATH_TO_SEMANTIC:
                # Convert mathematical trait to semantic character
                semantic_trait = self._math_to_character(trait)
                converted_traits.append(semantic_trait)
                print(f"    → Added semantic trait to results: {semantic_trait.name}")
            else:
                # Convert semantic trait to mathematical representation
                math_trait = self._character_to_math(trait)
                converted_traits.append(math_trait)
                print(f"    → Created math trait: {math_trait.name}")
        
        print(f"✅ Character-based conversion complete: {len(converted_traits)} traits")
        return converted_traits
    
    def _convert_pattern_matching(self, context: ConversionContext) -> List[Union[MathematicalTrait, SemanticTrait]]:
        """Convert traits using pattern matching approach"""
        converted_traits = []
        
        print(f"🔍 Converting {len(context.source_traits)} traits using pattern matching method")
        
        # Group traits by patterns
        pattern_groups = self._group_traits_by_patterns(context.source_traits)
        
        for pattern, traits in pattern_groups.items():
            print(f"  Processing pattern: {pattern} with {len(traits)} traits")
            
            if context.direction == ConversionDirection.MATH_TO_SEMANTIC:
                semantic_pattern = self._match_math_pattern_to_semantic(pattern, traits)
                converted_traits.extend(semantic_pattern)
                print(f"    → Added {len(semantic_pattern)} semantic traits from pattern")
            else:
                math_pattern = self._match_semantic_pattern_to_math(pattern, traits)
                converted_traits.extend(math_pattern)
                print(f"    → Added {len(math_pattern)} math traits from pattern")
        
        print(f"✅ Pattern matching conversion complete: {len(converted_traits)} traits")
        return converted_traits
    
    def _convert_recursive_synthesis(self, context: ConversionContext) -> List[Union[MathematicalTrait, SemanticTrait]]:
        """Convert traits using recursive synthesis approach"""
        return self._convert_recursive_formation(context)
    
    def _convert_recursive_formation(self, context: ConversionContext) -> List[Union[MathematicalTrait, SemanticTrait]]:
        """Convert traits using recursive typewriter approach"""
        if context.direction != ConversionDirection.MATH_TO_SEMANTIC:
            raise ValueError("Recursive formation only supports math-to-semantic conversion")
        
        # Initialize recursive formation
        formation_patterns = []
        
        for trait in context.source_traits:
            # Create character formation
            characters = self._extract_characters_from_trait(trait)
            
            # Form words from characters
            words = self._form_words_from_characters(characters)
            
            # Form sentences from words
            sentences = self._form_sentences_from_words(words)
            
            # Create formation pattern
            pattern = FormationPattern(
                pattern_uuid=uuid.uuid4(),
                formation_type=FormationType.CHARACTER,
                characters=characters,
                word=words[0] if words else None,
                sentence=sentences[0] if sentences else None,
                mathematical_consistency=1.0
            )
            
            formation_patterns.append(pattern)
        
        # Convert patterns to semantic traits
        semantic_traits = []
        for pattern in formation_patterns:
            semantic_trait = SemanticTrait(
                trait_uuid=uuid.uuid4(),
                name=f"recursive_formation_{pattern.pattern_uuid}",
                category=TraitCategory.FORMATION,
                complexity=context.target_complexity,
                semantic_properties={
                    'sentences': [pattern.sentence] if pattern.sentence else [],
                    'words': [pattern.word] if pattern.word else [],
                    'characters': pattern.characters,
                    'formation_type': pattern.formation_type.value,
                    'mathematical_consistency': pattern.mathematical_consistency
                }
            )
            semantic_traits.append(semantic_trait)
        
        return semantic_traits
    
    def _convert_hybrid_synthesis(self, context: ConversionContext) -> List[Union[MathematicalTrait, SemanticTrait]]:
        """Convert traits using hybrid synthesis approach"""
        converted_traits = []
        
        # Apply multiple conversion strategies
        strategies = ['character_based', 'pattern_matching']
        
        for strategy in strategies:
            try:
                # Create sub-context for this strategy
                sub_context = ConversionContext(
                    conversion_id=uuid.uuid4(),
                    direction=context.direction,
                    source_traits=context.source_traits,
                    target_complexity=context.target_complexity,
                    conversion_strategy=strategy
                )
                
                # Execute sub-conversion
                conversion_method = getattr(self, f"_convert_{strategy}")
                sub_results = conversion_method(sub_context)
                
                # Synthesize results
                converted_traits.extend(sub_results)
                
            except Exception as e:
                print(f"Warning: Strategy {strategy} failed: {e}")
                continue
        
        # Remove duplicates and optimize
        converted_traits = self._deduplicate_and_optimize(converted_traits)
        
        return converted_traits
    
    def _math_to_character(self, math_trait: MathematicalTrait) -> SemanticTrait:
        """Convert mathematical trait to semantic character"""
        print(f"    🔧 Converting math trait '{math_trait.name}' to character...")
        
        # Extract character from mathematical properties
        character = self._extract_character_from_math(math_trait)
        print(f"    📝 Extracted character: '{character}'")
        
        # Create semantic trait
        print(f"    🏗️ Creating SemanticTrait object...")
        try:
            semantic_trait = SemanticTrait(
                trait_uuid=uuid.uuid4(),
                name=f"char_{math_trait.name}",
                category=TraitCategory.CHARACTER,
                complexity=SemanticComplexity.BASIC,
                semantic_properties={
                    'character': character,
                    'source_trait': str(math_trait.trait_uuid),
                    'conversion_method': 'character_based'
                }
            )
            print(f"    ✅ Created semantic trait: {semantic_trait.name}")
            return semantic_trait
        except Exception as e:
            print(f"    ❌ Failed to create semantic trait: {e}")
            raise
    
    def _character_to_math(self, semantic_trait: SemanticTrait) -> MathematicalTrait:
        """Convert semantic character to mathematical trait"""
        # Extract mathematical properties from character
        math_properties = self._extract_math_from_character(semantic_trait)
        
        # Create mathematical trait
        math_trait = MathematicalTrait(
            trait_uuid=uuid.uuid4(),
            name=f"math_{semantic_trait.name}",
            category=TraitCategory.MATHEMATICAL,
            complexity=SemanticComplexity.BASIC,
            mathematical_properties=math_properties
        )
        
        return math_trait
    
    def _extract_character_from_math(self, math_trait: MathematicalTrait) -> str:
        """Extract character representation from mathematical trait"""
        # Simple character extraction based on trait properties
        trait_hash = hashlib.md5(str(math_trait.mathematical_properties).encode()).hexdigest()
        char_index = int(trait_hash[:8], 16) % 26
        return chr(ord('a') + char_index)
    
    def _extract_math_from_character(self, semantic_trait: SemanticTrait) -> Dict[str, Any]:
        """Extract mathematical properties from semantic character"""
        char_value = ord(semantic_trait.semantic_properties.get('character', 'a')[0])
        
        return {
            'type': 'character_encoding',
            'value': char_value,
            'consistency': 0.8,  # Default consistency
            'convergence': 0.8,  # Default convergence rate
            'ascii_value': char_value,
            'character': semantic_trait.semantic_properties.get('character', 'a')[0]
        }
    
    def _group_traits_by_patterns(self, traits: List[Union[MathematicalTrait, SemanticTrait]]) -> Dict[str, List]:
        """Group traits by their patterns"""
        pattern_groups = defaultdict(list)
        
        for trait in traits:
            if isinstance(trait, MathematicalTrait):
                pattern = self._extract_math_pattern(trait)
            else:
                pattern = self._extract_semantic_pattern(trait)
            
            pattern_groups[pattern].append(trait)
        
        return dict(pattern_groups)
    
    def _extract_math_pattern(self, trait: MathematicalTrait) -> str:
        """Extract pattern from mathematical trait"""
        return f"math_{trait.category.value}_{trait.complexity.value}"
    
    def _extract_semantic_pattern(self, trait: SemanticTrait) -> str:
        """Extract pattern from semantic trait"""
        return f"semantic_{trait.category.value}_{trait.complexity.value}"
    
    def _match_math_pattern_to_semantic(self, pattern: str, traits: List[MathematicalTrait]) -> List[SemanticTrait]:
        """Match mathematical pattern to semantic representation"""
        semantic_traits = []
        
        for trait in traits:
            semantic_trait = self._math_to_character(trait)
            semantic_traits.append(semantic_trait)
        
        return semantic_traits
    
    def _match_semantic_pattern_to_math(self, pattern: str, traits: List[SemanticTrait]) -> List[MathematicalTrait]:
        """Match semantic pattern to mathematical representation"""
        math_traits = []
        
        for trait in traits:
            math_trait = self._character_to_math(trait)
            math_traits.append(math_trait)
        
        return math_traits
    
    def _extract_characters_from_trait(self, trait: Union[MathematicalTrait, SemanticTrait]) -> List[str]:
        """Extract characters from trait for recursive formation"""
        if isinstance(trait, MathematicalTrait):
            return [self._extract_character_from_math(trait)]
        else:
            return list(trait.semantic_meaning) if trait.semantic_meaning else ['a']
    
    def _form_words_from_characters(self, characters: List[str]) -> List[str]:
        """Form words from characters using recursive typewriter approach"""
        words = []
        current_word = ""
        
        for char in characters:
            current_word += char
            if len(current_word) >= 3:  # Minimum word length
                words.append(current_word)
                current_word = ""
        
        if current_word:  # Add remaining characters
            words.append(current_word)
        
        return words if words else ['default']
    
    def _form_sentences_from_words(self, words: List[str]) -> List[str]:
        """Form sentences from words using recursive typewriter approach"""
        if not words:
            return ["Default sentence."]
        
        # Simple sentence formation
        sentence = " ".join(words) + "."
        return [sentence]
    
    def _deduplicate_and_optimize(self, traits: List[Union[MathematicalTrait, SemanticTrait]]) -> List[Union[MathematicalTrait, SemanticTrait]]:
        """Remove duplicates and optimize trait list"""
        seen = set()
        optimized = []
        
        for trait in traits:
            trait_id = f"{type(trait).__name__}_{trait.trait_name}"
            if trait_id not in seen:
                seen.add(trait_id)
                optimized.append(trait)
        
        return optimized
    
    def _calculate_semantic_coherence(self, traits: List[Union[MathematicalTrait, SemanticTrait]]) -> float:
        """Calculate semantic coherence of converted traits"""
        if not traits:
            return 0.0
        
        coherence_scores = []
        for trait in traits:
            if isinstance(trait, SemanticTrait):
                # SemanticTrait doesn't have mathematical_consistency, use a default
                coherence_scores.append(0.8)  # Default coherence score
            else:
                coherence_scores.append(trait.consistency_score)
        
        return statistics.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_mathematical_consistency(self, traits: List[Union[MathematicalTrait, SemanticTrait]]) -> float:
        """Calculate mathematical consistency of converted traits"""
        if not traits:
            return 0.0
        
        consistency_scores = []
        for trait in traits:
            if isinstance(trait, MathematicalTrait):
                consistency_scores.append(trait.consistency_score)
            else:
                # SemanticTrait doesn't have mathematical_consistency, use a default
                consistency_scores.append(0.8)  # Default consistency score
        
        return statistics.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_average_complexity(self, traits: List[Union[MathematicalTrait, SemanticTrait]]) -> SemanticComplexity:
        """Calculate average complexity of traits"""
        if not traits:
            return SemanticComplexity.BASIC
        
        complexity_values = []
        for trait in traits:
            if isinstance(trait, SemanticTrait):
                complexity_values.append(trait.complexity.value)
            else:
                # Map mathematical traits to complexity
                complexity_values.append(1)  # Default to BASIC
        
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
    
    def _update_trait_mappings(self, context: ConversionContext, converted_traits: List[Union[MathematicalTrait, SemanticTrait]]):
        """Update trait mappings after conversion"""
        for i, source_trait in enumerate(context.source_traits):
            if i < len(converted_traits):
                converted_trait = converted_traits[i]
                
                mapping = TraitMapping(
                    mapping_id=uuid.uuid4(),
                    mathematical_trait=source_trait if isinstance(source_trait, MathematicalTrait) else None,
                    semantic_trait=source_trait if isinstance(source_trait, SemanticTrait) else None,
                    confidence=0.8,  # Default confidence
                    bidirectional=context.direction == ConversionDirection.BIDIRECTIONAL
                )
                
                self.trait_mappings[mapping.mapping_id] = mapping
    
    def _update_conversion_metrics(self, result: ConversionResult):
        """Update conversion performance metrics"""
        self.conversion_metrics['total_conversions'] += 1
        
        if result.status == ConversionStatus.COMPLETED:
            self.conversion_metrics['successful_conversions'] += 1
        else:
            self.conversion_metrics['failed_conversions'] += 1
        
        # Update success rate
        total = self.conversion_metrics['total_conversions']
        successful = self.conversion_metrics['successful_conversions']
        self.conversion_metrics['success_rate'] = successful / total if total > 0 else 0.0
        
        # Update averages
        if result.conversion_metrics.get('processing_time'):
            current_avg = self.conversion_metrics['average_conversion_time']
            total_conversions = self.conversion_metrics['total_conversions']
            new_avg = (current_avg * (total_conversions - 1) + result.conversion_metrics['processing_time']) / total_conversions
            self.conversion_metrics['average_conversion_time'] = new_avg
        
        # Update coherence and consistency averages
        if result.semantic_coherence > 0:
            current_avg = self.conversion_metrics['semantic_coherence_avg']
            total_conversions = self.conversion_metrics['total_conversions']
            new_avg = (current_avg * (total_conversions - 1) + result.semantic_coherence) / total_conversions
            self.conversion_metrics['semantic_coherence_avg'] = new_avg
        
        if result.mathematical_consistency > 0:
            current_avg = self.conversion_metrics['mathematical_consistency_avg']
            total_conversions = self.conversion_metrics['total_conversions']
            new_avg = (current_avg * (total_conversions - 1) + result.mathematical_consistency) / total_conversions
            self.conversion_metrics['mathematical_consistency_avg'] = new_avg
    
    def _handle_conversion_request(self, event_data: Dict[str, Any]):
        """Handle conversion request events"""
        try:
            source_traits = event_data.get('source_traits', [])
            direction = ConversionDirection(event_data.get('direction', 'math_to_semantic'))
            target_complexity = SemanticComplexity(event_data.get('target_complexity', 'basic'))
            strategy = event_data.get('strategy', 'auto')
            
            result = self.convert_traits(source_traits, direction, target_complexity, strategy)
            
            # Publish completion event
            self.event_bridge.publish_semantic_event(
                WordFormationEvent(
                    event_id=uuid.uuid4(),
                    timestamp=datetime.utcnow(),
                    payload={
                        'conversion_id': str(result.conversion_id),
                        'status': result.status.value,
                        'trait_count': len(result.converted_traits)
                    }
                )
            )
            
        except Exception as e:
            print(f"Error handling conversion request: {e}")
    
    def _handle_formation_complete(self, event_data: Dict[str, Any]):
        """Handle formation completion events"""
        # Update mappings based on completed formations
        formation_id = event_data.get('formation_id')
        if formation_id:
            # Update related trait mappings
            pass
    
    def get_conversion_status(self, conversion_id: uuid.UUID) -> Optional[ConversionResult]:
        """Get status of specific conversion"""
        return self.completed_conversions.get(conversion_id)
    
    def get_conversion_metrics(self) -> Dict[str, Any]:
        """Get current conversion performance metrics"""
        return self.conversion_metrics.copy()
    
    def get_trait_mappings(self) -> Dict[uuid.UUID, TraitMapping]:
        """Get all trait mappings"""
        return self.trait_mappings.copy()
    
    def clear_completed_conversions(self, older_than: timedelta = timedelta(hours=24)):
        """Clear old completed conversions"""
        cutoff_time = datetime.utcnow() - older_than
        
        with self._lock:
            old_conversions = [
                conv_id for conv_id, result in self.completed_conversions.items()
                if result.completion_time and result.completion_time < cutoff_time
            ]
            
            for conv_id in old_conversions:
                del self.completed_conversions[conv_id]
            
            print(f"Cleared {len(old_conversions)} old conversions")
