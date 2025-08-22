# rebuild/mathematical_semantic_api.py
"""
Mathematical Semantic API - Semantic foundation query interface
Allows recursive typewriter to query semantic knowledge using mathematical operations
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
    SemanticTrait, MathematicalTrait, FormationPattern,
    SemanticComplexity, TraitCategory, SemanticViolation
)
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager
from local_semantic_database import LocalSemanticDatabase, SemanticReference, SemanticRelationship

class QueryType(Enum):
    """Types of semantic queries"""
    WORD_LOOKUP = "word_lookup"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    RELATIONSHIP_QUERY = "relationship_query"
    CONCEPT_EXPANSION = "concept_expansion"
    EMOTIONAL_ANALYSIS = "emotional_analysis"
    FORMATION_GUIDANCE = "formation_guidance"

class QueryResult(Enum):
    """Result status of semantic queries"""
    SUCCESS = "success"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    ERROR = "error"

@dataclass
class SemanticQuery:
    """Mathematical semantic query"""
    query_id: uuid.UUID
    query_type: QueryType
    input_data: Dict[str, Any]
    mathematical_constraints: Dict[str, Any]
    target_complexity: SemanticComplexity
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticQueryResult:
    """Result of semantic query"""
    query_id: uuid.UUID
    query_type: QueryType
    result_status: QueryResult
    semantic_data: List[SemanticReference]
    mathematical_properties: Dict[str, Any]
    confidence_score: float
    processing_time: float
    error_details: Optional[str] = None

@dataclass
class FormationGuidance:
    """Guidance for recursive typewriter formation"""
    guidance_id: uuid.UUID
    target_word: str
    semantic_context: Dict[str, Any]
    mathematical_constraints: Dict[str, Any]
    formation_suggestions: List[str]
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)

class MathematicalSemanticAPI:
    """
    Mathematical Semantic API wrapper
    Provides semantic foundation queries using mathematical operations
    """
    
    def __init__(self,
                 state_manager: SemanticStateManager,
                 event_bridge: SemanticEventBridge,
                 violation_monitor: SemanticViolationMonitor,
                 checkpoint_manager: SemanticCheckpointManager,
                 uuid_anchor: UUIDanchor,
                 trait_convergence: TraitConvergenceEngine,
                 semantic_database: LocalSemanticDatabase):
        
        # Core dependencies
        self.state_manager = state_manager
        self.event_bridge = event_bridge
        self.violation_monitor = violation_monitor
        self.checkpoint_manager = checkpoint_manager
        self.uuid_anchor = uuid_anchor
        self.trait_convergence = trait_convergence
        self.semantic_database = semantic_database
        
        # Query state
        self.active_queries: Dict[uuid.UUID, SemanticQuery] = {}
        self.query_history: Dict[uuid.UUID, SemanticQueryResult] = {}
        self.formation_guidance_cache: Dict[str, FormationGuidance] = {}
        
        # Performance tracking
        self.query_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_query_time': 0.0,
            'average_confidence': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize query strategies
        self._initialize_query_strategies()
        
        # Register with event bridge
        self.event_bridge.register_handler("SEMANTIC_QUERY_REQUEST", self._handle_query_request)
        self.event_bridge.register_handler("FORMATION_GUIDANCE_REQUEST", self._handle_formation_guidance_request)
        
        print(f"🔗 MathematicalSemanticAPI initialized with {len(self.query_strategies)} query strategies")
    
    def _initialize_query_strategies(self):
        """Initialize available query strategies"""
        self.query_strategies = {
            'exact_match': {
                'description': 'Exact word lookup in semantic database',
                'complexity_range': [SemanticComplexity.BASIC, SemanticComplexity.INTERMEDIATE],
                'confidence_threshold': 0.8
            },
            'semantic_similarity': {
                'description': 'Find semantically similar words',
                'complexity_range': [SemanticComplexity.INTERMEDIATE, SemanticComplexity.ADVANCED],
                'confidence_threshold': 0.6
            },
            'relationship_traversal': {
                'description': 'Traverse semantic relationships',
                'complexity_range': [SemanticComplexity.INTERMEDIATE, SemanticComplexity.EXPERT],
                'confidence_threshold': 0.7
            },
            'concept_expansion': {
                'description': 'Expand concepts through semantic relationships',
                'complexity_range': [SemanticComplexity.ADVANCED, SemanticComplexity.EXPERT],
                'confidence_threshold': 0.5
            },
            'emotional_analysis': {
                'description': 'Analyze emotional properties of words',
                'complexity_range': [SemanticComplexity.BASIC, SemanticComplexity.ADVANCED],
                'confidence_threshold': 0.7
            },
            'formation_guidance': {
                'description': 'Provide guidance for recursive typewriter formation',
                'complexity_range': [SemanticComplexity.INTERMEDIATE, SemanticComplexity.EXPERT],
                'confidence_threshold': 0.6
            }
        }
    
    def query_semantic_foundation(self,
                                 query_type: QueryType,
                                 input_data: Dict[str, Any],
                                 mathematical_constraints: Dict[str, Any],
                                 target_complexity: SemanticComplexity) -> SemanticQueryResult:
        """
        Query semantic foundation using mathematical operations
        """
        start_time = datetime.utcnow()
        
        with self._lock:
            query_id = uuid.uuid4()
            
            # Create query
            query = SemanticQuery(
                query_id=query_id,
                query_type=query_type,
                input_data=input_data,
                mathematical_constraints=mathematical_constraints,
                target_complexity=target_complexity
            )
            
            # Store active query
            self.active_queries[query_id] = query
            
            try:
                # Execute query based on type
                if query_type == QueryType.WORD_LOOKUP:
                    result = self._execute_word_lookup(query)
                elif query_type == QueryType.SEMANTIC_SIMILARITY:
                    result = self._execute_semantic_similarity(query)
                elif query_type == QueryType.RELATIONSHIP_QUERY:
                    result = self._execute_relationship_query(query)
                elif query_type == QueryType.CONCEPT_EXPANSION:
                    result = self._execute_concept_expansion(query)
                elif query_type == QueryType.EMOTIONAL_ANALYSIS:
                    result = self._execute_emotional_analysis(query)
                elif query_type == QueryType.FORMATION_GUIDANCE:
                    result = self._execute_formation_guidance(query)
                else:
                    result = self._execute_generic_query(query)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                result.processing_time = processing_time
                
                # Update metrics
                self._update_query_metrics(result)
                
                # Store in history
                self.query_history[query_id] = result
                
                # Remove from active queries
                del self.active_queries[query_id]
                
                return result
                
            except Exception as e:
                # Handle error
                error_result = SemanticQueryResult(
                    query_id=query_id,
                    query_type=query_type,
                    result_status=QueryResult.ERROR,
                    semantic_data=[],
                    mathematical_properties={},
                    confidence_score=0.0,
                    processing_time=(datetime.utcnow() - start_time).total_seconds(),
                    error_details=str(e)
                )
                
                # Update metrics
                self._update_query_metrics(error_result)
                
                # Store in history
                self.query_history[query_id] = error_result
                
                # Remove from active queries
                del self.active_queries[query_id]
                
                return error_result
    
    def _execute_word_lookup(self, query: SemanticQuery) -> SemanticQueryResult:
        """Execute exact word lookup"""
        word = query.input_data.get('word', '').lower()
        
        # Look up word in semantic database
        reference = self.semantic_database.get_semantic_data(word)
        
        if reference:
            return SemanticQueryResult(
                query_id=query.query_id,
                query_type=query.query_type,
                result_status=QueryResult.SUCCESS,
                semantic_data=[reference],
                mathematical_properties=reference.mathematical_properties,
                confidence_score=0.9,
                processing_time=0.0
            )
        else:
            return SemanticQueryResult(
                query_id=query.query_id,
                query_type=query.query_type,
                result_status=QueryResult.NOT_FOUND,
                semantic_data=[],
                mathematical_properties={},
                confidence_score=0.0,
                processing_time=0.0
            )
    
    def _execute_semantic_similarity(self, query: SemanticQuery) -> SemanticQueryResult:
        """Execute semantic similarity search"""
        target_word = query.input_data.get('word', '').lower()
        similarity_threshold = query.mathematical_constraints.get('similarity_threshold', 0.5)
        
        # Get target reference
        target_reference = self.semantic_database.get_semantic_data(target_word)
        if not target_reference:
            return SemanticQueryResult(
                query_id=query.query_id,
                query_type=query.query_type,
                result_status=QueryResult.NOT_FOUND,
                semantic_data=[],
                mathematical_properties={},
                confidence_score=0.0,
                processing_time=0.0
            )
        
        # Find similar words
        similar_references = []
        for reference in self.semantic_database.semantic_references.values():
            if reference.word.lower() != target_word:
                similarity = self._calculate_semantic_similarity(target_reference, reference)
                if similarity >= similarity_threshold:
                    similar_references.append(reference)
        
        # Sort by similarity
        similar_references.sort(key=lambda r: self._calculate_semantic_similarity(target_reference, r), reverse=True)
        
        return SemanticQueryResult(
            query_id=query.query_id,
            query_type=query.query_type,
            result_status=QueryResult.SUCCESS if similar_references else QueryResult.PARTIAL,
            semantic_data=similar_references[:10],  # Limit to top 10
            mathematical_properties={
                'similarity_threshold': similarity_threshold,
                'total_similar_words': len(similar_references)
            },
            confidence_score=0.7 if similar_references else 0.3,
            processing_time=0.0
        )
    
    def _calculate_semantic_similarity(self, ref1: SemanticReference, ref2: SemanticReference) -> float:
        """Calculate semantic similarity between two references"""
        # Compare mathematical properties
        math_similarity = self._compare_mathematical_properties(ref1.mathematical_properties, ref2.mathematical_properties)
        
        # Compare semantic properties
        semantic_similarity = self._compare_semantic_properties(ref1.semantic_properties, ref2.semantic_properties)
        
        # Compare trait properties
        trait_similarity = self._compare_trait_properties(ref1, ref2)
        
        # Weighted average
        similarity = (math_similarity * 0.4) + (semantic_similarity * 0.4) + (trait_similarity * 0.2)
        return max(0.0, min(1.0, similarity))
    
    def _compare_mathematical_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
        """Compare mathematical properties"""
        if not props1 or not props2:
            return 0.0
        
        # Compare key properties
        length_diff = abs(props1.get('length', 0) - props2.get('length', 0)) / max(props1.get('length', 1), props2.get('length', 1), 1)
        complexity_diff = abs(props1.get('complexity', 0) - props2.get('complexity', 0))
        frequency_diff = abs(props1.get('frequency', 0) - props2.get('frequency', 0))
        
        # Calculate similarity (1 - normalized difference)
        similarity = 1.0 - ((length_diff + complexity_diff + frequency_diff) / 3.0)
        return max(0.0, min(1.0, similarity))
    
    def _compare_semantic_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
        """Compare semantic properties"""
        if not props1 or not props2:
            return 0.0
        
        # Compare categorical properties
        category_match = 1.0 if props1.get('category') == props2.get('category') else 0.0
        pos_match = 1.0 if props1.get('pos') == props2.get('pos') else 0.0
        
        # Compare continuous properties
        valence_diff = abs(props1.get('valence', 0.5) - props2.get('valence', 0.5))
        arousal_diff = abs(props1.get('arousal', 0.5) - props2.get('arousal', 0.5))
        dominance_diff = abs(props1.get('dominance', 0.5) - props2.get('dominance', 0.5))
        
        # Calculate similarity
        continuous_similarity = 1.0 - ((valence_diff + arousal_diff + dominance_diff) / 3.0)
        categorical_similarity = (category_match + pos_match) / 2.0
        
        similarity = (continuous_similarity * 0.6) + (categorical_similarity * 0.4)
        return max(0.0, min(1.0, similarity))
    
    def _compare_trait_properties(self, ref1: SemanticReference, ref2: SemanticReference) -> float:
        """Compare trait properties"""
        # Compare convergence stability, violation pressure, and trait intensity
        stability_diff = abs(ref1.convergence_stability - ref2.convergence_stability)
        vp_diff = abs(ref1.violation_pressure - ref2.violation_pressure)
        intensity_diff = abs(ref1.trait_intensity - ref2.trait_intensity)
        
        # Calculate similarity
        similarity = 1.0 - ((stability_diff + vp_diff + intensity_diff) / 3.0)
        return max(0.0, min(1.0, similarity))
    
    def _execute_relationship_query(self, query: SemanticQuery) -> SemanticQueryResult:
        """Execute relationship query"""
        word = query.input_data.get('word', '').lower()
        relationship_type = query.input_data.get('relationship_type', 'all')
        
        # Get relationships for word
        relationships = self.semantic_database.get_semantic_relationships(word)
        
        # Filter by relationship type if specified
        if relationship_type != 'all':
            relationships = [rel for rel in relationships if rel.relationship_type == relationship_type]
        
        # Get related references
        related_references = []
        for rel in relationships:
            target_word = rel.target_word if rel.source_word.lower() == word else rel.source_word
            reference = self.semantic_database.get_semantic_data(target_word)
            if reference:
                related_references.append(reference)
        
        return SemanticQueryResult(
            query_id=query.query_id,
            query_type=query.query_type,
            result_status=QueryResult.SUCCESS if related_references else QueryResult.PARTIAL,
            semantic_data=related_references,
            mathematical_properties={
                'relationship_type': relationship_type,
                'total_relationships': len(relationships)
            },
            confidence_score=0.8 if related_references else 0.4,
            processing_time=0.0
        )
    
    def _execute_concept_expansion(self, query: SemanticQuery) -> SemanticQueryResult:
        """Execute concept expansion query"""
        concept = query.input_data.get('concept', '').lower()
        expansion_depth = query.mathematical_constraints.get('expansion_depth', 2)
        
        # Get concept reference
        concept_ref = self.semantic_database.get_semantic_data(concept)
        if not concept_ref:
            return SemanticQueryResult(
                query_id=query.query_id,
                query_type=query.query_type,
                result_status=QueryResult.NOT_FOUND,
                semantic_data=[],
                mathematical_properties={},
                confidence_score=0.0,
                processing_time=0.0
            )
        
        # Expand concept through relationships
        expanded_references = set([concept_ref])
        current_level = {concept_ref}
        
        for depth in range(expansion_depth):
            next_level = set()
            for ref in current_level:
                # Get relationships for this reference
                relationships = self.semantic_database.get_semantic_relationships(ref.word)
                for rel in relationships:
                    target_word = rel.target_word if rel.source_word.lower() == ref.word.lower() else rel.source_word
                    target_ref = self.semantic_database.get_semantic_data(target_word)
                    if target_ref and target_ref not in expanded_references:
                        next_level.add(target_ref)
                        expanded_references.add(target_ref)
            current_level = next_level
        
        return SemanticQueryResult(
            query_id=query.query_id,
            query_type=query.query_type,
            result_status=QueryResult.SUCCESS,
            semantic_data=list(expanded_references),
            mathematical_properties={
                'expansion_depth': expansion_depth,
                'total_expanded_concepts': len(expanded_references)
            },
            confidence_score=0.6,
            processing_time=0.0
        )
    
    def _execute_emotional_analysis(self, query: SemanticQuery) -> SemanticQueryResult:
        """Execute emotional analysis query"""
        word = query.input_data.get('word', '').lower()
        
        # Get word reference
        reference = self.semantic_database.get_semantic_data(word)
        if not reference:
            return SemanticQueryResult(
                query_id=query.query_id,
                query_type=query.query_type,
                result_status=QueryResult.NOT_FOUND,
                semantic_data=[],
                mathematical_properties={},
                confidence_score=0.0,
                processing_time=0.0
            )
        
        # Extract emotional properties
        emotional_properties = {
            'valence': reference.semantic_properties.get('valence', 0.5),
            'arousal': reference.semantic_properties.get('arousal', 0.5),
            'dominance': reference.semantic_properties.get('dominance', 0.5),
            'emotional_intensity': reference.mathematical_properties.get('emotional_intensity', 0.3),
            'is_emotion': reference.data_type.value == 'emotion'
        }
        
        return SemanticQueryResult(
            query_id=query.query_id,
            query_type=query.query_type,
            result_status=QueryResult.SUCCESS,
            semantic_data=[reference],
            mathematical_properties=emotional_properties,
            confidence_score=0.8,
            processing_time=0.0
        )
    
    def _execute_formation_guidance(self, query: SemanticQuery) -> SemanticQueryResult:
        """Execute formation guidance query"""
        target_concept = query.input_data.get('target_concept', '').lower()
        formation_context = query.input_data.get('formation_context', {})
        
        # Get target reference
        target_ref = self.semantic_database.get_semantic_data(target_concept)
        if not target_ref:
            return SemanticQueryResult(
                query_id=query.query_id,
                query_type=query.query_type,
                result_status=QueryResult.NOT_FOUND,
                semantic_data=[],
                mathematical_properties={},
                confidence_score=0.0,
                processing_time=0.0
            )
        
        # Generate formation guidance
        guidance = self._generate_formation_guidance(target_ref, formation_context, query.mathematical_constraints)
        
        # Cache guidance
        self.formation_guidance_cache[target_concept] = guidance
        
        return SemanticQueryResult(
            query_id=query.query_id,
            query_type=query.query_type,
            result_status=QueryResult.SUCCESS,
            semantic_data=[target_ref],
            mathematical_properties={
                'formation_suggestions': guidance.formation_suggestions,
                'confidence_score': guidance.confidence_score
            },
            confidence_score=guidance.confidence_score,
            processing_time=0.0
        )
    
    def _generate_formation_guidance(self, target_ref: SemanticReference, context: Dict[str, Any], constraints: Dict[str, Any]) -> FormationGuidance:
        """Generate formation guidance for recursive typewriter"""
        suggestions = []
        
        # Analyze target reference properties
        complexity = target_ref.mathematical_properties.get('complexity', 0.5)
        emotional_intensity = target_ref.mathematical_properties.get('emotional_intensity', 0.3)
        abstractness = target_ref.mathematical_properties.get('abstractness', 0.3)
        
        # Generate suggestions based on properties
        if complexity > 0.7:
            suggestions.append("Use recursive synthesis for complex formation")
        if emotional_intensity > 0.6:
            suggestions.append("Apply emotional intensity scaling")
        if abstractness > 0.6:
            suggestions.append("Use abstract concept formation patterns")
        
        # Add context-specific suggestions
        if context.get('formation_type') == 'word':
            suggestions.append("Apply word formation validation rules")
        elif context.get('formation_type') == 'sentence':
            suggestions.append("Use sentence structure patterns")
        elif context.get('formation_type') == 'dialogue':
            suggestions.append("Apply dialogue coherence rules")
        
        # Add mathematical constraint suggestions
        if constraints.get('min_length'):
            suggestions.append(f"Ensure minimum length: {constraints['min_length']}")
        if constraints.get('max_complexity'):
            suggestions.append(f"Limit complexity to: {constraints['max_complexity']}")
        
        # Calculate confidence based on available data
        confidence = min(0.9, 0.3 + (len(suggestions) * 0.1))
        
        return FormationGuidance(
            guidance_id=uuid.uuid4(),
            target_word=target_ref.word,
            semantic_context=target_ref.semantic_properties,
            mathematical_constraints=constraints,
            formation_suggestions=suggestions,
            confidence_score=confidence
        )
    
    def _execute_generic_query(self, query: SemanticQuery) -> SemanticQueryResult:
        """Execute generic query fallback"""
        return SemanticQueryResult(
            query_id=query.query_id,
            query_type=query.query_type,
            result_status=QueryResult.ERROR,
            semantic_data=[],
            mathematical_properties={},
            confidence_score=0.0,
            processing_time=0.0,
            error_details="Unknown query type"
        )
    
    def _update_query_metrics(self, result: SemanticQueryResult):
        """Update query performance metrics"""
        self.query_metrics['total_queries'] += 1
        
        if result.result_status == QueryResult.SUCCESS:
            self.query_metrics['successful_queries'] += 1
        elif result.result_status == QueryResult.ERROR:
            self.query_metrics['failed_queries'] += 1
        
        # Update averages
        total_successful = self.query_metrics['successful_queries']
        if total_successful > 0:
            # Update average query time
            current_avg = self.query_metrics['average_query_time']
            new_avg = ((current_avg * (total_successful - 1)) + result.processing_time) / total_successful
            self.query_metrics['average_query_time'] = new_avg
            
            # Update average confidence
            current_conf_avg = self.query_metrics['average_confidence']
            new_conf_avg = ((current_conf_avg * (total_successful - 1)) + result.confidence_score) / total_successful
            self.query_metrics['average_confidence'] = new_conf_avg
    
    def get_query_metrics(self) -> Dict[str, Any]:
        """Get query performance metrics"""
        with self._lock:
            return self.query_metrics.copy()
    
    def get_formation_guidance(self, target_concept: str) -> Optional[FormationGuidance]:
        """Get cached formation guidance"""
        with self._lock:
            return self.formation_guidance_cache.get(target_concept.lower())
    
    def _handle_query_request(self, event_data: Dict[str, Any]):
        """Handle semantic query request events"""
        try:
            query_type = QueryType(event_data.get('query_type'))
            input_data = event_data.get('input_data', {})
            mathematical_constraints = event_data.get('mathematical_constraints', {})
            target_complexity = SemanticComplexity(event_data.get('target_complexity', 'BASIC'))
            
            result = self.query_semantic_foundation(
                query_type=query_type,
                input_data=input_data,
                mathematical_constraints=mathematical_constraints,
                target_complexity=target_complexity
            )
            
            # Publish result event
            self.event_bridge.publish_semantic_event({
                'event_type': 'SEMANTIC_QUERY_RESULT',
                'query_id': str(result.query_id),
                'result': asdict(result)
            })
            
        except Exception as e:
            print(f"Error handling query request: {e}")
    
    def _handle_formation_guidance_request(self, event_data: Dict[str, Any]):
        """Handle formation guidance request events"""
        try:
            target_concept = event_data.get('target_concept', '')
            formation_context = event_data.get('formation_context', {})
            mathematical_constraints = event_data.get('mathematical_constraints', {})
            
            # Create query for formation guidance
            query = SemanticQuery(
                query_id=uuid.uuid4(),
                query_type=QueryType.FORMATION_GUIDANCE,
                input_data={
                    'target_concept': target_concept,
                    'formation_context': formation_context
                },
                mathematical_constraints=mathematical_constraints,
                target_complexity=SemanticComplexity.INTERMEDIATE
            )
            
            result = self._execute_formation_guidance(query)
            
            # Publish result event
            self.event_bridge.publish_semantic_event({
                'event_type': 'FORMATION_GUIDANCE_RESULT',
                'target_concept': target_concept,
                'guidance': asdict(result)
            })
            
        except Exception as e:
            print(f"Error handling formation guidance request: {e}")
