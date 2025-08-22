"""
Semantic Data Structures - Core data types for semantic system
Defines all data structures used throughout the semantic system
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
from enum import Enum

class EvolutionStage(Enum):
    """Evolution stages of semantic understanding"""
    INITIALIZATION = "initialization"
    SEMANTIC_FOUNDATION = "semantic_foundation"
    GUIDED_LEARNING = "guided_learning"
    PATTERN_RECOGNITION = "pattern_recognition"
    AUTONOMOUS_FORMATION = "autonomous_formation"
    SEMANTIC_TRANSCENDENCE = "semantic_transcendence"
    PURE_MATHEMATICAL = "pure_mathematical"

class CheckpointType(Enum):
    """Types of semantic checkpoints"""
    GENESIS = "genesis"
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    MILESTONE = "milestone"
    REGRESSION_PROTECTION = "regression_protection"
    EMERGENCY = "emergency"
    SAFETY = "safety"
    AB_TEST_BASELINE = "ab_test_baseline"

class FormationType(Enum):
    """Types of semantic formations"""
    CHARACTER = "character"
    WORD = "word"
    SENTENCE = "sentence"
    DIALOGUE = "dialogue"

class ExperimentType(Enum):
    """Types of semantic experiments"""
    NOVEL_FORMATION = "novel_formation"
    PATTERN_VARIATION = "pattern_variation"
    AGGRESSIVE_EVOLUTION = "aggressive_evolution"
    BOUNDARY_TESTING = "boundary_testing"
    RECOVERY_TESTING = "recovery_testing"

class RegressionSeverity(Enum):
    """Severity levels for performance regression"""
    NONE = "none"
    WARNING = "warning"
    SEVERE = "severe"
    CRITICAL = "critical"

class RollbackReason(Enum):
    """Reasons for semantic state rollback"""
    AUTOMATIC_REGRESSION_PROTECTION = "automatic_regression_protection"
    MANUAL_ROLLBACK = "manual_rollback"
    SAFETY_THRESHOLD_VIOLATION = "safety_threshold_violation"
    MATHEMATICAL_INCONSISTENCY = "mathematical_inconsistency"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"

class EvolutionStrategy(Enum):
    """Semantic evolution strategies"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"

@dataclass
class SemanticCheckpoint:
    """Semantic system checkpoint with mathematical integrity"""
    checkpoint_id: uuid.UUID
    timestamp: datetime
    semantic_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    evolution_stage: EvolutionStage
    mathematical_hash: str
    parent_checkpoint: Optional[uuid.UUID] = None
    checkpoint_type: CheckpointType = CheckpointType.MANUAL
    description: str = ""

@dataclass
class FormationPattern:
    """Pattern of semantic formation with mathematical properties"""
    pattern_uuid: uuid.UUID
    formation_type: FormationType
    characters: List[str]
    word: Optional[str] = None
    sentence: Optional[str] = None
    dialogue: Optional[str] = None
    semantic_traits: Dict[str, Any] = field(default_factory=dict)
    violation_pressure: float = 0.0
    formation_success: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    mathematical_consistency: float = 1.0

@dataclass
class SemanticEvolutionBranch:
    """Branch of semantic evolution for A/B testing"""
    branch_uuid: uuid.UUID
    base_checkpoint: uuid.UUID
    strategy: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    mathematical_consistency: float = 1.0
    adoption_decision: Optional[bool] = None
    evolution_progress: float = 0.0

@dataclass
class EvolutionValidation:
    """Validation result for semantic evolution"""
    evolution_direction: str  # "improving", "degrading", "stable"
    performance_delta: float
    mathematical_consistency: bool
    recommendation: str

@dataclass
class SafetyNet:
    """Multi-layer safety net for semantic evolution"""
    pre_evolution_checkpoint: SemanticCheckpoint
    sandbox_environment: Dict[str, Any] = field(default_factory=dict)
    vp_monitoring: Dict[str, Any] = field(default_factory=dict)
    regression_baseline: Dict[str, Any] = field(default_factory=dict)
    emergency_protocols: Dict[str, Any] = field(default_factory=dict)
    safety_validation: bool = True

@dataclass
class RegressionMonitoring:
    """Monitoring result for performance regression"""
    regression_status: Dict[str, Any] = field(default_factory=dict)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

@dataclass
class SemanticHealth:
    """Health status of semantic system"""
    formation_stability: float
    semantic_stability: float
    checkpoint_integrity: bool
    evolution_progress: float
    system_coherence: float
    last_checkpoint: Optional[datetime] = None
    total_checkpoints: int = 0
    current_stage: EvolutionStage = EvolutionStage.INITIALIZATION

@dataclass
class SemanticMilestone:
    """Milestone in semantic evolution"""
    milestone_uuid: uuid.UUID
    milestone_type: str
    description: str
    timestamp: datetime
    evolution_stage: EvolutionStage
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    mathematical_validation: bool = True

@dataclass
class SemanticViolation:
    """Semantic violation with mathematical quantification"""
    violation_uuid: uuid.UUID
    violation_type: str
    severity: RegressionSeverity
    violation_pressure: float
    formation_pattern: Optional[FormationPattern] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    mathematical_consistency: float = 0.0

@dataclass
class SemanticGuidance:
    """Mathematical guidance for semantic operations"""
    guidance_uuid: uuid.UUID
    guidance_type: str
    mathematical_properties: Dict[str, float] = field(default_factory=dict)
    convergence_stability: float = 1.0
    violation_pressure: float = 0.0
    trait_intensity: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticLearning:
    """Learning result from semantic operations"""
    learning_uuid: uuid.UUID
    learning_type: str
    formation_pattern: FormationPattern
    mathematical_insights: Dict[str, Any] = field(default_factory=dict)
    evolution_contribution: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticConsensus:
    """Consensus result for multi-agent semantic operations"""
    consensus_uuid: uuid.UUID
    consensus_type: str
    agent_states: List[Dict[str, Any]] = field(default_factory=list)
    consensus_decision: Dict[str, Any] = field(default_factory=dict)
    mathematical_validation: bool = True
    confidence_level: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticGovernance:
    """Governance decision for semantic evolution"""
    governance_uuid: uuid.UUID
    decision_type: str
    decision: str  # "approved", "rejected", "requires_amendment"
    proposal: Dict[str, Any] = field(default_factory=dict)
    mathematical_justification: str = ""
    constitutional_impact: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticMetrics:
    """Comprehensive metrics for semantic system"""
    metrics_uuid: uuid.UUID
    timestamp: datetime
    formation_metrics: Dict[str, float] = field(default_factory=dict)
    evolution_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    health_metrics: Dict[str, float] = field(default_factory=dict)
    mathematical_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class SemanticEvent:
    """Base class for semantic events"""
    event_uuid: uuid.UUID
    event_type: str
    timestamp: datetime
    payload: Dict[str, Any] = field(default_factory=dict)
    mathematical_validation: bool = True

@dataclass
class CharacterFormationEvent(SemanticEvent):
    """Event for character formation"""
    character: str = ""
    formation_success: bool = False
    violation_pressure: float = 0.0
    mathematical_consistency: float = 0.0

@dataclass
class WordFormationEvent(SemanticEvent):
    """Event for word formation"""
    word: str = ""
    characters: List[str] = field(default_factory=list)
    formation_success: bool = False
    violation_pressure: float = 0.0
    mathematical_consistency: float = 0.0

@dataclass
class SentenceFormationEvent(SemanticEvent):
    """Event for sentence formation"""
    sentence: str = ""
    words: List[str] = field(default_factory=list)
    formation_success: bool = False
    violation_pressure: float = 0.0
    mathematical_consistency: float = 0.0

@dataclass
class DialogueFormationEvent(SemanticEvent):
    """Event for dialogue formation"""
    dialogue: str = ""
    sentences: List[str] = field(default_factory=list)
    formation_success: bool = False
    violation_pressure: float = 0.0
    mathematical_consistency: float = 0.0

@dataclass
class CommunicationEvent(SemanticEvent):
    """Event for communication formation"""
    communication: str = ""
    dialogues: List[str] = field(default_factory=list)
    formation_success: bool = False
    violation_pressure: float = 0.0
    mathematical_consistency: float = 0.0

@dataclass
class SemanticMilestoneEvent(SemanticEvent):
    """Event for semantic milestone achievement"""
    milestone: Optional[SemanticMilestone] = None
    evolution_stage: EvolutionStage = EvolutionStage.INITIALIZATION
    performance_improvement: float = 0.0

@dataclass
class SemanticViolationEvent(SemanticEvent):
    """Event for semantic violation detection"""
    violation: Optional[SemanticViolation] = None
    automatic_response: str = ""
    safety_triggered: bool = False

@dataclass
class SemanticCheckpointEvent(SemanticEvent):
    """Event for checkpoint creation"""
    checkpoint: Optional[SemanticCheckpoint] = None
    checkpoint_type: CheckpointType = CheckpointType.MANUAL
    trigger: str = ""

@dataclass
class SemanticRollbackEvent(SemanticEvent):
    """Event for semantic state rollback"""
    rollback_reason: RollbackReason = RollbackReason.MANUAL_ROLLBACK
    from_checkpoint: Optional[uuid.UUID] = None
    to_checkpoint: Optional[uuid.UUID] = None
    restoration_success: bool = False

@dataclass
class SemanticEvolutionEvent(SemanticEvent):
    """Event for semantic evolution"""
    evolution_type: str = ""
    evolution_stage: EvolutionStage = EvolutionStage.INITIALIZATION
    performance_delta: float = 0.0
    mathematical_consistency: bool = True

@dataclass
class SemanticGovernanceEvent(SemanticEvent):
    """Event for semantic governance decisions"""
    governance: Optional[SemanticGovernance] = None
    constitutional_impact: str = ""
    amendment_required: bool = False

# Add missing trait-related classes and enums
class TraitCategory(Enum):
    """Categories of traits"""
    BASIC = "basic"
    SEMANTIC = "semantic"
    LINGUISTIC = "linguistic"
    SYNTHESIS = "synthesis"
    PHILOSOPHICAL = "philosophical"
    TECHNICAL = "technical"
    ABSTRACT = "abstract"
    MATHEMATICAL = "mathematical"
    FORMATION = "formation"  # Added for integration compatibility
    GENERAL = "general"      # Added for integration compatibility
    CHARACTER = "character"

class SemanticComplexity(Enum):
    """Complexity levels for semantic operations"""
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

class ConversionType(Enum):
    """Types of trait conversion"""
    CHARACTER_BASED = "character_based"
    PATTERN_MATCHING = "pattern_matching"
    RECURSIVE_FORMATION = "recursive_formation"
    HYBRID_SYNTHESIS = "hybrid_synthesis"

@dataclass
class MathematicalTrait:
    """Mathematical trait with properties"""
    trait_uuid: uuid.UUID
    name: str
    category: TraitCategory
    complexity: SemanticComplexity
    mathematical_properties: Dict[str, Any]
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticTrait:
    """Semantic trait with understanding"""
    trait_uuid: uuid.UUID
    name: str
    category: TraitCategory
    complexity: SemanticComplexity
    semantic_properties: Dict[str, Any]
    mathematical_anchor: Optional[uuid.UUID] = None
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TraitConversion:
    """Trait conversion record"""
    conversion_uuid: uuid.UUID
    source_trait: Union[MathematicalTrait, SemanticTrait]
    target_trait: Union[MathematicalTrait, SemanticTrait]
    conversion_type: ConversionType
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
