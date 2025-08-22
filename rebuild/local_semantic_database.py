# rebuild/local_semantic_database.py
"""
Local Semantic Database - Pre-processed semantic foundation
Stores semantic libraries as mathematical references for recursive typewriter
"""

import uuid
import hashlib
import json
import pickle
import os
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

class SemanticSource(Enum):
    """Sources of semantic data"""
    WORDNET = "wordnet"
    CONCEPTNET = "conceptnet"
    NRC_EMOTION = "nrc_emotion"
    INTERNAL = "internal"
    EVOLVED = "evolved"

class SemanticDataType(Enum):
    """Types of semantic data"""
    WORD = "word"
    CONCEPT = "concept"
    EMOTION = "emotion"
    RELATIONSHIP = "relationship"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"

@dataclass
class SemanticReference:
    """Mathematical reference for semantic data"""
    reference_id: uuid.UUID
    word: str
    source: SemanticSource
    data_type: SemanticDataType
    mathematical_properties: Dict[str, Any]
    semantic_properties: Dict[str, Any]
    convergence_stability: float
    violation_pressure: float
    trait_intensity: float
    relationships: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticRelationship:
    """Mathematical representation of semantic relationships"""
    relationship_id: uuid.UUID
    source_word: str
    target_word: str
    relationship_type: str
    strength: float
    mathematical_properties: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticDatabaseMetrics:
    """Metrics for semantic database performance"""
    total_words: int
    total_concepts: int
    total_emotions: int
    total_relationships: int
    average_convergence_stability: float
    average_violation_pressure: float
    average_trait_intensity: float
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticFoundationCache:
    """Cache for semantic foundation data"""
    references: Dict[str, SemanticReference]
    relationships: Dict[uuid.UUID, SemanticRelationship]
    word_index: Dict[str, List[str]]
    concept_index: Dict[str, List[str]]
    emotion_index: Dict[str, List[str]]
    metrics: SemanticDatabaseMetrics
    cache_timestamp: datetime
    cache_version: str = "1.0"

class LocalSemanticDatabase:
    """
    Local semantic database with pre-processed mathematical references
    Provides semantic foundation for recursive typewriter operations
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
        
        # Database state
        self.semantic_references: Dict[str, SemanticReference] = {}
        self.semantic_relationships: Dict[uuid.UUID, SemanticRelationship] = {}
        self.word_index: Dict[str, List[str]] = defaultdict(list)
        self.concept_index: Dict[str, List[str]] = defaultdict(list)
        self.emotion_index: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.database_metrics = SemanticDatabaseMetrics(
            total_words=0,
            total_concepts=0,
            total_emotions=0,
            total_relationships=0,
            average_convergence_stability=0.0,
            average_violation_pressure=0.0,
            average_trait_intensity=0.0
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache file paths
        self.cache_dir = "semantic_cache"
        self.cache_file = os.path.join(self.cache_dir, "semantic_foundation_cache.pkl")
        self.metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize semantic foundation (with caching)
        self._initialize_semantic_foundation()
        
        # Register with event bridge
        self.event_bridge.register_handler("SEMANTIC_DATABASE_QUERY", self._handle_database_query)
        self.event_bridge.register_handler("SEMANTIC_REFERENCE_UPDATE", self._handle_reference_update)
        
        print(f"📚 LocalSemanticDatabase initialized with {len(self.semantic_references)} references")
    
    def _initialize_semantic_foundation(self):
        """Initialize semantic foundation with caching support"""
        print("🔥 Initializing semantic foundation...")
        
        # Try to load from cache first
        if self._load_semantic_foundation_from_cache():
            print("✅ Semantic foundation loaded from cache")
            return
        
        # If cache doesn't exist or is invalid, build from scratch
        print("🔄 Cache not found or invalid - building semantic foundation from scratch...")
        self._build_semantic_foundation_from_scratch()
        
        # Save to cache for future use
        self._save_semantic_foundation_to_cache()
        print("💾 Semantic foundation saved to cache")
    
    def _load_semantic_foundation_from_cache(self) -> bool:
        """Load semantic foundation from cache file"""
        try:
            if not os.path.exists(self.cache_file):
                print("📁 No cache file found")
                return False
            
            if not os.path.exists(self.metadata_file):
                print("📁 No cache metadata found")
                return False
            
            # Check cache metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is permanent or expired
            if metadata.get('permanent_cache', False):
                print("🔒 Using permanent cache (no expiration)")
            else:
                cache_age = datetime.utcnow() - datetime.fromisoformat(metadata['cache_timestamp'])
                max_age = timedelta(days=30)  # Cache expires after 30 days (much longer for massive loads)
                
                if cache_age > max_age:
                    print(f"⏰ Cache expired (age: {cache_age})")
                    return False
            
            # Load cache data
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restore database state
            self.semantic_references = cache_data.references
            self.semantic_relationships = cache_data.relationships
            self.word_index = cache_data.word_index
            self.concept_index = cache_data.concept_index
            self.emotion_index = cache_data.emotion_index
            self.database_metrics = cache_data.metrics
            
            print(f"📚 Loaded {len(self.semantic_references)} references from cache")
            print(f"🔗 Loaded {len(self.semantic_relationships)} relationships from cache")
            print(f"📊 Cache age: {cache_age}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return False
    
    def _save_semantic_foundation_to_cache(self):
        """Save semantic foundation to cache file"""
        try:
            # Create cache data
            cache_data = SemanticFoundationCache(
                references=self.semantic_references,
                relationships=self.semantic_relationships,
                word_index=self.word_index,
                concept_index=self.concept_index,
                emotion_index=self.emotion_index,
                metrics=self.database_metrics,
                cache_timestamp=datetime.utcnow()
            )
            
            # Save cache data
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save metadata
            metadata = {
                'cache_timestamp': cache_data.cache_timestamp.isoformat(),
                'cache_version': cache_data.cache_version,
                'reference_count': len(self.semantic_references),
                'relationship_count': len(self.semantic_relationships),
                'total_words': self.database_metrics.total_words,
                'total_concepts': self.database_metrics.total_concepts,
                'total_emotions': self.database_metrics.total_emotions,
                'total_relationships': self.database_metrics.total_relationships
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"💾 Cache saved: {len(self.semantic_references)} references, {len(self.semantic_relationships)} relationships")
            
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
    
    def _build_semantic_foundation_from_scratch(self):
        """Build semantic foundation from scratch using massive integration"""
        print("🚀 Building MASSIVE semantic foundation from scratch...")
        
        # Use the REAL massive semantic library integration
        from semantic_library_integration import SemanticLibraryIntegration, SemanticLibraryType
        
        # Create the massive integration system
        massive_integration = SemanticLibraryIntegration(
            self.state_manager, self.event_bridge, self.violation_monitor,
            self.checkpoint_manager, self.uuid_anchor, self.trait_convergence
        )
        
        # Start MASSIVE integration with ALL libraries
        print("🚀 Starting MASSIVE semantic library integration...")
        integration_result = massive_integration.start_massive_integration([
            SemanticLibraryType.WORDNET,
            SemanticLibraryType.NRC_EMOTION,
            SemanticLibraryType.CONCEPTNET,
            SemanticLibraryType.TEXTBLOB
        ], test_mode=False)  # Use full integration for cache
        
        # Convert the massive integration results to our database format
        self.semantic_references = self._convert_massive_integration_to_references(massive_integration)
        self.semantic_relationships = self._build_mathematical_reference_database()
        
        # Update metrics
        self._update_database_metrics()
        
        print(f"✅ MASSIVE semantic foundation built: {len(self.semantic_references)} words, {len(self.semantic_relationships)} relationships")
        print(f"🔥 Integration result: {integration_result.get('total_traits_created', 0)} traits created")
    
    def clear_cache(self):
        """Clear the semantic foundation cache"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                print("🗑️ Cache file removed")
            
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
                print("🗑️ Cache metadata removed")
            
            print("✅ Semantic foundation cache cleared")
            
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
    
    def make_cache_permanent(self):
        """Make the current cache permanent (no expiration)"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Set a very far future expiration (100 years)
                metadata['permanent_cache'] = True
                metadata['cache_expires'] = '2124-01-01T00:00:00'
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print("🔒 Cache marked as permanent (no expiration)")
            else:
                print("⚠️ No cache metadata found to make permanent")
                
        except Exception as e:
            print(f"❌ Failed to make cache permanent: {e}")
    
    def force_rebuild_foundation(self):
        """Force rebuild of semantic foundation from scratch"""
        print("🔄 Force rebuilding semantic foundation...")
        
        # Clear cache
        self.clear_cache()
        
        # Rebuild from scratch
        self._build_semantic_foundation_from_scratch()
        
        # Save new cache
        self._save_semantic_foundation_to_cache()
        
        print("✅ Semantic foundation force rebuilt and cached")
    
    def _convert_massive_integration_to_references(self, massive_integration) -> Dict[str, SemanticReference]:
        """Convert massive integration results to semantic references"""
        references = {}
        
        print(f"🔄 Converting ALL {len(massive_integration.processed_traits)} processed semantic traits to references...")
        
        # Convert ALL processed traits (this is where the 147,306 traits are!)
        trait_count = 0
        for trait_id, trait in massive_integration.processed_traits.items():
            # Convert SemanticTrait to SemanticReference
            reference = SemanticReference(
                reference_id=trait.trait_uuid,
                word=trait.name,
                source=self._get_source_from_trait(trait),
                data_type=self._get_data_type_from_trait(trait),
                mathematical_properties={
                    'convergence_stability': trait.convergence_stability,
                    'violation_pressure': trait.violation_pressure,
                    'trait_intensity': trait.trait_intensity,
                    'complexity': trait.complexity.value,
                    'category': trait.category.value
                },
                semantic_properties=trait.semantic_properties,
                convergence_stability=trait.convergence_stability,
                violation_pressure=trait.violation_pressure,
                trait_intensity=trait.trait_intensity
            )
            references[trait_id] = reference
            trait_count += 1
            
            # Progress indicator for large datasets
            if trait_count % 10000 == 0:
                print(f"   Processed {trait_count} traits...")
        
        print(f"✅ Converted ALL {len(references)} semantic references from processed traits!")
        print(f"   Total traits processed: {trait_count}")
        
        return references
    
    def _get_source_from_trait(self, trait) -> SemanticSource:
        """Get semantic source from trait properties"""
        semantic_props = trait.semantic_properties
        
        # Check for source indicators in semantic properties
        if 'definitions' in semantic_props or 'synsets' in semantic_props:
            return SemanticSource.WORDNET
        elif 'emotion_scores' in semantic_props or 'valence' in semantic_props:
            return SemanticSource.NRC_EMOTION
        elif 'relationship_type' in semantic_props or 'source_concept' in semantic_props:
            return SemanticSource.CONCEPTNET
        elif 'polarity' in semantic_props or 'subjectivity' in semantic_props:
            return SemanticSource.INTERNAL  # TextBlob
        else:
            return SemanticSource.INTERNAL
    
    def _get_data_type_from_trait(self, trait) -> SemanticDataType:
        """Get data type from trait properties"""
        semantic_props = trait.semantic_properties
        
        if 'emotion_scores' in semantic_props:
            return SemanticDataType.EMOTION
        elif 'relationship_type' in semantic_props:
            return SemanticDataType.RELATIONSHIP
        elif 'definitions' in semantic_props:
            return SemanticDataType.WORD
        else:
            return SemanticDataType.WORD
    
    def _build_semantic_trait_database(self) -> Dict[str, SemanticReference]:
        """Build semantic trait database from core vocabulary"""
        semantic_database = {}
        
        # Core vocabulary words with mathematical properties
        core_words = [
            # Basic concepts
            "love", "hate", "joy", "sadness", "anger", "fear", "surprise", "trust",
            "good", "bad", "big", "small", "fast", "slow", "hot", "cold",
            "light", "dark", "strong", "weak", "rich", "poor", "happy", "sad",
            
            # Abstract concepts
            "truth", "beauty", "justice", "freedom", "peace", "war", "life", "death",
            "time", "space", "energy", "matter", "mind", "body", "soul", "spirit",
            "knowledge", "wisdom", "understanding", "learning", "growth", "change",
            
            # Mathematical concepts
            "number", "quantity", "measure", "calculate", "compute", "solve",
            "pattern", "sequence", "order", "chaos", "random", "determined",
            "infinite", "finite", "zero", "one", "many", "all", "none",
            
            # Linguistic concepts
            "word", "language", "meaning", "communication", "expression", "speech",
            "write", "read", "speak", "listen", "understand", "explain", "describe",
            "question", "answer", "statement", "command", "request", "promise",
            
            # Emotional concepts
            "emotion", "feeling", "mood", "passion", "desire", "hope", "dream",
            "worry", "anxiety", "confidence", "courage", "pride", "shame", "guilt",
            "gratitude", "forgiveness", "compassion", "empathy", "sympathy",
            
            # Social concepts
            "friend", "family", "community", "society", "culture", "tradition",
            "relationship", "connection", "bond", "trust", "loyalty", "betrayal",
            "cooperation", "competition", "conflict", "resolution", "agreement",
            
            # Action concepts
            "create", "destroy", "build", "break", "move", "stop", "start", "end",
            "begin", "finish", "continue", "pause", "wait", "hurry", "rest",
            "work", "play", "study", "teach", "learn", "practice", "improve",
            
            # Physical concepts
            "body", "mind", "heart", "brain", "hand", "eye", "ear", "mouth",
            "walk", "run", "jump", "fly", "swim", "climb", "fall", "rise",
            "touch", "see", "hear", "smell", "taste", "feel", "sense"
        ]
        
        for word in core_words:
            # Convert word to mathematical semantic reference
            reference = self._convert_word_to_semantic_reference(word)
            semantic_database[word] = reference
            
            # Index by type
            if reference.data_type == SemanticDataType.WORD:
                self.word_index[word.lower()].append(word)
            elif reference.data_type == SemanticDataType.CONCEPT:
                self.concept_index[word.lower()].append(word)
            elif reference.data_type == SemanticDataType.EMOTION:
                self.emotion_index[word.lower()].append(word)
        
        return semantic_database
    
    def _convert_word_to_semantic_reference(self, word: str) -> SemanticReference:
        """Convert word to mathematical semantic reference"""
        # Generate mathematical properties from word characteristics
        word_hash = hash(word.lower())
        word_length = len(word)
        
        # Calculate mathematical properties
        mathematical_properties = {
            "length": word_length,
            "complexity": min(1.0, word_length / 12.0),
            "frequency": self._calculate_word_frequency(word),
            "emotional_intensity": self._calculate_emotional_intensity(word),
            "abstractness": self._calculate_abstractness(word),
            "concreteness": self._calculate_concreteness(word)
        }
        
        # Determine semantic properties
        semantic_properties = {
            "pos": self._determine_part_of_speech(word),
            "category": self._determine_semantic_category(word),
            "valence": self._calculate_semantic_valence(word),
            "arousal": self._calculate_semantic_arousal(word),
            "dominance": self._calculate_semantic_dominance(word)
        }
        
        # Calculate trait properties
        convergence_stability = self._calculate_convergence_stability(word, mathematical_properties)
        violation_pressure = self._calculate_violation_pressure(word, mathematical_properties)
        trait_intensity = self._calculate_trait_intensity(word, mathematical_properties)
        
        # Determine data type
        data_type = self._determine_data_type(word, semantic_properties)
        
        # Determine source (simulated from core vocabulary)
        source = SemanticSource.INTERNAL
        
        return SemanticReference(
            reference_id=uuid.uuid4(),
            word=word,
            source=source,
            data_type=data_type,
            mathematical_properties=mathematical_properties,
            semantic_properties=semantic_properties,
            convergence_stability=convergence_stability,
            violation_pressure=violation_pressure,
            trait_intensity=trait_intensity,
            relationships=[]
        )
    
    def _calculate_word_frequency(self, word: str) -> float:
        """Calculate word frequency (simulated)"""
        # Simulate word frequency based on word characteristics
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        if word.lower() in common_words:
            return 0.9
        elif len(word) <= 3:
            return 0.7
        elif len(word) <= 5:
            return 0.5
        else:
            return 0.3
    
    def _calculate_emotional_intensity(self, word: str) -> float:
        """Calculate emotional intensity of word"""
        emotional_words = {
            "love": 0.9, "hate": 0.9, "joy": 0.8, "sadness": 0.8, "anger": 0.8,
            "fear": 0.8, "surprise": 0.7, "trust": 0.7, "gratitude": 0.7,
            "compassion": 0.7, "empathy": 0.7, "pride": 0.6, "shame": 0.6,
            "guilt": 0.6, "hope": 0.6, "worry": 0.6, "anxiety": 0.6
        }
        return emotional_words.get(word.lower(), 0.3)
    
    def _calculate_abstractness(self, word: str) -> float:
        """Calculate abstractness of word"""
        abstract_words = {
            "truth", "beauty", "justice", "freedom", "peace", "war", "life", "death",
            "time", "space", "energy", "matter", "mind", "soul", "spirit",
            "knowledge", "wisdom", "understanding", "learning", "growth", "change"
        }
        return 0.8 if word.lower() in abstract_words else 0.3
    
    def _calculate_concreteness(self, word: str) -> float:
        """Calculate concreteness of word"""
        concrete_words = {
            "hand", "eye", "ear", "mouth", "body", "brain", "heart",
            "walk", "run", "jump", "fly", "swim", "climb", "fall", "rise",
            "touch", "see", "hear", "smell", "taste", "feel"
        }
        return 0.8 if word.lower() in concrete_words else 0.3
    
    def _determine_part_of_speech(self, word: str) -> str:
        """Determine part of speech (simplified)"""
        # Simple POS detection based on word patterns
        if word.endswith(('ing', 'ed', 'er', 'est')):
            return "verb" if word.endswith(('ing', 'ed')) else "adjective"
        elif word.endswith(('ly')):
            return "adverb"
        elif word.endswith(('tion', 'sion', 'ness', 'ment')):
            return "noun"
        elif word in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
            return "determiner"
        elif word in ['and', 'or', 'but', 'because', 'if', 'when']:
            return "conjunction"
        else:
            return "noun"  # Default to noun
    
    def _determine_semantic_category(self, word: str) -> str:
        """Determine semantic category"""
        categories = {
            "emotion": ["love", "hate", "joy", "sadness", "anger", "fear", "surprise", "trust"],
            "physical": ["body", "hand", "eye", "ear", "mouth", "brain", "heart"],
            "action": ["walk", "run", "jump", "fly", "swim", "climb", "fall", "rise"],
            "abstract": ["truth", "beauty", "justice", "freedom", "peace", "war", "life", "death"],
            "mathematical": ["number", "quantity", "measure", "calculate", "compute", "solve"],
            "linguistic": ["word", "language", "meaning", "communication", "expression"]
        }
        
        for category, words in categories.items():
            if word.lower() in words:
                return category
        return "general"
    
    def _calculate_semantic_valence(self, word: str) -> float:
        """Calculate semantic valence (positive/negative)"""
        positive_words = {"love", "joy", "good", "happy", "beautiful", "peace", "freedom", "truth", "wisdom"}
        negative_words = {"hate", "sadness", "anger", "fear", "bad", "sad", "ugly", "war", "death", "lies"}
        
        if word.lower() in positive_words:
            return 0.8
        elif word.lower() in negative_words:
            return 0.2
        else:
            return 0.5  # Neutral
    
    def _calculate_semantic_arousal(self, word: str) -> float:
        """Calculate semantic arousal (calm/excited)"""
        high_arousal = {"anger", "fear", "surprise", "excitement", "passion", "energy"}
        low_arousal = {"sadness", "peace", "calm", "rest", "sleep", "quiet"}
        
        if word.lower() in high_arousal:
            return 0.8
        elif word.lower() in low_arousal:
            return 0.2
        else:
            return 0.5  # Moderate
    
    def _calculate_semantic_dominance(self, word: str) -> float:
        """Calculate semantic dominance (submissive/dominant)"""
        dominant_words = {"power", "strength", "control", "authority", "lead", "command"}
        submissive_words = {"weak", "follow", "obey", "submit", "surrender", "yield"}
        
        if word.lower() in dominant_words:
            return 0.8
        elif word.lower() in submissive_words:
            return 0.2
        else:
            return 0.5  # Balanced
    
    def _calculate_convergence_stability(self, word: str, properties: Dict[str, Any]) -> float:
        """Calculate convergence stability for word"""
        # Base stability on word properties
        frequency = properties.get("frequency", 0.5)
        complexity = properties.get("complexity", 0.5)
        emotional_intensity = properties.get("emotional_intensity", 0.3)
        
        # More frequent, less complex words are more stable
        stability = (frequency * 0.4) + ((1.0 - complexity) * 0.3) + ((1.0 - emotional_intensity) * 0.3)
        return max(0.1, min(0.9, stability))
    
    def _calculate_violation_pressure(self, word: str, properties: Dict[str, Any]) -> float:
        """Calculate violation pressure for word"""
        # More complex, emotional words have higher VP
        complexity = properties.get("complexity", 0.5)
        emotional_intensity = properties.get("emotional_intensity", 0.3)
        abstractness = properties.get("abstractness", 0.3)
        
        vp = (complexity * 0.3) + (emotional_intensity * 0.4) + (abstractness * 0.3)
        return max(0.1, min(0.9, vp))
    
    def _calculate_trait_intensity(self, word: str, properties: Dict[str, Any]) -> float:
        """Calculate trait intensity for word"""
        # Emotional and abstract words have higher intensity
        emotional_intensity = properties.get("emotional_intensity", 0.3)
        abstractness = properties.get("abstractness", 0.3)
        complexity = properties.get("complexity", 0.5)
        
        intensity = (emotional_intensity * 0.4) + (abstractness * 0.3) + (complexity * 0.3)
        return max(0.1, min(0.9, intensity))
    
    def _determine_data_type(self, word: str, semantic_properties: Dict[str, Any]) -> SemanticDataType:
        """Determine semantic data type"""
        category = semantic_properties.get("category", "general")
        
        if category == "emotion":
            return SemanticDataType.EMOTION
        elif category in ["abstract", "mathematical", "linguistic"]:
            return SemanticDataType.CONCEPT
        else:
            return SemanticDataType.WORD
    
    def _build_mathematical_reference_database(self) -> Dict[uuid.UUID, SemanticRelationship]:
        """Build mathematical reference database for semantic relationships"""
        relationships = {}
        
        # Define core semantic relationships
        relationship_definitions = [
            # Synonyms
            ("love", "adore", "synonym", 0.8),
            ("hate", "despise", "synonym", 0.8),
            ("joy", "happiness", "synonym", 0.8),
            ("sadness", "sorrow", "synonym", 0.8),
            
            # Antonyms
            ("love", "hate", "antonym", 0.9),
            ("joy", "sadness", "antonym", 0.9),
            ("good", "bad", "antonym", 0.9),
            ("big", "small", "antonym", 0.9),
            
            # Hypernyms
            ("love", "emotion", "hypernym", 0.7),
            ("hate", "emotion", "hypernym", 0.7),
            ("joy", "emotion", "hypernym", 0.7),
            ("sadness", "emotion", "hypernym", 0.7),
            
            # Hyponyms
            ("emotion", "love", "hyponym", 0.7),
            ("emotion", "hate", "hyponym", 0.7),
            ("emotion", "joy", "hyponym", 0.7),
            ("emotion", "sadness", "hyponym", 0.7),
            
            # Related concepts
            ("love", "compassion", "related", 0.6),
            ("hate", "anger", "related", 0.6),
            ("joy", "happiness", "related", 0.8),
            ("sadness", "sorrow", "related", 0.8),
        ]
        
        for source_word, target_word, rel_type, strength in relationship_definitions:
            if source_word in self.semantic_references and target_word in self.semantic_references:
                relationship = SemanticRelationship(
                    relationship_id=uuid.uuid4(),
                    source_word=source_word,
                    target_word=target_word,
                    relationship_type=rel_type,
                    strength=strength,
                    mathematical_properties={
                        "relationship_strength": strength,
                        "bidirectional": rel_type in ["synonym", "antonym"],
                        "hierarchical": rel_type in ["hypernym", "hyponym"]
                    }
                )
                relationships[relationship.relationship_id] = relationship
                
                # Add to word relationships
                self.semantic_references[source_word].relationships.append(target_word)
        
        return relationships
    
    def _update_database_metrics(self):
        """Update database performance metrics"""
        if not self.semantic_references:
            return
        
        # Calculate averages
        convergence_stabilities = [ref.convergence_stability for ref in self.semantic_references.values()]
        violation_pressures = [ref.violation_pressure for ref in self.semantic_references.values()]
        trait_intensities = [ref.trait_intensity for ref in self.semantic_references.values()]
        
        # Count by type
        word_count = len([ref for ref in self.semantic_references.values() if ref.data_type == SemanticDataType.WORD])
        concept_count = len([ref for ref in self.semantic_references.values() if ref.data_type == SemanticDataType.CONCEPT])
        emotion_count = len([ref for ref in self.semantic_references.values() if ref.data_type == SemanticDataType.EMOTION])
        
        self.database_metrics = SemanticDatabaseMetrics(
            total_words=word_count,
            total_concepts=concept_count,
            total_emotions=emotion_count,
            total_relationships=len(self.semantic_relationships),
            average_convergence_stability=statistics.mean(convergence_stabilities),
            average_violation_pressure=statistics.mean(violation_pressures),
            average_trait_intensity=statistics.mean(trait_intensities)
        )
    
    def get_semantic_data(self, word: str) -> Optional[SemanticReference]:
        """Get semantic data for word"""
        with self._lock:
            reference = self.semantic_references.get(word.lower())
            if reference:
                reference.last_accessed = datetime.utcnow()
            return reference
    
    def get_semantic_relationships(self, word: str) -> List[SemanticRelationship]:
        """Get semantic relationships for word"""
        with self._lock:
            relationships = []
            for rel in self.semantic_relationships.values():
                if rel.source_word.lower() == word.lower() or rel.target_word.lower() == word.lower():
                    relationships.append(rel)
            return relationships
    
    def search_semantic_references(self, query: str, data_type: Optional[SemanticDataType] = None) -> List[SemanticReference]:
        """Search semantic references"""
        with self._lock:
            results = []
            for reference in self.semantic_references.values():
                if data_type and reference.data_type != data_type:
                    continue
                if query.lower() in reference.word.lower():
                    results.append(reference)
            return results
    
    def get_database_metrics(self) -> SemanticDatabaseMetrics:
        """Get database performance metrics"""
        with self._lock:
            return self.database_metrics
    
    def _handle_database_query(self, event_data: Dict[str, Any]):
        """Handle semantic database query events"""
        try:
            query_type = event_data.get('query_type')
            word = event_data.get('word')
            
            if query_type == 'get_semantic_data':
                result = self.get_semantic_data(word)
                # Publish result event
                self.event_bridge.publish_semantic_event({
                    'event_type': 'SEMANTIC_DATA_RETRIEVED',
                    'word': word,
                    'data': asdict(result) if result else None
                })
            
            elif query_type == 'get_relationships':
                relationships = self.get_semantic_relationships(word)
                # Publish result event
                self.event_bridge.publish_semantic_event({
                    'event_type': 'SEMANTIC_RELATIONSHIPS_RETRIEVED',
                    'word': word,
                    'relationships': [asdict(rel) for rel in relationships]
                })
                
        except Exception as e:
            print(f"Error handling database query: {e}")
    
    def _handle_reference_update(self, event_data: Dict[str, Any]):
        """Handle semantic reference update events"""
        try:
            word = event_data.get('word')
            updates = event_data.get('updates', {})
            
            if word in self.semantic_references:
                reference = self.semantic_references[word]
                
                # Update reference properties
                for key, value in updates.items():
                    if hasattr(reference, key):
                        setattr(reference, key, value)
                
                # Update metrics
                self._update_database_metrics()
                
        except Exception as e:
            print(f"Error handling reference update: {e}")
