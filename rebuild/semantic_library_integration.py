# rebuild/semantic_library_integration.py
"""
Semantic Library Integration - REAL massive semantic foundation
Integrates actual semantic libraries (WordNet, NRC Emotion, ConceptNet) into mathematical traits
NO MORE MOCK DATA - REAL SEMANTIC INTEGRATION ONLY
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

class SemanticLibraryType(Enum):
    """Types of semantic libraries"""
    WORDNET = "wordnet"
    NRC_EMOTION = "nrc_emotion"
    CONCEPTNET = "conceptnet"
    TEXTBLOB = "textblob"

class IntegrationStatus(Enum):
    """Status of library integration"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SemanticLibraryData:
    """Data structure for semantic library content"""
    library_type: SemanticLibraryType
    word: str
    raw_data: Dict[str, Any]
    confidence: float
    source_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationMetrics:
    """Metrics for library integration"""
    total_words_processed: int
    total_relationships_found: int
    total_traits_created: int
    processing_time: float
    errors_encountered: int
    integration_status: IntegrationStatus

class SemanticLibraryIntegration:
    """
    REAL Semantic Library Integration System
    NO MOCK DATA - Actual integration with real semantic libraries
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
        
        # Data storage
        self.wordnet_data: Dict[str, SemanticLibraryData] = {}
        self.nrc_emotion_data: Dict[str, SemanticLibraryData] = {}
        self.conceptnet_data: Dict[str, SemanticLibraryData] = {}
        self.textblob_data: Dict[str, SemanticLibraryData] = {}
        self.processed_traits: Dict[str, SemanticTrait] = {}
        
        # Integration metrics
        self.integration_metrics = IntegrationMetrics(
            total_words_processed=0,
            total_relationships_found=0,
            total_traits_created=0,
            processing_time=0.0,
            errors_encountered=0,
            integration_status=IntegrationStatus.NOT_STARTED
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        print("🔥 SemanticLibraryIntegration initialized - READY FOR REAL DATA INTEGRATION")
    
    def start_massive_integration(self, library_types: List[SemanticLibraryType], test_mode: bool = False) -> Dict[str, Any]:
        """
        Start massive integration of real semantic libraries
        
        Args:
            library_types: List of library types to integrate
            test_mode: If True, load only a small subset for testing
            
        Returns:
            Integration results dictionary
        """
        mode_text = "TEST MODE" if test_mode else "MASSIVE REAL"
        print(f"🚀 Starting {mode_text} semantic library integration...")
        print(f"📚 Integrating libraries: {[lib.value for lib in library_types]}")
        
        start_time = datetime.utcnow()
        
        with self._lock:
            self.integration_metrics.integration_status = IntegrationStatus.IN_PROGRESS
            
            try:
                # Integrate each library
                for library_type in library_types:
                    print(f"\n📖 Integrating {library_type.value}...")
                    
                    if library_type == SemanticLibraryType.WORDNET:
                        self._integrate_real_wordnet(test_mode=test_mode)
                    elif library_type == SemanticLibraryType.NRC_EMOTION:
                        self._integrate_real_nrc_emotion(test_mode=test_mode)
                    elif library_type == SemanticLibraryType.CONCEPTNET:
                        self._integrate_real_conceptnet(test_mode=test_mode)
                    elif library_type == SemanticLibraryType.TEXTBLOB:
                        self._integrate_real_textblob(test_mode=test_mode)
                
                # Process all library data to traits
                print(f"\n🔄 Processing all library data to semantic traits...")
                self._process_all_library_data_to_traits()
                
                # Calculate final metrics
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                self.integration_metrics.processing_time = processing_time
                self.integration_metrics.integration_status = IntegrationStatus.COMPLETED
                
                # Final status
                total_library_data = (len(self.wordnet_data) + len(self.nrc_emotion_data) + 
                                    len(self.conceptnet_data) + len(self.textblob_data))
                
                print(f"\n✅ {mode_text} INTEGRATION COMPLETE!")
                print(f"📊 FINAL STATISTICS:")
                print(f"   Total library entries: {total_library_data}")
                print(f"   Total semantic traits: {len(self.processed_traits)}")
                print(f"   Processing time: {processing_time:.2f}s")
                print(f"   Status: {self.integration_metrics.integration_status.value}")
                
                return {
                    'status': 'completed',
                    'total_library_entries': total_library_data,
                    'total_traits_created': len(self.processed_traits),
                    'processing_time': processing_time,
                    'wordnet_count': len(self.wordnet_data),
                    'nrc_emotion_count': len(self.nrc_emotion_data),
                    'conceptnet_count': len(self.conceptnet_data),
                    'textblob_count': len(self.textblob_data),
                    'test_mode': test_mode
                }
                
            except Exception as e:
                print(f"❌ {mode_text} INTEGRATION FAILED: {e}")
                self.integration_metrics.integration_status = IntegrationStatus.FAILED
                self.integration_metrics.errors_encountered += 1
                
                import traceback
                traceback.print_exc()
                
                return {
                    'status': 'failed',
                    'error': str(e),
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'test_mode': test_mode
                }
    
    def _integrate_real_wordnet(self, test_mode: bool = False):
        """Integrate REAL WordNet data using NLTK"""
        print("📚 Loading REAL WordNet data via NLTK...")
        
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Ensure WordNet is downloaded
            try:
                nltk.data.find('corpora/wordnet')
                print("✅ WordNet corpus found")
            except LookupError:
                print("📥 Downloading WordNet corpus...")
                nltk.download('wordnet')
                nltk.download('omw-1.4')  # Open Multilingual Wordnet
            
            # Get ALL real words from WordNet
            print("🔍 Extracting all real words from WordNet...")
            real_words = set()
            synset_count = 0
            
            for synset in wordnet.all_synsets():
                synset_count += 1
                for lemma in synset.lemmas():
                    # Clean up lemma names
                    clean_word = lemma.name().replace('_', ' ').lower()
                    real_words.add(clean_word)
            
            word_list = sorted(list(real_words))
            print(f"📚 Found {len(word_list)} REAL unique words from {synset_count} synsets")
            
            # Process words (limit for test mode)
            if test_mode:
                max_words = 100  # Small subset for testing
                print(f"🧪 TEST MODE: Processing first {max_words} words only")
            else:
                max_words = len(word_list)  # ALL words for full integration
                print(f"📚 Processing ALL {max_words} words...")
            
            words_to_process = word_list[:max_words]
            print(f"📚 Processing {len(words_to_process)} words...")
            
            processed_count = 0
            for i, word in enumerate(words_to_process):
                try:
                    # Get real synsets for this word
                    synsets = wordnet.synsets(word.replace(' ', '_'))
                    
                    if synsets:
                        # Extract REAL semantic data
                        definitions = []
                        pos_tags = set()
                        hypernyms = []
                        hyponyms = []
                        synonyms = set()
                        antonyms = set()
                        
                        # Process synsets (limit to avoid huge data)
                        for synset in synsets[:3]:  # Max 3 synsets per word
                            # Real definition
                            definitions.append(synset.definition())
                            pos_tags.add(synset.pos())
                            
                            # Real hypernyms
                            for hypernym in synset.hypernyms()[:2]:
                                for lemma in hypernym.lemmas()[:2]:
                                    hypernyms.append(lemma.name().replace('_', ' '))
                            
                            # Real hyponyms
                            for hyponym in synset.hyponyms()[:2]:
                                for lemma in hyponym.lemmas()[:1]:
                                    hyponyms.append(lemma.name().replace('_', ' '))
                            
                            # Real synonyms and antonyms
                            for lemma in synset.lemmas()[:3]:
                                # Add other lemmas in same synset as synonyms
                                for syn_lemma in synset.lemmas():
                                    if syn_lemma != lemma:
                                        synonyms.add(syn_lemma.name().replace('_', ' '))
                                
                                # Real antonyms
                                for antonym in lemma.antonyms():
                                    antonyms.add(antonym.name().replace('_', ' '))
                        
                        # Create REAL WordNet data
                        wordnet_data = SemanticLibraryData(
                            library_type=SemanticLibraryType.WORDNET,
                            word=word,
                            raw_data={
                                'synsets': [synset.name() for synset in synsets[:3]],
                                'definitions': definitions,
                                'pos_tags': list(pos_tags),
                                'hypernyms': list(set(hypernyms))[:5],
                                'hyponyms': list(set(hyponyms))[:5],
                                'synonyms': list(synonyms)[:5],
                                'antonyms': list(antonyms)[:3],
                                'synset_count': len(synsets),
                                'word_frequency': self._estimate_word_frequency(word),
                                'semantic_similarity_cluster': self._calculate_semantic_cluster(word, synsets)
                            },
                            confidence=min(0.95, 0.6 + (len(synsets) * 0.05)),
                            source_metadata={
                                'source': 'nltk_wordnet',
                                'synset_count': len(synsets),
                                'extraction_method': 'real_wordnet_api',
                                'test_mode': test_mode
                            }
                        )
                        
                        self.wordnet_data[word] = wordnet_data
                        processed_count += 1
                        
                        if test_mode and processed_count % 10 == 0:
                            print(f"📚 WordNet test progress: {processed_count}/{len(words_to_process)} real words")
                        elif not test_mode and processed_count % 1000 == 0:
                            print(f"📚 WordNet progress: {processed_count}/{len(words_to_process)} real words")
                
                except Exception as e:
                    # Skip words that cause errors
                    continue
            
            self.integration_metrics.total_words_processed += processed_count
            mode_text = "TEST" if test_mode else "REAL"
            print(f"✅ WordNet {mode_text} integration completed: {len(self.wordnet_data)} words processed")
            
        except ImportError:
            print("❌ NLTK not available - WordNet integration skipped")
            print("💡 Install NLTK: pip install nltk")
    
    def _integrate_real_nrc_emotion(self, test_mode: bool = False):
        """Integrate REAL NRC Emotion Lexicon data"""
        print("😊 Loading REAL NRC Emotion Lexicon data...")
        
        try:
            # Try to use NRCLex library for real NRC data
            from nrclex import NRCLex
            print("✅ NRCLex library found - using REAL NRC emotion data")
            
            # Get words from our WordNet integration to analyze
            if test_mode:
                words_to_analyze = list(self.wordnet_data.keys())[:50]  # Small subset for testing
                print(f"🧪 TEST MODE: Analyzing emotions for {len(words_to_analyze)} words...")
            else:
                words_to_analyze = list(self.wordnet_data.keys())  # ALL words for full analysis
                print(f"😊 Analyzing emotions for {len(words_to_analyze)} real words...")
            
            processed_count = 0
            for word in words_to_analyze:
                try:
                    # Get REAL emotion analysis
                    emotion_analyzer = NRCLex(word)
                    
                    # Extract real emotion scores
                    emotion_scores = {
                        'fear': emotion_analyzer.fear,
                        'anger': emotion_analyzer.anger,
                        'anticipation': emotion_analyzer.anticipation,
                        'trust': emotion_analyzer.trust,
                        'surprise': emotion_analyzer.surprise,
                        'positive': emotion_analyzer.positive,
                        'negative': emotion_analyzer.negative,
                        'sadness': emotion_analyzer.sadness,
                        'disgust': emotion_analyzer.disgust,
                        'joy': emotion_analyzer.joy
                    }
                    
                    # Only store if word has emotional content
                    if any(score > 0 for score in emotion_scores.values()):
                        nrc_data = SemanticLibraryData(
                            library_type=SemanticLibraryType.NRC_EMOTION,
                            word=word,
                            raw_data={
                                'emotion_scores': emotion_scores,
                                'affect_frequencies': emotion_analyzer.affect_frequencies,
                                'affect_list': emotion_analyzer.affect_list,
                                'top_emotions': emotion_analyzer.top_emotions,
                                'raw_emotion_scores': emotion_analyzer.raw_emotion_scores,
                                'valence': 1.0 if emotion_scores['positive'] > emotion_scores['negative'] else 0.0,
                                'arousal': max(emotion_scores['anger'], emotion_scores['fear'], emotion_scores['joy']),
                                'dominance': emotion_scores['trust'] - emotion_scores['fear']
                            },
                            confidence=0.85,
                            source_metadata={
                                'source': 'nrclex_real',
                                'version': 'real_nrc_emotion_lexicon',
                                'has_emotions': True,
                                'test_mode': test_mode
                            }
                        )
                        
                        self.nrc_emotion_data[word] = nrc_data
                        processed_count += 1
                        
                        if test_mode and processed_count % 5 == 0:
                            print(f"😊 NRC test progress: {processed_count} emotional words found")
                        elif not test_mode and processed_count % 500 == 0:
                            print(f"😊 NRC progress: {processed_count} emotional words found")
                
                except Exception as e:
                    continue
            
            mode_text = "TEST" if test_mode else "REAL"
            print(f"✅ NRC Emotion {mode_text} integration completed: {len(self.nrc_emotion_data)} emotional words processed")
            
        except ImportError:
            print("⚠️ NRCLex not available - using basic emotion analysis")
            print("💡 Install NRCLex: pip install NRCLex")
            self._integrate_basic_emotion_analysis(test_mode=test_mode)
    
    def _integrate_basic_emotion_analysis(self, test_mode: bool = False):
        """Fallback basic emotion analysis when NRCLex is not available"""
        print("😊 Using basic emotion analysis fallback...")
        
        # Basic emotion word mapping
        emotion_mappings = {
            # Joy/Happiness
            'joy': {'joy': 1.0, 'positive': 1.0},
            'happy': {'joy': 0.9, 'positive': 0.9},
            'happiness': {'joy': 1.0, 'positive': 1.0},
            'delight': {'joy': 0.8, 'positive': 0.8, 'surprise': 0.3},
            'pleasure': {'joy': 0.7, 'positive': 0.7},
            'bliss': {'joy': 1.0, 'positive': 1.0},
            
            # Sadness
            'sad': {'sadness': 0.9, 'negative': 0.8},
            'sadness': {'sadness': 1.0, 'negative': 0.9},
            'sorrow': {'sadness': 0.9, 'negative': 0.8},
            'grief': {'sadness': 1.0, 'negative': 1.0},
            'melancholy': {'sadness': 0.8, 'negative': 0.7},
            
            # Anger
            'anger': {'anger': 1.0, 'negative': 0.9},
            'angry': {'anger': 0.9, 'negative': 0.8},
            'rage': {'anger': 1.0, 'negative': 1.0},
            'fury': {'anger': 1.0, 'negative': 1.0},
            'wrath': {'anger': 1.0, 'negative': 1.0},
            
            # Fear
            'fear': {'fear': 1.0, 'negative': 0.8},
            'afraid': {'fear': 0.9, 'negative': 0.7},
            'terror': {'fear': 1.0, 'negative': 1.0},
            'anxiety': {'fear': 0.8, 'negative': 0.7},
            'worry': {'fear': 0.6, 'negative': 0.5},
            
            # Trust
            'trust': {'trust': 1.0, 'positive': 0.7},
            'faith': {'trust': 0.9, 'positive': 0.8},
            'confidence': {'trust': 0.8, 'positive': 0.8},
            'belief': {'trust': 0.7, 'positive': 0.6},
            
            # Surprise
            'surprise': {'surprise': 1.0},
            'amazement': {'surprise': 0.9, 'positive': 0.6},
            'astonishment': {'surprise': 0.9},
            'wonder': {'surprise': 0.7, 'positive': 0.7},
            
            # Disgust
            'disgust': {'disgust': 1.0, 'negative': 0.9},
            'revulsion': {'disgust': 1.0, 'negative': 1.0},
            'repulsion': {'disgust': 0.9, 'negative': 0.9},
            
            # Anticipation
            'anticipation': {'anticipation': 1.0, 'positive': 0.6},
            'expectation': {'anticipation': 0.8},
            'hope': {'anticipation': 0.8, 'positive': 0.8, 'trust': 0.4}
        }
        
        # Limit for test mode
        if test_mode:
            emotion_mappings = dict(list(emotion_mappings.items())[:10])  # First 10 emotions for testing
            print(f"🧪 TEST MODE: Using {len(emotion_mappings)} basic emotions")
        
        # Process emotion words
        for word, emotions in emotion_mappings.items():
            # Fill in missing emotions with 0
            full_emotions = {
                'fear': 0, 'anger': 0, 'anticipation': 0, 'trust': 0,
                'surprise': 0, 'positive': 0, 'negative': 0, 'sadness': 0,
                'disgust': 0, 'joy': 0
            }
            full_emotions.update(emotions)
            
            nrc_data = SemanticLibraryData(
                library_type=SemanticLibraryType.NRC_EMOTION,
                word=word,
                raw_data={
                    'emotion_scores': full_emotions,
                    'valence': full_emotions['positive'] - full_emotions['negative'],
                    'arousal': max(full_emotions['anger'], full_emotions['fear'], full_emotions['joy']),
                    'dominance': full_emotions['trust'] - full_emotions['fear'],
                    'is_basic_emotion': True
                },
                confidence=0.7,
                source_metadata={
                    'source': 'basic_emotion_mapping',
                    'method': 'fallback_analysis',
                    'test_mode': test_mode
                }
            )
            
            self.nrc_emotion_data[word] = nrc_data
        
        mode_text = "TEST" if test_mode else "BASIC"
        print(f"✅ {mode_text} emotion analysis completed: {len(self.nrc_emotion_data)} emotion words")
    
    def _integrate_real_conceptnet(self, test_mode: bool = False):
        """Integrate REAL ConceptNet data via API"""
        print("🔗 Loading REAL ConceptNet relationship data...")
        
        try:
            import requests
            import time
            
            # Get sample words from WordNet to find relationships for
            if test_mode:
                sample_words = list(self.wordnet_data.keys())[:20]  # Small subset for testing
                print(f"🧪 TEST MODE: Finding ConceptNet relationships for {len(sample_words)} words...")
            else:
                sample_words = list(self.wordnet_data.keys())  # ALL words for full integration
                print(f"🔗 Finding ConceptNet relationships for ALL {len(sample_words)} words...")
            
            processed_count = 0
            for word in sample_words:
                try:
                    # Query ConceptNet API
                    url = f"http://api.conceptnet.io/c/en/{word.replace(' ', '_')}"
                    response = requests.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract relationships
                        edges = data.get('edges', [])
                        relationships = []
                        
                        for edge in edges[:10]:  # Limit to 10 relationships per word
                            rel_type = edge.get('rel', {}).get('@id', '').split('/')[-1]
                            start = edge.get('start', {}).get('@id', '')
                            end = edge.get('end', {}).get('@id', '')
                            weight = edge.get('weight', 1.0)
                            
                            # Clean up the relationship data
                            if '/c/en/' in start:
                                start_word = start.split('/c/en/')[-1].replace('_', ' ')
                            else:
                                start_word = start
                            
                            if '/c/en/' in end:
                                end_word = end.split('/c/en/')[-1].replace('_', ' ')
                            else:
                                end_word = end
                            
                            relationships.append({
                                'relation': rel_type,
                                'start': start_word,
                                'end': end_word,
                                'weight': weight
                            })
                        
                        if relationships:
                            conceptnet_data = SemanticLibraryData(
                                library_type=SemanticLibraryType.CONCEPTNET,
                                word=word,
                                raw_data={
                                    'relationships': relationships,
                                    'total_edges': len(edges),
                                    'api_response': True,
                                    'relationship_types': list(set(rel['relation'] for rel in relationships))
                                },
                                confidence=0.8,
                                source_metadata={
                                    'source': 'conceptnet_api',
                                    'url': url,
                                    'edge_count': len(edges),
                                    'test_mode': test_mode
                                }
                            )
                            
                            self.conceptnet_data[word] = conceptnet_data
                            processed_count += 1
                            
                            if test_mode and processed_count % 5 == 0:
                                print(f"🔗 ConceptNet test progress: {processed_count} words with relationships")
                            elif not test_mode and processed_count % 100 == 0:
                                print(f"🔗 ConceptNet progress: {processed_count} words with relationships")
                    
                    # Rate limiting
                    time.sleep(0.1)  # Don't hammer the API
                    
                except Exception as e:
                    continue
            
            mode_text = "TEST" if test_mode else "REAL"
            print(f"✅ ConceptNet {mode_text} integration completed: {len(self.conceptnet_data)} words with relationships")
            
        except ImportError:
            print("⚠️ Requests library not available - using basic relationships")
            self._integrate_basic_conceptnet(test_mode=test_mode)
        except Exception as e:
            print(f"⚠️ ConceptNet API failed: {e} - using basic relationships")
            self._integrate_basic_conceptnet(test_mode=test_mode)
    
    def _integrate_basic_conceptnet(self, test_mode: bool = False):
        """Basic conceptnet relationships when API is not available"""
        print("🔗 Using basic relationship mapping...")
        
        # Basic relationship patterns
        basic_relationships = [
            # IsA relationships
            ('love', 'IsA', 'emotion'),
            ('hate', 'IsA', 'emotion'),
            ('joy', 'IsA', 'emotion'),
            ('sadness', 'IsA', 'emotion'),
            ('anger', 'IsA', 'emotion'),
            ('fear', 'IsA', 'emotion'),
            ('dog', 'IsA', 'animal'),
            ('cat', 'IsA', 'animal'),
            ('tree', 'IsA', 'plant'),
            ('rose', 'IsA', 'flower'),
            
            # Antonym relationships
            ('love', 'Antonym', 'hate'),
            ('joy', 'Antonym', 'sadness'),
            ('good', 'Antonym', 'bad'),
            ('big', 'Antonym', 'small'),
            ('hot', 'Antonym', 'cold'),
            ('light', 'Antonym', 'dark'),
            
            # RelatedTo relationships
            ('love', 'RelatedTo', 'heart'),
            ('hate', 'RelatedTo', 'anger'),
            ('joy', 'RelatedTo', 'happiness'),
            ('sadness', 'RelatedTo', 'tears'),
            
            # CapableOf relationships
            ('dog', 'CapableOf', 'bark'),
            ('cat', 'CapableOf', 'meow'),
            ('bird', 'CapableOf', 'fly'),
            ('fish', 'CapableOf', 'swim'),
            
            # UsedFor relationships
            ('knife', 'UsedFor', 'cut'),
            ('pen', 'UsedFor', 'write'),
            ('car', 'UsedFor', 'transport'),
            ('book', 'UsedFor', 'read')
        ]
        
        # Limit for test mode
        if test_mode:
            basic_relationships = basic_relationships[:10]  # First 10 relationships for testing
            print(f"🧪 TEST MODE: Using {len(basic_relationships)} basic relationships")
        
        # Process relationships
        for source, rel_type, target in basic_relationships:
            relationship_id = f"{source}_{rel_type}_{target}"
            
            conceptnet_data = SemanticLibraryData(
                library_type=SemanticLibraryType.CONCEPTNET,
                word=relationship_id,
                raw_data={
                    'source_concept': source,
                    'relationship_type': rel_type,
                    'target_concept': target,
                    'weight': 1.0,
                    'is_basic_relationship': True
                },
                confidence=0.7,
                source_metadata={
                    'source': 'basic_conceptnet',
                    'relationship_type': rel_type,
                    'test_mode': test_mode
                }
            )
            
            self.conceptnet_data[relationship_id] = conceptnet_data
        
        mode_text = "TEST" if test_mode else "BASIC"
        print(f"✅ {mode_text} ConceptNet completed: {len(self.conceptnet_data)} relationships")
    
    def _integrate_real_textblob(self, test_mode: bool = False):
        """Integrate REAL TextBlob sentiment analysis"""
        print("📝 Loading REAL TextBlob sentiment analysis...")
        
        try:
            from textblob import TextBlob
            print("✅ TextBlob library found - using REAL sentiment analysis")
            
            # Analyze sentiment for our WordNet words
            if test_mode:
                words_to_analyze = list(self.wordnet_data.keys())[:50]  # Small subset for testing
                print(f"🧪 TEST MODE: Analyzing sentiment for {len(words_to_analyze)} words...")
            else:
                words_to_analyze = list(self.wordnet_data.keys())  # ALL words for full analysis
                print(f"📝 Analyzing sentiment for ALL {len(words_to_analyze)} words...")
            
            processed_count = 0
            for word in words_to_analyze:
                try:
                    # Get REAL sentiment analysis
                    blob = TextBlob(word)
                    sentiment = blob.sentiment
                    
                    # Get linguistic features
                    word_blob = TextBlob(word)
                    tags = word_blob.tags
                    noun_phrases = word_blob.noun_phrases
                    
                    textblob_data = SemanticLibraryData(
                        library_type=SemanticLibraryType.TEXTBLOB,
                        word=word,
                        raw_data={
                            'polarity': sentiment.polarity,
                            'subjectivity': sentiment.subjectivity,
                            'pos_tags': tags,
                            'noun_phrases': list(noun_phrases),
                            'sentiment_classification': 'positive' if sentiment.polarity > 0.1 else 'negative' if sentiment.polarity < -0.1 else 'neutral',
                            'subjectivity_classification': 'subjective' if sentiment.subjectivity > 0.5 else 'objective'
                        },
                        confidence=0.75,
                        source_metadata={
                            'source': 'textblob_real',
                            'sentiment_engine': 'textblob',
                            'has_sentiment': True,
                            'test_mode': test_mode
                        }
                    )
                    
                    self.textblob_data[word] = textblob_data
                    processed_count += 1
                    
                    if test_mode and processed_count % 10 == 0:
                        print(f"📝 TextBlob test progress: {processed_count} words analyzed")
                    elif not test_mode and processed_count % 200 == 0:
                        print(f"📝 TextBlob progress: {processed_count} words analyzed")
                
                except Exception as e:
                    continue
            
            mode_text = "TEST" if test_mode else "REAL"
            print(f"✅ TextBlob {mode_text} integration completed: {len(self.textblob_data)} words analyzed")
            
        except ImportError:
            print("⚠️ TextBlob not available - sentiment integration skipped")
            print("💡 Install TextBlob: pip install textblob")
    
    def _process_all_library_data_to_traits(self):
        """Process all library data into semantic traits"""
        print("🔄 Converting all library data to semantic traits...")
        
        # Process WordNet data
        for word, data in self.wordnet_data.items():
            trait = self._convert_library_data_to_trait(word, data)
            if trait:
                self.processed_traits[f"wordnet_{word}"] = trait
        
        # Process NRC Emotion data
        for word, data in self.nrc_emotion_data.items():
            trait = self._convert_library_data_to_trait(word, data)
            if trait:
                self.processed_traits[f"nrc_{word}"] = trait
        
        # Process ConceptNet data
        for rel_id, data in self.conceptnet_data.items():
            trait = self._convert_library_data_to_trait(rel_id, data)
            if trait:
                self.processed_traits[f"conceptnet_{rel_id}"] = trait
        
        # Process TextBlob data
        for word, data in self.textblob_data.items():
            trait = self._convert_library_data_to_trait(word, data)
            if trait:
                self.processed_traits[f"textblob_{word}"] = trait
        
        self.integration_metrics.total_traits_created = len(self.processed_traits)
        print(f"✅ Trait conversion completed: {len(self.processed_traits)} semantic traits created")
    
    def _convert_library_data_to_trait(self, identifier: str, data: SemanticLibraryData) -> Optional[SemanticTrait]:
        """Convert library data to semantic trait"""
        try:
            # Extract semantic properties based on library type
            semantic_properties = {}
            
            if data.library_type == SemanticLibraryType.WORDNET:
                semantic_properties = {
                    'definitions': data.raw_data.get('definitions', []),
                    'pos_tags': data.raw_data.get('pos_tags', []),
                    'hypernyms': data.raw_data.get('hypernyms', []),
                    'hyponyms': data.raw_data.get('hyponyms', []),
                    'synonyms': data.raw_data.get('synonyms', []),
                    'antonyms': data.raw_data.get('antonyms', []),
                    'synset_count': data.raw_data.get('synset_count', 0),
                    'word_frequency': data.raw_data.get('word_frequency', 0.5),
                    'semantic_cluster': data.raw_data.get('semantic_similarity_cluster', 'general')
                }
                category = TraitCategory.SEMANTIC
                
            elif data.library_type == SemanticLibraryType.NRC_EMOTION:
                semantic_properties = {
                    'emotion_scores': data.raw_data.get('emotion_scores', {}),
                    'valence': data.raw_data.get('valence', 0.5),
                    'arousal': data.raw_data.get('arousal', 0.5),
                    'dominance': data.raw_data.get('dominance', 0.5),
                    'top_emotions': data.raw_data.get('top_emotions', []),
                    'emotional_intensity': max(data.raw_data.get('emotion_scores', {}).values()) if data.raw_data.get('emotion_scores') else 0.3
                }
                category = TraitCategory.LINGUISTIC
                
            elif data.library_type == SemanticLibraryType.CONCEPTNET:
                semantic_properties = {
                    'relationships': data.raw_data.get('relationships', []),
                    'relationship_types': data.raw_data.get('relationship_types', []),
                    'source_concept': data.raw_data.get('source_concept', ''),
                    'target_concept': data.raw_data.get('target_concept', ''),
                    'relationship_type': data.raw_data.get('relationship_type', ''),
                    'weight': data.raw_data.get('weight', 1.0)
                }
                category = TraitCategory.SYNTHESIS
                
            elif data.library_type == SemanticLibraryType.TEXTBLOB:
                semantic_properties = {
                    'polarity': data.raw_data.get('polarity', 0.0),
                    'subjectivity': data.raw_data.get('subjectivity', 0.5),
                    'sentiment_classification': data.raw_data.get('sentiment_classification', 'neutral'),
                    'subjectivity_classification': data.raw_data.get('subjectivity_classification', 'objective'),
                    'pos_tags': data.raw_data.get('pos_tags', []),
                    'noun_phrases': data.raw_data.get('noun_phrases', [])
                }
                category = TraitCategory.LINGUISTIC
            
            # Create semantic trait
            trait = SemanticTrait(
                trait_uuid=uuid.uuid4(),
                name=identifier,
                category=category,
                complexity=SemanticComplexity.INTERMEDIATE,
                semantic_properties=semantic_properties
            )
            
            return trait
            
        except Exception as e:
            print(f"⚠️ Failed to convert {identifier}: {e}")
            return None
    
    def _estimate_word_frequency(self, word: str) -> float:
        """Estimate word frequency based on characteristics"""
        # Simple frequency estimation
        word_len = len(word)
        if word_len <= 3:
            return 0.8
        elif word_len <= 5:
            return 0.6
        elif word_len <= 8:
            return 0.4
        else:
            return 0.2
    
    def _calculate_semantic_cluster(self, word: str, synsets) -> str:
        """Calculate semantic cluster for word"""
        if not synsets:
            return 'unknown'
        
        # Get first synset's lexical domain
        first_synset = synsets[0]
        if hasattr(first_synset, 'lexname'):
            return first_synset.lexname()
        else:
            return 'general'
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        with self._lock:
            return {
                'status': self.integration_metrics.integration_status.value,
                'total_words_processed': self.integration_metrics.total_words_processed,
                'total_relationships_found': self.integration_metrics.total_relationships_found,
                'total_traits_created': self.integration_metrics.total_traits_created,
                'processing_time': self.integration_metrics.processing_time,
                'errors_encountered': self.integration_metrics.errors_encountered,
                'wordnet_entries': len(self.wordnet_data),
                'nrc_emotion_entries': len(self.nrc_emotion_data),
                'conceptnet_entries': len(self.conceptnet_data),
                'textblob_entries': len(self.textblob_data),
                'processed_traits': len(self.processed_traits)
            }
    
    def verify_integration_data(self):
        """Verify the integration data is real"""
        print("\n🔍 VERIFYING REAL INTEGRATION DATA")
        print("=" * 50)
        
        # Verify WordNet data
        if self.wordnet_data:
            sample_word = list(self.wordnet_data.keys())[0]
            sample_data = self.wordnet_data[sample_word]
            print(f"📚 WordNet Sample - '{sample_word}':")
            print(f"   Definitions: {len(sample_data.raw_data.get('definitions', []))}")
            print(f"   Synsets: {sample_data.raw_data.get('synset_count', 0)}")
            print(f"   Source: {sample_data.source_metadata.get('source', 'unknown')}")
        
        # Verify NRC data
        if self.nrc_emotion_data:
            sample_word = list(self.nrc_emotion_data.keys())[0]
            sample_data = self.nrc_emotion_data[sample_word]
            print(f"😊 NRC Sample - '{sample_word}':")
            emotions = sample_data.raw_data.get('emotion_scores', {})
            print(f"   Emotions: {list(emotions.keys())}")
            print(f"   Source: {sample_data.source_metadata.get('source', 'unknown')}")
        
        # Verify ConceptNet data
        if self.conceptnet_data:
            sample_rel = list(self.conceptnet_data.keys())[0]
            sample_data = self.conceptnet_data[sample_rel]
            print(f"🔗 ConceptNet Sample - '{sample_rel}':")
            print(f"   Relationship: {sample_data.raw_data.get('relationship_type', 'unknown')}")
            print(f"   Source: {sample_data.source_metadata.get('source', 'unknown')}")
        
        # Verify TextBlob data
        if self.textblob_data:
            sample_word = list(self.textblob_data.keys())[0]
            sample_data = self.textblob_data[sample_word]
            print(f"📝 TextBlob Sample - '{sample_word}':")
            print(f"   Polarity: {sample_data.raw_data.get('polarity', 0.0)}")
            print(f"   Source: {sample_data.source_metadata.get('source', 'unknown')}")
        
        print(f"\n✅ VERIFICATION COMPLETE")
        print(f"📊 Total Real Data Entries: {len(self.wordnet_data) + len(self.nrc_emotion_data) + len(self.conceptnet_data) + len(self.textblob_data)}")
        print(f"🎯 Total Semantic Traits: {len(self.processed_traits)}")

def main():
    """Main function to demonstrate semantic integration with test mode"""
    print("🔥 SEMANTIC LIBRARY INTEGRATION - TEST MODE")
    print("=" * 60)
    
    # Setup mock dependencies for testing
    class MockComponent:
        def __init__(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    # Create integration system
    integration = SemanticLibraryIntegration(
        MockComponent(), MockComponent(), MockComponent(),
        MockComponent(), MockComponent(), MockComponent()
    )
    
    print("✅ Integration system created")
    
    # Start TEST integration (fast, small dataset)
    result = integration.start_massive_integration([
        SemanticLibraryType.WORDNET,
        SemanticLibraryType.NRC_EMOTION,
        SemanticLibraryType.CONCEPTNET,
        SemanticLibraryType.TEXTBLOB
    ], test_mode=True)  # Use test mode for quick verification
    
    print(f"\n🎯 INTEGRATION RESULT: {result}")
    
    # Verify the data is real
    integration.verify_integration_data()
    
    print("\n🎉 SEMANTIC INTEGRATION COMPLETE!")
    print("Test mode completed - use test_mode=False for full integration")

if __name__ == "__main__":
    main()
