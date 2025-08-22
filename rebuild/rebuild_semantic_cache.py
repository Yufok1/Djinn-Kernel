# rebuild/rebuild_semantic_cache.py
"""Rebuild the semantic foundation cache with real data"""

import time
from datetime import datetime

def rebuild_semantic_cache():
    """Rebuild the semantic foundation cache with real data"""
    print("🔄 REBUILDING SEMANTIC FOUNDATION CACHE")
    print("=" * 50)
    
    # Setup mock dependencies
    class MockComponent:
        def __init__(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    print("📚 Loading semantic foundation with real data...")
    start_time = time.time()
    
    # Import and create the database (will rebuild from scratch)
    from local_semantic_database import LocalSemanticDatabase
    
    db = LocalSemanticDatabase(
        MockComponent(), MockComponent(), MockComponent(),
        MockComponent(), MockComponent(), MockComponent()
    )
    
    build_time = time.time() - start_time
    print(f"⏱️ Built in {build_time:.2f} seconds")
    
    # Show what we have
    print(f"\n📊 SEMANTIC FOUNDATION STATS:")
    print(f"   References: {len(db.semantic_references)}")
    print(f"   Relationships: {len(db.semantic_relationships)}")
    print(f"   Word Index: {len(db.word_index)}")
    print(f"   Concept Index: {len(db.concept_index)}")
    print(f"   Emotion Index: {len(db.emotion_index)}")
    
    # Show sample data to verify it's real
    if db.semantic_references:
        print(f"\n🔍 SAMPLE REAL SEMANTIC REFERENCES:")
        sample_keys = list(db.semantic_references.keys())[:5]
        for key in sample_keys:
            ref = db.semantic_references[key]
            print(f"   '{ref.word}' ({ref.source.value}) - {ref.data_type.value}")
            print(f"     Stability: {ref.convergence_stability:.3f}")
            print(f"     VP: {ref.violation_pressure:.3f}")
            print(f"     Intensity: {ref.trait_intensity:.3f}")
    
    # Check if we have real data (not fake massive_trait_X)
    fake_count = sum(1 for key in db.semantic_references.keys() if key.startswith('massive_trait_'))
    real_count = len(db.semantic_references) - fake_count
    
    print(f"\n✅ CACHE REBUILD COMPLETE!")
    print(f"   Real references: {real_count}")
    print(f"   Fake references: {fake_count}")
    
    if real_count > fake_count:
        print(f"   ✅ Cache contains mostly real semantic data!")
    else:
        print(f"   ❌ Cache still contains fake data - needs investigation")
    
    return db

if __name__ == "__main__":
    rebuild_semantic_cache()
