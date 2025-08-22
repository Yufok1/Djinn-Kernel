# rebuild/test_semantic_caching.py
"""
Test script to verify semantic foundation caching
Ensures the system doesn't reload massive data every time
"""

import time
import os
from datetime import datetime

def test_semantic_caching():
    """Test the semantic caching system"""
    print("🧪 TESTING SEMANTIC FOUNDATION CACHING")
    print("=" * 50)
    
    # Setup mock dependencies
    class MockComponent:
        def __init__(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    # Import the database
    from local_semantic_database import LocalSemanticDatabase
    
    print("📚 Creating LocalSemanticDatabase...")
    
    # First run - should build from scratch
    print("\n🔄 FIRST RUN - Building from scratch...")
    start_time = time.time()
    
    db1 = LocalSemanticDatabase(
        MockComponent(), MockComponent(), MockComponent(),
        MockComponent(), MockComponent(), MockComponent()
    )
    
    first_run_time = time.time() - start_time
    print(f"⏱️ First run completed in {first_run_time:.2f} seconds")
    print(f"📊 References: {len(db1.semantic_references)}")
    print(f"🔗 Relationships: {len(db1.semantic_relationships)}")
    
    # Second run - should load from cache
    print("\n🔄 SECOND RUN - Loading from cache...")
    start_time = time.time()
    
    db2 = LocalSemanticDatabase(
        MockComponent(), MockComponent(), MockComponent(),
        MockComponent(), MockComponent(), MockComponent()
    )
    
    second_run_time = time.time() - start_time
    print(f"⏱️ Second run completed in {second_run_time:.2f} seconds")
    print(f"📊 References: {len(db2.semantic_references)}")
    print(f"🔗 Relationships: {len(db2.semantic_relationships)}")
    
    # Verify cache files exist
    cache_file = "semantic_cache/semantic_foundation_cache.pkl"
    metadata_file = "semantic_cache/cache_metadata.json"
    
    print(f"\n📁 Cache file exists: {os.path.exists(cache_file)}")
    print(f"📁 Metadata file exists: {os.path.exists(metadata_file)}")
    
    if os.path.exists(metadata_file):
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"📊 Cache metadata: {metadata}")
    
    # Performance comparison
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    print(f"\n🚀 CACHING PERFORMANCE:")
    print(f"   First run (build): {first_run_time:.2f}s")
    print(f"   Second run (cache): {second_run_time:.2f}s")
    print(f"   Speedup: {speedup:.1f}x faster")
    
    # Verify data consistency
    print(f"\n✅ DATA CONSISTENCY CHECK:")
    print(f"   References match: {len(db1.semantic_references) == len(db2.semantic_references)}")
    print(f"   Relationships match: {len(db1.semantic_relationships) == len(db2.semantic_relationships)}")
    
    if len(db1.semantic_references) > 0 and len(db2.semantic_references) > 0:
        # Check a sample reference
        sample_key = list(db1.semantic_references.keys())[0]
        ref1 = db1.semantic_references[sample_key]
        ref2 = db2.semantic_references[sample_key]
        
        print(f"   Sample reference '{sample_key}' matches: {ref1.word == ref2.word}")
        print(f"   Sample source matches: {ref1.source == ref2.source}")
    
    print(f"\n🎉 CACHING TEST COMPLETE!")
    
    if speedup > 2.0:  # At least 2x faster
        print("✅ Caching is working effectively!")
    else:
        print("⚠️ Caching may not be working optimally")
    
    return {
        'first_run_time': first_run_time,
        'second_run_time': second_run_time,
        'speedup': speedup,
        'cache_exists': os.path.exists(cache_file),
        'data_consistent': len(db1.semantic_references) == len(db2.semantic_references)
    }

def test_cache_clear_and_rebuild():
    """Test cache clearing and rebuilding"""
    print("\n🧪 TESTING CACHE CLEAR AND REBUILD")
    print("=" * 50)
    
    # Setup mock dependencies
    class MockComponent:
        def __init__(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    from local_semantic_database import LocalSemanticDatabase
    
    # Create database
    db = LocalSemanticDatabase(
        MockComponent(), MockComponent(), MockComponent(),
        MockComponent(), MockComponent(), MockComponent()
    )
    
    initial_count = len(db.semantic_references)
    print(f"📊 Initial references: {initial_count}")
    
    # Clear cache
    print("🗑️ Clearing cache...")
    db.clear_cache()
    
    # Verify cache is cleared
    cache_file = "semantic_cache/semantic_foundation_cache.pkl"
    metadata_file = "semantic_cache/cache_metadata.json"
    
    print(f"📁 Cache file exists after clear: {os.path.exists(cache_file)}")
    print(f"📁 Metadata file exists after clear: {os.path.exists(metadata_file)}")
    
    # Force rebuild
    print("🔄 Force rebuilding...")
    start_time = time.time()
    db.force_rebuild_foundation()
    rebuild_time = time.time() - start_time
    
    print(f"⏱️ Rebuild completed in {rebuild_time:.2f} seconds")
    print(f"📊 References after rebuild: {len(db.semantic_references)}")
    
    # Verify cache is restored
    print(f"📁 Cache file exists after rebuild: {os.path.exists(cache_file)}")
    print(f"📁 Metadata file exists after rebuild: {os.path.exists(metadata_file)}")
    
    print("✅ Cache clear and rebuild test complete!")

if __name__ == "__main__":
    # Run caching test
    results = test_semantic_caching()
    
    # Run cache clear test
    test_cache_clear_and_rebuild()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Caching speedup: {results['speedup']:.1f}x")
    print(f"   Cache files created: {results['cache_exists']}")
    print(f"   Data consistency: {results['data_consistent']}")
    
    if results['speedup'] > 2.0 and results['cache_exists'] and results['data_consistent']:
        print("🎉 ALL TESTS PASSED! Semantic caching is working correctly.")
    else:
        print("❌ Some tests failed. Check the implementation.")
