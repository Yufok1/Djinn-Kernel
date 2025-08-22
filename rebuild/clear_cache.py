# rebuild/clear_cache.py
"""Clear the semantic foundation cache"""

import os

def clear_cache():
    """Clear the semantic foundation cache"""
    cache_dir = "semantic_cache"
    cache_file = os.path.join(cache_dir, "semantic_foundation_cache.pkl")
    metadata_file = os.path.join(cache_dir, "cache_metadata.json")
    
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("🗑️ Cache file removed")
    
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
        print("🗑️ Cache metadata removed")
    
    print("✅ Semantic foundation cache cleared")

if __name__ == "__main__":
    clear_cache()
