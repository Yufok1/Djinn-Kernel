# Test full integration with all libraries working
from semantic_library_integration import SemanticLibraryIntegration, SemanticLibraryType, IntegrationStatus
from semantic_state_manager import SemanticStateManager
from semantic_event_bridge import SemanticEventBridge
from semantic_violation_monitor import SemanticViolationMonitor
from semantic_checkpoint_manager import SemanticCheckpointManager

# Mock dependencies
class MockUUIDanchor:
    def anchor_trait(self, word):
        import uuid
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(word))

class MockDjinnEventBus:
    def __init__(self):
        self.events = []
        self.handlers = {}
        self.subscriptions = {}
    def register_handler(self, event_type, handler):
        self.handlers[event_type] = handler
    def subscribe(self, event_type, handler):
        self.subscriptions[event_type] = handler
    def publish(self, event_data):
        self.events.append(event_data)

class MockViolationMonitor:
    def calculate_violation_pressure(self, trait_data):
        return 0.5

class MockTraitConvergenceEngine:
    def calculate_convergence_stability(self, trait_data):
        return 0.7

class MockTemporalIsolationManager:
    def create_isolation_context(self):
        return "test_context"

def main():
    print("🚀 TESTING FULL SEMANTIC LIBRARY INTEGRATION")
    print("=" * 60)
    
    # Setup components
    print("🔧 Setting up components...")
    uuid_anchor = MockUUIDanchor()
    event_bus = MockDjinnEventBus()
    violation_monitor = MockViolationMonitor()
    trait_convergence = MockTraitConvergenceEngine()
    temporal_isolation = MockTemporalIsolationManager()
    
    state_manager = SemanticStateManager(event_bus, uuid_anchor, violation_monitor)
    event_bridge = SemanticEventBridge(event_bus, state_manager, violation_monitor, temporal_isolation)
    semantic_violation_monitor = SemanticViolationMonitor(violation_monitor, temporal_isolation, state_manager, event_bridge)
    checkpoint_manager = SemanticCheckpointManager(state_manager, event_bridge, semantic_violation_monitor, uuid_anchor)
    
    # Create integration system
    integration = SemanticLibraryIntegration(
        state_manager, event_bridge, semantic_violation_monitor, 
        checkpoint_manager, uuid_anchor, trait_convergence
    )
    
    print("✅ Components initialized")
    
    # Test each library individually first
    print("\n🧪 Testing individual libraries...")
    
    # Test NRC Emotion
    print("\n😊 Testing NRC Emotion...")
    result = integration.start_massive_integration([SemanticLibraryType.NRC_EMOTION])
    print(f"Started: {result['status']}")
    
    import time
    start_time = time.time()
    while integration.integration_metrics.integration_status == IntegrationStatus.IN_PROGRESS:
        time.sleep(1)
        if time.time() - start_time > 10:
            break
    
    status = integration.get_integration_status()
    print(f"NRC Results: {status['total_words_processed']} words, {status['total_traits_created']} traits")
    
    # Test TextBlob
    print("\n📝 Testing TextBlob...")
    integration.integration_metrics.integration_status = IntegrationStatus.NOT_STARTED
    integration.integration_metrics.total_words_processed = 0
    integration.integration_metrics.total_traits_created = 0
    
    result = integration.start_massive_integration([SemanticLibraryType.TEXTBLOB])
    print(f"Started: {result['status']}")
    
    start_time = time.time()
    while integration.integration_metrics.integration_status == IntegrationStatus.IN_PROGRESS:
        time.sleep(1)
        if time.time() - start_time > 10:
            break
    
    status = integration.get_integration_status()
    print(f"TextBlob Results: {status['total_words_processed']} words, {status['total_traits_created']} traits")
    
    # Test full integration
    print("\n🎯 Testing FULL INTEGRATION...")
    integration.integration_metrics.integration_status = IntegrationStatus.NOT_STARTED
    integration.integration_metrics.total_words_processed = 0
    integration.integration_metrics.total_traits_created = 0
    
    result = integration.start_massive_integration()
    print(f"Started: {result['status']}")
    
    start_time = time.time()
    while integration.integration_metrics.integration_status == IntegrationStatus.IN_PROGRESS:
        time.sleep(2)
        elapsed = time.time() - start_time
        status = integration.get_integration_status()
        print(f"⏱️ {elapsed:.1f}s - Words: {status['total_words_processed']:,}, Traits: {status['total_traits_created']:,}")
        
        if elapsed > 60:  # Stop after 1 minute for demo
            break
    
    # Final results
    final_status = integration.get_integration_status()
    print(f"\n🎉 FINAL RESULTS:")
    print(f"Total Words: {final_status['total_words_processed']:,}")
    print(f"Total Traits: {final_status['total_traits_created']:,}")
    print(f"Total Relationships: {final_status['total_relationships_found']:,}")
    print(f"Library Coverage: {final_status['library_coverage']}")
    
    # Test trait retrieval
    print(f"\n🔍 Testing trait retrieval...")
    test_words = ["love", "hate", "good", "bad", "computer", "science"]
    found_count = 0
    
    for word in test_words:
        traits = integration.get_traits_for_word(word)
        if traits:
            trait = traits[0]
            print(f"  ✅ {word}: {trait.complexity} | {trait.category}")
            found_count += 1
        else:
            print(f"  ❌ {word}: No traits found")
    
    print(f"\n📈 SUMMARY: {found_count}/{len(test_words)} words found")
    
    if final_status['total_traits_created'] > 0:
        print(f"\n✅ SUCCESS! All libraries are working!")
        print(f"The system now has {final_status['total_traits_created']:,} semantic traits!")
        return True
    else:
        print(f"\n❌ No traits were created. Need to investigate further.")
        return False

if __name__ == "__main__":
    main()
