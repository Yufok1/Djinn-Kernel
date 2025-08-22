# Debug script to test NRC Emotion and TextBlob integrations
import sys
import traceback

def test_nrc_emotion():
    """Test NRC Emotion Lexicon integration"""
    print("😊 Testing NRC Emotion Lexicon...")
    
    try:
        from nrclex import NRCLex
        
        # Test with a few emotion words
        test_words = ['love', 'hate', 'joy', 'sadness', 'anger', 'fear']
        
        for word in test_words:
            try:
                print(f"  Testing '{word}'...")
                emotion_data = NRCLex(word)
                
                print(f"    Fear: {emotion_data.fear}")
                print(f"    Anger: {emotion_data.anger}")
                print(f"    Joy: {emotion_data.joy}")
                print(f"    Sadness: {emotion_data.sadness}")
                print(f"    Positive: {emotion_data.positive}")
                print(f"    Negative: {emotion_data.negative}")
                print(f"    Affect dict: {emotion_data.affect_dict}")
                print(f"    Top emotions: {emotion_data.top_emotions}")
                print()
                
            except Exception as e:
                print(f"    ❌ Error with '{word}': {e}")
                traceback.print_exc()
                
    except ImportError as e:
        print(f"❌ NRC Lexicon not installed: {e}")
    except Exception as e:
        print(f"❌ NRC Emotion test failed: {e}")
        traceback.print_exc()

def test_textblob():
    """Test TextBlob sentiment analysis"""
    print("📝 Testing TextBlob...")
    
    try:
        from textblob import TextBlob
        
        # Test with a few words
        test_words = ['love', 'hate', 'good', 'bad', 'beautiful', 'ugly']
        
        for word in test_words:
            try:
                print(f"  Testing '{word}'...")
                blob = TextBlob(word)
                
                print(f"    Polarity: {blob.sentiment.polarity}")
                print(f"    Subjectivity: {blob.sentiment.subjectivity}")
                print(f"    Words: {blob.words}")
                print(f"    Noun phrases: {blob.noun_phrases}")
                print(f"    Tags: {blob.tags}")
                print()
                
            except Exception as e:
                print(f"    ❌ Error with '{word}': {e}")
                traceback.print_exc()
                
    except ImportError as e:
        print(f"❌ TextBlob not installed: {e}")
    except Exception as e:
        print(f"❌ TextBlob test failed: {e}")
        traceback.print_exc()

def test_wordnet():
    """Test WordNet to compare"""
    print("📚 Testing WordNet...")
    
    try:
        import nltk
        from nltk.corpus import wordnet
        
        # Test with a few words
        test_words = ['dog', 'cat', 'computer', 'love']
        
        for word in test_words:
            try:
                print(f"  Testing '{word}'...")
                synsets = wordnet.synsets(word)
                
                if synsets:
                    synset = synsets[0]  # First synset
                    print(f"    Definition: {synset.definition()}")
                    print(f"    POS: {synset.pos()}")
                    print(f"    Lemmas: {[lemma.name() for lemma in synset.lemmas()]}")
                    print(f"    Hypernyms: {[h.name() for h in synset.hypernyms()]}")
                    print()
                else:
                    print(f"    No synsets found for '{word}'")
                    
            except Exception as e:
                print(f"    ❌ Error with '{word}': {e}")
                traceback.print_exc()
                
    except ImportError as e:
        print(f"❌ NLTK/WordNet not installed: {e}")
    except Exception as e:
        print(f"❌ WordNet test failed: {e}")
        traceback.print_exc()

def main():
    print("🔍 DEBUGGING SEMANTIC LIBRARY INTEGRATIONS")
    print("=" * 60)
    
    print("\n1. Testing WordNet (should work):")
    test_wordnet()
    
    print("\n2. Testing NRC Emotion Lexicon:")
    test_nrc_emotion()
    
    print("\n3. Testing TextBlob:")
    test_textblob()
    
    print("\n🎯 DEBUG COMPLETE")

if __name__ == "__main__":
    main()
