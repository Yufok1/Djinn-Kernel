# Test NRC Lexicon API to find correct attributes
from nrclex import NRCLex

print("🔍 Testing NRC Lexicon API...")

# Test with a simple word
word = "love"
print(f"Testing word: '{word}'")

try:
    emotion_data = NRCLex(word)
    
    print(f"Type: {type(emotion_data)}")
    print(f"Dir: {dir(emotion_data)}")
    
    # Try to access different attributes
    print(f"Affect dict: {emotion_data.affect_dict}")
    print(f"Affect frequencies: {emotion_data.affect_frequencies}")
    print(f"Top emotions: {emotion_data.top_emotions}")
    
    # Try to access individual emotion scores
    print(f"Raw affect: {emotion_data.affect}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
