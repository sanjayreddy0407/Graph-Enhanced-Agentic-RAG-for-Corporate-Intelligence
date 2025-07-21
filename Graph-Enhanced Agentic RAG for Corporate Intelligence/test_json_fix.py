"""Direct test for JSON extraction issues."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_json_extraction():
    """Test the specific JSON extraction issue."""
    print("🔍 Testing JSON extraction...")
    
    try:
        from knowledge_extractor import KnowledgeExtractor
        
        # Initialize extractor
        extractor = KnowledgeExtractor()
        
        # Test problematic response (similar to what caused the error)
        test_responses = [
            '\n "companies": ["Microsoft", "Apple"]',
            '{\n  "companies": ["Microsoft", "Apple"],\n  "people": ["Tim Cook"]\n}',
            'Here is the JSON:\n{\n  "companies": ["Microsoft"]\n}',
            '```json\n{\n  "companies": ["Microsoft"]\n}\n```'
        ]
        
        print("Testing JSON extraction on various malformed responses...")
        
        for i, response in enumerate(test_responses):
            print(f"\nTest {i+1}: {response[:50]}...")
            result = extractor._extract_json_from_response(response)
            print(f"  Result: {result}")
        
        # Test actual entity extraction
        print("\n📝 Testing entity extraction...")
        sample_text = "Microsoft CEO Satya Nadella reported Q3 revenue of $50 billion."
        entities = extractor.extract_entities(sample_text)
        
        print(f"✅ Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  • {entity.name} ({entity.type})")
        
        return len(entities) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_json_extraction()
    print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}")
