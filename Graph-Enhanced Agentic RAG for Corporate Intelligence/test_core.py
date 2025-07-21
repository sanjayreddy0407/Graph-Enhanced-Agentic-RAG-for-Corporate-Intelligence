"""Quick test to verify document processing and knowledge extraction is working."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import DocumentProcessor
from src.knowledge_extractor import KnowledgeExtractor

def test_system():
    """Test the core document processing and knowledge extraction."""
    
    print("🧪 Testing Graph-Enhanced RAG System Core Components")
    print("=" * 60)
    
    # Initialize processors
    doc_processor = DocumentProcessor()
    knowledge_extractor = KnowledgeExtractor()
    
    # Test with sample text (simulating PDF content)
    sample_text = """
    Microsoft Corporation reported strong quarterly results for Q3 2023.
    CEO Satya Nadella announced that Azure revenue grew by 31% year-over-year.
    The company's total revenue reached $52.9 billion, representing a 7% increase.
    Microsoft acquired LinkedIn for $26.2 billion in 2016.
    Apple Inc. CEO Tim Cook discussed the company's commitment to innovation.
    Amazon.com Inc. reported AWS revenue of $21.4 billion in the same quarter.
    """
    
    print("\n1️⃣  Testing Knowledge Extraction...")
    print("-" * 40)
    
    # Test entity extraction
    entities = knowledge_extractor.extract_entities(sample_text)
    print(f"✅ Extracted {len(entities)} entities:")
    for entity in entities[:5]:  # Show first 5
        print(f"   • {entity.name} ({entity.type})")
    
    # Test relation extraction
    relations = knowledge_extractor.extract_relations(sample_text, entities)
    print(f"\n✅ Extracted {len(relations)} relations:")
    for relation in relations[:3]:  # Show first 3
        print(f"   • {relation.subject} → {relation.predicate} → {relation.object}")
    
    print("\n2️⃣  Testing Text Processing...")
    print("-" * 40)
    
    # Test chunking
    chunks = doc_processor.chunk_text(sample_text, chunk_size=100, overlap=20)
    print(f"✅ Created {len(chunks)} text chunks")
    
    print(f"\n3️⃣  System Status Summary:")
    print("-" * 40)
    print(f"   📄 Document Processing: {'✅ Working' if len(chunks) > 0 else '❌ Failed'}")
    print(f"   🧠 Entity Extraction: {'✅ Working' if len(entities) > 0 else '❌ Failed'}")
    print(f"   🔗 Relation Extraction: {'✅ Working' if len(relations) > 0 else '❌ Failed'}")
    
    if len(entities) > 0 and len(relations) > 0:
        print(f"\n🎉 Core system components are working correctly!")
        print(f"   Your RAG system should now be able to process documents.")
        return True
    else:
        print(f"\n⚠️  Some components need attention.")
        return False

if __name__ == "__main__":
    test_system()
