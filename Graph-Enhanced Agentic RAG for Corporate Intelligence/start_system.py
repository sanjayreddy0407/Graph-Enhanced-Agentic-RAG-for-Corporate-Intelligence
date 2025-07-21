"""Startup script to get the Graph-Enhanced RAG system running quickly."""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install minimal requirements."""
    print("📦 Installing minimal requirements...")
    
    minimal_packages = [
        "streamlit>=1.24.0",
        "sentence-transformers>=2.2.0", 
        "scikit-learn>=1.0.0",
        "PyMuPDF>=1.23.0",
        "pdfminer.six>=20220524",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0"
    ]
    
    for package in minimal_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package.split('>=')[0]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def test_system():
    """Test core system components."""
    print("\n🧪 Testing core system components...")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from document_processor import DocumentProcessor
        from knowledge_extractor import KnowledgeExtractor
        from simple_vector_store import VectorStore
        
        # Quick functionality test
        doc_processor = DocumentProcessor()
        knowledge_extractor = KnowledgeExtractor()
        vector_store = VectorStore()
        
        # Test with sample text
        sample_text = "Microsoft CEO Satya Nadella announced strong Q3 2023 results."
        entities = knowledge_extractor.extract_entities(sample_text)
        
        if len(entities) > 0:
            print("✅ Knowledge extraction working")
            return True
        else:
            print("⚠️  Knowledge extraction may need attention")
            return False
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def start_app():
    """Start the Streamlit application."""
    print("\n🚀 Starting Streamlit application...")
    print("   Access your app at: http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_simple.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"❌ Failed to start app: {e}")

def main():
    """Main startup sequence."""
    print("🧠 Graph-Enhanced Agentic RAG System Startup")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually.")
        return
    
    # Test system
    if test_system():
        print("✅ System tests passed!")
    else:
        print("⚠️  Some components may not work perfectly, but proceeding...")
    
    # Start application
    start_app()

if __name__ == "__main__":
    main()
