# 🧠 Graph-Enhanced Agentic RAG for Corporate Intelligence

> An intelligent Q&A system that understands business documents by combining semantic search with knowledge graphs, powered by Llama 3.2 AI.

## 🎯 What Does This System Do?

Imagine having an AI assistant that can read your company's annual reports, SEC filings, and business documents, then answer complex questions like:

- *"What were Microsoft's main challenges in 2023 and how did they address them?"*
- *"Which companies did Apple acquire and what were the strategic reasons?"*
- *"Show me the relationship between executive decisions and revenue growth"*

This system makes that possible by:
1. **Reading** your PDF documents and extracting key information
2. **Understanding** relationships between companies, people, and financial data
3. **Answering** your questions with evidence from the documents
4. **Visualizing** knowledge as interactive graphs

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Requirements
```bash
# Install essential packages
pip install streamlit sentence-transformers scikit-learn PyMuPDF requests python-dotenv pandas numpy
```

### Step 2: Set Up Llama 3.2 (Optional - uses fallback if not available)
```bash
# Download and install Ollama from https://ollama.ai/download
# Then pull Llama 3.2:
ollama pull llama3.2

# Start Ollama service:
ollama serve
```

### Step 3: Run the System
```bash
# Start the application
streamlit run app_simple.py
```

### Step 4: Upload & Query
1. Open http://localhost:8501 in your browser
2. Upload your PDF documents using the sidebar
3. Click "Process" to extract knowledge
4. Ask questions in natural language!

## 💡 How It Works (Simple Explanation)

1. **Document Processing** 📄
   - Extracts text from your PDF files
   - Breaks text into manageable chunks
   - Cleans and prepares data for analysis

2. **Knowledge Extraction** 🧠  
   - Uses Llama 3.2 AI to find entities (companies, people, financial data)
   - Discovers relationships between entities
   - Creates a knowledge graph of your business data

3. **Smart Search** 🔍
   - Vector search finds semantically similar content
   - Graph search discovers relationships and connections
   - Combines both for comprehensive answers

4. **Question Answering** 💬
   - Routes your question to the best search method
   - Retrieves relevant information from documents
   - Generates human-readable answers with sources

## 🛠️ System Architecture

```
📄 PDF Documents → 🔄 Processing → 🧠 AI Extraction → 📊 Knowledge Base → 💬 Q&A Interface
                                                           ↓
                   📚 Vector Search ← → 🕸️ Graph Search → ✨ Smart Answers
```

### Core Components:

- **Document Processor**: Extracts and cleans text from PDFs
- **Knowledge Extractor**: Uses Llama 3.2 to find entities and relationships  
- **Vector Store**: Semantic search using TF-IDF (sklearn-based)
- **Graph Store**: Relationship storage and traversal
- **Multi-Agent System**: Routes queries and synthesizes answers

## 📁 Project Structure

```
📂 Your Project/
├── 🚀 app.py          # Main application (START HERE)
├── ⚙️  config.py             # Configuration settings
├── 📋 requirements_minimal.txt # Essential packages
├── 🧪 test_core.py           # Test core functionality
├── 
├── 📂 src/                   # Core system components
│   ├── document_processor.py # PDF text extraction
│   ├── knowledge_extractor.py # AI-powered entity extraction
│   ├── simple_vector_store.py # Semantic search
│   └── utils/                 # Visualization tools
│
├── 📂 data/                  # Your documents and processed data
│   ├── documents/            # Upload PDFs here
│   └── vector_db/            # Search index storage
│
└── 📂 notebooks/             # Jupyter exploration notebooks
    └── exploration.ipynb     # System demonstration
```

## 🎯 Key Features

### ✅ What Works Out of the Box:
- **PDF Document Upload** - Drag & drop your business documents
- **Automatic Knowledge Extraction** - Finds companies, people, financial data
- **Natural Language Questions** - Ask questions like you would a human
- **Smart Search** - Combines semantic similarity with relationship data
- **Interactive Web Interface** - Clean, user-friendly Streamlit app
- **Real-time Processing** - See results as documents are processed

### 🔧 Configuration Options:
- **AI Model**: Llama 3.2 (local) or fallback extraction
- **Search Method**: TF-IDF vector search (no complex dependencies)
- **Entity Types**: Companies, people, financial metrics, products, dates
- **Relationship Types**: CEO_OF, ACQUIRED, PARTNER_WITH, etc.

## 📊 Sample Use Cases

| Question Type | Example | How It Works |
|---------------|---------|--------------|
| **Factual** | "What was Apple's revenue in Q3?" | → Vector search finds financial sections |
| **Relational** | "Who is the CEO of Microsoft?" | → Graph search finds person-company relationships |
| **Analytical** | "What challenges did Amazon face and how did they respond?" | → Hybrid search combines context + relationships |
| **Comparative** | "Compare Google and Microsoft's AI strategies" | → Multi-document analysis with entity linking |

## 🚦 System Status & Troubleshooting

### ✅ If Everything Works:
- Documents process without errors
- Entities and relationships are extracted
- Questions return relevant answers
- Knowledge graph shows in sidebar

### ⚠️ If You See Issues:

**"No entities extracted"**
- The system uses fallback extraction methods
- Documents will still be searchable via text content
- Try with different document types

**"Ollama connection failed"**
- System automatically uses built-in extraction
- For better results, install Ollama and Llama 3.2
- Check if Ollama service is running on port 11434

**"Import errors"**
- Run: `pip install -r requirements_minimal.txt`
- Make sure you're in the project directory
- Try running `python test_core.py` to check

## 🔄 Workflow Example

1. **📤 Upload**: Drop "Apple_Annual_Report_2023.pdf" into the interface
2. **⚙️ Processing**: System extracts text → finds "Tim Cook", "iPhone", "$394.3 billion revenue"
3. **🕸️ Knowledge Graph**: Creates relationships → "Tim Cook" → CEO_OF → "Apple Inc"
4. **❓ Query**: Ask "Who leads Apple and what were their main products?"
5. **🎯 Answer**: "Tim Cook is the CEO of Apple Inc. Main products include iPhone, iPad, Mac..." (with source references)

## 🎉 Success Metrics

Your system is working well when you see:
- ✅ Documents processed: 3+ files
- ✅ Entities extracted: 50+ items  
- ✅ Relationships found: 10+ connections
- ✅ Query responses: Relevant answers with sources
- ✅ Processing time: < 30 seconds per document

## 🎉 WORKING DEMO
<img width="1904" height="683" alt="Screenshot 2025-07-21 164626" src="https://github.com/user-attachments/assets/97d76e39-dd7e-42ae-a306-59a44c8f691f" />
<img width="1467" height="869" alt="Screenshot 2025-07-21 164800" src="https://github.com/user-attachments/assets/11ad9291-f244-42ed-97a2-5cc0709abc87" />
<img width="1483" height="627" alt="Screenshot 2025-07-21 164831" src="https://github.com/user-attachments/assets/77032004-3462-4e5f-b14c-446db5b64c1e" />
<img width="1054" height="515" alt="Screenshot 2025-07-21 164854" src="https://github.com/user-attachments/assets/a3e58bdf-96c6-4988-8544-774c1782a54c" />
<img width="1062" height="384" alt="Screenshot 2025-07-21 164917" src="https://github.com/user-attachments/assets/15286591-6daf-4576-a7b4-df2479df3bb5" />
<img width="1462" height="865" alt="Screenshot 2025-07-21 165103" src="https://github.com/user-attachments/assets/7a9a33e4-579a-4edb-9f22-7dde2894de56" />
<img width="1498" height="664" alt="Screenshot 2025-07-21 165118" src="https://github.com/user-attachments/assets/21516661-3509-4a8d-b2db-bbfb4761cca7" />
<img width="1216" height="641" alt="Screenshot 2025-07-21 165206" src="https://github.com/user-attachments/assets/911f1771-2ecb-4a75-9af6-192c79a343c8" />
<img width="1499" height="762" alt="Screenshot 2025-07-21 165239" src="https://github.com/user-attachments/assets/b6a34514-2cd8-4cda-9671-7fb8c6e42d64" />


## 🤝 Need Help?

1. **Test Core System**: Run `python test_core.py`
2. **Check Logs**: Look for error messages in the terminal
3. **Start Simple**: Try with 1-2 page PDF documents first
4. **Verify Setup**: Ensure all packages installed correctly

## 📄 License

MIT License - Feel free to use, modify, and distribute for your business intelligence needs.

---

**🎯 Ready to get started?** Run `streamlit run app.py` and upload your first business document!
