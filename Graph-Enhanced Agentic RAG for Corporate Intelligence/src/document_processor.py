"""Document processing module for extracting and cleaning text from PDFs."""

import os
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pdfminer
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section of a document."""
    title: str
    content: str
    page_numbers: List[int]
    metadata: Dict[str, Any]

class DocumentProcessor:
    """Handles PDF processing and text extraction."""
    
    def __init__(self):
        self.section_patterns = [
            r'(?i)(risk\s+factors?)',
            r'(?i)(management.s?\s+discussion)',
            r'(?i)(financial\s+statements?)',
            r'(?i)(business\s+overview)',
            r'(?i)(executive\s+summary)',
            r'(?i)(operations?)',
            r'(?i)(legal\s+proceedings?)',
            r'(?i)(market\s+risk)',
        ]
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (fitz)."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {e}")
            return ""
    
    def extract_text_pdfminer(self, pdf_path: str) -> str:
        """Extract text using pdfminer."""
        try:
            laparams = LAParams(
                char_margin=1.0,
                word_margin=0.1,
                line_margin=0.5,
                boxes_flow=0.5,
                all_texts=False
            )
            return extract_text(pdf_path, laparams=laparams)
        except Exception as e:
            logger.error(f"Error extracting text with pdfminer: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely headers/footers
            if len(line) > 20:
                cleaned_lines.append(line)
        
        # Remove special characters and fix encoding issues
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\$\%]', ' ', text)
        
        return text.strip()
    
    def identify_sections(self, text: str) -> List[DocumentSection]:
        """Identify and extract document sections."""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            is_section_header = False
            for pattern in self.section_patterns:
                if re.search(pattern, line):
                    is_section_header = True
                    break
            
            # Also check for typical header formatting (short lines, all caps, etc.)
            if (len(line) < 100 and 
                (line.isupper() or 
                 any(keyword in line.lower() for keyword in ['item', 'part', 'section']))):
                is_section_header = True
            
            if is_section_header:
                # Save previous section
                if current_section:
                    sections.append(DocumentSection(
                        title=current_section,
                        content='\n'.join(current_content),
                        page_numbers=[1],  # Simplified - would need better page tracking
                        metadata={'line_number': i}
                    ))
                
                # Start new section
                current_section = line
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append(DocumentSection(
                title=current_section,
                content='\n'.join(current_content),
                page_numbers=[1],
                metadata={'line_number': len(lines)}
            ))
        
        return sections
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_idx': i,
                'end_idx': min(i + chunk_size, len(words)),
                'word_count': len(chunk_words)
            })
            
            # Break if we've reached the end
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def process_document(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Returns:
            Dict containing extracted text, sections, chunks, and metadata
        """
        logger.info(f"Processing document: {pdf_path}")
        
        # Extract text using primary method (PyMuPDF)
        raw_text = self.extract_text_pymupdf(pdf_path)
        
        # Fallback to pdfminer if PyMuPDF fails
        if not raw_text or len(raw_text) < 100:
            logger.warning("PyMuPDF extraction failed or insufficient, trying pdfminer...")
            raw_text = self.extract_text_pdfminer(pdf_path)
        
        if not raw_text:
            raise ValueError("Failed to extract text from PDF")
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Identify sections
        sections = self.identify_sections(cleaned_text)
        
        # Create chunks
        chunks = self.chunk_text(cleaned_text, chunk_size, overlap)
        
        # Extract basic metadata
        filename = os.path.basename(pdf_path)
        metadata = {
            'filename': filename,
            'file_path': pdf_path,
            'total_characters': len(cleaned_text),
            'total_words': len(cleaned_text.split()),
            'total_sections': len(sections),
            'total_chunks': len(chunks)
        }
        
        return {
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'sections': sections,
            'chunks': chunks,
            'metadata': metadata
        }
    
    def save_processed_document(self, processed_doc: Dict[str, Any], output_path: str):
        """Save processed document to file."""
        import json
        
        # Convert sections to serializable format
        serializable_doc = {
            'cleaned_text': processed_doc['cleaned_text'],
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'page_numbers': section.page_numbers,
                    'metadata': section.metadata
                } for section in processed_doc['sections']
            ],
            'chunks': processed_doc['chunks'],
            'metadata': processed_doc['metadata']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_doc, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed document saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Process a sample document
    # pdf_path = "data/documents/sample_report.pdf"
    # processed = processor.process_document(pdf_path)
    # processor.save_processed_document(processed, "data/processed/sample_report.json")
    
    print("DocumentProcessor initialized successfully!")
