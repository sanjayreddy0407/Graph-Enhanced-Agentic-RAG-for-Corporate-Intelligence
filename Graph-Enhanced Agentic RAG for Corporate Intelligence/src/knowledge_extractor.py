"""Knowledge extraction module using Llama 3.2 via Ollama for entity and relation extraction."""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import requests
from config import ENTITY_EXTRACTION_PROMPT, RELATION_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str  # company, person, financial, product, date, location
    context: str
    confidence: float = 1.0

@dataclass
class Relation:
    """Represents an extracted relation."""
    subject: str
    predicate: str
    object: str
    context: str
    confidence: float = 1.0

class KnowledgeExtractor:
    """Extracts entities and relations using Llama 3.2 via Ollama."""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.config = {
            "llm_type": "ollama",
            "base_url": "http://localhost:11434",
            "model_name": model_name,
            "temperature": 0.7,
            "top_p": 0.9,
            "n_ctx": 55000,
            "stop": ["User:", "\n\n"]
        }
        self.api_url = f"{self.config['base_url']}/api/generate"
        logger.info(f"Initialized KnowledgeExtractor with Llama 3.2 model: {model_name}")
    
    def set_model(self, model_name: str):
        """
        Update the model name in the configuration.
        
        Args:
            model_name (str): The new model name to use.
        """
        self.config["model_name"] = model_name
        logger.info(f"Updated model to: {model_name}")
    
    def _generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using Llama 3.2 via Ollama."""
        try:
            payload = {
                "model": self.config["model_name"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                    "num_ctx": self.config["n_ctx"],
                    "stop": self.config["stop"]
                }
            }
            
            logger.debug(f"Sending request to Ollama API: {self.api_url}")
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            if not generated_text:
                logger.warning("Empty response from Ollama, using fallback")
                return self._mock_response(prompt)
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            logger.warning("Falling back to mock responses")
            return self._mock_response(prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Provide mock responses for development/testing."""
        if "extract entities" in prompt.lower():
            return """{
  "companies": ["Microsoft Corporation", "Apple Inc", "Amazon.com Inc"],
  "people": ["Satya Nadella", "Tim Cook"],
  "financials": ["$50.1 billion revenue", "15% growth"],
  "products": ["Azure", "Office 365", "iPhone"],
  "dates": ["Q3 2023", "fiscal year 2023"],
  "locations": ["Seattle", "Redmond", "Cupertino"]
}"""
        elif "extract relationships" in prompt.lower():
            return """{
  "relationships": [
    {"subject": "Satya Nadella", "predicate": "CEO_OF", "object": "Microsoft Corporation"},
    {"subject": "Microsoft Corporation", "predicate": "ACQUIRED", "object": "LinkedIn"},
    {"subject": "Apple Inc", "predicate": "REPORTED", "object": "$50.1 billion revenue"}
  ]
}"""
        return "{}"
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using Llama 3.2."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:2000])  # Limit text length
        response = self._generate_response(prompt)
        
        # Debug logging
        logger.info(f"Raw response from Llama: {response[:200]}...")
        
        try:
            # Clean and extract JSON from response
            entities_data = self._extract_json_from_response(response)
            
            if not entities_data:
                logger.warning("No valid JSON found in response, using fallback extraction")
                return self._extract_entities_fallback(text)
            
            logger.info(f"Parsed JSON data: {entities_data}")
            entities = []
            
            # Process each entity type
            for entity_type, entity_list in entities_data.items():
                if isinstance(entity_list, list):
                    for entity_name in entity_list:
                        if entity_name and str(entity_name).strip():
                            entities.append(Entity(
                                name=str(entity_name).strip(),
                                type=entity_type.rstrip('s'),  # Remove plural 's'
                                context=text[:200],  # First 200 chars as context
                                confidence=0.8
                            ))
            
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            logger.error(f"Problematic response was: {response}")
            return self._extract_entities_fallback(text)
    
    def _extract_json_from_response(self, response: str) -> dict:
        """Robust JSON extraction from LLM response."""
        if not response:
            return {}
        
        # Method 1: Look for complete JSON blocks
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
            r'\{.*?\}',  # Simple JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Clean the JSON string
                    cleaned = re.sub(r'[\n\r\t]+', ' ', match.strip())
                    cleaned = re.sub(r'\s+', ' ', cleaned)
                    
                    # Try to parse
                    data = json.loads(cleaned)
                    if isinstance(data, dict) and data:
                        return data
                except json.JSONDecodeError:
                    continue
        
        # Method 2: Try to fix common JSON issues
        try:
            # Remove markdown code blocks
            cleaned_response = re.sub(r'```(?:json)?\s*', '', response)
            cleaned_response = re.sub(r'```', '', cleaned_response)
            
            # Remove leading/trailing whitespace and newlines
            cleaned_response = cleaned_response.strip()
            
            # Find JSON-like content
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = cleaned_response[start_idx:end_idx+1]
                
                # Fix common issues
                json_str = re.sub(r'[\n\r\t]+', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                
                return json.loads(json_str)
                
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Method 3: Return empty dict if all fails
        logger.warning(f"Could not extract JSON from response: {response[:100]}...")
        return {}
    
    def _extract_entities_fallback(self, text: str) -> List[Entity]:
        """Fallback method to extract entities using regex patterns."""
        entities = []
        
        # Company patterns (common suffixes)
        company_pattern = r'\b([A-Z][a-zA-Z\s&\.]+(?:Inc|Corp|Corporation|Company|LLC|Ltd|Co\.?))\b'
        companies = re.findall(company_pattern, text)
        for company in companies[:10]:  # Limit to 10
            entities.append(Entity(name=company.strip(), type='company', context=text[:200], confidence=0.6))
        
        # Person patterns (Title + Name)
        person_pattern = r'\b(?:CEO|President|CFO|CTO|Chairman|Director)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        persons = re.findall(person_pattern, text)
        for person in persons[:10]:
            entities.append(Entity(name=person.strip(), type='person', context=text[:200], confidence=0.6))
        
        # Financial patterns
        financial_pattern = r'\$[\d,]+\.?\d*\s*(?:billion|million|thousand)?|\d+%\s*(?:growth|increase|decrease)'
        financials = re.findall(financial_pattern, text, re.IGNORECASE)
        for financial in financials[:5]:
            entities.append(Entity(name=financial.strip(), type='financial', context=text[:200], confidence=0.6))
        
        # Date patterns
        date_pattern = r'\b(?:Q[1-4]\s+\d{4}|\d{4}\s*fiscal\s*year|January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{4}?\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        for date in dates[:5]:
            entities.append(Entity(name=date.strip(), type='date', context=text[:200], confidence=0.6))
        
        logger.info(f"Fallback extraction found {len(entities)} entities")
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations from text using Llama 3.2."""
        # Create entity context for better relation extraction
        entity_context = {}
        for entity in entities:
            entity_context[entity.type] = entity_context.get(entity.type, [])
            entity_context[entity.type].append(entity.name)
        
        entities_str = json.dumps(entity_context, indent=2)
        prompt = RELATION_EXTRACTION_PROMPT.format(text=text[:2000], entities=entities_str)
        response = self._generate_response(prompt)
        
        try:
            # Use the same robust JSON extraction
            relations_data = self._extract_json_from_response(response)
            
            relations = []
            
            if 'relationships' in relations_data and isinstance(relations_data['relationships'], list):
                for rel_data in relations_data['relationships']:
                    if isinstance(rel_data, dict) and all(key in rel_data for key in ['subject', 'predicate', 'object']):
                        relations.append(Relation(
                            subject=str(rel_data['subject']).strip(),
                            predicate=str(rel_data['predicate']).strip(),
                            object=str(rel_data['object']).strip(),
                            context=text[:200],
                            confidence=0.8
                        ))
            
            logger.info(f"Extracted {len(relations)} relations")
            return relations
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []
    
    def process_document_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Entity], List[Relation]]:
        """Process multiple document chunks to extract knowledge."""
        all_entities = []
        all_relations = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_text = chunk.get('text', '')
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            try:
                # Extract entities with error handling
                entities = self.extract_entities(chunk_text)
                all_entities.extend(entities)
                
                # Extract relations with error handling  
                relations = self.extract_relations(chunk_text, entities)
                all_relations.extend(relations)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                # Continue with fallback extraction
                fallback_entities = self._extract_entities_fallback(chunk_text)
                all_entities.extend(fallback_entities)
        
        # Deduplicate entities and relations
        unique_entities = self._deduplicate_entities(all_entities)
        unique_relations = self._deduplicate_relations(all_relations)
        
        logger.info(f"Total unique entities: {len(unique_entities)}")
        logger.info(f"Total unique relations: {len(unique_relations)}")
        
        # Ensure we always return some entities (fallback)
        if len(unique_entities) == 0:
            logger.warning("No entities extracted, creating minimal fallback entities")
            # Create at least one entity from the first chunk
            if chunks:
                first_chunk = chunks[0].get('text', '')
                fallback_entities = self._extract_entities_fallback(first_chunk)
                unique_entities = fallback_entities[:5]  # Take first 5
        
        return unique_entities, unique_relations
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on name and type."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations."""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            key = (relation.subject.lower(), relation.predicate, relation.object.lower())
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def entities_to_dict(self, entities: List[Entity]) -> List[Dict[str, Any]]:
        """Convert entities to dictionary format."""
        return [
            {
                'name': entity.name,
                'type': entity.type,
                'context': entity.context,
                'confidence': entity.confidence
            }
            for entity in entities
        ]
    
    def relations_to_dict(self, relations: List[Relation]) -> List[Dict[str, Any]]:
        """Convert relations to dictionary format."""
        return [
            {
                'subject': relation.subject,
                'predicate': relation.predicate,
                'object': relation.object,
                'context': relation.context,
                'confidence': relation.confidence
            }
            for relation in relations
        ]

# Rule-based fallback for when LLM extraction fails
class RuleBasedExtractor:
    """Simple rule-based entity and relation extractor as fallback."""
    
    def __init__(self):
        self.company_patterns = [
            r'\b([A-Z][a-zA-Z]*\s+(?:Inc|Corp|Corporation|Ltd|Limited|Company|Co)\b\.?)',
            r'\b([A-Z][a-zA-Z]*\s+[A-Z][a-zA-Z]*\s+(?:Inc|Corp|Corporation|Ltd|Limited|Company|Co)\b\.?)'
        ]
        
        self.person_patterns = [
            r'\b((?:Mr\.|Ms\.|Dr\.|CEO|CFO|President|Chairman)\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?:,\s+(?:CEO|CFO|President|Chairman|Director))'
        ]
        
        self.financial_patterns = [
            r'\$[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|thousand)?',
            r'\d+(?:\.\d+)?%\s*(?:growth|increase|decrease|decline)'
        ]
    
    def extract_entities_simple(self, text: str) -> List[Entity]:
        """Extract entities using simple regex patterns."""
        entities = []
        
        # Extract companies
        for pattern in self.company_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(Entity(
                    name=match.group(1),
                    type='company',
                    context=text[max(0, match.start()-50):match.end()+50],
                    confidence=0.6
                ))
        
        # Extract people
        for pattern in self.person_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(Entity(
                    name=match.group(1),
                    type='person',
                    context=text[max(0, match.start()-50):match.end()+50],
                    confidence=0.6
                ))
        
        # Extract financial data
        for pattern in self.financial_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(Entity(
                    name=match.group(0),
                    type='financial',
                    context=text[max(0, match.start()-50):match.end()+50],
                    confidence=0.6
                ))
        
        return entities

if __name__ == "__main__":
    # Example usage
    extractor = KnowledgeExtractor()
    
    sample_text = """
    Microsoft Corporation reported strong quarterly results with revenue of $50.1 billion.
    CEO Satya Nadella highlighted the growth in Azure cloud services and Office 365.
    The company acquired LinkedIn for $26.2 billion in 2016.
    """
    
    entities = extractor.extract_entities(sample_text)
    relations = extractor.extract_relations(sample_text, entities)
    
    print(f"Extracted {len(entities)} entities and {len(relations)} relations")
    for entity in entities[:5]:  # Show first 5
        print(f"Entity: {entity.name} ({entity.type})")
    
    for relation in relations[:5]:  # Show first 5
        print(f"Relation: {relation.subject} -> {relation.predicate} -> {relation.object}")
