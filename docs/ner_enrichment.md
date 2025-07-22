# NER-Based Medical Enrichment

This document describes the NER-based medical entity enrichment feature for the healthcare RAG system.

## Overview

The NER enrichment feature uses BioBERT-based models to automatically extract medical entities from documents during ingestion. This provides structured metadata that improves retrieval accuracy and enables advanced filtering capabilities.

## Features

### 1. **Medical Entity Extraction**
- **Drugs/Medications**: Detects drug names, including brand and generic names
- **Diseases/Conditions**: Identifies medical conditions and diagnoses
- **Procedures/Tests**: Recognizes medical procedures and diagnostic tests
- **Dosages**: Extracts dosage information and frequencies
- **Anatomy**: Identifies body parts and organs

### 2. **Entity Processing**
- **Abbreviation Expansion**: Converts medical abbreviations (e.g., "HTN" → "hypertension")
- **Entity Normalization**: Standardizes entity representations
- **Confidence Scoring**: Assigns confidence scores to extracted entities
- **Relationship Detection**: Links related entities (e.g., drug-dosage pairs)

### 3. **Retrieval Enhancement**
- **Structured Filtering**: Filter results by specific entity types
- **Entity-Based Boosting**: Boost chunks containing relevant entities
- **Improved Precision**: Combine semantic search with entity matching

## Usage

### Enabling NER During Ingestion

```bash
# Basic ingestion (without NER)
curl -X POST "http://localhost:8000/ingestion/api/v1/ingest" \
  -F "file=@medical_document.pdf" \
  -F "title=Document Title" \
  -F "type=GUIDELINE" \
  -F "description=Document description" \
  -F "date=2024-01-15" \
  -F "embedding_type=pubmedbert" \
  -F "chunker_type=healthcare"

# With NER enrichment
curl -X POST "http://localhost:8000/ingestion/api/v1/ingest" \
  -F "file=@medical_document.pdf" \
  -F "title=Document Title" \
  -F "type=GUIDELINE" \
  -F "description=Document description" \
  -F "date=2024-01-15" \
  -F "embedding_type=pubmedbert" \
  -F "chunker_type=healthcare" \
  -F "enable_ner=true"
```

### Chunk Metadata Structure

When NER is enabled, chunks include additional metadata:

```json
{
  "chunk": "Lisinopril 10mg once daily for hypertension...",
  "chunk_type": "text",
  "page": 1,
  
  // NER-enriched fields
  "ner_entities": [
    {
      "text": "Lisinopril",
      "type": "drug",
      "confidence": 0.95,
      "normalized": "lisinopril",
      "start": 0,
      "end": 10
    },
    {
      "text": "10mg",
      "type": "dosage",
      "confidence": 0.92,
      "normalized": "10mg",
      "start": 11,
      "end": 15
    },
    {
      "text": "hypertension",
      "type": "disease",
      "confidence": 0.98,
      "normalized": "hypertension",
      "start": 31,
      "end": 43
    }
  ],
  
  "entity_summary": {
    "drug": ["lisinopril"],
    "dosage": ["10mg"],
    "disease": ["hypertension"]
  },
  
  "has_medical_entities": true,
  "drugs": ["lisinopril"],
  "diseases": ["hypertension"],
  "answer_types": ["medication_info", "dosage_info"]
}
```

### Retrieval with Entity Filters

The retrieval API can leverage entity metadata for improved search:

```python
# Future enhancement - entity-based filtering
search_request = {
    "query": "lisinopril dosage for heart failure",
    "filters": {
        "drugs": ["lisinopril"],
        "diseases": ["heart failure"],
        "entity_types": ["dosage"]
    }
}
```

## Performance Considerations

### Processing Time
- NER adds ~50-200ms per chunk depending on text length
- Models are cached after first use
- Batch processing available for multiple chunks

### Storage Impact
- Entity metadata increases chunk size by 30-50%
- Qdrant payload size increases proportionally
- Consider storage costs for large corpora

### Resource Requirements
- NER models require ~1-2GB memory each
- GPU acceleration recommended for production
- CPU inference is possible but slower

## Available NER Models

### 1. **General Medical NER** (Default)
- Model: `d4data/biomedical-ner-all`
- Coverage: Comprehensive medical entities
- Best for: General medical documents

### 2. **Disease-Focused NER**
- Model: `alvaroalon2/biobert_diseases_ner`
- Coverage: Disease and condition entities
- Best for: Clinical guidelines, diagnosis documents

### 3. **Clinical Notes NER**
- Model: `emilyalsentzer/Bio_ClinicalBERT`
- Coverage: Clinical entities from EHR notes
- Best for: Clinical documentation

## Testing

### Run NER Enrichment Test
```bash
cd scripts
python test_ner_enrichment.py
```

### Test Different Models
```bash
python test_ner_enrichment.py --test-models
```

## Architecture

```
┌─────────────────┐
│  Document PDF   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LlamaParse    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Healthcare      │
│ Chunker         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   NER Model     │ ◄── BioBERT-based
│  (if enabled)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Entity Processor │ ◄── Normalization, Relations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Enriched Chunks  │
│ with Entities   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Qdrant       │
│  Vector Store   │
└─────────────────┘
```

## Best Practices

1. **When to Enable NER**
   - Documents with specific drug/disease information
   - When structured search is required
   - For compliance/audit requirements
   - When entity-level analytics are needed

2. **When to Skip NER**
   - General narrative documents
   - Non-medical content
   - When processing speed is critical
   - Limited computational resources

3. **Model Selection**
   - Use general model for mixed content
   - Use specialized models for focused domains
   - Consider model size vs accuracy tradeoffs

## Future Enhancements

1. **UMLS Integration**
   - Link entities to UMLS concepts
   - Enable semantic relationships
   - Support medical ontologies

2. **Custom Entity Types**
   - Add organization-specific entities
   - Custom medical device recognition
   - Proprietary drug name mapping

3. **Relationship Extraction**
   - Drug-drug interactions
   - Disease-symptom relationships
   - Treatment-outcome associations

4. **Real-time Processing**
   - Streaming NER for large documents
   - Incremental entity extraction
   - Progressive enrichment

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```bash
   # Manually download models
   python scripts/download_models.py --include-ner
   ```

2. **Out of Memory**
   - Reduce batch size in NER config
   - Use CPU instead of GPU for small workloads
   - Process documents sequentially

3. **Slow Processing**
   - Enable GPU acceleration
   - Use smaller, specialized models
   - Implement caching for repeated entities

### Debug Mode

Enable detailed NER logging:
```python
import logging
logging.getLogger("shared.medical_ner").setLevel(logging.DEBUG)
```

## Conclusion

NER-based enrichment provides powerful capabilities for medical document processing, enabling structured search and improved retrieval accuracy. While it adds processing overhead, the benefits for medical use cases often justify the additional complexity.

For questions or issues, please refer to the main project documentation or open an issue on GitHub.
