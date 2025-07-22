# Query-Time Enhancement with NER Integration

This document describes the query-time enhancement system that works in conjunction with NER-enriched chunks to provide highly relevant search results.

## Overview

The query-time enhancement system analyzes user queries to:
1. Extract medical entities using NER
2. Detect query intent (dosage lookup, side effects, etc.)
3. Generate query variants for better recall
4. Apply intelligent filtering and boosting during retrieval

## Architecture

```
User Query
    ↓
┌─────────────────────┐
│  Query Enhancer     │
├─────────────────────┤
│ • Medical Analyzer  │ ← Abbreviation expansion, spell correction
│ • Intent Detection  │ ← Classify query purpose
│ • Entity Extraction │ ← NER for drugs, diseases, etc.
│ • Variant Generator │ ← Create search variations
└─────────────────────┘
    ↓
Enhanced Query Object
    ↓
┌─────────────────────┐
│  Search Strategy    │
├─────────────────────┤
│ • Entity Filters    │ ← Filter by drugs, diseases
│ • Boost Parameters  │ ← Section & answer type boosts
│ • Query Variants    │ ← Multiple search queries
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Vector Search      │
├─────────────────────┤
│ • Embedding Search  │ ← Semantic similarity
│ • Metadata Filters  │ ← Entity-based filtering
│ • Score Boosting    │ ← Intent-aware ranking
└─────────────────────┘
    ↓
Ranked Results
```

## Key Components

### 1. Query Enhancement (`QueryEnhancer`)

The main orchestrator that:
- Analyzes medical queries
- Detects intent with confidence scores
- Extracts entities and their types
- Generates search strategies

### 2. Medical Query Analyzer

Performs medical-specific query processing:
- **Abbreviation Expansion**: "HTN" → "hypertension"
- **Spell Correction**: "metaformin" → "metformin"
- **Entity Extraction**: Identifies drugs, diseases, procedures
- **Synonym Generation**: Creates variations for better matching

### 3. Intent Detection

Maps queries to specific intents:
- `DOSAGE_LOOKUP`: Questions about drug dosing
- `SIDE_EFFECTS_LOOKUP`: Adverse reaction queries
- `CONTRAINDICATIONS_LOOKUP`: When not to use medications
- `DISEASE_INFO`: Disease-related queries
- `TREATMENT_LOOKUP`: Treatment options
- `DRUG_INTERACTION`: Drug interaction queries
- `PROCEDURE_INFO`: Medical procedure information

### 4. Search Strategy Generation

Creates comprehensive search plans:
```json
{
  "query_texts": ["metformin dosage diabetes", "metformin dose", "glucophage dosage"],
  "filters": {
    "drugs": ["metformin", "glucophage"],
    "answer_types": ["dosage_info", "medication_info"]
  },
  "boost_params": {
    "boost_sections": ["dosage", "administration"],
    "boost_weight": 1.3
  },
  "intent": "dosage_lookup",
  "use_entity_filtering": true
}
```

## Scoring System

### Base Score
- Vector similarity from embedding search (0.0 - 1.0)

### Boost Factors
1. **Entity Match**: +25% if chunk contains query entities
2. **Section Match**: +30% for relevant sections
3. **Answer Type Match**: +15% for matching answer types
4. **Chunk Type Match**: +20% for relevant chunk types

### Example Scoring
```
Query: "metformin dosage for diabetes"
Chunk: From "DOSAGE AND ADMINISTRATION" section, contains "metformin"

Base similarity: 0.85
Entity boost (metformin): 0.85 × 1.25 = 1.06
Section boost (dosage): 1.06 × 1.30 = 1.38
Answer type boost: 1.38 × 1.15 = 1.59
Final score: 1.59
```

## Query Processing Examples

### Example 1: Dosage Query
```
Input: "What is the dosage of metformin for diabetes?"
Enhanced: "metformin dosage diabetes mellitus"
Entities: {drugs: ["metformin"], diseases: ["diabetes"]}
Intent: DOSAGE_LOOKUP
Filters: drugs=["metformin"], answer_types=["dosage_info"]
Boosts: sections=["dosage", "administration"]
```

### Example 2: Side Effects Query
```
Input: "lisinopril side effects"
Enhanced: "lisinopril adverse reactions side effects"
Entities: {drugs: ["lisinopril"]}
Intent: SIDE_EFFECTS_LOOKUP
Filters: drugs=["lisinopril"], answer_types=["side_effects"]
Boosts: sections=["adverse_reactions", "warnings"]
```

### Example 3: Abbreviation Handling
```
Input: "HTN treatment guidelines"
Enhanced: "hypertension treatment guidelines"
Entities: {diseases: ["hypertension"]}
Intent: TREATMENT_LOOKUP
Filters: diseases=["hypertension"], answer_types=["treatment"]
```

## API Usage

### Search Endpoint with Enhancement

```bash
curl -X POST "http://localhost:8001/retrieval/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "metformin dosage for type 2 diabetes",
    "embedding_types": ["pubmedbert"]
  }'
```

### Response Structure
```json
{
  "message": "Search logic executed successfully.",
  "enhanced_query": "metformin dosage type 2 diabetes mellitus",
  "query_variants": [
    "metformin dose diabetes",
    "glucophage dosage type 2 diabetes"
  ],
  "llm_response": "...",
  "query_analysis": {
    "intent": "dosage_lookup",
    "confidence": 0.92,
    "entities": {
      "drug": ["metformin"],
      "disease": ["diabetes mellitus", "type 2 diabetes"]
    },
    "filters_applied": {
      "drugs": ["metformin", "glucophage"],
      "answer_types": ["dosage_info", "medication_info"]
    },
    "boost_params": {
      "boost_sections": ["dosage", "administration"],
      "boost_weight": 1.3
    }
  }
}
```

## Benefits

### 1. **Improved Precision**
- Entity filtering ensures only relevant chunks are considered
- Intent-based boosting prioritizes the right content

### 2. **Better Recall**
- Query variants capture different phrasings
- Abbreviation expansion finds more matches
- Synonym inclusion catches related terms

### 3. **Explainability**
- Clear indication of why results were selected
- Transparency in filtering and boosting
- Detailed query analysis in response

### 4. **Performance**
- Metadata filtering reduces search space
- Early filtering improves response time
- Focused search on relevant collections

## Configuration

### Intent Configurations
Each intent has specific search parameters:

```python
QueryIntent.DOSAGE_LOOKUP: {
    "answer_types": ["dosage_info", "medication_info"],
    "boost_sections": ["dosage", "administration"],
    "entity_types": ["drug", "dosage"],
    "boost_weight": 1.3
}
```

### Customization
- Adjust boost weights for different use cases
- Add new intents for specialized queries
- Modify entity extraction patterns

## Testing

### Run Query Enhancement Tests
```bash
python scripts/test_query_enhancement.py
```

### Example Test Output
```
Query: "What is the dosage of metformin for diabetes?"
Intent: dosage_lookup (confidence: 0.92)
Entities found:
  - drug: metformin
  - disease: diabetes
Query variants:
  - metformin dosage
  - metformin dose
  - how much metformin
Search filters:
  - drugs: ['metformin']
  - answer_types: ['dosage_info', 'medication_info']
```

## Best Practices

1. **Query Design**
   - Use specific medical terms when possible
   - Include relevant context (disease, patient type)
   - Specify the type of information needed

2. **System Configuration**
   - Adjust boost weights based on user feedback
   - Monitor which intents are most common
   - Fine-tune entity extraction patterns

3. **Performance Optimization**
   - Cache frequently used query enhancements
   - Pre-compute embeddings for common variants
   - Use appropriate embedding models

## Integration with NER

The query enhancement system works seamlessly with NER-enriched chunks:

1. **During Ingestion**: Chunks are enriched with entities
2. **During Query**: Entities are extracted from queries
3. **During Retrieval**: Entity matching boosts relevance
4. **Result**: Highly relevant, entity-aware search results

## Future Enhancements

1. **Learning from Feedback**
   - Adjust boost weights based on click-through rates
   - Improve intent detection with user feedback

2. **Query Expansion**
   - Use medical ontologies for synonym expansion
   - Include related concepts automatically

3. **Personalization**
   - Adapt to user preferences
   - Learn from query history

4. **Multi-lingual Support**
   - Handle queries in multiple languages
   - Cross-lingual entity matching

## Conclusion

The query-time enhancement system, combined with NER-enriched chunks, provides a powerful medical information retrieval system that understands user intent, extracts relevant entities, and delivers highly relevant results through intelligent filtering and boosting strategies.
