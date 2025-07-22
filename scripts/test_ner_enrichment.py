#!/usr/bin/env python3
"""
Test script to demonstrate NER-based medical enrichment during ingestion.
"""
import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.utils.enhanced_healthcare_chunker import EnhancedHealthcareChunker
from shared.medical_ner import get_medical_ner_model, MedicalEntityProcessor

console = Console()

# Test medical document
MEDICAL_DOCUMENT = {
    "job_id": "ner_test",
    "pages": [{
        "page": 1,
        "width": 612,
        "height": 792,
        "items": [
            {
                "type": "heading",
                "level": 1,
                "md": "LISINOPRIL",
                "bBox": {"x": 50, "y": 50, "w": 500, "h": 30}
            },
            {
                "type": "text",
                "md": "Lisinopril is an ACE inhibitor used to treat hypertension and heart failure. It is also used to improve survival after myocardial infarction.",
                "bBox": {"x": 50, "y": 100, "w": 500, "h": 40}
            },
            {
                "type": "heading",
                "level": 2,
                "md": "DOSAGE AND ADMINISTRATION",
                "bBox": {"x": 50, "y": 160, "w": 500, "h": 25}
            },
            {
                "type": "text",
                "md": "For hypertension: Initial dose 10 mg once daily. Usual dosage range: 20-40 mg per day. Maximum dose: 80 mg per day. For CHF: Initial dose 2.5-5 mg once daily.",
                "bBox": {"x": 50, "y": 190, "w": 500, "h": 60}
            },
            {
                "type": "heading",
                "level": 2,
                "md": "CONTRAINDICATIONS",
                "bBox": {"x": 50, "y": 270, "w": 500, "h": 25}
            },
            {
                "type": "text",
                "md": "History of angioedema related to previous ACE inhibitor therapy. Concomitant use with aliskiren in patients with diabetes.",
                "bBox": {"x": 50, "y": 300, "w": 500, "h": 40}
            },
            {
                "type": "heading",
                "level": 2,
                "md": "ADVERSE REACTIONS",
                "bBox": {"x": 50, "y": 360, "w": 500, "h": 25}
            },
            {
                "type": "text",
                "md": "Common side effects include dizziness, headache, cough, hyperkalemia, and hypotension. Rare but serious: angioedema and renal impairment.",
                "bBox": {"x": 50, "y": 390, "w": 500, "h": 40}
            }
        ]
    }]
}


def test_ner_enrichment():
    """Test the NER enrichment functionality."""
    console.print(Panel("NER-BASED MEDICAL ENRICHMENT TEST", style="bold blue"))
    
    # Initialize components
    console.print("\n[bold]1. Initializing Components...[/bold]")
    
    # Initialize enhanced chunker
    chunker = EnhancedHealthcareChunker.get_instance()
    if not hasattr(EnhancedHealthcareChunker, '_tokenizer_instance'):
        EnhancedHealthcareChunker.init_tokenizer_for_worker()
    
    # Enable NER
    chunker._ner_enabled = True
    chunker.initialize_ner()
    
    console.print("   ✓ Enhanced chunker initialized")
    console.print("   ✓ NER model loaded")
    
    # Test NER on sample text
    console.print("\n[bold]2. Testing NER Extraction...[/bold]")
    
    test_text = "Lisinopril 10mg once daily for hypertension. Monitor for hyperkalemia and renal function."
    
    ner_model = chunker._ner_model
    processor = chunker._entity_processor
    
    # Extract entities
    start_time = time.time()
    ner_result = ner_model.extract_entities(test_text)
    ner_result = processor.process_entities(ner_result)
    processing_time = (time.time() - start_time) * 1000
    
    console.print(f"\n   Processing time: {processing_time:.2f}ms")
    
    # Display entities
    entity_table = Table(show_header=True, header_style="bold magenta")
    entity_table.add_column("Entity", width=20)
    entity_table.add_column("Type", width=15)
    entity_table.add_column("Confidence", width=10)
    entity_table.add_column("Normalized", width=20)
    
    for entity in ner_result.entities:
        entity_table.add_row(
            entity.text,
            entity.entity_type.value,
            f"{entity.confidence:.3f}",
            entity.normalized_text or "-"
        )
    
    console.print(entity_table)
    
    # Chunk document with NER
    console.print("\n[bold]3. Chunking Document with NER Enrichment...[/bold]")
    
    metadata = {
        "original_filename": "lisinopril_prescribing_info.pdf",
        "parse_type": "pdf",
        "enable_ner": True
    }
    
    chunks = chunker.chunk_json(MEDICAL_DOCUMENT, metadata)
    console.print(f"\n   Generated {len(chunks)} enriched chunks")
    
    # Analyze enriched chunks
    console.print("\n[bold]4. Chunk Analysis:[/bold]")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        console.print(f"\n[yellow]Chunk {i+1}:[/yellow]")
        console.print(f"Text: {chunk['chunk'][:100]}...")
        
        if "ner_entities" in chunk and chunk["ner_entities"]:
            console.print("\n[green]Extracted Entities:[/green]")
            
            entity_summary = chunk.get("entity_summary", {})
            for entity_type, entities in entity_summary.items():
                console.print(f"  {entity_type}: {', '.join(entities)}")
            
            # Show detailed entity info
            console.print("\n[cyan]Detailed Entities:[/cyan]")
            for entity in chunk["ner_entities"][:5]:  # Show first 5
                console.print(f"  - '{entity['text']}' ({entity['type']}) [conf: {entity['confidence']:.2f}]")
        else:
            console.print("  [dim]No entities found[/dim]")
        
        console.print(f"\nAnswer Types: {chunk.get('answer_types', [])}")
        console.print(f"Has Medical Entities: {chunk.get('has_medical_entities', False)}")
    
    # Summary statistics
    console.print("\n[bold]5. Summary Statistics:[/bold]")
    
    total_entities = sum(len(chunk.get("ner_entities", [])) for chunk in chunks)
    chunks_with_entities = sum(1 for chunk in chunks if chunk.get("has_medical_entities", False))
    
    # Entity type distribution
    entity_types = {}
    for chunk in chunks:
        for entity in chunk.get("ner_entities", []):
            entity_type = entity["type"]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    console.print(f"   Total entities extracted: {total_entities}")
    console.print(f"   Chunks with entities: {chunks_with_entities}/{len(chunks)}")
    console.print("\n   Entity type distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        console.print(f"     - {entity_type}: {count}")
    
    # Show how this improves retrieval
    console.print("\n[bold]6. Retrieval Benefits:[/bold]")
    
    console.print("\n   [green]Without NER:[/green] Vector similarity only")
    console.print("   - Query: 'lisinopril dosage for heart failure'")
    console.print("   - Relies entirely on embedding similarity")
    
    console.print("\n   [green]With NER:[/green] Structured filtering + vector similarity")
    console.print("   - Can filter chunks containing drug='lisinopril'")
    console.print("   - Can boost chunks with disease='heart failure'")
    console.print("   - Can prioritize chunks with answer_type='dosage_info'")
    
    # Performance considerations
    console.print("\n[bold]7. Performance Considerations:[/bold]")
    
    avg_entities_per_chunk = total_entities / len(chunks) if chunks else 0
    console.print(f"   Average entities per chunk: {avg_entities_per_chunk:.1f}")
    console.print(f"   Estimated NER processing time per chunk: ~{processing_time:.0f}ms")
    console.print(f"   Total estimated processing time for document: ~{len(chunks) * processing_time:.0f}ms")
    
    # Storage impact
    regular_chunk_size = len(json.dumps({"chunk": chunks[0]["chunk"], "page": 1}))
    enriched_chunk_size = len(json.dumps(chunks[0]))
    storage_increase = ((enriched_chunk_size - regular_chunk_size) / regular_chunk_size) * 100
    
    console.print(f"\n   Storage impact:")
    console.print(f"   - Regular chunk size: ~{regular_chunk_size} bytes")
    console.print(f"   - Enriched chunk size: ~{enriched_chunk_size} bytes")
    console.print(f"   - Storage increase: ~{storage_increase:.0f}%")
    
    console.print("\n[bold green]✅ NER ENRICHMENT TEST COMPLETE![/bold green]")


def test_ner_models():
    """Test different NER models to compare performance."""
    console.print("\n[bold]Testing Different NER Models:[/bold]")
    
    test_texts = [
        "Patient prescribed metformin 500mg BID for type 2 diabetes mellitus.",
        "MRI showed evidence of hepatocellular carcinoma. Recommended sorafenib therapy.",
        "CBC revealed anemia with hemoglobin 8.2 g/dL. Started on iron supplementation.",
    ]
    
    models_to_test = [
        ("d4data/biomedical-ner-all", "General medical NER"),
        ("alvaroalon2/biobert_diseases_ner", "Disease-focused NER"),
    ]
    
    for model_name, description in models_to_test:
        console.print(f"\n[yellow]Model: {model_name}[/yellow]")
        console.print(f"Description: {description}")
        
        try:
            model = get_medical_ner_model(model_name)
            processor = MedicalEntityProcessor()
            
            for text in test_texts[:1]:  # Test first text
                result = model.extract_entities(text)
                result = processor.process_entities(result)
                
                console.print(f"\nText: '{text}'")
                console.print(f"Entities found: {len(result.entities)}")
                for entity in result.entities:
                    console.print(f"  - {entity.text} ({entity.entity_type.value})")
        except Exception as e:
            console.print(f"[red]Error testing model: {e}[/red]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NER medical enrichment")
    parser.add_argument("--test-models", action="store_true", help="Test different NER models")
    args = parser.parse_args()
    
    if args.test_models:
        test_ner_models()
    else:
        test_ner_enrichment()
