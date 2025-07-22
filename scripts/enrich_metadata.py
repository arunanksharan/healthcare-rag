#!/usr/bin/env python3
"""
Script to enrich existing chunks in Qdrant with intent-based metadata.
This allows the new query processing system to work with existing data.
"""
import logging
import asyncio
from typing import Dict, List, Any
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models

from shared.embeddings import EmbeddingType, get_collection_name
from shared.query_analysis import MedicalQueryAnalyzer, MedicalEntityType
from data_retrieval.core.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
medical_analyzer = MedicalQueryAnalyzer()


def analyze_chunk_content(chunk_text: str, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a chunk to extract medical entities and determine content type.
    Only adds missing fields, preserves existing metadata from chunker.
    """
    # If chunk already has the new metadata, skip analysis
    if all(key in existing_metadata for key in ["answer_types", "medical_entities", "entity_types"]):
        return {}
    
    # Extract medical entities
    analysis = medical_analyzer.analyze_query(chunk_text)
    
    # Use answer_types from chunk if available, otherwise analyze
    answer_types = existing_metadata.get("answer_types")
    if not answer_types:
        answer_types = []
        text_lower = chunk_text.lower()
        
        # Check for definition content
        if any(phrase in text_lower for phrase in ["is a", "refers to", "defined as", "means"]):
            answer_types.append("definition")
        
        # Check for dosage content
        if any(term in text_lower for term in ["mg", "ml", "dose", "dosage", "administration"]):
            answer_types.append("dosage")
        
        # Check for side effects
        if any(term in text_lower for term in ["side effect", "adverse", "reaction", "complication"]):
            answer_types.append("side_effects")
        
        # Check for contraindications
        if any(term in text_lower for term in ["contraindication", "do not use", "avoid", "interaction"]):
            answer_types.append("contraindications")
        
        # Check for treatment content
        if any(term in text_lower for term in ["treatment", "therapy", "management", "protocol"]):
            answer_types.append("treatment")
        
        # Check for diagnostic content
        if any(term in text_lower for term in ["diagnosis", "diagnostic", "symptom", "sign", "test"]):
            answer_types.append("diagnosis")
    
    # Extract entity information if not present
    medical_entities = existing_metadata.get("medical_entities", [])
    entity_types = existing_metadata.get("entity_types", [])
    
    if not medical_entities:
        for entity in analysis.entities:
            medical_entities.append(entity.normalized_form)
            entity_types.append(entity.entity_type.value)
    
    # Determine boost_section based on section_type if available
    boost_section = existing_metadata.get("boost_section")
    if not boost_section and "section_type" in existing_metadata:
        section_mapping = {
            "medications": "dosage",
            "dosage": "dosage",
            "contraindications": "contraindications",
            "adverse_reactions": "side_effects",
            "treatment": "treatment",
            "diagnosis": "diagnosis",
        }
        boost_section = section_mapping.get(existing_metadata["section_type"])
    
    # Build update dict with only new fields
    update_dict = {}
    
    if "answer_types" not in existing_metadata:
        update_dict["answer_types"] = answer_types if answer_types else ["general"]
    
    if "medical_entities" not in existing_metadata:
        update_dict["medical_entities"] = medical_entities
    
    if "entity_types" not in existing_metadata:
        update_dict["entity_types"] = list(set(entity_types))
    
    if "has_medical_content" not in existing_metadata:
        update_dict["has_medical_content"] = len(medical_entities) > 0
    
    if boost_section and "boost_section" not in existing_metadata:
        update_dict["boost_section"] = boost_section
    
    return update_dict


async def enrich_collection(collection_name: str, batch_size: int = 100):
    """
    Enrich all points in a collection with medical metadata.
    """
    logger.info(f"Starting enrichment for collection: {collection_name}")
    
    # Get collection info
    try:
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        logger.info(f"Collection has {total_points} points")
    except Exception as e:
        logger.error(f"Error accessing collection {collection_name}: {e}")
        return
    
    # Process in batches
    offset = None
    processed = 0
    
    with tqdm(total=total_points, desc=f"Enriching {collection_name}") as pbar:
        while True:
            # Scroll through points
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not records:
                break
            
            # Process each record
            updates = []
            for record in records:
                payload = record.payload or {}
                text = payload.get("text", "")
                
                if not text:
                    continue
                
                # Analyze chunk content, preserving existing metadata
                new_metadata = analyze_chunk_content(text, payload)
                
                # Only update if there are new fields to add
                if new_metadata:
                    new_payload = {
                        **payload,
                        **new_metadata,
                        "enrichment_version": "2.0",
                    }
                else:
                    # Skip if already has all metadata
                    continue
                
                updates.append(
                    models.PointStruct(
                        id=record.id,
                        payload=new_payload,
                    )
                )
            
            # Update points in batch
            if updates:
                client.upsert(
                    collection_name=collection_name,
                    points=updates,
                )
                processed += len(updates)
                pbar.update(len(updates))
            
            # Move to next batch
            offset = next_offset
            if offset is None:
                break
    
    logger.info(f"Enriched {processed} points in {collection_name}")


async def enrich_all_collections():
    """
    Enrich all medical embedding collections.
    """
    # Get all embedding types
    embedding_types = [
        EmbeddingType.PUBMEDBERT,
        EmbeddingType.BIOBERT,
        EmbeddingType.SCIBERT,
        EmbeddingType.SPECTER,
        EmbeddingType.OPENAI,
    ]
    
    for embedding_type in embedding_types:
        collection_name = get_collection_name(settings.qdrant_collection_name, embedding_type)
        
        # Check if collection exists
        try:
            client.get_collection(collection_name)
            await enrich_collection(collection_name)
        except Exception as e:
            logger.info(f"Collection {collection_name} not found or error: {e}")
            continue


def main():
    """
    Main enrichment process.
    """
    logger.info("Starting metadata enrichment process...")
    
    # Run enrichment
    asyncio.run(enrich_all_collections())
    
    logger.info("Metadata enrichment complete!")


if __name__ == "__main__":
    main()
