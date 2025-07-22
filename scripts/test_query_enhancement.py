#!/usr/bin/env python3
"""
Test script to demonstrate query-time enhancement with NER integration.
"""
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.query_analysis import QueryEnhancer, QueryIntent

console = Console()

# Test queries representing different medical intents
TEST_QUERIES = [
    "What is the dosage of metformin for diabetes?",
    "lisinopril side effects",
    "contraindications for aspirin",
    "symptoms of myocardial infarction",
    "treatment for hypertension",
    "warfarin coumadin drug interactions",
    "how to perform colonoscopy",
    "dm2 management guidelines"
]


def test_query_enhancement():
    """Test the query enhancement functionality."""
    console.print(Panel("QUERY-TIME ENHANCEMENT TEST", style="bold blue"))
    
    # Initialize query enhancer
    console.print("\n[bold]1. Initializing Query Enhancer...[/bold]")
    enhancer = QueryEnhancer()
    console.print("   ✓ Query enhancer initialized")
    
    # Process each test query
    console.print("\n[bold]2. Processing Test Queries:[/bold]")
    
    for i, query in enumerate(TEST_QUERIES, 1):
        console.print(f"\n[yellow]Query {i}: '{query}'[/yellow]")
        
        try:
            # Enhance the query
            enhanced = enhancer.enhance_query(query)
            
            # Display results
            console.print(f"[green]Intent:[/green] {enhanced.intent.value} (confidence: {enhanced.intent_confidence:.2f})")
            
            # Show entities
            if enhanced.entities:
                console.print("[green]Entities found:[/green]")
                for entity_type, entities in enhanced.entities.items():
                    console.print(f"  - {entity_type}: {', '.join(entities)}")
            else:
                console.print("  [dim]No entities extracted[/dim]")
            
            # Show query variants
            if enhanced.query_variants:
                console.print("[green]Query variants:[/green]")
                for variant in enhanced.query_variants[:3]:  # Show first 3
                    console.print(f"  - {variant}")
            
            # Show filters that would be applied
            if enhanced.filters:
                console.print("[green]Search filters:[/green]")
                for filter_key, filter_values in enhanced.filters.items():
                    console.print(f"  - {filter_key}: {filter_values}")
            
            # Show boost parameters
            if enhanced.boost_params:
                console.print("[green]Boost parameters:[/green]")
                console.print(f"  - Boost sections: {enhanced.boost_params.get('boost_sections', [])}")
                console.print(f"  - Boost weight: {enhanced.boost_params.get('boost_weight', 1.0)}")
                
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
    
    # Demonstrate search strategy
    console.print("\n[bold]3. Search Strategy Example:[/bold]")
    
    example_query = "metformin dosage for type 2 diabetes"
    console.print(f"\n[yellow]Example query: '{example_query}'[/yellow]")
    
    enhanced = enhancer.enhance_query(example_query)
    strategy = enhancer.get_search_strategy(enhanced)
    
    console.print("\n[green]Complete search strategy:[/green]")
    console.print(json.dumps(strategy, indent=2))
    
    # Show how the search would work
    console.print("\n[bold]4. How Enhanced Search Works:[/bold]")
    
    console.print("\n[cyan]Step 1: Query Analysis[/cyan]")
    console.print(f"  - Original: '{example_query}'")
    console.print(f"  - Cleaned: '{enhanced.cleaned_query}'")
    console.print(f"  - Intent: {enhanced.intent.value}")
    console.print(f"  - Entities: {enhanced.entities}")
    
    console.print("\n[cyan]Step 2: Vector Search[/cyan]")
    console.print("  - Generate embeddings for query variants")
    console.print("  - Search in Qdrant with cosine similarity")
    
    console.print("\n[cyan]Step 3: Apply Filters & Boosts[/cyan]")
    console.print("  - Filter chunks containing 'metformin' in drugs field")
    console.print("  - Boost chunks with answer_type='dosage_info' (+15%)")
    console.print("  - Boost chunks from 'dosage' sections (+30%)")
    console.print("  - Boost chunks with entity matches (+25%)")
    
    console.print("\n[cyan]Step 4: Score Calculation[/cyan]")
    console.print("  Example chunk scoring:")
    console.print("  - Base vector similarity: 0.85")
    console.print("  - Entity match (metformin): 0.85 × 1.25 = 1.06")
    console.print("  - Section match (dosage): 1.06 × 1.30 = 1.38")
    console.print("  - Answer type match: 1.38 × 1.15 = 1.59")
    console.print("  - Final score: 1.59")
    
    console.print("\n[bold green]✅ QUERY ENHANCEMENT TEST COMPLETE![/bold green]")


def test_intent_mapping():
    """Test how different intents map to search configurations."""
    console.print("\n[bold]Intent Mapping Test:[/bold]")
    
    enhancer = QueryEnhancer()
    
    # Create a table showing intent configurations
    intent_table = Table(show_header=True, header_style="bold magenta")
    intent_table.add_column("Intent", width=25)
    intent_table.add_column("Answer Types", width=30)
    intent_table.add_column("Boost Sections", width=30)
    intent_table.add_column("Weight", width=10)
    
    for intent in QueryIntent:
        config = enhancer.intent_configs.get(intent, {})
        intent_table.add_row(
            intent.value,
            ", ".join(config.get("answer_types", [])),
            ", ".join(config.get("boost_sections", [])),
            str(config.get("boost_weight", 1.0))
        )
    
    console.print(intent_table)


def demonstrate_retrieval_improvement():
    """Show how query enhancement improves retrieval."""
    console.print("\n[bold]Retrieval Improvement Demonstration:[/bold]")
    
    queries = [
        ("dm2", "Type 2 diabetes query with abbreviation"),
        ("aspirin for MI", "Multiple medical abbreviations"),
        ("metaformin dose", "Misspelled drug name"),
    ]
    
    enhancer = QueryEnhancer()
    
    for query, description in queries:
        console.print(f"\n[yellow]{description}[/yellow]")
        console.print(f"Original query: '{query}'")
        
        enhanced = enhancer.enhance_query(query)
        console.print(f"Enhanced query: '{enhanced.cleaned_query}'")
        
        if enhanced.query_variants:
            console.print("Variants generated:")
            for variant in enhanced.query_variants[:3]:
                console.print(f"  - {variant}")
        
        console.print(f"Entities extracted: {enhanced.entities}")


if __name__ == "__main__":
    test_query_enhancement()
    test_intent_mapping()
    demonstrate_retrieval_improvement()
