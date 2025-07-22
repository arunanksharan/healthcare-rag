#!/usr/bin/env python3
"""
Test script to verify the enhanced query processing system.
"""
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from shared.query_analysis import EnhancedQueryProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize console for pretty printing
console = Console()

# Test queries covering different intents
TEST_QUERIES = [
    # Dosage queries
    "metformin dosage for diabetes",
    "what is the dose of lisinopril",
    "how much aspirin for heart attack",
    
    # Side effects queries
    "side effects of atorvastatin",
    "adverse reactions to amoxicillin",
    "what are the complications of chemotherapy",
    
    # Diagnosis queries
    "how to diagnose diabetes",
    "symptoms of heart failure",
    "diagnostic criteria for hypertension",
    
    # Treatment queries
    "treatment for pneumonia",
    "how to manage chronic pain",
    "therapy options for depression",
    
    # Definition queries
    "what is COPD",
    "define myocardial infarction",
    "hypertension meaning",
    
    # Contraindications
    "when not to use metformin",
    "drug interactions with warfarin",
    "contraindications for aspirin",
    
    # Abbreviations and misspellings
    "tx for dm2",
    "sx of mi",
    "diabetis treatment",
    "metropolol side effects",
    
    # Complex queries
    "management of type 2 diabetes with metformin contraindications",
    "chest pain differential diagnosis and treatment options",
]


def test_query_processing():
    """Test the enhanced query processor with various medical queries."""
    processor = EnhancedQueryProcessor()
    
    console.print("\n[bold blue]Testing Enhanced Query Processing System[/bold blue]\n")
    
    for query in TEST_QUERIES:
        console.print(f"\n[bold yellow]Query:[/bold yellow] {query}")
        
        try:
            # Process query
            result = processor.process_query(query)
            
            # Create results table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Attribute", style="cyan", width=25)
            table.add_column("Value", style="green")
            
            # Add intent information
            table.add_row(
                "Intent", 
                f"{result.primary_intent.value} (confidence: {result.intent_confidence:.2f})"
            )
            
            # Add entities
            if result.analysis.entities:
                entities_str = ", ".join([
                    f"{e.text} ({e.entity_type.value})"
                    for e in result.analysis.entities
                ])
                table.add_row("Entities", entities_str)
            
            # Add abbreviations expanded
            if result.analysis.expanded_abbreviations:
                abbrev_str = ", ".join([
                    f"{k} → {v}"
                    for k, v in result.analysis.expanded_abbreviations.items()
                ])
                table.add_row("Abbreviations", abbrev_str)
            
            # Add corrections
            if result.analysis.corrected_terms:
                correct_str = ", ".join([
                    f"{k} → {v}"
                    for k, v in result.analysis.corrected_terms.items()
                ])
                table.add_row("Corrections", correct_str)
            
            # Add cleaned query
            table.add_row("Cleaned Query", result.analysis.cleaned_query)
            
            # Add top enhanced queries
            if result.enhanced_queries:
                table.add_row("Enhanced Query 1", result.enhanced_queries[0])
                if len(result.enhanced_queries) > 1:
                    table.add_row("Enhanced Query 2", result.enhanced_queries[1])
            
            # Add retrieval strategy
            strategy = result.retrieval_strategy
            table.add_row("Chunk Types", ", ".join(strategy.get("chunk_types", [])))
            table.add_row("Boost Sections", ", ".join(strategy.get("boost_sections", [])))
            table.add_row("Precision Required", str(strategy.get("precision_required", False)))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
            logger.error(f"Error: {e}", exc_info=True)


def test_specific_scenarios():
    """Test specific medical scenarios."""
    processor = EnhancedQueryProcessor()
    
    console.print("\n[bold blue]Testing Specific Medical Scenarios[/bold blue]\n")
    
    scenarios = [
        {
            "name": "Emergency Dosage Query",
            "query": "epinephrine dose for anaphylaxis emergency",
            "expected_intent": "dosage",
            "expected_precision": True,
        },
        {
            "name": "Drug Safety Query",
            "query": "is it safe to take ibuprofen with warfarin",
            "expected_intent": "contraindications",
            "expected_precision": True,
        },
        {
            "name": "Clinical Guideline Query",
            "query": "latest guidelines for hypertension management 2024",
            "expected_intent": "treatment",
            "expected_precision": False,
        },
    ]
    
    for scenario in scenarios:
        panel = Panel(
            f"[bold]{scenario['name']}[/bold]\nQuery: {scenario['query']}",
            title="Scenario Test",
            border_style="blue"
        )
        console.print(panel)
        
        result = processor.process_query(scenario['query'])
        
        # Check intent
        intent_match = result.primary_intent.value == scenario['expected_intent']
        console.print(
            f"Intent Match: {'✓' if intent_match else '✗'} "
            f"(Expected: {scenario['expected_intent']}, Got: {result.primary_intent.value})"
        )
        
        # Check precision requirement
        precision_match = result.retrieval_strategy['precision_required'] == scenario['expected_precision']
        console.print(
            f"Precision Match: {'✓' if precision_match else '✗'} "
            f"(Expected: {scenario['expected_precision']}, Got: {result.retrieval_strategy['precision_required']})"
        )
        
        console.print()


def main():
    """Run all tests."""
    test_query_processing()
    test_specific_scenarios()
    
    console.print("\n[bold green]Query processing tests complete![/bold green]\n")


if __name__ == "__main__":
    main()
