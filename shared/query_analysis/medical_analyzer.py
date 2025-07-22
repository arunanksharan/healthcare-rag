"""
Medical query analysis including entity extraction, abbreviation expansion, and spell correction.
"""
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MedicalEntityType(str, Enum):
    """Types of medical entities."""
    DRUG = "drug"
    DISEASE = "disease"
    SYMPTOM = "symptom"
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    LAB_TEST = "lab_test"
    DOSAGE_FORM = "dosage_form"
    MEDICAL_DEVICE = "medical_device"


@dataclass
class MedicalEntity:
    """Represents a medical entity found in text."""
    text: str
    entity_type: MedicalEntityType
    normalized_form: str
    confidence: float
    synonyms: List[str] = None
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []


@dataclass
class QueryAnalysisResult:
    """Complete analysis result for a medical query."""
    original_query: str
    cleaned_query: str
    entities: List[MedicalEntity]
    expanded_abbreviations: Dict[str, str]
    corrected_terms: Dict[str, str]
    query_variants: List[str]
    medical_terms_found: List[str]


class MedicalQueryAnalyzer:
    """Analyzes medical queries for entities, abbreviations, and corrections."""
    
    def __init__(self):
        # Common medical abbreviations
        self.medical_abbreviations = {
            # Diseases/Conditions
            'dm': 'diabetes mellitus',
            'dm1': 'diabetes mellitus type 1',
            'dm2': 'diabetes mellitus type 2',
            't1dm': 'type 1 diabetes mellitus',
            't2dm': 'type 2 diabetes mellitus',
            'htn': 'hypertension',
            'mi': 'myocardial infarction',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'cad': 'coronary artery disease',
            'cvd': 'cardiovascular disease',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'gerd': 'gastroesophageal reflux disease',
            'ibs': 'irritable bowel syndrome',
            'uti': 'urinary tract infection',
            'uri': 'upper respiratory infection',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'tia': 'transient ischemic attack',
            'cva': 'cerebrovascular accident',
            'ms': 'multiple sclerosis',
            'ra': 'rheumatoid arthritis',
            'oa': 'osteoarthritis',
            'tb': 'tuberculosis',
            'hiv': 'human immunodeficiency virus',
            'aids': 'acquired immunodeficiency syndrome',
            
            # Symptoms/Signs
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'ha': 'headache',
            'n/v': 'nausea and vomiting',
            'abd': 'abdominal',
            'gi': 'gastrointestinal',
            
            # Medications/Treatment
            'abx': 'antibiotics',
            'tx': 'treatment',
            'rx': 'prescription',
            'dx': 'diagnosis',
            'sx': 'symptoms',
            'hx': 'history',
            'pmh': 'past medical history',
            'prn': 'as needed',
            'po': 'by mouth',
            'iv': 'intravenous',
            'im': 'intramuscular',
            'sq': 'subcutaneous',
            'subq': 'subcutaneous',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'qd': 'once daily',
            'qod': 'every other day',
            'qhs': 'at bedtime',
            'ac': 'before meals',
            'pc': 'after meals',
            
            # Tests/Procedures
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'cmp': 'comprehensive metabolic panel',
            'lfts': 'liver function tests',
            'tsh': 'thyroid stimulating hormone',
            'hba1c': 'hemoglobin a1c',
            'ekz': 'electrocardiogram',
            'ecg': 'electrocardiogram',
            'echo': 'echocardiogram',
            'cxr': 'chest x-ray',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'us': 'ultrasound',
            
            # Anatomy
            'gi': 'gastrointestinal',
            'cv': 'cardiovascular',
            'gu': 'genitourinary',
            'cns': 'central nervous system',
            
            # Units
            'mg': 'milligrams',
            'mcg': 'micrograms',
            'ml': 'milliliters',
            'l': 'liters',
        }
        
        # Common medical misspellings
        self.common_misspellings = {
            # Diseases
            'diabetis': 'diabetes',
            'diabets': 'diabetes',
            'hypertenion': 'hypertension',
            'hypertention': 'hypertension',
            'astma': 'asthma',
            'athsma': 'asthma',
            'neumonia': 'pneumonia',
            'pneumoia': 'pneumonia',
            'diarhea': 'diarrhea',
            'diarrea': 'diarrhea',
            
            # Medications
            'metropolol': 'metoprolol',
            'metroprolol': 'metoprolol',
            'metaformin': 'metformin',
            'metformine': 'metformin',
            'lisinipril': 'lisinopril',
            'lisinapril': 'lisinopril',
            'amoxicilin': 'amoxicillin',
            'amoxacillin': 'amoxicillin',
            'ibuprophen': 'ibuprofen',
            'ibuprofin': 'ibuprofen',
            'acetominophen': 'acetaminophen',
            'acetaminophin': 'acetaminophen',
            'omeprezole': 'omeprazole',
            'omeprazol': 'omeprazole',
            
            # Procedures
            'colonoscapy': 'colonoscopy',
            'endoscapy': 'endoscopy',
            'mamogram': 'mammogram',
            'mamography': 'mammography',
        }
        
        # Medical entity patterns
        self.entity_patterns = {
            MedicalEntityType.DRUG: [
                r'\b(\w+)(cillin|cycline|mycin|statin|pril|sartan|olol|azole|prazole|pine|done|pam|zepam)\b',
                r'\b(aspirin|insulin|metformin|lisinopril|atorvastatin|levothyroxine|amlodipine)\b',
            ],
            MedicalEntityType.DISEASE: [
                r'\b(\w+)(itis|osis|emia|oma|pathy|syndrome|disease|disorder)\b',
                r'\b(diabetes|hypertension|cancer|asthma|arthritis|pneumonia)\b',
            ],
            MedicalEntityType.SYMPTOM: [
                r'\b(pain|ache|fever|cough|nausea|vomiting|diarrhea|fatigue|weakness)\b',
                r'\b(\w+)(algia|dynia)\b',  # Pain-related
            ],
            MedicalEntityType.PROCEDURE: [
                r'\b(\w+)(scopy|ectomy|otomy|plasty|graphy|gram)\b',
                r'\b(surgery|biopsy|examination|screening|test)\b',
            ],
            MedicalEntityType.LAB_TEST: [
                r'\b(blood|urine|lab|test)\s+\w+',
                r'\b(\w+)\s+(level|count|test|panel)\b',
            ],
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for entity_type, patterns in self.entity_patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def analyze_query(self, query: str) -> QueryAnalysisResult:
        """
        Perform complete analysis of a medical query.
        
        Args:
            query: The user's query string
            
        Returns:
            QueryAnalysisResult with all analysis components
        """
        # Step 1: Clean and normalize
        cleaned_query = self._clean_query(query)
        
        # Step 2: Expand abbreviations
        expanded_query, expansions = self._expand_abbreviations(cleaned_query)
        
        # Step 3: Correct misspellings
        corrected_query, corrections = self._correct_misspellings(expanded_query)
        
        # Step 4: Extract medical entities
        entities = self._extract_entities(corrected_query)
        
        # Step 5: Find medical terms
        medical_terms = self._find_medical_terms(corrected_query)
        
        # Step 6: Generate query variants
        variants = self._generate_variants(
            corrected_query,
            entities,
            expansions,
            corrections
        )
        
        return QueryAnalysisResult(
            original_query=query,
            cleaned_query=corrected_query,
            entities=entities,
            expanded_abbreviations=expansions,
            corrected_terms=corrections,
            query_variants=variants,
            medical_terms_found=medical_terms
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Remove extra whitespace
        cleaned = ' '.join(query.split())
        
        # Normalize punctuation
        cleaned = re.sub(r'\s+([,.])', r'\1', cleaned)
        cleaned = re.sub(r'([,.])(\w)', r'\1 \2', cleaned)
        
        return cleaned
    
    def _expand_abbreviations(self, query: str) -> Tuple[str, Dict[str, str]]:
        """Expand medical abbreviations in the query."""
        expanded = query
        expansions = {}
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_abbrevs = sorted(
            self.medical_abbreviations.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for abbrev, full_form in sorted_abbrevs:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, expanded, re.IGNORECASE):
                expanded = re.sub(pattern, full_form, expanded, flags=re.IGNORECASE)
                expansions[abbrev] = full_form
        
        return expanded, expansions
    
    def _correct_misspellings(self, query: str) -> Tuple[str, Dict[str, str]]:
        """Correct common medical misspellings."""
        corrected = query
        corrections = {}
        
        for misspelling, correct in self.common_misspellings.items():
            pattern = r'\b' + re.escape(misspelling) + r'\b'
            if re.search(pattern, corrected, re.IGNORECASE):
                corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)
                corrections[misspelling] = correct
        
        return corrected, corrections
    
    def _extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text."""
        entities = []
        found_texts = set()  # Avoid duplicates
        
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    entity_text = match.group(0).lower()
                    
                    if entity_text not in found_texts:
                        found_texts.add(entity_text)
                        
                        # Create entity
                        entity = MedicalEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            normalized_form=entity_text,  # TODO: Add normalization
                            confidence=0.8,  # TODO: Calculate based on pattern
                            synonyms=self._get_synonyms(entity_text, entity_type)
                        )
                        entities.append(entity)
        
        return entities
    
    def _get_synonyms(self, term: str, entity_type: MedicalEntityType) -> List[str]:
        """Get synonyms for a medical term."""
        # Simple synonym mapping - in production, use medical ontology
        synonym_map = {
            'diabetes': ['diabetes mellitus', 'dm'],
            'hypertension': ['high blood pressure', 'htn', 'elevated blood pressure'],
            'myocardial infarction': ['heart attack', 'mi', 'acute mi'],
            'metformin': ['glucophage', 'fortamet', 'glumetza'],
            'acetaminophen': ['tylenol', 'paracetamol', 'apap'],
        }
        
        return synonym_map.get(term.lower(), [])
    
    def _find_medical_terms(self, text: str) -> List[str]:
        """Find all medical terms in the text."""
        medical_terms = []
        
        # Look for medical suffixes
        medical_suffixes = [
            'itis', 'osis', 'emia', 'oma', 'pathy',
            'algia', 'dynia', 'scopy', 'ectomy',
            'otomy', 'plasty', 'graphy', 'gram'
        ]
        
        words = text.lower().split()
        for word in words:
            for suffix in medical_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    medical_terms.append(word)
                    break
        
        return list(set(medical_terms))
    
    def _generate_variants(
        self,
        query: str,
        entities: List[MedicalEntity],
        expansions: Dict[str, str],
        corrections: Dict[str, str]
    ) -> List[str]:
        """Generate query variants for better retrieval."""
        variants = [query]
        
        # Add original query if different
        original_form = query
        for expanded, abbrev in [(v, k) for k, v in expansions.items()]:
            original_form = original_form.replace(expanded, abbrev)
        for correct, misspell in [(v, k) for k, v in corrections.items()]:
            original_form = original_form.replace(correct, misspell)
        
        if original_form != query:
            variants.append(original_form)
        
        # Add entity synonyms
        for entity in entities:
            for synonym in entity.synonyms:
                variant = query.replace(entity.text, synonym)
                if variant != query:
                    variants.append(variant)
        
        # Add focused queries for each entity
        for entity in entities:
            if entity.entity_type == MedicalEntityType.DRUG:
                variants.append(f"{entity.text} medication information")
                variants.append(f"{entity.text} drug")
            elif entity.entity_type == MedicalEntityType.DISEASE:
                variants.append(f"{entity.text} condition")
                variants.append(f"{entity.text} disease")
        
        return list(set(variants))  # Remove duplicates
