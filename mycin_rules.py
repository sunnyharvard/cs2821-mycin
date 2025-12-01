"""
MYCIN Expert System Rules
=========================

This module contains the complete rule base for MYCIN, an expert system for
diagnosing bacterial infections and recommending antibiotic treatments.

Each rule follows the MYCIN structure:
- IF conditions (antecedents)
- THEN conclusion (consequent)
- Certainty Factor (CF) ranging from -1.0 to 1.0

The rules are organized into categories:
1. Organism Identification Rules
2. Infection Site Rules
3. Treatment Recommendation Rules
4. Clinical Context Rules
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class CertaintyFactor:
    """Certainty Factor utilities for MYCIN rules"""
    STRONG_SUGGESTIVE = 0.8
    SUGGESTIVE = 0.7
    WEAKLY_SUGGESTIVE = 0.4
    VERY_WEAK = 0.2
    STRONG_DISSUASIVE = -0.8
    DISSUASIVE = -0.5


@dataclass
class RuleCondition:
    """A single condition in a rule's IF clause"""
    parameter: str  # e.g., "gram_stain", "morphology", "site"
    operator: str   # e.g., "is", "is_not", "greater_than", "less_than"
    value: Any      # e.g., "positive", "coccus", "blood"


@dataclass
class Rule:
    """A MYCIN rule"""
    rule_id: str
    category: str
    conditions: List[RuleCondition]
    conclusion: Dict[str, Any]  # e.g., {"identity": "pseudomonas", "certainty": 0.4}
    certainty_factor: float
    description: str


# ============================================================================
# QUESTIONS / PARAMETERS
# ============================================================================

# QUESTIONS dictionary maps MYCIN parameter names to question descriptions
# Note: Actual patient data uses question text from question_en_output.txt
# The mycin_question_mapping.py module handles mapping actual questions to these parameters
QUESTIONS = {
    # Patient Demographics (from CSV demographics columns)
    "age": "What is the patient's age?",
    "sex": "What is the patient's sex?",
    "weight": "What is the patient's weight (kg)?",
    
    # Infection Site (may need LLM inference)
    "site": "What is the site of the infection?",
    "culture_site": "What is the site of the culture?",
    
    # Organism Characteristics (typically require LLM inference from symptoms)
    "gram_stain": "What is the gram stain of the organism? (positive/negative) - LLM inferred",
    "morphology": "What is the morphology of the organism? (coccus/rod/spirillum) - LLM inferred",
    "growth_pattern": "How does the organism grow? (chains/pairs/clusters/single) - LLM inferred",
    "aerobicity": "Is the organism aerobic, anaerobic, or facultative? - LLM inferred",
    "spore_forming": "Is the organism spore-forming? - LLM inferred",
    
    # Clinical Context (mapped from actual questions)
    "burn": "Does the patient have burns? (serious/moderate/none) - LLM inferred",
    "immunocompromised": "Is the patient immunocompromised? (mapped from immunosuppressed/HIV questions)",
    "recent_surgery": "Has the patient had recent surgery? (mapped from 'Have you had surgery within the last month?')",
    "hospital_acquired": "Is this a hospital-acquired infection? - LLM inferred",
    "community_acquired": "Is this a community-acquired infection? - LLM inferred",
    "previous_antibiotics": "Has the patient received previous antibiotics? - LLM inferred",
    
    # Lab Results (typically require LLM inference)
    "white_blood_count": "What is the white blood cell count? - LLM inferred",
    "fever": "Does the patient have fever? (mapped from 'Do you have a fever (either felt or measured with a thermometer)?')",
    "cerebrospinal_fluid": "What are the CSF findings? - LLM inferred",
    "urine_culture": "What are the urine culture results? - LLM inferred",
    "sputum_culture": "What are the sputum culture results? - LLM inferred",
    "blood_culture": "What are the blood culture results? - LLM inferred",
    
    # Symptoms (mapped from actual questions)
    "cough": "Does the patient have a cough? (mapped from 'Do you have a cough?')",
    "dyspnea": "Does the patient have dyspnea? (mapped from shortness of breath questions)",
    "chest_pain": "Does the patient have chest pain? (mapped from chest pain questions)",
    "headache": "Does the patient have a headache? - LLM inferred",
    "nuchal_rigidity": "Does the patient have nuchal rigidity? (mapped from neck stiffness questions)",
    "urinary_symptoms": "Does the patient have urinary symptoms? - LLM inferred",
    "abdominal_pain": "Does the patient have abdominal pain? (mapped from pain questions)",
    
    # Drug Sensitivity (require LLM inference)
    "penicillin_sensitive": "Is the organism sensitive to penicillin? - LLM inferred",
    "cephalosporin_sensitive": "Is the organism sensitive to cephalosporins? - LLM inferred",
    "aminoglycoside_sensitive": "Is the organism sensitive to aminoglycosides? - LLM inferred",
    "allergy_penicillin": "Is the patient allergic to penicillin? - LLM inferred",
    "allergy_sulfa": "Is the patient allergic to sulfa drugs? - LLM inferred",
    
    # Additional parameters that can be inferred from available questions
    "respiratory_condition": "Does the patient have a respiratory condition? (mapped from COPD/asthma questions)",
    "sore_throat": "Does the patient have a sore throat? (mapped from 'Do you have a sore throat?')",
    "nasal_symptoms": "Does the patient have nasal symptoms? (mapped from nasal congestion/discharge questions)",
    "viral_infection": "Has the patient had a recent viral infection? (mapped from 'Have you recently had a viral infection?')",
    "contact_exposure": "Has the patient been in contact with infected individuals? (mapped from contact questions)",
    "travel_history": "Has the patient traveled recently? (mapped from 'Have you traveled out of the country in the last 4 weeks?')",
}


# ============================================================================
# ORGANISM IDENTIFICATION RULES
# ============================================================================

ORGANISM_IDENTIFICATION_RULES = [
    # Rule 37: Streptococcus identification
    Rule(
        rule_id="R037",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "positive"),
            RuleCondition("morphology", "is", "coccus"),
            RuleCondition("growth_pattern", "is", "chains"),
        ],
        conclusion={"identity": "streptococcus"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If gram stain is positive, morphology is coccus, and growth is in chains, then suggestive evidence for Streptococcus"
    ),
    
    # Rule 52: Pseudomonas identification
    Rule(
        rule_id="R052",
        category="organism_identification",
        conditions=[
            RuleCondition("culture_site", "is", "blood"),
            RuleCondition("gram_stain", "is", "negative"),
            RuleCondition("morphology", "is", "rod"),
            RuleCondition("burn", "is", "serious"),
        ],
        conclusion={"identity": "pseudomonas"},
        certainty_factor=CertaintyFactor.WEAKLY_SUGGESTIVE,
        description="If culture site is blood, gram negative rod, and patient has serious burns, then weakly suggestive evidence for Pseudomonas"
    ),
    
    # Rule 50: Staphylococcus identification
    Rule(
        rule_id="R050",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "positive"),
            RuleCondition("morphology", "is", "coccus"),
            RuleCondition("growth_pattern", "is", "clusters"),
        ],
        conclusion={"identity": "staphylococcus"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If gram positive coccus in clusters, then suggestive evidence for Staphylococcus"
    ),
    
    # Rule 51: E. coli identification
    Rule(
        rule_id="R051",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "negative"),
            RuleCondition("morphology", "is", "rod"),
            RuleCondition("culture_site", "is", "urine"),
            RuleCondition("aerobicity", "is", "facultative"),
        ],
        conclusion={"identity": "e_coli"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If gram negative rod in urine culture, facultative, then suggestive evidence for E. coli"
    ),
    
    # Rule 53: Klebsiella identification
    Rule(
        rule_id="R053",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "negative"),
            RuleCondition("morphology", "is", "rod"),
            RuleCondition("culture_site", "is", "sputum"),
            RuleCondition("hospital_acquired", "is", True),
        ],
        conclusion={"identity": "klebsiella"},
        certainty_factor=CertaintyFactor.WEAKLY_SUGGESTIVE,
        description="If gram negative rod in sputum, hospital-acquired, then weakly suggestive evidence for Klebsiella"
    ),
    
    # Rule 54: Neisseria meningitidis identification
    Rule(
        rule_id="R054",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "negative"),
            RuleCondition("morphology", "is", "coccus"),
            RuleCondition("culture_site", "is", "cerebrospinal_fluid"),
            RuleCondition("nuchal_rigidity", "is", True),
        ],
        conclusion={"identity": "neisseria_meningitidis"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If gram negative coccus in CSF with nuchal rigidity, then strongly suggestive evidence for N. meningitidis"
    ),
    
    # Rule 55: Haemophilus influenzae identification
    Rule(
        rule_id="R055",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "negative"),
            RuleCondition("morphology", "is", "rod"),
            RuleCondition("culture_site", "is", "cerebrospinal_fluid"),
            RuleCondition("age", "less_than", 5),
        ],
        conclusion={"identity": "haemophilus_influenzae"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If gram negative rod in CSF and patient is under 5 years, then suggestive evidence for H. influenzae"
    ),
    
    # Rule 56: Enterococcus identification
    Rule(
        rule_id="R056",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "positive"),
            RuleCondition("morphology", "is", "coccus"),
            RuleCondition("culture_site", "is", "urine"),
            RuleCondition("growth_pattern", "is", "pairs"),
        ],
        conclusion={"identity": "enterococcus"},
        certainty_factor=CertaintyFactor.WEAKLY_SUGGESTIVE,
        description="If gram positive coccus in pairs from urine, then weakly suggestive evidence for Enterococcus"
    ),
    
    # Rule 57: Clostridium identification
    Rule(
        rule_id="R057",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "positive"),
            RuleCondition("morphology", "is", "rod"),
            RuleCondition("spore_forming", "is", True),
            RuleCondition("aerobicity", "is", "anaerobic"),
        ],
        conclusion={"identity": "clostridium"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If gram positive spore-forming anaerobic rod, then suggestive evidence for Clostridium"
    ),
    
    # Rule 58: Proteus identification
    Rule(
        rule_id="R058",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "negative"),
            RuleCondition("morphology", "is", "rod"),
            RuleCondition("culture_site", "is", "urine"),
            RuleCondition("hospital_acquired", "is", True),
        ],
        conclusion={"identity": "proteus"},
        certainty_factor=CertaintyFactor.WEAKLY_SUGGESTIVE,
        description="If gram negative rod in urine, hospital-acquired, then weakly suggestive evidence for Proteus"
    ),
    
    # Rule 59: Bacteroides identification
    Rule(
        rule_id="R059",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "negative"),
            RuleCondition("morphology", "is", "rod"),
            RuleCondition("aerobicity", "is", "anaerobic"),
            RuleCondition("culture_site", "is", "abdominal"),
        ],
        conclusion={"identity": "bacteroides"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If gram negative anaerobic rod from abdominal site, then suggestive evidence for Bacteroides"
    ),
    
    # Rule 60: Pneumococcus (S. pneumoniae) identification
    Rule(
        rule_id="R060",
        category="organism_identification",
        conditions=[
            RuleCondition("gram_stain", "is", "positive"),
            RuleCondition("morphology", "is", "coccus"),
            RuleCondition("culture_site", "is", "sputum"),
            RuleCondition("cough", "is", True),
            RuleCondition("fever", "is", True),
        ],
        conclusion={"identity": "streptococcus_pneumoniae"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If gram positive coccus in sputum with cough and fever, then suggestive evidence for S. pneumoniae"
    ),
]


# ============================================================================
# INFECTION SITE RULES
# ============================================================================

INFECTION_SITE_RULES = [
    # Rule 100: Meningitis
    Rule(
        rule_id="R100",
        category="infection_site",
        conditions=[
            RuleCondition("headache", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("nuchal_rigidity", "is", True),
            RuleCondition("culture_site", "is", "cerebrospinal_fluid"),
        ],
        conclusion={"infection_site": "meningitis"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If headache, fever, nuchal rigidity, and CSF culture positive, then strongly suggestive evidence for meningitis"
    ),
    
    # Rule 101: Pneumonia
    Rule(
        rule_id="R101",
        category="infection_site",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("culture_site", "is", "sputum"),
        ],
        conclusion={"infection_site": "pneumonia"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If cough, dyspnea, fever, and sputum culture positive, then strongly suggestive evidence for pneumonia"
    ),
    
    # Rule 102: Urinary Tract Infection
    Rule(
        rule_id="R102",
        category="infection_site",
        conditions=[
            RuleCondition("urinary_symptoms", "is", True),
            RuleCondition("culture_site", "is", "urine"),
            RuleCondition("fever", "is", True),
        ],
        conclusion={"infection_site": "urinary_tract_infection"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If urinary symptoms, positive urine culture, and fever, then strongly suggestive evidence for UTI"
    ),
    
    # Rule 103: Bacteremia/Septicemia
    Rule(
        rule_id="R103",
        category="infection_site",
        conditions=[
            RuleCondition("culture_site", "is", "blood"),
            RuleCondition("fever", "is", True),
            RuleCondition("white_blood_count", "greater_than", 12000),
        ],
        conclusion={"infection_site": "bacteremia"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If positive blood culture, fever, and elevated WBC, then strongly suggestive evidence for bacteremia"
    ),
    
    # Rule 104: Wound Infection
    Rule(
        rule_id="R104",
        category="infection_site",
        conditions=[
            RuleCondition("culture_site", "is", "wound"),
            RuleCondition("recent_surgery", "is", True),
        ],
        conclusion={"infection_site": "wound_infection"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If positive wound culture and recent surgery, then suggestive evidence for wound infection"
    ),
    
    # Rule 105: Abdominal Infection
    Rule(
        rule_id="R105",
        category="infection_site",
        conditions=[
            RuleCondition("abdominal_pain", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("culture_site", "is", "abdominal"),
        ],
        conclusion={"infection_site": "abdominal_infection"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If abdominal pain, fever, and positive abdominal culture, then suggestive evidence for abdominal infection"
    ),
]


# ============================================================================
# TREATMENT RECOMMENDATION RULES
# ============================================================================

TREATMENT_RULES = [
    # Rule 200: Penicillin for Streptococcus
    Rule(
        rule_id="R200",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "streptococcus"),
            RuleCondition("penicillin_sensitive", "is", True),
            RuleCondition("allergy_penicillin", "is", False),
        ],
        conclusion={"recommended_drug": "penicillin", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Streptococcus, sensitive to penicillin, and no allergy, then strongly recommend penicillin"
    ),
    
    # Rule 201: Vancomycin for Penicillin-Allergic Streptococcus
    Rule(
        rule_id="R201",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "streptococcus"),
            RuleCondition("allergy_penicillin", "is", True),
        ],
        conclusion={"recommended_drug": "vancomycin", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Streptococcus and patient is allergic to penicillin, then recommend vancomycin"
    ),
    
    # Rule 202: Gentamicin for Pseudomonas
    Rule(
        rule_id="R202",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "pseudomonas"),
            RuleCondition("aminoglycoside_sensitive", "is", True),
        ],
        conclusion={"recommended_drug": "gentamicin", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Pseudomonas and sensitive to aminoglycosides, then recommend gentamicin"
    ),
    
    # Rule 203: Cephalosporin for E. coli
    Rule(
        rule_id="R203",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "e_coli"),
            RuleCondition("cephalosporin_sensitive", "is", True),
            RuleCondition("allergy_penicillin", "is", False),
        ],
        conclusion={"recommended_drug": "cephalosporin", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is E. coli and sensitive to cephalosporins, then recommend cephalosporin"
    ),
    
    # Rule 204: Ampicillin for E. coli (alternative)
    Rule(
        rule_id="R204",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "e_coli"),
            RuleCondition("penicillin_sensitive", "is", True),
            RuleCondition("allergy_penicillin", "is", False),
        ],
        conclusion={"recommended_drug": "ampicillin", "dose": "standard"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If organism is E. coli and sensitive to penicillin, then suggest ampicillin"
    ),
    
    # Rule 205: Methicillin/Oxacillin for Staphylococcus
    Rule(
        rule_id="R205",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "staphylococcus"),
            RuleCondition("penicillin_sensitive", "is", False),
        ],
        conclusion={"recommended_drug": "methicillin", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Staphylococcus and not sensitive to penicillin, then recommend methicillin/oxacillin"
    ),
    
    # Rule 206: Vancomycin for Resistant Staphylococcus
    Rule(
        rule_id="R206",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "staphylococcus"),
            RuleCondition("penicillin_sensitive", "is", False),
            RuleCondition("hospital_acquired", "is", True),
        ],
        conclusion={"recommended_drug": "vancomycin", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Staphylococcus, resistant to penicillin, and hospital-acquired, then recommend vancomycin"
    ),
    
    # Rule 207: Ceftriaxone for Neisseria meningitidis
    Rule(
        rule_id="R207",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "neisseria_meningitidis"),
            RuleCondition("infection_site", "is", "meningitis"),
        ],
        conclusion={"recommended_drug": "ceftriaxone", "dose": "high"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is N. meningitidis causing meningitis, then strongly recommend high-dose ceftriaxone"
    ),
    
    # Rule 208: Ampicillin for Haemophilus influenzae
    Rule(
        rule_id="R208",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "haemophilus_influenzae"),
            RuleCondition("penicillin_sensitive", "is", True),
            RuleCondition("allergy_penicillin", "is", False),
        ],
        conclusion={"recommended_drug": "ampicillin", "dose": "high"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is H. influenzae and sensitive to penicillin, then recommend high-dose ampicillin"
    ),
    
    # Rule 209: Chloramphenicol for H. influenzae (penicillin-allergic)
    Rule(
        rule_id="R209",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "haemophilus_influenzae"),
            RuleCondition("allergy_penicillin", "is", True),
        ],
        conclusion={"recommended_drug": "chloramphenicol", "dose": "standard"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If organism is H. influenzae and patient is allergic to penicillin, then suggest chloramphenicol"
    ),
    
    # Rule 210: Penicillin + Gentamicin for Enterococcus
    Rule(
        rule_id="R210",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "enterococcus"),
            RuleCondition("penicillin_sensitive", "is", True),
            RuleCondition("allergy_penicillin", "is", False),
        ],
        conclusion={"recommended_drug": "penicillin_gentamicin", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Enterococcus, then recommend combination of penicillin and gentamicin"
    ),
    
    # Rule 211: Metronidazole for Bacteroides
    Rule(
        rule_id="R211",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "bacteroides"),
        ],
        conclusion={"recommended_drug": "metronidazole", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Bacteroides, then recommend metronidazole"
    ),
    
    # Rule 212: Penicillin for Clostridium
    Rule(
        rule_id="R212",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "clostridium"),
            RuleCondition("allergy_penicillin", "is", False),
        ],
        conclusion={"recommended_drug": "penicillin", "dose": "high"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Clostridium, then recommend high-dose penicillin"
    ),
    
    # Rule 213: Ciprofloxacin for Proteus
    Rule(
        rule_id="R213",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "proteus"),
        ],
        conclusion={"recommended_drug": "ciprofloxacin", "dose": "standard"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If organism is Proteus, then suggest ciprofloxacin"
    ),
    
    # Rule 214: Ceftazidime for Klebsiella
    Rule(
        rule_id="R214",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "klebsiella"),
            RuleCondition("cephalosporin_sensitive", "is", True),
        ],
        conclusion={"recommended_drug": "ceftazidime", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is Klebsiella and sensitive to cephalosporins, then recommend ceftazidime"
    ),
    
    # Rule 215: Penicillin for S. pneumoniae
    Rule(
        rule_id="R215",
        category="treatment",
        conditions=[
            RuleCondition("identity", "is", "streptococcus_pneumoniae"),
            RuleCondition("penicillin_sensitive", "is", True),
            RuleCondition("allergy_penicillin", "is", False),
        ],
        conclusion={"recommended_drug": "penicillin", "dose": "standard"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="If organism is S. pneumoniae and sensitive to penicillin, then recommend penicillin"
    ),
]


# ============================================================================
# CLINICAL CONTEXT RULES
# ============================================================================

CLINICAL_CONTEXT_RULES = [
    # Rule 300: Hospital-acquired infection increases risk
    Rule(
        rule_id="R300",
        category="clinical_context",
        conditions=[
            RuleCondition("hospital_acquired", "is", True),
            RuleCondition("previous_antibiotics", "is", True),
        ],
        conclusion={"resistance_risk": "high"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If hospital-acquired infection and previous antibiotics, then high risk of resistance"
    ),
    
    # Rule 301: Immunocompromised patient needs broader coverage
    Rule(
        rule_id="R301",
        category="clinical_context",
        conditions=[
            RuleCondition("immunocompromised", "is", True),
        ],
        conclusion={"treatment_approach": "broad_spectrum"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If patient is immunocompromised, then recommend broad-spectrum antibiotics"
    ),
    
    # Rule 302: Serious burns increase risk of Pseudomonas
    Rule(
        rule_id="R302",
        category="clinical_context",
        conditions=[
            RuleCondition("burn", "is", "serious"),
            RuleCondition("culture_site", "is", "blood"),
        ],
        conclusion={"pseudomonas_risk": "high"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="If patient has serious burns and positive blood culture, then high risk of Pseudomonas"
    ),
]


# ============================================================================
# COMPLETE RULE BASE
# ============================================================================

ALL_RULES = (
    ORGANISM_IDENTIFICATION_RULES +
    INFECTION_SITE_RULES +
    TREATMENT_RULES +
    CLINICAL_CONTEXT_RULES
)

RULE_BY_ID = {rule.rule_id: rule for rule in ALL_RULES}

RULES_BY_CATEGORY = {
    "organism_identification": ORGANISM_IDENTIFICATION_RULES,
    "infection_site": INFECTION_SITE_RULES,
    "treatment": TREATMENT_RULES,
    "clinical_context": CLINICAL_CONTEXT_RULES,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    # Print summary
    print(f"MYCIN Rule Base Summary")
    print(f"=" * 60)
    print(f"Total Rules: {len(ALL_RULES)}")
    print(f"  - Organism Identification: {len(ORGANISM_IDENTIFICATION_RULES)}")
    print(f"  - Infection Site: {len(INFECTION_SITE_RULES)}")
    print(f"  - Treatment: {len(TREATMENT_RULES)}")
    print(f"  - Clinical Context: {len(CLINICAL_CONTEXT_RULES)}")
    print(f"\nTotal Questions/Parameters: {len(QUESTIONS)}")

