"""
MYCIN-Style Rules for General Medical Diagnosis
================================================

This module contains MYCIN-style rules for diagnosing general medical conditions
based on patient symptoms and evidence. Designed to work with the evaluation dataset.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class CertaintyFactor:
    """Certainty Factor utilities for MYCIN rules"""
    STRONG_SUGGESTIVE = 0.8
    SUGGESTIVE = 0.7
    MODERATE = 0.6
    WEAKLY_SUGGESTIVE = 0.4
    VERY_WEAK = 0.2
    STRONG_DISSUASIVE = -0.8
    DISSUASIVE = -0.5


@dataclass
class RuleCondition:
    """A single condition in a rule's IF clause"""
    parameter: str
    operator: str  # "is", "is_not", "greater_than", "less_than", "contains"
    value: Any


@dataclass
class Rule:
    """A MYCIN-style rule for medical diagnosis"""
    rule_id: str
    category: str
    conditions: List[RuleCondition]
    conclusion: Dict[str, Any]  # {"diagnosis": "condition_name"}
    certainty_factor: float
    description: str


# ============================================================================
# QUESTIONS / PARAMETERS
# ============================================================================

QUESTIONS = {
    # Demographics
    "age": "What is the patient's age?",
    "sex": "What is the patient's sex?",
    
    # Respiratory symptoms
    "cough": "Does the patient have a cough?",
    "productive_cough": "Does the patient have a productive cough?",
    "dyspnea": "Does the patient have shortness of breath?",
    "wheezing": "Does the patient have wheezing?",
    "fever": "Does the patient have a fever?",
    "sore_throat": "Does the patient have a sore throat?",
    "nasal_congestion": "Does the patient have nasal congestion?",
    "hoarse_voice": "Does the patient have hoarseness?",
    
    # Pain symptoms
    "chest_pain": "Does the patient have chest pain?",
    "chest_pain_at_rest": "Does the patient have chest pain at rest?",
    "chest_pain_exertion": "Does chest pain worsen with exertion?",
    "chest_pain_breathing": "Does chest pain worsen with breathing?",
    "abdominal_pain": "Does the patient have abdominal pain?",
    "pain_location": "Where is the pain located?",
    "pain_character": "What is the character of the pain?",
    "pain_radiates": "Does the pain radiate?",
    
    # GI symptoms
    "heartburn": "Does the patient have heartburn?",
    "burning_sensation": "Does the patient have a burning sensation?",
    "worse_lying_down": "Are symptoms worse when lying down?",
    "worse_after_eating": "Are symptoms worse after eating?",
    "black_stools": "Does the patient have black stools?",
    "hiatal_hernia": "Does the patient have a hiatal hernia?",
    "alcohol_abuse": "Does the patient abuse alcohol?",
    "overweight": "Is the patient overweight?",
    
    # Cardiac symptoms
    "palpitations": "Does the patient have palpitations?",
    "syncope": "Does the patient have syncope?",
    
    # Neurological symptoms
    "headache": "Does the patient have a headache?",
    "confusion": "Does the patient have confusion?",
    "muscle_spasms": "Does the patient have muscle spasms?",
    "tongue_protrusion": "Does the patient have trouble keeping tongue in mouth?",
    "eyelid_droop": "Does the patient have eyelid drooping?",
    "recent_antipsychotics": "Has the patient taken antipsychotics recently?",
    
    # Other symptoms
    "contact_exposure": "Has the patient been in contact with sick people?",
    "travel_history": "Has the patient traveled recently?",
    "smoking": "Does the patient smoke?",
    "copd": "Does the patient have COPD?",
    "asthma": "Does the patient have asthma?",
    "daycare_exposure": "Does the patient attend/work in daycare?",
    "household_size": "How many people live in the household?",
    "pneumothorax_history": "Has the patient had pneumothorax before?",
    "family_pneumothorax": "Has family had pneumothorax?",
    "intense_coughing_fits": "Does the patient have intense coughing fits?",
    "premature_birth": "Was the patient born prematurely?",
    
    # Additional parameters for new rules
    "ear_pain": "Does the patient have ear pain?",
    "recent_cold": "Has the patient had a recent cold?",
    "dyspnea_at_rest": "Does the patient have dyspnea at rest?",
    "heart_failure": "Does the patient have heart failure?",
    "facial_pain": "Does the patient have facial pain?",
    "greenish_discharge": "Does the patient have greenish nasal discharge?",
    "itchy_nose": "Does the patient have an itchy nose?",
    "allergy_history": "Does the patient have a history of allergies?",
    "allergy_exposure": "Has the patient been exposed to an allergen?",
    "swelling": "Does the patient have swelling?",
    "pale_skin": "Does the patient have pale skin?",
    "fatigue": "Does the patient have fatigue?",
    "anemia_history": "Does the patient have a history of anemia?",
    "irregular_heartbeat": "Does the patient have an irregular heartbeat?",
    "vomiting_blood": "Has the patient vomited blood?",
    "chronic_cough": "Does the patient have a chronic cough?",
    "recurrent_infections": "Does the patient have recurrent infections?",
    "cardiac_symptoms": "Does the patient have cardiac symptoms?",
    "chronic_sinusitis": "Does the patient have chronic sinusitis?",
    "nasal_polyps": "Does the patient have nasal polyps?",
    "severe_headache": "Does the patient have a severe headache?",
    "family_cluster_headache": "Does the patient have a family history of cluster headaches?",
    "stridor": "Does the patient have stridor?",
    "ebola_contact": "Has the patient been in contact with Ebola?",
    "bleeding": "Does the patient have bleeding?",
    "weakness_limbs": "Does the patient have limb weakness?",
    "numbness": "Does the patient have numbness?",
    "recent_infection": "Has the patient had a recent infection?",
    "hiv_risk": "Does the patient have HIV risk factors?",
    "groin_pain": "Does the patient have groin pain?",
    "groin_swelling": "Does the patient have groin swelling?",
    "pain_with_coughing": "Does the patient have pain with coughing?",
    "suffocating_feeling": "Does the patient have a suffocating feeling?",
    "inability_to_breathe": "Does the patient have an inability to breathe?",
    "localized_swelling": "Does the patient have localized swelling?",
    "pain_at_site": "Does the patient have pain at the site?",
    "muscle_weakness": "Does the patient have muscle weakness?",
    "weakness_worse_fatigue": "Does weakness worsen with fatigue?",
    "recent_viral_infection": "Has the patient had a recent viral infection?",
    "rapid_heartbeat": "Does the patient have a rapid heartbeat?",
    "weight_loss": "Has the patient had weight loss?",
    "family_pancreatic_cancer": "Does the patient have a family history of pancreatic cancer?",
    "anxiety": "Does the patient have anxiety?",
    "feeling_dying": "Does the patient feel like they are dying?",
    "chest_pain_improves_forward": "Does chest pain improve when leaning forward?",
    "pericarditis_history": "Does the patient have a history of pericarditis?",
    "dvt_history": "Does the patient have a history of DVT?",
    "joint_pain": "Does the patient have joint pain?",
    "rash": "Does the patient have a rash?",
    "fish_consumption": "Has the patient consumed fish?",
    "nausea": "Does the patient have nausea?",
    "flushing": "Does the patient have flushing?",
    "chest_pain_movement": "Does chest pain worsen with movement?",
}

ASK_FIRST_PARAMETERS = set(QUESTIONS.keys())


# ============================================================================
# DIAGNOSIS RULES
# ============================================================================

def create_medical_rules() -> List[Rule]:
    """Create MYCIN-style rules for general medical diagnosis"""
    
    rules = []
    
    # ========================================================================
    # GERD Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="GERD001",
        category="GI",
        conditions=[
            RuleCondition("heartburn", "is", True),
            RuleCondition("burning_sensation", "is", True),
            RuleCondition("worse_lying_down", "is", True),
        ],
        conclusion={"diagnosis": "GERD"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="GERD: heartburn + burning sensation + worse when lying down"
    ))
    
    rules.append(Rule(
        rule_id="GERD002",
        category="GI",
        conditions=[
            RuleCondition("burning_sensation", "is", True),
            RuleCondition("hiatal_hernia", "is", True),
            RuleCondition("worse_lying_down", "is", True),
        ],
        conclusion={"diagnosis": "GERD"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="GERD: burning sensation + hiatal hernia + worse when lying down"
    ))
    
    rules.append(Rule(
        rule_id="GERD003",
        category="GI",
        conditions=[
            RuleCondition("heartburn", "is", True),
            RuleCondition("worse_after_eating", "is", True),
            RuleCondition("overweight", "is", True),
        ],
        conclusion={"diagnosis": "GERD"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="GERD: heartburn + worse after eating + overweight"
    ))
    
    # ========================================================================
    # Bronchitis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="BRONCH001",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("productive_cough", "is", True),
            RuleCondition("fever", "is", True),
        ],
        conclusion={"diagnosis": "Bronchitis"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Bronchitis: cough + productive cough + fever"
    ))
    
    rules.append(Rule(
        rule_id="BRONCH002",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("smoking", "is", True),
        ],
        conclusion={"diagnosis": "Bronchitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Bronchitis: cough + fever + smoking"
    ))
    
    rules.append(Rule(
        rule_id="BRONCH003",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("productive_cough", "is", True),
            RuleCondition("copd", "is", True),
        ],
        conclusion={"diagnosis": "Bronchitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Bronchitis: cough + productive cough + COPD"
    ))
    
    # ========================================================================
    # Pneumonia Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="PNEUM001",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("productive_cough", "is", True),
        ],
        conclusion={"diagnosis": "Pneumonia"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Pneumonia: cough + fever + dyspnea + productive cough"
    ))
    
    rules.append(Rule(
        rule_id="PNEUM002",
        category="Respiratory",
        conditions=[
            RuleCondition("fever", "is", True),
            RuleCondition("chest_pain", "is", True),
            RuleCondition("chest_pain_breathing", "is", True),
            RuleCondition("cough", "is", True),
        ],
        conclusion={"diagnosis": "Pneumonia"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Pneumonia: fever + chest pain + pain with breathing + cough"
    ))
    
    # ========================================================================
    # Influenza Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="FLU001",
        category="Respiratory",
        conditions=[
            RuleCondition("fever", "is", True),
            RuleCondition("sore_throat", "is", True),
            RuleCondition("cough", "is", True),
            RuleCondition("contact_exposure", "is", True),
        ],
        conclusion={"diagnosis": "Influenza"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Influenza: fever + sore throat + cough + contact exposure"
    ))
    
    rules.append(Rule(
        rule_id="FLU002",
        category="Respiratory",
        conditions=[
            RuleCondition("fever", "is", True),
            RuleCondition("nasal_congestion", "is", True),
            RuleCondition("cough", "is", True),
            RuleCondition("contact_exposure", "is", True),
        ],
        conclusion={"diagnosis": "Influenza"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Influenza: fever + nasal congestion + cough + contact exposure"
    ))
    
    # ========================================================================
    # URTI Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="URTI001",
        category="Respiratory",
        conditions=[
            RuleCondition("sore_throat", "is", True),
            RuleCondition("cough", "is", True),
            RuleCondition("nasal_congestion", "is", True),
        ],
        conclusion={"diagnosis": "URTI"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="URTI: sore throat + cough + nasal congestion"
    ))
    
    # ========================================================================
    # Acute Laryngitis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="LARYNG001",
        category="Respiratory",
        conditions=[
            RuleCondition("hoarse_voice", "is", True),
            RuleCondition("sore_throat", "is", True),
            RuleCondition("cough", "is", True),
        ],
        conclusion={"diagnosis": "Acute laryngitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Acute laryngitis: hoarse voice + sore throat + cough"
    ))
    
    rules.append(Rule(
        rule_id="LARYNG002",
        category="Respiratory",
        conditions=[
            RuleCondition("hoarse_voice", "is", True),
            RuleCondition("sore_throat", "is", True),
            RuleCondition("recent_cold", "is", True),
        ],
        conclusion={"diagnosis": "Acute laryngitis"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Acute laryngitis: hoarse voice + sore throat + recent cold"
    ))
    
    rules.append(Rule(
        rule_id="LARYNG003",
        category="Respiratory",
        conditions=[
            RuleCondition("hoarse_voice", "is", True),
            RuleCondition("cough", "is", True),
            RuleCondition("contact_exposure", "is", True),
        ],
        conclusion={"diagnosis": "Acute laryngitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Acute laryngitis: hoarse voice + cough + contact exposure"
    ))
    
    # ========================================================================
    # Bronchospasm / Asthma Exacerbation Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="ASTHMA001",
        category="Respiratory",
        conditions=[
            RuleCondition("wheezing", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("copd", "is", True),
        ],
        conclusion={"diagnosis": "Bronchospasm / acute asthma exacerbation"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Asthma exacerbation: wheezing + dyspnea + COPD"
    ))
    
    rules.append(Rule(
        rule_id="ASTHMA002",
        category="Respiratory",
        conditions=[
            RuleCondition("wheezing", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("asthma", "is", True),
        ],
        conclusion={"diagnosis": "Bronchospasm / acute asthma exacerbation"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Asthma exacerbation: wheezing + dyspnea + asthma"
    ))
    
    # ========================================================================
    # Spontaneous Pneumothorax Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="PNEUMO001",
        category="Respiratory",
        conditions=[
            RuleCondition("chest_pain", "is", True),
            RuleCondition("chest_pain_breathing", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("pneumothorax_history", "is", True),
        ],
        conclusion={"diagnosis": "Spontaneous pneumothorax"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Pneumothorax: chest pain + pain with breathing + dyspnea + history"
    ))
    
    rules.append(Rule(
        rule_id="PNEUMO002",
        category="Respiratory",
        conditions=[
            RuleCondition("chest_pain", "is", True),
            RuleCondition("chest_pain_breathing", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("family_pneumothorax", "is", True),
        ],
        conclusion={"diagnosis": "Spontaneous pneumothorax"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Pneumothorax: chest pain + pain with breathing + dyspnea + family history"
    ))
    
    # ========================================================================
    # Acute Dystonic Reactions Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="DYSTON001",
        category="Neurological",
        conditions=[
            RuleCondition("muscle_spasms", "is", True),
            RuleCondition("tongue_protrusion", "is", True),
            RuleCondition("recent_antipsychotics", "is", True),
        ],
        conclusion={"diagnosis": "Acute dystonic reactions"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Dystonic reaction: muscle spasms + tongue protrusion + recent antipsychotics"
    ))
    
    rules.append(Rule(
        rule_id="DYSTON002",
        category="Neurological",
        conditions=[
            RuleCondition("muscle_spasms", "is", True),
            RuleCondition("eyelid_droop", "is", True),
            RuleCondition("recent_antipsychotics", "is", True),
        ],
        conclusion={"diagnosis": "Acute dystonic reactions"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Dystonic reaction: muscle spasms + eyelid droop + recent antipsychotics"
    ))
    
    # ========================================================================
    # Whooping Cough Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="WHOOP001",
        category="Respiratory",
        conditions=[
            RuleCondition("intense_coughing_fits", "is", True),
            RuleCondition("cough", "is", True),
            RuleCondition("premature_birth", "is", True),
        ],
        conclusion={"diagnosis": "Whooping cough"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Whooping cough: intense coughing fits + cough + premature birth"
    ))
    
    # ========================================================================
    # Viral Pharyngitis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="PHARYNG001",
        category="Respiratory",
        conditions=[
            RuleCondition("sore_throat", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("contact_exposure", "is", True),
        ],
        conclusion={"diagnosis": "Viral pharyngitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Viral pharyngitis: sore throat + fever + contact exposure"
    ))
    
    # ========================================================================
    # Tuberculosis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="TB001",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("productive_cough", "is", True),
            RuleCondition("contact_exposure", "is", True),
        ],
        conclusion={"diagnosis": "Tuberculosis"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Tuberculosis: cough + fever + productive cough + contact exposure"
    ))
    
    # ========================================================================
    # Possible NSTEMI / STEMI Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="MI001",
        category="Cardiac",
        conditions=[
            RuleCondition("chest_pain", "is", True),
            RuleCondition("chest_pain_at_rest", "is", True),
            RuleCondition("chest_pain_exertion", "is", True),
        ],
        conclusion={"diagnosis": "Possible NSTEMI / STEMI"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="MI: chest pain at rest + pain with exertion"
    ))
    
    # ========================================================================
    # Stable/Unstable Angina Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="ANGINA001",
        category="Cardiac",
        conditions=[
            RuleCondition("chest_pain", "is", True),
            RuleCondition("chest_pain_exertion", "is", True),
            RuleCondition("chest_pain_at_rest", "is_not", True),
        ],
        conclusion={"diagnosis": "Stable angina"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Stable angina: chest pain with exertion, not at rest"
    ))
    
    rules.append(Rule(
        rule_id="ANGINA002",
        category="Cardiac",
        conditions=[
            RuleCondition("chest_pain", "is", True),
            RuleCondition("chest_pain_at_rest", "is", True),
        ],
        conclusion={"diagnosis": "Unstable angina"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Unstable angina: chest pain at rest"
    ))
    
    # ========================================================================
    # Acute COPD Exacerbation Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="COPD001",
        category="Respiratory",
        conditions=[
            RuleCondition("copd", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("wheezing", "is", True),
            RuleCondition("cough", "is", True),
        ],
        conclusion={"diagnosis": "Acute COPD exacerbation / infection"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="COPD exacerbation: COPD + dyspnea + wheezing + cough"
    ))
    
    rules.append(Rule(
        rule_id="COPD002",
        category="Respiratory",
        conditions=[
            RuleCondition("copd", "is", True),
            RuleCondition("productive_cough", "is", True),
            RuleCondition("fever", "is", True),
        ],
        conclusion={"diagnosis": "Acute COPD exacerbation / infection"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="COPD exacerbation: COPD + productive cough + fever"
    ))
    
    # ========================================================================
    # Acute Otitis Media Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="OTITIS001",
        category="ENT",
        conditions=[
            RuleCondition("fever", "is", True),
            RuleCondition("ear_pain", "is", True),
            RuleCondition("recent_cold", "is", True),
        ],
        conclusion={"diagnosis": "Acute otitis media"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Otitis media: fever + ear pain + recent cold"
    ))
    
    rules.append(Rule(
        rule_id="OTITIS002",
        category="ENT",
        conditions=[
            RuleCondition("ear_pain", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("nasal_congestion", "is", True),
        ],
        conclusion={"diagnosis": "Acute otitis media"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Otitis media: ear pain + fever + nasal congestion"
    ))
    
    rules.append(Rule(
        rule_id="OTITIS003",
        category="ENT",
        conditions=[
            RuleCondition("ear_pain", "is", True),
            RuleCondition("daycare_exposure", "is", True),
        ],
        conclusion={"diagnosis": "Acute otitis media"},
        certainty_factor=CertaintyFactor.WEAKLY_SUGGESTIVE,
        description="Otitis media: ear pain + daycare exposure (common in children)"
    ))
    
    # ========================================================================
    # Acute Pulmonary Edema Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="EDEMA001",
        category="Respiratory",
        conditions=[
            RuleCondition("dyspnea", "is", True),
            RuleCondition("dyspnea_at_rest", "is", True),
            RuleCondition("heart_failure", "is", True),
        ],
        conclusion={"diagnosis": "Acute pulmonary edema"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Pulmonary edema: dyspnea at rest + heart failure"
    ))
    
    # ========================================================================
    # Acute Rhinosinusitis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="RHINOSIN001",
        category="Respiratory",
        conditions=[
            RuleCondition("nasal_congestion", "is", True),
            RuleCondition("facial_pain", "is", True),
            RuleCondition("greenish_discharge", "is", True),
        ],
        conclusion={"diagnosis": "Acute rhinosinusitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Acute rhinosinusitis: nasal congestion + facial pain + greenish discharge"
    ))
    
    rules.append(Rule(
        rule_id="RHINOSIN002",
        category="Respiratory",
        conditions=[
            RuleCondition("nasal_congestion", "is", True),
            RuleCondition("facial_pain", "is", True),
            RuleCondition("fever", "is", True),
        ],
        conclusion={"diagnosis": "Acute rhinosinusitis"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Acute rhinosinusitis: nasal congestion + facial pain + fever"
    ))
    
    rules.append(Rule(
        rule_id="RHINOSIN003",
        category="Respiratory",
        conditions=[
            RuleCondition("nasal_congestion", "is", True),
            RuleCondition("greenish_discharge", "is", True),
            RuleCondition("recent_cold", "is", True),
        ],
        conclusion={"diagnosis": "Acute rhinosinusitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Acute rhinosinusitis: nasal congestion + greenish discharge + recent cold"
    ))
    
    # ========================================================================
    # Allergic Sinusitis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="ALLERGICSIN001",
        category="Respiratory",
        conditions=[
            RuleCondition("nasal_congestion", "is", True),
            RuleCondition("itchy_nose", "is", True),
            RuleCondition("allergy_history", "is", True),
        ],
        conclusion={"diagnosis": "Allergic sinusitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Allergic sinusitis: nasal congestion + itchy nose + allergy history"
    ))
    
    rules.append(Rule(
        rule_id="ALLERGICSIN002",
        category="Respiratory",
        conditions=[
            RuleCondition("nasal_congestion", "is", True),
            RuleCondition("allergy_exposure", "is", True),
            RuleCondition("itchy_nose", "is", True),
        ],
        conclusion={"diagnosis": "Allergic sinusitis"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Allergic sinusitis: nasal congestion + allergy exposure + itchy nose"
    ))
    
    # ========================================================================
    # Anaphylaxis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="ANAPHYL001",
        category="Allergic",
        conditions=[
            RuleCondition("allergy_exposure", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("swelling", "is", True),
        ],
        conclusion={"diagnosis": "Anaphylaxis"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Anaphylaxis: allergy exposure + dyspnea + swelling"
    ))
    
    rules.append(Rule(
        rule_id="ANAPHYL002",
        category="Allergic",
        conditions=[
            RuleCondition("allergy_exposure", "is", True),
            RuleCondition("swelling", "is", True),
            RuleCondition("suffocating_feeling", "is", True),
        ],
        conclusion={"diagnosis": "Anaphylaxis"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Anaphylaxis: allergy exposure + swelling + suffocating feeling"
    ))
    
    rules.append(Rule(
        rule_id="ANAPHYL003",
        category="Allergic",
        conditions=[
            RuleCondition("allergy_exposure", "is", True),
            RuleCondition("inability_to_breathe", "is", True),
        ],
        conclusion={"diagnosis": "Anaphylaxis"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Anaphylaxis: allergy exposure + inability to breathe"
    ))
    
    # ========================================================================
    # Anemia Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="ANEMIA001",
        category="Hematological",
        conditions=[
            RuleCondition("pale_skin", "is", True),
            RuleCondition("fatigue", "is", True),
            RuleCondition("anemia_history", "is", True),
        ],
        conclusion={"diagnosis": "Anemia"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Anemia: pale skin + fatigue + anemia history"
    ))
    
    rules.append(Rule(
        rule_id="ANEMIA002",
        category="Hematological",
        conditions=[
            RuleCondition("pale_skin", "is", True),
            RuleCondition("fatigue", "is", True),
            RuleCondition("vomiting_blood", "is", True),
        ],
        conclusion={"diagnosis": "Anemia"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Anemia: pale skin + fatigue + vomiting blood (blood loss)"
    ))
    
    rules.append(Rule(
        rule_id="ANEMIA003",
        category="Hematological",
        conditions=[
            RuleCondition("pale_skin", "is", True),
            RuleCondition("fatigue", "is", True),
            RuleCondition("black_stools", "is", True),
        ],
        conclusion={"diagnosis": "Anemia"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Anemia: pale skin + fatigue + black stools (GI bleeding)"
    ))
    
    # ========================================================================
    # Atrial Fibrillation Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="AFIB001",
        category="Cardiac",
        conditions=[
            RuleCondition("palpitations", "is", True),
            RuleCondition("irregular_heartbeat", "is", True),
            RuleCondition("dyspnea", "is", True),
        ],
        conclusion={"diagnosis": "Atrial fibrillation"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Atrial fibrillation: palpitations + irregular heartbeat + dyspnea"
    ))
    
    rules.append(Rule(
        rule_id="AFIB002",
        category="Cardiac",
        conditions=[
            RuleCondition("irregular_heartbeat", "is", True),
            RuleCondition("rapid_heartbeat", "is", True),
            RuleCondition("palpitations", "is", True),
        ],
        conclusion={"diagnosis": "Atrial fibrillation"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Atrial fibrillation: irregular + rapid heartbeat + palpitations"
    ))
    
    rules.append(Rule(
        rule_id="AFIB003",
        category="Cardiac",
        conditions=[
            RuleCondition("irregular_heartbeat", "is", True),
            RuleCondition("syncope", "is", True),
        ],
        conclusion={"diagnosis": "Atrial fibrillation"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Atrial fibrillation: irregular heartbeat + syncope"
    ))
    
    # ========================================================================
    # Boerhaave Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="BOERHAAVE001",
        category="GI",
        conditions=[
            RuleCondition("vomiting_blood", "is", True),
            RuleCondition("chest_pain", "is", True),
            RuleCondition("alcohol_abuse", "is", True),
        ],
        conclusion={"diagnosis": "Boerhaave"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Boerhaave: vomiting blood + chest pain + alcohol abuse"
    ))
    
    # ========================================================================
    # Bronchiectasis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="BRONCHIECT001",
        category="Respiratory",
        conditions=[
            RuleCondition("chronic_cough", "is", True),
            RuleCondition("productive_cough", "is", True),
            RuleCondition("recurrent_infections", "is", True),
        ],
        conclusion={"diagnosis": "Bronchiectasis"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Bronchiectasis: chronic cough + productive cough + recurrent infections"
    ))
    
    # ========================================================================
    # Bronchiolitis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="BRONCHIOL001",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("wheezing", "is", True),
            RuleCondition("age", "less_than", 2),
        ],
        conclusion={"diagnosis": "Bronchiolitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Bronchiolitis: cough + wheezing + young age"
    ))
    
    rules.append(Rule(
        rule_id="BRONCHIOL002",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("wheezing", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("age", "less_than", 2),
        ],
        conclusion={"diagnosis": "Bronchiolitis"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Bronchiolitis: cough + wheezing + fever + young age"
    ))
    
    rules.append(Rule(
        rule_id="BRONCHIOL003",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("age", "less_than", 2),
        ],
        conclusion={"diagnosis": "Bronchiolitis"},
        certainty_factor=CertaintyFactor.WEAKLY_SUGGESTIVE,
        description="Bronchiolitis: cough + dyspnea + young age"
    ))
    
    # ========================================================================
    # Chagas Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="CHAGAS001",
        category="Infectious",
        conditions=[
            RuleCondition("travel_history", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("cardiac_symptoms", "is", True),
        ],
        conclusion={"diagnosis": "Chagas"},
        certainty_factor=CertaintyFactor.VERY_WEAK,
        description="Chagas: travel history + fever + cardiac symptoms"
    ))
    
    # ========================================================================
    # Chronic Rhinosinusitis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="CHRONICSIN001",
        category="Respiratory",
        conditions=[
            RuleCondition("nasal_congestion", "is", True),
            RuleCondition("chronic_sinusitis", "is", True),
            RuleCondition("nasal_polyps", "is", True),
        ],
        conclusion={"diagnosis": "Chronic rhinosinusitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Chronic rhinosinusitis: nasal congestion + chronic sinusitis + polyps"
    ))
    
    # ========================================================================
    # Cluster Headache Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="CLUSTER001",
        category="Neurological",
        conditions=[
            RuleCondition("headache", "is", True),
            RuleCondition("severe_headache", "is", True),
            RuleCondition("family_cluster_headache", "is", True),
        ],
        conclusion={"diagnosis": "Cluster headache"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Cluster headache: severe headache + family history"
    ))
    
    # ========================================================================
    # Croup Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="CROUP001",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("stridor", "is", True),
            RuleCondition("hoarse_voice", "is", True),
        ],
        conclusion={"diagnosis": "Croup"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Croup: cough + stridor + hoarse voice"
    ))
    
    rules.append(Rule(
        rule_id="CROUP002",
        category="Respiratory",
        conditions=[
            RuleCondition("cough", "is", True),
            RuleCondition("stridor", "is", True),
            RuleCondition("fever", "is", True),
        ],
        conclusion={"diagnosis": "Croup"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Croup: cough + stridor + fever"
    ))
    
    rules.append(Rule(
        rule_id="CROUP003",
        category="Respiratory",
        conditions=[
            RuleCondition("stridor", "is", True),
            RuleCondition("hoarse_voice", "is", True),
            RuleCondition("age", "less_than", 5),
        ],
        conclusion={"diagnosis": "Croup"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Croup: stridor + hoarse voice + young age"
    ))
    
    # ========================================================================
    # Ebola Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="EBOLA001",
        category="Infectious",
        conditions=[
            RuleCondition("ebola_contact", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("bleeding", "is", True),
        ],
        conclusion={"diagnosis": "Ebola"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Ebola: contact + fever + bleeding"
    ))
    
    # ========================================================================
    # Epiglottitis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="EPIGLOTT001",
        category="Respiratory",
        conditions=[
            RuleCondition("sore_throat", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("stridor", "is", True),
        ],
        conclusion={"diagnosis": "Epiglottitis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Epiglottitis: sore throat + dyspnea + stridor"
    ))
    
    rules.append(Rule(
        rule_id="EPIGLOTT002",
        category="Respiratory",
        conditions=[
            RuleCondition("sore_throat", "is", True),
            RuleCondition("inability_to_breathe", "is", True),
            RuleCondition("fever", "is", True),
        ],
        conclusion={"diagnosis": "Epiglottitis"},
        certainty_factor=CertaintyFactor.STRONG_SUGGESTIVE,
        description="Epiglottitis: sore throat + inability to breathe + fever"
    ))
    
    rules.append(Rule(
        rule_id="EPIGLOTT003",
        category="Respiratory",
        conditions=[
            RuleCondition("stridor", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("fever", "is", True),
        ],
        conclusion={"diagnosis": "Epiglottitis"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Epiglottitis: stridor + dyspnea + fever"
    ))
    
    # ========================================================================
    # Guillain-Barré Syndrome Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="GBS001",
        category="Neurological",
        conditions=[
            RuleCondition("weakness_limbs", "is", True),
            RuleCondition("numbness", "is", True),
            RuleCondition("recent_infection", "is", True),
        ],
        conclusion={"diagnosis": "Guillain-Barré syndrome"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="GBS: limb weakness + numbness + recent infection"
    ))
    
    # ========================================================================
    # HIV Initial Infection Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="HIV001",
        category="Infectious",
        conditions=[
            RuleCondition("hiv_risk", "is", True),
            RuleCondition("fever", "is", True),
            RuleCondition("fatigue", "is", True),
        ],
        conclusion={"diagnosis": "HIV (initial infection)"},
        certainty_factor=CertaintyFactor.VERY_WEAK,
        description="HIV: risk factors + fever + fatigue"
    ))
    
    # ========================================================================
    # Inguinal Hernia Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="HERNIA001",
        category="GI",
        conditions=[
            RuleCondition("groin_pain", "is", True),
            RuleCondition("groin_swelling", "is", True),
            RuleCondition("pain_with_coughing", "is", True),
        ],
        conclusion={"diagnosis": "Inguinal hernia"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Inguinal hernia: groin pain + swelling + pain with coughing"
    ))
    
    # ========================================================================
    # Laryngospasm Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="LARYNGOSP001",
        category="Respiratory",
        conditions=[
            RuleCondition("suffocating_feeling", "is", True),
            RuleCondition("inability_to_breathe", "is", True),
            RuleCondition("recent_antipsychotics", "is", True),
        ],
        conclusion={"diagnosis": "Larygospasm"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Laryngospasm: suffocating feeling + inability to breathe + antipsychotics"
    ))
    
    # ========================================================================
    # Localized Edema Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="EDEMA_LOCAL001",
        category="General",
        conditions=[
            RuleCondition("swelling", "is", True),
            RuleCondition("localized_swelling", "is", True),
            RuleCondition("pain_at_site", "is", True),
        ],
        conclusion={"diagnosis": "Localized edema"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Localized edema: localized swelling + pain"
    ))
    
    # ========================================================================
    # Myasthenia Gravis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="MYASTHENIA001",
        category="Neurological",
        conditions=[
            RuleCondition("muscle_weakness", "is", True),
            RuleCondition("weakness_worse_fatigue", "is", True),
            RuleCondition("eyelid_droop", "is", True),
        ],
        conclusion={"diagnosis": "Myasthenia gravis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Myasthenia gravis: muscle weakness + worse with fatigue + eyelid droop"
    ))
    
    # ========================================================================
    # Myocarditis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="MYOCARD001",
        category="Cardiac",
        conditions=[
            RuleCondition("chest_pain", "is", True),
            RuleCondition("dyspnea", "is", True),
            RuleCondition("recent_viral_infection", "is", True),
        ],
        conclusion={"diagnosis": "Myocarditis"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Myocarditis: chest pain + dyspnea + recent viral infection"
    ))
    
    # ========================================================================
    # PSVT Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="PSVT001",
        category="Cardiac",
        conditions=[
            RuleCondition("palpitations", "is", True),
            RuleCondition("rapid_heartbeat", "is", True),
            RuleCondition("syncope", "is", True),
        ],
        conclusion={"diagnosis": "PSVT"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="PSVT: palpitations + rapid heartbeat + syncope"
    ))
    
    # ========================================================================
    # Pancreatic Neoplasm Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="PANCREATIC001",
        category="Oncological",
        conditions=[
            RuleCondition("abdominal_pain", "is", True),
            RuleCondition("weight_loss", "is", True),
            RuleCondition("family_pancreatic_cancer", "is", True),
        ],
        conclusion={"diagnosis": "Pancreatic neoplasm"},
        certainty_factor=CertaintyFactor.VERY_WEAK,
        description="Pancreatic neoplasm: abdominal pain + weight loss + family history"
    ))
    
    # ========================================================================
    # Panic Attack Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="PANIC001",
        category="Psychiatric",
        conditions=[
            RuleCondition("anxiety", "is", True),
            RuleCondition("palpitations", "is", True),
            RuleCondition("feeling_dying", "is", True),
        ],
        conclusion={"diagnosis": "Panic attack"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Panic attack: anxiety + palpitations + feeling of dying"
    ))
    
    # ========================================================================
    # Pericarditis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="PERICARD001",
        category="Cardiac",
        conditions=[
            RuleCondition("chest_pain", "is", True),
            RuleCondition("chest_pain_improves_forward", "is", True),
            RuleCondition("pericarditis_history", "is", True),
        ],
        conclusion={"diagnosis": "Pericarditis"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Pericarditis: chest pain + improves leaning forward + history"
    ))
    
    # ========================================================================
    # Pulmonary Embolism Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="PE001",
        category="Respiratory",
        conditions=[
            RuleCondition("dyspnea", "is", True),
            RuleCondition("chest_pain", "is", True),
            RuleCondition("dvt_history", "is", True),
        ],
        conclusion={"diagnosis": "Pulmonary embolism"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Pulmonary embolism: dyspnea + chest pain + DVT history"
    ))
    
    # ========================================================================
    # Pulmonary Neoplasm Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="LUNG_CANCER001",
        category="Oncological",
        conditions=[
            RuleCondition("chronic_cough", "is", True),
            RuleCondition("weight_loss", "is", True),
            RuleCondition("smoking", "is", True),
        ],
        conclusion={"diagnosis": "Pulmonary neoplasm"},
        certainty_factor=CertaintyFactor.VERY_WEAK,
        description="Pulmonary neoplasm: chronic cough + weight loss + smoking"
    ))
    
    # ========================================================================
    # SLE Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="SLE001",
        category="Autoimmune",
        conditions=[
            RuleCondition("joint_pain", "is", True),
            RuleCondition("rash", "is", True),
            RuleCondition("fatigue", "is", True),
        ],
        conclusion={"diagnosis": "SLE"},
        certainty_factor=CertaintyFactor.VERY_WEAK,
        description="SLE: joint pain + rash + fatigue"
    ))
    
    # ========================================================================
    # Sarcoidosis Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="SARCOID001",
        category="Autoimmune",
        conditions=[
            RuleCondition("dyspnea", "is", True),
            RuleCondition("cough", "is", True),
            RuleCondition("weight_loss", "is", True),
        ],
        conclusion={"diagnosis": "Sarcoidosis"},
        certainty_factor=CertaintyFactor.VERY_WEAK,
        description="Sarcoidosis: dyspnea + cough + weight loss"
    ))
    
    # ========================================================================
    # Scombroid Food Poisoning Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="SCOMBROID001",
        category="GI",
        conditions=[
            RuleCondition("fish_consumption", "is", True),
            RuleCondition("nausea", "is", True),
            RuleCondition("flushing", "is", True),
        ],
        conclusion={"diagnosis": "Scombroid food poisoning"},
        certainty_factor=CertaintyFactor.SUGGESTIVE,
        description="Scombroid: fish consumption + nausea + flushing"
    ))
    
    # ========================================================================
    # Spontaneous Rib Fracture Rules
    # ========================================================================
    rules.append(Rule(
        rule_id="RIB_FRACTURE001",
        category="Musculoskeletal",
        conditions=[
            RuleCondition("chest_pain", "is", True),
            RuleCondition("chest_pain_breathing", "is", True),
            RuleCondition("chest_pain_movement", "is", True),
        ],
        conclusion={"diagnosis": "Spontaneous rib fracture"},
        certainty_factor=CertaintyFactor.MODERATE,
        description="Rib fracture: chest pain + pain with breathing + pain with movement"
    ))
    
    return rules


# Create the rules list
ALL_RULES = create_medical_rules()

