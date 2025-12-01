"""
MYCIN Question Mapping
======================

Maps actual questions from question_en_output.txt to MYCIN parameters.
This allows the system to work with the actual question text in patient data.
"""

from typing import Dict, Optional

# Mapping from actual question text to MYCIN parameter names
# This maps the questions from question_en_output.txt to MYCIN parameters
QUESTION_TO_PARAMETER: Dict[str, str] = {
    # Fever and temperature
    "Do you have a fever (either felt or measured with a thermometer)?": "fever",
    
    # Respiratory symptoms
    "Do you have a cough?": "cough",
    "Are you experiencing shortness of breath or difficulty breathing in a significant way?": "dyspnea",
    "Do you feel out of breath with minimal physical effort?": "dyspnea",
    "Do you have bouts of choking or shortness of breath that wake you up at night?": "dyspnea",
    "Do you wheeze while inhaling or is your breathing noisy after coughing spells?": "dyspnea",
    "Have you noticed a wheezing sound when you exhale?": "dyspnea",
    "Do you have a cough that produces colored or more abundant sputum than usual?": "cough",
    "Have you been coughing up blood?": "cough",
    "Do you have intense coughing fits?": "cough",
    
    # Chest and pain
    "Do you have chest pain even at rest?": "chest_pain",
    "Do you have pain that is increased when you breathe in deeply?": "chest_pain",
    "Do you have symptoms that are increased with physical exertion but alleviated with rest?": "chest_pain",
    
    # Headache and neurological
    "Do you have a headache?": "headache",  # Note: not in the list, but we'll map similar
    "Have you felt confused or disorientated lately?": "headache",  # Proxy for neurological symptoms
    
    # Neck and meningitis symptoms
    "Do you feel that muscle spasms or soreness in your neck are keeping you from turning your head to one side?": "nuchal_rigidity",
    
    # Urinary symptoms
    "Have you had any vaginal discharge?": "urinary_symptoms",  # Related to genitourinary
    
    # Abdominal
    "Do you feel pain somewhere?": "abdominal_pain",
    "Do you have pain somewhere, related to your reason for consulting?": "abdominal_pain",
    "Do you feel your abdomen is bloated or distended (swollen due to pressure from inside)?": "abdominal_pain",
    "Are you feeling nauseous or do you feel like vomiting?": "abdominal_pain",
    "Have you recently thrown up blood or something resembling coffee beans?": "abdominal_pain",
    "Have you recently had stools that were black (like coal)?": "abdominal_pain",
    "Have you had diarrhea or an increase in stool frequency?": "abdominal_pain",
    
    # Immunocompromised and hospital-acquired
    "Are you immunosuppressed?": "immunocompromised",
    "Are you infected with the human immunodeficiency virus (HIV)?": "immunocompromised",
    "Do you have metastatic cancer?": "immunocompromised",
    "Do you have an active cancer?": "immunocompromised",
    "Have you been treated in hospital recently for nausea, agitation, intoxication or aggressive behavior and received medication via an intravenous or intramuscular route?": "hospital_acquired",
    
    # Surgery
    "Have you had surgery within the last month?": "recent_surgery",
    "Have you ever had surgery to remove lymph nodes?": "recent_surgery",
    
    # Age-related (for demographics)
    # Note: Age is typically in demographics, not questions
    
    # Chronic conditions that affect infection risk
    "Do you have chronic kidney failure?": "immunocompromised",
    "Do you have liver cirrhosis?": "immunocompromised",
    "Do you have cystic fibrosis?": "immunocompromised",
    "Do you have diabetes?": "immunocompromised",
    
    # Respiratory conditions
    "Do you have asthma or have you ever had to use a bronchodilator in the past?": "respiratory_condition",
    "Do you have a chronic obstructive pulmonary disease (COPD)?": "respiratory_condition",
    "Do you have severe Chronic Obstructive Pulmonary Disease (COPD)?": "respiratory_condition",
    "Have you ever had pneumonia?": "respiratory_condition",
    "Have you ever had fluid in your lungs?": "respiratory_condition",
    
    # Sore throat and upper respiratory
    "Do you have a sore throat?": "sore_throat",
    "Do you have nasal congestion or a clear runny nose?": "nasal_symptoms",
    "Do you have greenish or yellowish nasal discharge?": "nasal_symptoms",
    "Have you had a cold in the last 2 weeks?": "nasal_symptoms",
    "Have you recently had a viral infection?": "viral_infection",
    
    # Contact and exposure
    "Have you been in contact with a person with similar symptoms in the past 2 weeks?": "contact_exposure",
    "Have you been in contact with someone who has had pertussis (whoooping cough)?": "contact_exposure",
    
    # Travel
    "Have you traveled out of the country in the last 4 weeks?": "travel_history",
}

# Reverse mapping: from MYCIN parameter to list of possible question texts
PARAMETER_TO_QUESTIONS: Dict[str, list] = {}
for question, param in QUESTION_TO_PARAMETER.items():
    if param not in PARAMETER_TO_QUESTIONS:
        PARAMETER_TO_QUESTIONS[param] = []
    PARAMETER_TO_QUESTIONS[param].append(question)


def get_parameter_from_question(question_text: str) -> Optional[str]:
    """
    Get MYCIN parameter name from actual question text.
    
    Args:
        question_text: The actual question text from patient data
        
    Returns:
        MYCIN parameter name, or None if not found
    """
    # Exact match
    if question_text in QUESTION_TO_PARAMETER:
        return QUESTION_TO_PARAMETER[question_text]
    
    # Try case-insensitive match
    question_lower = question_text.lower().strip()
    for q, param in QUESTION_TO_PARAMETER.items():
        if q.lower().strip() == question_lower:
            return param
    
    return None


def get_questions_for_parameter(parameter: str) -> list:
    """
    Get all question texts that map to a MYCIN parameter.
    
    Args:
        parameter: MYCIN parameter name
        
    Returns:
        List of question texts
    """
    return PARAMETER_TO_QUESTIONS.get(parameter, [])


def load_questions_from_file(filepath: str = "outputs/question_en_output.txt") -> Dict[str, int]:
    """
    Load all questions from the question_en_output.txt file.
    Returns a dict mapping question text to line number (for reference).
    """
    questions = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                question = line.strip()
                if question:
                    questions[question] = line_num
    except FileNotFoundError:
        pass
    return questions


# Load all available questions
ALL_AVAILABLE_QUESTIONS = load_questions_from_file()

