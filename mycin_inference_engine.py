"""
MYCIN Inference Engine
======================

This module implements a MYCIN-style inference engine that:
1. Uses LLM only to answer individual questions about patient data
2. Evaluates rules programmatically (not via LLM)
3. Combines certainty factors using MYCIN's combination functions
4. Performs backward/forward chaining inference
"""

import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

from mycin_rules import (
    ALL_RULES,
    Rule,
    RuleCondition,
    QUESTIONS,
)


@dataclass
class Fact:
    """A fact with a certainty factor"""
    parameter: str
    value: Any
    certainty: float  # -1.0 to 1.0
    source_rules: List[str] = field(default_factory=list)


class MYCINInferenceEngine:
    """
    MYCIN inference engine that evaluates rules programmatically.
    LLM is only used to answer questions, not to evaluate rules.
    """
    
    def __init__(self, llm_question_answering_fn: Optional[Callable] = None):
        """
        Args:
            llm_question_answering_fn: Function that takes (question_key, patient_data) 
                                      and returns answer with optional certainty
        """
        self.llm_qa_fn = llm_question_answering_fn
        self.known_facts: Dict[str, Fact] = {}  # parameter -> Fact
        self.evaluated_rules: set = set()
        
    def set_fact(self, parameter: str, value: Any, certainty: float = 1.0, source_rule: Optional[str] = None):
        """Set a fact with certainty"""
        if parameter in self.known_facts:
            # Combine with existing fact using MYCIN combination
            existing = self.known_facts[parameter]
            if existing.value == value:
                # Same value: combine certainties
                combined_cf = self.combine_certainties(existing.certainty, certainty)
                self.known_facts[parameter] = Fact(
                    parameter=parameter,
                    value=value,
                    certainty=combined_cf,
                    source_rules=existing.source_rules + ([source_rule] if source_rule else [])
                )
            else:
                # Different values: conflict resolution (keep higher certainty)
                if certainty > existing.certainty:
                    self.known_facts[parameter] = Fact(
                        parameter=parameter,
                        value=value,
                        certainty=certainty,
                        source_rules=[source_rule] if source_rule else []
                    )
        else:
            self.known_facts[parameter] = Fact(
                parameter=parameter,
                value=value,
                certainty=certainty,
                source_rules=[source_rule] if source_rule else []
            )
    
    def get_fact(self, parameter: str) -> Optional[Fact]:
        """Get a fact if known"""
        return self.known_facts.get(parameter)
    
    def ask_llm_question(self, question_key: str, patient_data: Dict[str, Any]) -> Tuple[Any, float]:
        """
        Ask LLM to answer a question based on patient data.
        Returns (answer, certainty)
        """
        if self.llm_qa_fn:
            result = self.llm_qa_fn(question_key, patient_data)
            if isinstance(result, tuple):
                return result
            elif isinstance(result, dict):
                return result.get("answer"), result.get("certainty", 1.0)
            else:
                return result, 1.0
        else:
            # No LLM function: return unknown
            return None, 0.0
    
    def evaluate_condition(self, condition: RuleCondition, patient_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate a single condition.
        Returns (is_satisfied, certainty)
        """
        param = condition.parameter
        operator = condition.operator
        expected_value = condition.value
        
        # Check if we have this fact
        fact = self.get_fact(param)
        if fact is None:
            # Try to get from patient_data directly
            if param in patient_data:
                actual_value = patient_data[param]
                fact = Fact(param, actual_value, 1.0)
            else:
                # Ask LLM if available
                answer, certainty = self.ask_llm_question(param, patient_data)
                if answer is not None:
                    fact = Fact(param, answer, certainty)
                    self.set_fact(param, answer, certainty)
                else:
                    return False, 0.0
        
        actual_value = fact.value
        fact_certainty = fact.certainty
        
        # Evaluate condition based on operator
        if operator == "is":
            satisfied = (actual_value == expected_value)
        elif operator == "is_not":
            satisfied = (actual_value != expected_value)
        elif operator == "greater_than":
            try:
                satisfied = (float(actual_value) > float(expected_value))
            except (ValueError, TypeError):
                satisfied = False
        elif operator == "less_than":
            try:
                satisfied = (float(actual_value) < float(expected_value))
            except (ValueError, TypeError):
                satisfied = False
        elif operator == "greater_than_or_equal":
            try:
                satisfied = (float(actual_value) >= float(expected_value))
            except (ValueError, TypeError):
                satisfied = False
        elif operator == "less_than_or_equal":
            try:
                satisfied = (float(actual_value) <= float(expected_value))
            except (ValueError, TypeError):
                satisfied = False
        else:
            satisfied = False
        
        # Return satisfaction and certainty
        if satisfied:
            return True, fact_certainty
        else:
            return False, 0.0
    
    def evaluate_rule(self, rule: Rule, patient_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate a rule's conditions.
        Returns (all_conditions_met, min_certainty)
        """
        if rule.rule_id in self.evaluated_rules:
            # Already evaluated
            return False, 0.0
        
        condition_results = []
        for condition in rule.conditions:
            satisfied, certainty = self.evaluate_condition(condition, patient_data)
            condition_results.append((satisfied, certainty))
            if not satisfied:
                # Early exit: all conditions must be met
                return False, 0.0
        
        # All conditions satisfied
        # Use minimum certainty (conservative approach)
        min_certainty = min(cf for _, cf in condition_results) if condition_results else 0.0
        return True, min_certainty
    
    def apply_rule(self, rule: Rule, patient_data: Dict[str, Any]) -> bool:
        """
        Evaluate and apply a rule if conditions are met.
        Returns True if rule was applied.
        """
        conditions_met, condition_cf = self.evaluate_rule(rule, patient_data)
        
        if conditions_met:
            # Calculate conclusion certainty: rule_cf * condition_cf
            conclusion_cf = rule.certainty_factor * condition_cf
            
            # Apply conclusion
            for key, value in rule.conclusion.items():
                self.set_fact(key, value, conclusion_cf, source_rule=rule.rule_id)
            
            self.evaluated_rules.add(rule.rule_id)
            return True
        
        return False
    
    def forward_chain(self, patient_data: Dict[str, Any], max_iterations: int = 10) -> List[str]:
        """
        Forward chaining: apply all applicable rules.
        Returns list of applied rule IDs.
        """
        applied_rules = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            new_rules_applied = False
            
            for rule in ALL_RULES:
                if rule.rule_id not in self.evaluated_rules:
                    if self.apply_rule(rule, patient_data):
                        applied_rules.append(rule.rule_id)
                        new_rules_applied = True
            
            if not new_rules_applied:
                break
        
        return applied_rules
    
    @staticmethod
    def combine_certainties(cf1: float, cf2: float) -> float:
        """
        Combine two certainty factors using MYCIN's combination function.
        CF_combined = CF1 + CF2 * (1 - |CF1|)
        """
        if cf1 >= 0 and cf2 >= 0:
            return cf1 + cf2 * (1 - cf1)
        elif cf1 < 0 and cf2 < 0:
            return cf1 + cf2 * (1 + cf1)
        else:
            # Opposite signs
            return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))
    
    def get_diagnosis(self) -> Dict[str, Any]:
        """
        Get the final diagnosis based on accumulated facts.
        Returns dict with organism identity, infection site, and treatment recommendations.
        """
        # Get organism identities
        organisms = {}
        for param, fact in self.known_facts.items():
            if param == "identity":
                organisms[fact.value] = fact.certainty
        
        # Get infection sites
        sites = {}
        for param, fact in self.known_facts.items():
            if param == "infection_site":
                sites[fact.value] = fact.certainty
        
        # Get treatment recommendations
        treatments = {}
        for param, fact in self.known_facts.items():
            if param == "recommended_drug":
                drug = fact.value
                dose_fact = self.get_fact("dose")
                dose = dose_fact.value if dose_fact else "standard"
                treatments[drug] = {
                    "certainty": fact.certainty,
                    "dose": dose,
                    "source_rules": fact.source_rules
                }
        
        # Get top predictions
        top_organism = max(organisms.items(), key=lambda x: x[1]) if organisms else (None, 0.0)
        top_site = max(sites.items(), key=lambda x: x[1]) if sites else (None, 0.0)
        top_treatment = max(treatments.items(), key=lambda x: x[1]["certainty"]) if treatments else (None, None)
        
        # Build probability distributions
        organism_probs = self._normalize_to_probs(organisms)
        site_probs = self._normalize_to_probs(sites)
        
        return {
            "organism_identity": {
                "name": top_organism[0],
                "certainty": top_organism[1],
                "all_organisms": organisms,
                "probabilities": organism_probs
            },
            "infection_site": {
                "site": top_site[0],
                "certainty": top_site[1],
                "all_sites": sites,
                "probabilities": site_probs
            },
            "treatment": {
                "recommended_drug": top_treatment[0] if top_treatment else None,
                "details": top_treatment[1] if top_treatment else None,
                "all_treatments": treatments
            },
            "all_facts": {k: {"value": v.value, "certainty": v.certainty} 
                         for k, v in self.known_facts.items()}
        }
    
    @staticmethod
    def _normalize_to_probs(certainty_dict: Dict[str, float]) -> Dict[str, float]:
        """Convert certainty factors to probabilities (softmax-like)"""
        if not certainty_dict:
            return {}
        
        # Convert CFs to positive values
        positive_cfs = {k: max(0, v) for k, v in certainty_dict.items()}
        
        if not positive_cfs or sum(positive_cfs.values()) == 0:
            return {k: 1.0 / len(certainty_dict) for k in certainty_dict}
        
        # Normalize
        total = sum(positive_cfs.values())
        return {k: v / total for k, v in positive_cfs.items()}
    
    def reset(self):
        """Reset the inference engine state"""
        self.known_facts.clear()
        self.evaluated_rules.clear()


def create_llm_question_prompt(question_key: str, question_text: str, patient_data: Dict[str, Any]) -> str:
    """
    Create a prompt for LLM to answer a specific question based on patient data.
    """
    prompt = f"""You are a medical assistant. Based on the following patient information, answer this question:

Question: {question_text}

Patient Information:
{json.dumps(patient_data, indent=2)}

Please provide a concise answer. If the information is not available, respond with "UNKNOWN".
For yes/no questions, respond with true or false.
For categorical questions, respond with the category value.
For numerical questions, respond with the number.

Answer:"""
    return prompt


def simple_llm_qa_function(question_key: str, patient_data: Dict[str, Any], 
                           llm_call_fn: Optional[Callable] = None) -> Tuple[Any, float]:
    """
    Simple LLM question answering function.
    You should replace this with your actual LLM API call.
    
    Args:
        question_key: The MYCIN question key
        patient_data: Patient data dictionary
        llm_call_fn: Function that takes a prompt string and returns response
    
    Returns:
        (answer, certainty) tuple
    """
    question_text = QUESTIONS.get(question_key, question_key)
    prompt = create_llm_question_prompt(question_key, question_text, patient_data)
    
    if llm_call_fn:
        response = llm_call_fn(prompt)
        # Parse response (simple implementation - you may want to improve this)
        if isinstance(response, str):
            response_lower = response.strip().lower()
            if response_lower in ["true", "yes", "1"]:
                return True, 1.0
            elif response_lower in ["false", "no", "0"]:
                return False, 1.0
            elif response_lower == "unknown":
                return None, 0.0
            else:
                # Try to parse as number
                try:
                    return float(response), 1.0
                except ValueError:
                    return response, 1.0
        return response, 1.0
    else:
        # No LLM: return unknown
        return None, 0.0

