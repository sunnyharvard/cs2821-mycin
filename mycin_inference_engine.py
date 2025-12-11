"""
MYCIN Inference Engine
======================

This module implements a MYCIN-style inference engine that:
1. Uses LLM only to answer individual questions about patient data
2. Evaluates rules programmatically (not via LLM)
3. Combines certainty factors using MYCIN's combination functions
4. Performs backward chaining inference (goal-directed)
"""

import json
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field

# Import rules dynamically - will be patched by pipeline
# Default to medical rules
try:
    from mycin_medical_rules import (
    ALL_RULES,
    Rule,
    RuleCondition,
    QUESTIONS,
    ASK_FIRST_PARAMETERS,
)
except ImportError:
    # Fallback - create minimal stubs
    ALL_RULES = []
    QUESTIONS = {}
    ASK_FIRST_PARAMETERS = set()
    
    @dataclass
    class RuleCondition:
        parameter: str
        operator: str
        value: Any
    
    @dataclass
    class Rule:
        rule_id: str
        category: str
        conditions: List[RuleCondition]
        conclusion: Dict[str, Any]
        certainty_factor: float
        description: str


@dataclass
class Fact:
    """A fact with a certainty factor"""
    parameter: str
    value: Any
    certainty: float  # -1.0 to 1.0
    source_rules: List[str] = field(default_factory=list)


class MYCINInferenceEngine:
    """
    MYCIN inference engine that evaluates rules programmatically using Backward Chaining.
    """
    
    def __init__(self, llm_question_answering_fn: Optional[Callable] = None):
        """
        Args:
            llm_question_answering_fn: Function that takes (question_key, patient_data) 
                                      and returns answer with optional certainty
        """
        self.llm_qa_fn = llm_question_answering_fn
        self.known_facts: Dict[str, List[Fact]] = {}  # parameter -> List[Fact] (one per value)
        self.traced_rules: Set[str] = set() # Rules currently being evaluated (loop detection)
        self.asked_questions: Set[str] = set() # Parameters already asked to user
        
        # Index rules by conclusion parameter for efficiency
        self.rules_by_conclusion: Dict[str, List[Rule]] = {}
        for rule in ALL_RULES:
            for concl_param in rule.conclusion.keys():
                if concl_param not in self.rules_by_conclusion:
                    self.rules_by_conclusion[concl_param] = []
                self.rules_by_conclusion[concl_param].append(rule)

    def get_facts(self, parameter: str) -> List[Fact]:
        """Get all known facts for a parameter"""
        return self.known_facts.get(parameter, [])

    def get_fact_certainty(self, parameter: str, value: Any) -> float:
        """Get certainty of a specific parameter=value"""
        facts = self.get_facts(parameter)
        for fact in facts:
            if fact.value == value:
                return fact.certainty
        return 0.0

    def update_fact(self, parameter: str, value: Any, certainty: float, source_rule: Optional[str] = None):
        """
        Update certainty for a fact.
        Combines with existing certainty using MYCIN's cf-or function.
        """
        if abs(certainty) < 0.001: # Ignore negligible updates
            return

        existing_facts = self.known_facts.get(parameter, [])
        
        # Check if we already have a fact for this value
        found = False
        for fact in existing_facts:
            if fact.value == value:
                # Combine certainties
                old_cf = fact.certainty
                new_cf = self.combine_certainties_or(old_cf, certainty)
                fact.certainty = new_cf
                if source_rule:
                    fact.source_rules.append(source_rule)
                found = True
                break
        
        if not found:
            new_fact = Fact(parameter, value, certainty, [source_rule] if source_rule else [])
            existing_facts.append(new_fact)
            self.known_facts[parameter] = existing_facts

    def find_out(self, parameter: str, patient_data: Dict[str, Any]) -> None:
        """
        The core backward chaining function.
        Find the value(s) of this parameter, unless already known.
        """
        # If we have any facts about this parameter with absolute certainty (1.0 or -1.0), 
        # or if we have exhausted sources, we might stop. 
        # But MYCIN typically gathers all evidence.
        # For simple optimization: if we already asked the user, we don't ask again.
        # If we already ran rules, we might not want to re-run (memoization could be added).
        
        # Check if we should ask first or use rules first
        ask_first = parameter in ASK_FIRST_PARAMETERS
        
        if ask_first:
            # 1. Ask User/LLM
            self.ask_vals(parameter, patient_data)
            # 2. Use Rules (if needed / as supplement)
            self.use_rules(parameter, patient_data)
        else:
            # 1. Use Rules
            self.use_rules(parameter, patient_data)
            # 2. Ask User/LLM (if rules didn't provide definitive answer)
            # Only ask if rules failed to provide ANY positive evidence? 
            # Or if parameter is askable?
            # In MYCIN, if rules fail, it asks.
            if not self.get_facts(parameter):
                 self.ask_vals(parameter, patient_data)

    def ask_vals(self, parameter: str, patient_data: Dict[str, Any]) -> None:
        """Ask the user/LLM for the value of the parameter."""
        if parameter in self.asked_questions:
            return
        
        self.asked_questions.add(parameter)
        
        # Check if data is already in patient_data (simulating "asking")
        if parameter in patient_data:
            val = patient_data[parameter]
            # Assume 1.0 certainty for provided data
            self.update_fact(parameter, val, 1.0, source_rule="user_input")
            return

        # If not in data, ask LLM
        if self.llm_qa_fn:
            answer, certainty = self.llm_qa_fn(parameter, patient_data)
            if answer is not None:
                self.update_fact(parameter, answer, certainty, source_rule="llm_qa")

    def use_rules(self, parameter: str, patient_data: Dict[str, Any]) -> None:
        """Try every rule associated with this parameter."""
        relevant_rules = self.rules_by_conclusion.get(parameter, [])
        for rule in relevant_rules:
            self.use_rule(rule, patient_data)

    def use_rule(self, rule: Rule, patient_data: Dict[str, Any]) -> None:
        """Apply a rule to the current situation."""
        if rule.rule_id in self.traced_rules:
            return # Avoid infinite loops
        
        self.traced_rules.add(rule.rule_id)
        
        # Evaluate premises
        # If any premise is known false, give up.
        # If every premise can be proved true, draw conclusions.
        
        # We need to calculate the minimum CF of all conditions (AND logic)
        min_cf = 1.0
        
        for condition in rule.conditions:
            # Ensure we have info about this parameter
            # Recursively call find_out
            self.find_out(condition.parameter, patient_data)
            
            # Evaluate condition
            cf = self.eval_condition(condition)
            
            # If false (CF <= 0.2 usually, but let's say <= 0 for strict AND), rule fails
            # MYCIN uses 0.2 cutoff.
            if cf <= 0.2: 
                min_cf = 0.0
                break
            
            min_cf = min(min_cf, cf)
        
        if min_cf > 0.2:
            # Rule succeeded
            for concl_param, concl_val in rule.conclusion.items():
                rule_cf = rule.certainty_factor
                weighted_cf = min_cf * rule_cf
                self.update_fact(concl_param, concl_val, weighted_cf, source_rule=rule.rule_id)
        
        self.traced_rules.remove(rule.rule_id)

    def eval_condition(self, condition: RuleCondition) -> float:
        """
        Evaluate a single condition against known facts.
        Returns the certainty that the condition is true.
        """
        param = condition.parameter
        op = condition.operator
        val = condition.value
        
        facts = self.get_facts(param)
        if not facts:
            return 0.0
            
        # Sum up evidence for the condition being true
        # (This is a simplification; MYCIN iterates over all values)
        
        total_cf = 0.0
        
        for fact in facts:
            fact_val = fact.value
            fact_cf = fact.certainty
            
            if fact_cf <= 0: continue # Only positive evidence counts towards "satisfying" a condition usually
            
            match = False
            if op == "is":
                match = (fact_val == val)
            elif op == "is_not":
                match = (fact_val != val)
            elif op == "greater_than":
                try: match = float(fact_val) > float(val)
                except: match = False
            elif op == "less_than":
                try: match = float(fact_val) < float(val)
                except: match = False
            # ... add other operators as needed
            
            if match:
                total_cf = self.combine_certainties_or(total_cf, fact_cf)
                
        return total_cf

    @staticmethod
    def combine_certainties_or(cf1: float, cf2: float) -> float:
        """
        Combine the certainty factors for (A or B).
        Used when two rules support the same conclusion.
        """
        if cf1 > 0 and cf2 > 0:
            return cf1 + cf2 - (cf1 * cf2)
        elif cf1 < 0 and cf2 < 0:
            return cf1 + cf2 + (cf1 * cf2)
        else:
            return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))

    def backward_chain(self, goal_parameter: str, patient_data: Dict[str, Any]) -> None:
        """Start backward chaining for a specific goal."""
        self.find_out(goal_parameter, patient_data)

    def get_diagnosis(self) -> Dict[str, Any]:
        """
        Get the final diagnosis based on accumulated facts.
        """
        # Helper to format output
        def format_facts(param_name):
            facts = self.get_facts(param_name)
            # Sort by certainty
            facts.sort(key=lambda x: x.certainty, reverse=True)
            return {
                "top": facts[0].value if facts else None,
                "certainty": facts[0].certainty if facts else 0.0,
                "all": {f.value: f.certainty for f in facts}
            }

        organism_data = format_facts("identity")
        site_data = format_facts("infection_site")
        
        # Treatment is a bit different, we might want to collect all recommended drugs
        drug_facts = self.get_facts("recommended_drug")
        drug_facts.sort(key=lambda x: x.certainty, reverse=True)
        treatments = {}
        for f in drug_facts:
            treatments[f.value] = {"certainty": f.certainty, "source_rules": f.source_rules}
            
        return {
            "organism_identity": {
                "name": organism_data["top"],
                "certainty": organism_data["certainty"],
                "probabilities": organism_data["all"] # Not strictly probs, but CFs
            },
            "infection_site": {
                "site": site_data["top"],
                "certainty": site_data["certainty"],
                "probabilities": site_data["all"]
            },
            "treatment": {
                "recommended_drug": drug_facts[0].value if drug_facts else None,
                "all_treatments": treatments
            },
            "all_facts": {
                k: [{"value": f.value, "certainty": f.certainty} for f in v]
                for k, v in self.known_facts.items()
            }
        }


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
    """
    question_text = QUESTIONS.get(question_key, question_key)
    prompt = create_llm_question_prompt(question_key, question_text, patient_data)
    
    if llm_call_fn:
        response = llm_call_fn(prompt)
        # Parse response
        if isinstance(response, str):
            response_lower = response.strip().lower()
            if response_lower in ["true", "yes", "1"]:
                return True, 1.0
            elif response_lower in ["false", "no", "0"]:
                return False, 1.0
            elif "unknown" in response_lower:
                return None, 0.0
            else:
                # Try to parse as number
                try:
                    return float(response), 1.0
                except ValueError:
                    # Clean up string (remove quotes etc)
                    return response.strip('"\' '), 1.0
        return response, 1.0
    else:
        return None, 0.0

