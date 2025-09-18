"""Utility functions for LLM."""
from src.consts import GBA_KEYS
from typing import List, Tuple
import re
from src.llm.phi_client import PhiClient
from src.llm.llm_client import LLMClient

def convert_to_dict(
        actions: List[str | Tuple[str, ...]], 
        keys: str=GBA_KEYS):
    """Convert a list of actions to a dictionary of button states."""
    # Create a dictionary with all values set to False initially
    results = []

    # Loop through each action and update the dictionary based on the action
    for action in actions:
        result_dict = {key: False for key in keys}
        if isinstance(action, str):
            if action in result_dict:
                result_dict[action] = True
        elif isinstance(action, tuple):
            for act in action:
                if act in result_dict:
                    result_dict[act] = True

            # Don't allow START and SELECT to be pressed simultaneously
            # Prevent restarting the emulator
            if result_dict.get("START") and result_dict.get("SELECT"):
                result_dict["START"] = False
                result_dict["SELECT"] = False
        results.append(result_dict)

    return results


def parse_actions_response(text: str) -> List[str | Tuple[str, ...]]:
    """
    Given a LLM response, look for the ```actions``` tag and parse it 
    into a list of actions.
    """
    # Look for content between ```actions and ``` markers
    pattern = r'```actions\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    combined_actions = []
    for match in matches:
        try:
            # Safely evaluate the string as Python code
            # Remove any comments after // or #
            action = re.sub(r'(?://|#).*$', '', match, flags=re.MULTILINE)
            actions = eval(action.strip())
            combined_actions.extend(actions)  # Use extend instead of append
        except:
            print(f"Failed to parse actions: {match}")
            return []
            
    return combined_actions  # Return single combined list


def convert_from_dict(button_states: List[dict]) -> List[str]:
    """Convert a dictionary of button states to a list of active buttons.
    
    Args:
        button_states: Dictionary mapping button names to their states (True/False)
        
    Returns:
        List of button names that are active (True)
    """
    return [key for key, value in button_states.items() if value]

def get_client(model, api_key, temperature, max_tokens, log_dir, api_base):
    if 'phi' in model:
        return PhiClient(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            log_dir=log_dir,
            api_base=api_base
        )
    return LLMClient(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        log_dir=log_dir,
        api_base=api_base
    )
