import numpy as np
from collections import OrderedDict


# Quantize the state space and action space
def quantize_state(state: OrderedDict, state_bins: dict) -> tuple:
    """
    
    Given the state and the bins for each state variable, quantize the state space.

    Args:
        state (dict): The state to be quantized.
        state_bins (dict): The bins used for quantizing each dimension of the state.

    Returns:
        tuple: The quantized representation of the state.
    """
    quantized_state = []
    for key, value in state.items():
        bins = state_bins[key]
        for i, (v, b) in enumerate(zip(value, bins)):
            quantized_state.append(np.digitize(v, b))
    
    return tuple(quantized_state)

    
    


def quantize_action(action: float, bins: list) -> int:
    """
    Quantize the action based on the provided bins. 
    """
    quantized_action = np.digitize(action, bins)
    return quantized_action



