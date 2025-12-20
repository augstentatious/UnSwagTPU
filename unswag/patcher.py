import torch
import torch.nn as nn
from .layers import UnSwagLinear

def _get_submodules(model, key):
    """
    Helper to navigate deep into the model structure (e.g., 'layers.0.mlp.gate_proj').
    Returns the parent module and the name of the child to replace.
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    return parent, target_name

def unswag_model(model, target_modules=None, verbose=True):
    """
    Surgically replaces standard nn.Linear layers with UnSwagLinear layers.
    
    Args:
        model: The PyTorch model (e.g., loaded via AutoModelForCausalLM).
        target_modules: List of string names to replace (e.g., ['q_proj', 'v_proj']).
                        If None, tries to replace ALL linear layers (risky for heads).
        verbose: Print logs of what got swapped.
    """
    if verbose:
        print("🦁 UnSwag Patcher: Scanning model for targets...")

    # If targets aren't specified, we define standard safe targets for Transformers
    if target_modules is None:
        # These are common names in Llama, Gemma, Mistral
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    replaced_count = 0
    
    # We iterate a copy of named_modules because we are modifying the dict in place
    for name, module in list(model.named_modules()):
        # Check if this module matches our target list
        # We look at the END of the name (e.g. 'layers.0.self_attn.q_proj' ends with 'q_proj')
        if any(name.endswith(target) for target in target_modules):
            if isinstance(module, nn.Linear):
                # 1. Identify the location
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                
                # 2. Create the UnSwag Replacement
                # We copy the weights and bias from the original layer
                new_layer = UnSwagLinear(
                    in_features=module.in_features, 
                    out_features=module.out_features, 
                    bias=(module.bias is not None)
                )
                
                # Copy Weights (In V2 we will quantize these here!)
                new_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_layer.bias.data = module.bias.data.clone()
                
                # Move to correct device (GPU/CPU)
                new_layer.to(module.weight.device)
                
                # 3. Perform the Transplant
                setattr(parent, child_name, new_layer)
                replaced_count += 1
                
    if verbose:
        print(f"🦁 Success: Swapped {replaced_count} layers to UnSwagLinear.")
        print("🦁 The model is now running on the UnSwag Protocol.")

    return model
