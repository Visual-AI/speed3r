import torch
import torch.nn as nn
from collections import OrderedDict
from ..models.pi3_sparse import Pi3_Sparse
from ..models.pi3 import Pi3

def load_old_ckpt_into_sparse_model(sparse_model: nn.Module, teacher_model_state_dict: dict):
    """
    Loads weights from an old Pi3 checkpoint into the new sparse Pi3 model.

    This function remaps the decoder weights from the original unified decoder
    to the new local_decoder and global_decoder, and handles other architectural
    changes like the removal of the register_token.

    Args:
        sparse_model (nn.Module): An instance of the new sparse Pi3 model.
        teacher_model_state_dict (dict): The state_dict loaded from the old model's checkpoint.
    """
    # Create a new state dictionary to hold the remapped weights
    new_state_dict = OrderedDict()

    print("Starting weight remapping process...")

    for key, value in teacher_model_state_dict.items():
        # --- Handle the decoder remapping ---
        if key.startswith('decoder.'):
            # Example key: "decoder.15.attn.qkv.weight"
            parts = key.split('.')
            block_idx = int(parts[1])
            rest_of_key = '.'.join(parts[2:])
            
            # New index for the split decoders
            new_idx = block_idx // 2

            if block_idx % 2 == 0:
                # Even-indexed blocks go to the local_decoder
                new_key = f"local_decoder.{new_idx}.{rest_of_key}"
                print(f"Remapping '{key}' -> '{new_key}'")
            else:
                # Odd-indexed blocks go to the global_decoder
                new_key = f"global_decoder.{new_idx}.{rest_of_key}"
                print(f"Remapping '{key}' -> '{new_key}'")
            
            new_state_dict[new_key] = value

        # --- Ignore the register_token from the old model ---
        elif key == 'register_token':
            print(f"Ignoring '{key}' from the old checkpoint as it's removed in the new model.")
            continue
            
        # --- Copy all other matching weights directly ---
        else:
            if key in sparse_model.state_dict():
                new_state_dict[key] = value
            else:
                print(f"Warning: Key '{key}' from old checkpoint not found in the new model. Skipping.")

    # --- Load the remapped state dictionary ---
    missing_keys, unexpected_keys = sparse_model.load_state_dict(new_state_dict, strict=False)

    if not missing_keys and not unexpected_keys:
        print("\nSuccessfully loaded and remapped all weights! The model is ready.")
    else:
        print("\nLoading finished with some discrepancies.")
        if missing_keys:
            print("\nMissing keys in new model (were not in the remapped checkpoint):")
            for k in missing_keys:
                print(f"  - {k}")
        if unexpected_keys:
            print("\nUnexpected keys in remapped checkpoint (not found in the new model):")
            for k in unexpected_keys:
                print(f"  - {k}")

    return sparse_model



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = Pi3.from_pretrained("yyfz233/Pi3").to("cpu").eval()
    teacher_model_state_dict = teacher_model.state_dict()

    sparse_model = Pi3_Sparse().to(device).eval()
    sparse_model = load_old_ckpt_into_sparse_model(sparse_model, teacher_model_state_dict)
