import torch
from nnsight import NNsight
from transformers import EsmForMaskedLM


def get_esm_submodule_and_access_method(nnsight_model: NNsight, hidden_layer_idx: int):
    """Get the submodule at the given hidden layer index and the access method (input or output)

    To patch the hidden state of layer i, ideally we would just adjust the .output attribute of that
    layer, however ESM-2 has a namespace clash where it already has a pre-defined .output attribute
    so nnsight instead mounts this at .nns_output, but I haven't had success using this attribute so
    instead we just adjust the .input attribute of the next layer (i+1) which is equivalent.
    """

    # In newer transformers (>=4.36), ESM models no longer have .esm prefix
    # Try both paths for backward compatibility
    try:
        encoder = nnsight_model.esm.encoder
    except AttributeError:
        encoder = nnsight_model.encoder

    n_layers = len(encoder.layer)

    # Confirm the hidden layer index is within bounds
    if hidden_layer_idx > n_layers:
        raise ValueError(f"Hidden layer index {hidden_layer_idx} is out of bounds")

    # The last hidden layer has an additional normalization step so we can access the output of that
    elif hidden_layer_idx == n_layers:
        return encoder.emb_layer_norm_after, "output"

    # The layers are indexed 0-5 in the list and we refer to them 1-6 so we don't need to adjust the index
    # in order to access the input of the next layer
    else:
        return encoder.layer[hidden_layer_idx], "input"


def get_esm_output_with_intervention(
    esm_model: EsmForMaskedLM,
    nnsight_model: NNsight,
    batch_tokens: torch.Tensor,
    batch_attn_mask: torch.Tensor,
    hidden_layer_idx: int,
    hidden_state_override: torch.Tensor | None = None,
):
    """Get model output with optional hidden state modification."""
    with torch.no_grad():
        orig_output = esm_model(
            batch_tokens, attention_mask=batch_attn_mask, output_hidden_states=True
        )

        if hidden_state_override is None:
            return orig_output.logits, orig_output.hidden_states[hidden_layer_idx]

        submodule, input_or_output = get_esm_submodule_and_access_method(
            nnsight_model, hidden_layer_idx
        )

        with nnsight_model.trace(
            batch_tokens, attention_mask=batch_attn_mask
        ) as tracer:
            embd_to_patch = (
                submodule.input[0][0]
                if input_or_output == "input"
                else submodule.output
            )
            embd_to_patch[:] = hidden_state_override
            modified_logits = nnsight_model.output.save()

        return modified_logits.value.logits, orig_output.hidden_states[hidden_layer_idx]
