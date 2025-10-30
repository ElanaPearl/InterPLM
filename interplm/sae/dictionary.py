"""
Defines the SAE classes.
(Originally built off of code from https://github.com/saprmarks/dictionary_learning/blob/2d586e417cd30473e1c608146df47eb5767e2527/dictionary.py)
"""

from abc import ABC, abstractmethod
from typing import Optional

import einops
import torch as t
import torch.nn as nn
import torch.nn.init as init

from interplm.utils import get_device


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    def __init__(self, normalize_to_sqrt_d=False):
        super().__init__()
        self.normalize_to_sqrt_d = normalize_to_sqrt_d
    
    @property
    def has_normalization_factors(self) -> bool:
        """Check if normalization factors have been calculated (not just initialized to ones)."""
        if not hasattr(self, 'activation_rescale_factor'):
            return False
        # Check if factors are not all ones (which is the default initialization)
        return not t.allclose(self.activation_rescale_factor, t.ones_like(self.activation_rescale_factor))

    def _unnormalize_output(self, x_normalized, original_norms):
        """Un-normalize output back to original scale if normalization was applied."""
        if self.normalize_to_sqrt_d:
            # For √d scaling: x_original = x_normalized * original_norms / √d
            d = x_normalized.shape[-1]
            sqrt_d = t.sqrt(t.tensor(d, dtype=x_normalized.dtype, device=x_normalized.device))
            return x_normalized * original_norms / sqrt_d
        return x_normalized

    def _normalize_input_and_get_norms(self, x):
        """
        Apply Anthropic's √d normalization to input and return normalized input + original norms.
        If normalization is disabled, returns original input and None.
        """
        if self.normalize_to_sqrt_d:
            # Save original norms
            original_norms = t.norm(x, dim=-1, keepdim=True)
            original_norms = t.clamp(original_norms, min=1e-8)
            
            # Apply Anthropic's √d scaling: normalize to unit vectors, then scale by √d
            d = x.shape[-1]
            sqrt_d = t.sqrt(t.tensor(d, dtype=x.dtype, device=x.device))
            normalized = t.nn.functional.normalize(x, p=2, dim=-1) * sqrt_d
            
            return normalized, original_norms
        return x, None

    @abstractmethod
    def encode(self, x, normalize_features: bool = False):
        """
        Encode a vector x in the activation space.
        
        Args:
            x: Input activations to encode
            normalize_features: If True, divide features by activation_rescale_factor
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @abstractmethod
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        """
        Encode a vector x in the activation space using a subset of features.
        """
        pass

    @classmethod
    def from_pretrained(cls, path: str | None = None, device=None):
        """
        Load a pretrained dictionary from a file.
        """
        raise NotImplementedError(
            "from_pretrained not implemented for this dictionary class"
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Call parent class load method first
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # If activation_rescale_factor wasn't in state dict, it will be in missing_keys
        # Remove it from missing_keys since we handle it here
        norm_factor_key = prefix + "activation_rescale_factor"
        if norm_factor_key in missing_keys:
            missing_keys.remove(norm_factor_key)
            # activation_rescale_factor buffer was already initialized to ones in __init__

        # Load normalize_to_sqrt_d parameter if it exists
        norm_key = prefix + "normalize_to_sqrt_d"
        if norm_key in state_dict:
            self.normalize_to_sqrt_d = state_dict[norm_key].item()
            # Remove from unexpected_keys if it's there
            if norm_key in unexpected_keys:
                unexpected_keys.remove(norm_key)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Save the normalize_to_sqrt_d parameter to state dict."""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "normalize_to_sqrt_d"] = t.tensor(self.normalize_to_sqrt_d, dtype=t.bool)
    
    def _make_contiguous(self):
        """Make all parameters contiguous (called automatically during init)."""
        for param in self.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()


class ReLUSAE(Dictionary):
    """
    ReLU SAE with separate pre-encoding bias and untied encoder/decoder weights.

    Architecture:
    - Pre-encoding bias applied before encoder
    - Decoder has no bias
    - Independent (untied) encoder and decoder weights
    - Decoder weights are unit-normed
    """

    def __init__(self, activation_dim, dict_size, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # Initialize encoder bias to zero
        nn.init.zeros_(self.encoder.bias)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

        self.register_buffer("activation_rescale_factor", t.ones(dict_size))
        self._make_contiguous()

    def encode(self, x, normalize_features: bool = False):
        features = nn.ReLU()(self.encoder(x - self.bias))
        if normalize_features:
            features /= self.activation_rescale_factor
        return features

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None, unnormalize=False):
        """
        Forward pass of an autoencoder.

        Args:
            x: activations to be autoencoded (unnormalized)
            output_features: if True, return the encoded features as well as the decoded x
            ghost_mask: if not None, run this autoencoder in "ghost mode" where features are masked
            unnormalize: if True, un-normalize output to original space (for injection/steering).
                        if False (default), keep output in normalized √d space (for training/analysis)
        """
        # Apply unit normalization and get original norms (calculated once)
        x, original_norms = self._normalize_input_and_get_norms(x)

        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)

            # Only un-normalize if explicitly requested (for fidelity/steering)
            if unnormalize:
                x_hat = self._unnormalize_output(x_hat, original_norms)

            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(
                f_ghost
            )  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)

            # Un-normalize both outputs
            x_ghost = self._unnormalize_output(x_ghost, original_norms)
            x_hat = self._unnormalize_output(x_hat, original_norms)

            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        x, _ = self._normalize_input_and_get_norms(x)
        features = t.nn.ReLU()((x - self.bias) @ encoder_w_subset.T + encoder_b_subset)
        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]
        return features

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        if device is None:
            device = get_device()
        state_dict = t.load(path, map_location=device)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        normalize_to_sqrt_d = state_dict.get("normalize_to_sqrt_d", t.tensor(False)).item()

        autoencoder = ReLUSAE(activation_dim, dict_size, normalize_to_sqrt_d=normalize_to_sqrt_d)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder = autoencoder.to(device)
        return autoencoder


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function. This is useful for treating neurons as features.

    """

    def __init__(self, activation_dim=None, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

        # Use nn.Identity for both encoder and decoder
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

        self.register_buffer("activation_rescale_factor", t.ones(activation_dim) if activation_dim else t.tensor([]))

    def encode(self, x, normalize_features: bool = False):
        x = self.encoder(x)
        if normalize_features:
            x = x / self.activation_rescale_factor
        return x

    def decode(self, f):
        return self.decoder(f)

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        features = x[:, feat_list]

        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]

        return features

    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x


class ReLUSAE_Tied(Dictionary, nn.Module):
    """
    ReLU SAE with tied encoder/decoder weights and dual biases.

    Architecture from https://transformer-circuits.pub/2024/april-update/index.html#training-saes

    Key differences from ReLUSAE:
    - Tied weights: encoder.weight = decoder.weight.T
    - Both encoder and decoder have bias parameters
    - Weights initialized to 0.1 norm (vs unit norm)
    - No separate pre-encoding bias parameter
    """

    def __init__(self, activation_dim, dict_size, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

        self.register_buffer("activation_rescale_factor", t.ones(dict_size))
        self._make_contiguous()

    def encode(self, x, normalize_features: bool = False):
        features = nn.ReLU()(self.encoder(x))
        if normalize_features:
            features /= self.activation_rescale_factor
        return features

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x, output_features=False, unnormalize=False):
        """
        Forward pass of an autoencoder.

        Args:
            x: activations to be autoencoded (unnormalized)
            output_features: if True, return the encoded features as well as the decoded x
            unnormalize: if True, un-normalize output to original space (for injection/steering).
                        if False (default), keep output in normalized √d space (for training/analysis)
        """
        # Apply unit normalization and get original norms (calculated once)
        x, original_norms = self._normalize_input_and_get_norms(x)

        f = self.encode(x)
        x_hat = self.decode(f)

        # Only un-normalize if explicitly requested (for fidelity/steering)
        if unnormalize:
            x_hat = self._unnormalize_output(x_hat, original_norms)

        if not output_features:
            return x_hat
        else:
            return x_hat, f

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        features = t.nn.ReLU()(x @ encoder_w_subset.T + encoder_b_subset)
        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]

        return features

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path, map_location=device)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        normalize_to_sqrt_d = state_dict.get("normalize_to_sqrt_d", t.tensor(False)).item()

        autoencoder = ReLUSAE_Tied(activation_dim, dict_size, normalize_to_sqrt_d=normalize_to_sqrt_d)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder = autoencoder.to(device)
        return autoencoder


class TopKSAE(Dictionary, nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    We abandoned it as we didn't notice a significant speedup and it added complications, which are noted
    in the TopKSAE class docstring in that branch.

    With some additional effort, you can train a Top-K SAE with the Triton kernels and modify the state dict for compatibility with this class.
    Notably, the Triton kernels currently have the decoder to be stored in nn.Parameter, not nn.Linear, and the decoder weights must also
    be stored in the same shape as the encoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        self.register_buffer("activation_rescale_factor", t.ones(dict_size))
        self._make_contiguous()

    def encode(
        self, x: t.Tensor, return_topk: bool = False, use_threshold: bool = False,
        normalize_features: bool = False
    ):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
            if normalize_features:
                encoded_acts_BF /= self.activation_rescale_factor
            if return_topk:
                post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)
                return (
                    encoded_acts_BF,
                    post_topk.values,
                    post_topk.indices,
                    post_relu_feat_acts_BF,
                )
            else:
                return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )
        
        if normalize_features:
            encoded_acts_BF /= self.activation_rescale_factor

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        # Apply unit normalization and get original norms (calculated once)
        x, original_norms = self._normalize_input_and_get_norms(x)
        
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        
        # Un-normalize the output
        x_hat_BD = self._unnormalize_output(x_hat_BD, original_norms)
        
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        post_relu_feat_acts_BF = t.nn.ReLU()(
            (x - self.b_dec) @ encoder_w_subset.T + encoder_b_subset
        )
        features = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]
        return features

    def from_pretrained(path, k: Optional[int] = None, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path, map_location=device)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        normalize_to_sqrt_d = state_dict.get("normalize_to_sqrt_d", t.tensor(False)).item()

        autoencoder = TopKSAE(activation_dim, dict_size, k, normalize_to_sqrt_d=normalize_to_sqrt_d)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder = autoencoder.to(device)
        return autoencoder


class JumpReLUSAE(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, device="cpu", normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        # Use standard encoder/decoder naming
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # Initialize decoder weights with kaiming uniform, then normalize to unit norm
        # Note: nn.Linear stores weights as (out_features, in_features) = (activation_dim, dict_size)
        # We normalize along dim=0 so each feature (column) has unit norm
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.decoder.weight.data = self.decoder.weight.data / self.decoder.weight.data.norm(dim=0, keepdim=True)

        # Initialize encoder as decoder transpose
        self.encoder.weight.data = self.decoder.weight.data.clone().T

        # Initialize biases to zero
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

        self.threshold = nn.Parameter(
            t.ones(dict_size, device=device) * 0.001
        )  # Appendix I

        self.apply_b_dec_to_input = False

        self.register_buffer("activation_rescale_factor", t.ones(dict_size))
        self._make_contiguous()

    def encode(self, x, output_pre_jump=False, normalize_features: bool = False):
        if self.apply_b_dec_to_input:
            x = x - self.decoder.bias
        pre_jump = self.encoder(x)

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))

        if normalize_features:
            f /= self.activation_rescale_factor

        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded (unnormalized)
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        # Apply unit normalization and get original norms (calculated once)
        x, original_norms = self._normalize_input_and_get_norms(x)
        
        f = self.encode(x)
        x_hat = self.decode(f)

        # Un-normalize the output
        x_hat = self._unnormalize_output(x_hat, original_norms)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.decoder.bias.data *= scale
        self.encoder.bias.data *= scale
        self.threshold.data *= scale

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        if self.apply_b_dec_to_input:
            x = x - self.decoder.bias
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        pre_jump = x @ encoder_w_subset.T + encoder_b_subset
        features = nn.ReLU()(pre_jump * (pre_jump > self.threshold[feat_list]))
        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]
        return features

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        dtype: t.dtype = t.float32,
        device: t.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path, map_location=device)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        normalize_to_sqrt_d = state_dict.get("normalize_to_sqrt_d", t.tensor(False)).item()

        autoencoder = JumpReLUSAE(activation_dim, dict_size, normalize_to_sqrt_d=normalize_to_sqrt_d)
        autoencoder.load_state_dict(state_dict)
        autoencoder = autoencoder.to(dtype=dtype, device=device)

        if device is not None:
            device = autoencoder.encoder.weight.device
        return autoencoder.to(dtype=dtype, device=device)


class BatchTopKSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        # Initialize normalization factor buffer with ones
        self.register_buffer("activation_rescale_factor", t.ones(dict_size))
        self._make_contiguous()

    def encode(
        self,
        x: t.Tensor,
        return_active: bool = False,
        use_threshold: bool = True,
        normalize_features: bool = False,
    ):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
        else:
            # Flatten and perform batch top-k
            flattened_acts = post_relu_feat_acts_BF.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)

            encoded_acts_BF = (
                t.zeros_like(post_relu_feat_acts_BF.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts_BF.shape)
            )

        if normalize_features:
            encoded_acts_BF /= self.activation_rescale_factor

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False, unnormalize: bool = False):
        """
        Forward pass of an autoencoder.

        Args:
            x: activations to be autoencoded (unnormalized)
            output_features: if True, return the encoded features as well as the decoded x
            unnormalize: if True, un-normalize output to original space (for injection/steering).
                        if False (default), keep output in normalized √d space (for training/analysis)
        """
        # Apply unit normalization and get original norms (calculated once)
        x, original_norms = self._normalize_input_and_get_norms(x)

        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        # Only un-normalize if explicitly requested (for fidelity/steering)
        if unnormalize:
            x_hat_BD = self._unnormalize_output(x_hat_BD, original_norms)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        encoded_acts_BF = t.nn.ReLU()(
            (x - self.b_dec) @ encoder_w_subset.T + encoder_b_subset
        )
        encoded_acts_BF = encoded_acts_BF * (encoded_acts_BF > self.threshold)

        if normalize_features:
            encoded_acts_BF /= self.activation_rescale_factor[feat_list]
        return encoded_acts_BF

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        k: int | None = None,
        threshold: float | None = None,
        device=None,
    ):
        """
        Load a pretrained dictionary from a file.
        """
        if device is None:
            device = get_device()
        state_dict = t.load(path, map_location=device)
        if k is None:
            print(f"Loading k={state_dict['k'].item()} from state dict")
            k = state_dict["k"].item()
        else:
            assert (
                k == state_dict["k"].item()
            ), f"k={k} != {state_dict['k'].item()}=state_dict['k']"

        if threshold is None:
            print(f"Loading threshold={state_dict['threshold'].item()} from state dict")
            threshold = state_dict["threshold"].item()
        else:
            assert (
                threshold == state_dict["threshold"].item()
            ), f"threshold={threshold} != {state_dict['threshold'].item()}=state_dict['threshold']"

        dict_size, activation_dim = state_dict["encoder.weight"].shape
        normalize_to_sqrt_d = state_dict.get("normalize_to_sqrt_d", t.tensor(False)).item()

        autoencoder = BatchTopKSAE(
            activation_dim=activation_dim, dict_size=dict_size, k=k, normalize_to_sqrt_d=normalize_to_sqrt_d
        ).to(device=device)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            device = autoencoder.encoder.weight.device

        return autoencoder.to(device=device)


# The next two functions could be replaced with the ConstrainedAdam Optimizer
@t.no_grad()
def set_decoder_norm_to_unit_norm(
    W_dec_DF: t.nn.Parameter, activation_dim: int, d_sae: int
) -> t.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape

    assert D == activation_dim
    assert F == d_sae

    eps = t.finfo(W_dec_DF.dtype).eps
    norm = t.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


@t.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    W_dec_DF: t.Tensor,
    W_dec_DF_grad: t.Tensor,
    activation_dim: int,
    d_sae: int,
) -> t.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape
    assert D == activation_dim
    assert F == d_sae

    normed_W_dec_DF = W_dec_DF / (t.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)

    parallel_component = einops.einsum(
        W_dec_DF_grad,
        normed_W_dec_DF,
        "d_in d_sae, d_in d_sae -> d_sae",
    )
    W_dec_DF_grad -= einops.einsum(
        parallel_component,
        normed_W_dec_DF,
        "d_sae, d_in d_sae -> d_in d_sae",
    )
    return W_dec_DF_grad
