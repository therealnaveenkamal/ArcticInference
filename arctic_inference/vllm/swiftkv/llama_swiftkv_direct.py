#!/usr/bin/env python3
"""
SwiftKV with Direct FlashInfer Integration
This bypasses vLLM's attention backend for better control and performance.
"""

import copy
from typing import Any, Iterable, List, Optional, Tuple, Union
import logging

import torch
from torch import nn

import vllm.distributed.parallel_state as parallel_state
from vllm.attention.backends.abstract import AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.llama import (LlamaAttention,
                                              LlamaDecoderLayer,
                                              LlamaMLP)
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

import arctic_inference.vllm.model_runner as model_runner
from arctic_inference.common.swiftkv.configs import LlamaSwiftKVConfig

# Try to import our direct FlashInfer module
try:
    from arctic_inference.vllm.swiftkv.direct_flashinfer import SwiftKVFlashInferAttention, FLASHINFER_AVAILABLE
except ImportError:
    logging.warning("Could not import direct FlashInfer - falling back to standard attention")
    SwiftKVFlashInferAttention = None
    FLASHINFER_AVAILABLE = False

logger = init_logger(__name__)


def get_attn_metadata_for_swiftkv():
    fwd_ctx = get_forward_context()
    if fwd_ctx.attn_metadata is None:
        return None
    meta = next(iter(fwd_ctx.attn_metadata.values()))
    assert all(m is meta for m in fwd_ctx.attn_metadata.values()), \
        "All attention metadata should be the same for LlamaSwiftKV."
    return meta


class LlamaSwiftKVDirectAttention(LlamaAttention):
    """
    SwiftKV attention layer using direct FlashInfer integration.
    """

    def __init__(
        self,
        config: LlamaSwiftKVConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        # Initialize parent
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=prefix,
            attn_type=attn_type)

        # SwiftKV projections
        self.q_proj_swiftkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_swiftkv",
        )

        self.kv_proj_swiftkv = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj_swiftkv",
        )

        # Direct FlashInfer attention
        if FLASHINFER_AVAILABLE and SwiftKVFlashInferAttention and cache_config:
            try:
                self.flashinfer_attn = SwiftKVFlashInferAttention(
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    page_size=cache_config.block_size,
                    dtype=torch.float16
                )
                self.use_direct_flashinfer = True
                logger.info(f"✅ Direct FlashInfer enabled for layer {prefix}")
            except Exception as e:
                logger.warning(f"Failed to initialize FlashInfer for {prefix}: {e}")
                self.flashinfer_attn = None
                self.use_direct_flashinfer = False
        else:
            self.flashinfer_attn = None
            self.use_direct_flashinfer = False
            logger.info(f"💡 Using standard attention for layer {prefix}")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute query
        q, _ = self.q_proj_swiftkv(hidden_states)
        q, _ = self.rotary_emb(positions, q, torch.empty_like(k))
        
        if self.use_direct_flashinfer and selected_indices is not None:
            # Use direct FlashInfer with SwiftKV selection
            try:
                # Reshape for attention
                q = q.view(-1, self.num_heads, self.head_size)
                k = k.view(-1, self.num_kv_heads, self.head_size)  
                v = v.view(-1, self.num_kv_heads, self.head_size)
                
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = get_attn_metadata_for_swiftkv()
                kv_cache = self.attn.kv_cache[forward_context.virtual_engine]
                
                attn_output = self.flashinfer_attn.forward_decode(
                    query=q,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    selected_indices=selected_indices,
                    scale=self.scale
                )
                
                # Reshape output
                attn_output = attn_output.view(-1, self.num_heads * self.head_size)
                output, _ = self.o_proj(attn_output)
                
                logger.debug(f"Direct FlashInfer processed {q.shape[0]} tokens")
                return output
                
            except Exception as e:
                logger.warning(f"FlashInfer failed, falling back to standard attention: {e}")
                # Fall through to standard attention

        # Standard attention fallback
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaSwiftKVDirectDecoderLayer(nn.Module):
    """
    SwiftKV decoder layer using direct FlashInfer attention.
    """

    def __init__(
        self,
        config: LlamaSwiftKVConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
            
        self.self_attn = LlamaSwiftKVDirectAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        selected_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
                
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            k=k_states,
            v=v_states,
            selected_indices=selected_indices,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# Modified decode runner to pass selected indices
@support_torch_compile
class LlamaSwiftKVDirectDecodeRunner(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, model: "LlamaSwiftKVDirectModel",
                 prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self._model = [model]

    @property
    def model(self) -> "LlamaSwiftKVDirectModel":
        return self._model[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        selected_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch._check(v_states.shape[0] == k_states.shape[0])
        num_layers = (self.config.num_hidden_layers -
                      self.config.num_key_value_layers)
        k_split = torch.chunk(k_states, num_layers, dim=-1)
        v_split = torch.chunk(v_states, num_layers, dim=-1)
        
        for idx, layer in enumerate(
                self.model.layers[self.config.num_key_value_layers:]):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                k_split[idx],
                v_split[idx],
                residual,
                selected_indices=selected_indices,  # Pass selection info
            )
            
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states


class LlamaSwiftKVDirectModel(nn.Module):
    """
    SwiftKV model with direct FlashInfer integration.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        
        logger.info(f"SwiftKV Direct Model initializing with FlashInfer: {FLASHINFER_AVAILABLE}")

        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=self.quant_config,
        )
        
        # Regular layers for prefill
        self.layers = torch.nn.ModuleList([
            LlamaDecoderLayer(config=config,
                              cache_config=vllm_config.cache_config,
                              quant_config=vllm_config.quant_config,
                              prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_key_value_layers)
        ])
        
        # SwiftKV layers with direct FlashInfer
        with model_runner.set_shift_parallel_mode(True):
            self.layers.extend([
                LlamaSwiftKVDirectDecoderLayer(config=config,
                                             cache_config=vllm_config.cache_config,
                                             quant_config=vllm_config.quant_config,
                                             prefix=f"{prefix}.layers.{idx}")
                for idx in range(config.num_key_value_layers,
                                 config.num_hidden_layers)
            ])
            self.norm_swiftkv = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        for param in self.layers[config.num_key_value_layers:].parameters():
            param.shift_parallel_mode = True

        # Initialize runners
        self._init_prefill_runner(vllm_config)
        self._init_decode_runner(vllm_config)

        from arctic_inference.py_custom_ops import try_load_torch_library
        self.use_custom_ops = True if try_load_torch_library() else False

    def _init_prefill_runner(self, vllm_config: VllmConfig):
        from arctic_inference.vllm.swiftkv.llama_swiftkv import LlamaSwiftKVPrefillRunner
        self.prefill_runner = LlamaSwiftKVPrefillRunner(
            vllm_config=vllm_config, model=self)

    def _init_decode_runner(self, vllm_config: VllmConfig):
        self.decode_runner = LlamaSwiftKVDirectDecodeRunner(
            vllm_config=vllm_config, model=self)

        # CUDA graph setup (same as original)
        config = vllm_config.model_config.hf_config
        if vllm_config.compilation_config.cudagraph_capture_sizes:
            self.cuda_graph_max_batch_size = max(
                vllm_config.compilation_config.cudagraph_capture_sizes)
            num_heads = self.layers[-1].self_attn.attn.num_kv_heads
            head_size = self.layers[-1].self_attn.attn.head_size
            num_kv = config.num_hidden_layers - config.num_key_value_layers
            kv_size = num_kv * num_heads * head_size
            self.decode_runner.inputs = {
                "hidden_states": torch.empty(self.cuda_graph_max_batch_size,
                                             config.hidden_size, device="cuda"),
                "residual": torch.empty(self.cuda_graph_max_batch_size,
                                        config.hidden_size, device="cuda"),
                "positions": torch.empty(self.cuda_graph_max_batch_size,
                                         dtype=torch.long, device="cuda"),
                "k_states": torch.empty(self.cuda_graph_max_batch_size,
                                        kv_size, device="cuda"),
                "v_states": torch.empty(self.cuda_graph_max_batch_size,
                                        kv_size, device="cuda"),
            }
        else:
            self.cuda_graph_max_batch_size = 0

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def swiftkv_select_simple(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, Optional[torch.Tensor]]:
        """
        Simple SwiftKV selection without complex metadata manipulation.
        Returns selected tensors + selection indices for direct FlashInfer.
        """
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = get_attn_metadata_for_swiftkv()
        
        if attn_metadata is None:
            # Graph capture mode
            if hidden_states.shape[0] <= self.cuda_graph_max_batch_size:
                inputs = self.decode_runner.inputs
                batch_size = hidden_states.shape[0]
                padded_size = self.vllm_config.pad_for_cudagraph(batch_size)
                return (inputs["hidden_states"][:padded_size],
                        inputs["residual"][:padded_size],
                        inputs["positions"][:padded_size],
                        inputs["k_states"][:padded_size],
                        inputs["v_states"][:padded_size],
                        None)  # No selection indices
            return hidden_states, residual, positions, k_states, v_states, None

        # Cache KV states (same as before)
        if self.use_custom_ops:
            key_caches = []
            value_caches = []
            k_scales = []
            v_scales = []
            num_heads = self.layers[-1].self_attn.attn.num_kv_heads
            head_size = self.layers[-1].self_attn.attn.head_size
            
            for idx, layer in enumerate(self.layers[self.config.num_key_value_layers:]):
                attn = layer.self_attn.attn
                kv_cache = attn.kv_cache[forward_context.virtual_engine]
                if kv_cache.numel():
                    key_caches.append(kv_cache[0])
                    value_caches.append(kv_cache[1])
                    k_scales.append(attn._k_scale)
                    v_scales.append(attn._v_scale)

            if len(key_caches) > 0:
                from arctic_inference.py_custom_ops import reshape_and_cache_flash_bulk
                reshape_and_cache_flash_bulk(
                    k_states, v_states, key_caches, value_caches, attn_metadata.slot_mapping,
                    attn.kv_cache_dtype, k_scales, v_scales, num_heads, head_size)
        else:
            # Standard caching
            num_layers = self.config.num_hidden_layers - self.config.num_key_value_layers
            k_split = k_states.chunk(num_layers, dim=-1)
            v_split = v_states.chunk(num_layers, dim=-1)

            for idx, layer in enumerate(self.layers[self.config.num_key_value_layers:]):
                attn = layer.self_attn.attn
                kv_cache = attn.kv_cache[forward_context.virtual_engine]
                if kv_cache.numel():
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        k_split[idx].view(-1, attn.num_kv_heads, attn.head_size),
                        v_split[idx].view(-1, attn.num_kv_heads, attn.head_size),
                        kv_cache[0],
                        kv_cache[1],
                        attn_metadata.slot_mapping,
                        attn.kv_cache_dtype,
                        attn._k_scale,
                        attn._v_scale,
                    )

        # Simple token selection - NO metadata manipulation
        if not hasattr(attn_metadata, 'swiftkv_logits_indices'):
            logger.debug("No SwiftKV selection - using all tokens")
            return hidden_states, residual, positions, k_states, v_states, None
            
        logits_indices = attn_metadata.swiftkv_logits_indices
        logger.debug(f"SwiftKV selecting {logits_indices.numel()} tokens out of {hidden_states.shape[0]}")

        # Simple indexing
        def index_tensor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.index_select(0, logits_indices)

        return (index_tensor(hidden_states),
                index_tensor(residual),
                index_tensor(positions),
                index_tensor(k_states),
                index_tensor(v_states),
                logits_indices)  # Return selection indices for FlashInfer

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states, residual, positions, k_states, v_states = (
            self.prefill_runner(input_ids, positions))

        orig_hidden_states = hidden_states
        hidden_states, residual, positions, k_states, v_states, selected_indices = (
            self.swiftkv_select_simple(
                hidden_states,
                residual,
                positions,
                k_states,
                v_states))

        with model_runner.set_shift_parallel_mode(True):
            hidden_states = self.decode_runner(
                hidden_states,
                residual,
                positions,
                k_states,
                v_states,
                selected_indices,  # Pass to decode runner
            )

        # Restore to original batch size
        if selected_indices is not None:
            batch_size = selected_indices.numel()
            orig_hidden_states[selected_indices] = hidden_states[:batch_size]
            return orig_hidden_states
        else:
            return hidden_states

    # Keep existing weight loading logic
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Same as original implementation
        from arctic_inference.vllm.swiftkv.llama_swiftkv import LlamaSwiftKVModel
        return LlamaSwiftKVModel.load_weights(self, weights)


class LlamaSwiftKVDirectForCausalLM(nn.Module):
    """
    SwiftKV for Causal LM with direct FlashInfer integration.
    """
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj_swiftkv": ["k_proj_swiftkv", "v_proj_swiftkv"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = LlamaSwiftKVDirectModel(vllm_config=vllm_config,
                                           prefix=maybe_prefix(prefix, "model"))

        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                logit_scale)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        assert intermediate_tensors is None and inputs_embeds is None
        model_output = self.model(input_ids, positions)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights) 

logger.info("🚀 SwiftKV Direct FlashInfer module loaded successfully!") 