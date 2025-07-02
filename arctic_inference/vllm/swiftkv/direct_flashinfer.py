#!/usr/bin/env python3
"""
Direct FlashInfer integration for SwiftKV - bypassing vLLM's attention backend.
This gives us full control over FlashInfer usage.
"""

import torch
from torch import nn
from typing import Optional, Tuple, List
import logging

try:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper
    )
    try:
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
    except ImportError:
        # Fallback if v1 utils not available
        def get_kv_cache_layout():
            return "NHD"  # Default layout
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    BatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None
    def get_kv_cache_layout():
        return "NHD"

from vllm.forward_context import get_forward_context

logger = logging.getLogger(__name__)

FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024


class SwiftKVFlashInferAttention(nn.Module):
    """
    Direct FlashInfer attention for SwiftKV - bypasses vLLM's attention backend.
    """
    
    def __init__(self, num_heads: int, num_kv_heads: int, head_size: int, 
                 page_size: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError("FlashInfer not available. Install with: pip install flashinfer")
            
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads  
        self.head_size = head_size
        self.page_size = page_size
        self.dtype = dtype
        
        # Create workspace buffer
        self.workspace_buffer = torch.empty(
            FLASHINFER_WORKSPACE_BUFFER_SIZE, 
            dtype=torch.uint8, 
            device="cuda"
        )
        
        # Wrappers for different scenarios
        self.decode_wrapper = None
        self.prefill_wrapper = None
        
        logger.info(f"SwiftKVFlashInferAttention initialized: "
                   f"heads={num_heads}, kv_heads={num_kv_heads}, head_size={head_size}")
    
    def _get_decode_wrapper(self):
        """Get or create decode wrapper."""
        if self.decode_wrapper is None:
            # Determine if we should use tensor cores
            use_tensor_cores = (self.num_heads // self.num_kv_heads > 4)
            
            self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                get_kv_cache_layout(),
                use_tensor_cores=use_tensor_cores
            )
        return self.decode_wrapper
    
    def _get_prefill_wrapper(self):
        """Get or create prefill wrapper."""
        if self.prefill_wrapper is None:
            self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                get_kv_cache_layout()
            )
        return self.prefill_wrapper
    
    def _extract_flashinfer_data(self, attn_metadata, selected_indices):
        """Extract data needed for FlashInfer from vLLM metadata."""
        
        # Basic token info
        num_selected_tokens = selected_indices.numel()
        
        # Extract KV cache info - this depends on your vLLM version
        # You may need to adjust these based on your specific metadata structure
        if hasattr(attn_metadata, 'paged_kv_indptr'):
            # FlashInfer v1 style
            paged_kv_indptr = attn_metadata.paged_kv_indptr
            paged_kv_indices = attn_metadata.paged_kv_indices  
            paged_kv_last_page_len = attn_metadata.paged_kv_last_page_len
        elif hasattr(attn_metadata, 'block_table'):
            # FlashAttention style - convert to FlashInfer format
            block_table = attn_metadata.block_table
            seq_lens = attn_metadata.seq_lens
            
            # Convert block table to paged format
            batch_size = block_table.shape[0]
            paged_kv_indices = block_table.flatten()
            
            # Create indptr
            blocks_per_seq = (seq_lens + self.page_size - 1) // self.page_size
            paged_kv_indptr = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=block_table.device),
                blocks_per_seq.cumsum(dim=0, dtype=torch.int32)
            ])
            
            # Last page lengths
            paged_kv_last_page_len = seq_lens % self.page_size
            paged_kv_last_page_len = torch.where(
                paged_kv_last_page_len == 0, 
                self.page_size, 
                paged_kv_last_page_len
            )
        else:
            raise ValueError("Cannot extract KV cache info from attention metadata")
        
        return {
            'num_tokens': num_selected_tokens,
            'paged_kv_indptr': paged_kv_indptr,
            'paged_kv_indices': paged_kv_indices, 
            'paged_kv_last_page_len': paged_kv_last_page_len
        }
    
    def forward_decode(self, query: torch.Tensor, kv_cache: torch.Tensor, 
                      attn_metadata, selected_indices: torch.Tensor,
                      scale: float = 1.0) -> torch.Tensor:
        """
        Forward pass for decode scenario (typical SwiftKV usage).
        
        Args:
            query: [num_selected_tokens, num_heads, head_size]  
            kv_cache: KV cache tensor
            attn_metadata: vLLM attention metadata
            selected_indices: Token indices selected by SwiftKV
            scale: Attention scale factor
            
        Returns:
            attention_output: [num_selected_tokens, num_heads, head_size]
        """
        
        # Extract FlashInfer data
        fi_data = self._extract_flashinfer_data(attn_metadata, selected_indices)
        
        # Get decode wrapper
        wrapper = self._get_decode_wrapper()
        
        # Plan the attention
        wrapper.plan(
            fi_data['paged_kv_indptr'],
            fi_data['paged_kv_indices'], 
            fi_data['paged_kv_last_page_len'],
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.page_size,
            pos_encoding_mode="NONE",  # vLLM handles RoPE
            sm_scale=scale,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
        )
        
        # Prepare output tensor
        output = torch.empty_like(query)
        
        # Run FlashInfer attention
        wrapper.run(
            query,
            kv_cache,
            out=output
        )
        
        logger.debug(f"FlashInfer decode: {fi_data['num_tokens']} tokens processed")
        return output
    
    def forward_prefill(self, query: torch.Tensor, key: torch.Tensor, 
                       value: torch.Tensor, kv_cache: torch.Tensor,
                       attn_metadata, scale: float = 1.0) -> torch.Tensor:
        """
        Forward pass for prefill scenario.
        
        Args:
            query: [num_tokens, num_heads, head_size]
            key: [num_tokens, num_kv_heads, head_size]  
            value: [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tensor
            attn_metadata: vLLM attention metadata
            scale: Attention scale factor
            
        Returns:
            attention_output: [num_tokens, num_heads, head_size]
        """
        
        # Extract data for prefill
        num_tokens = query.shape[0]
        
        # Create simple metadata for prefill (all tokens in one sequence)
        qo_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device=query.device)
        
        # For prefill, we typically have one sequence
        # You may need to adjust this based on your specific use case
        estimated_blocks = (num_tokens + self.page_size - 1) // self.page_size
        paged_kv_indptr = torch.tensor([0, estimated_blocks], dtype=torch.int32, device=query.device)
        paged_kv_indices = torch.arange(estimated_blocks, dtype=torch.int32, device=query.device)
        paged_kv_last_page_len = torch.tensor([num_tokens % self.page_size or self.page_size], 
                                            dtype=torch.int32, device=query.device)
        
        # Get prefill wrapper
        wrapper = self._get_prefill_wrapper()
        
        # Plan the attention
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            self.num_heads,
            self.num_kv_heads, 
            self.head_size,
            self.page_size,
            causal=True,
            sm_scale=scale,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
        )
        
        # Prepare output tensor
        output = torch.empty_like(query)
        
        # Run FlashInfer attention
        wrapper.run(
            query,
            kv_cache,
            out=output
        )
        
        logger.debug(f"FlashInfer prefill: {num_tokens} tokens processed")
        return output
    
    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None,
               value: Optional[torch.Tensor] = None, kv_cache: Optional[torch.Tensor] = None,
               attn_metadata=None, selected_indices: Optional[torch.Tensor] = None,
               scale: float = 1.0, is_prefill: bool = False) -> torch.Tensor:
        """
        Main forward pass - automatically chooses prefill or decode.
        
        Args:
            query: Query tensor
            key: Key tensor (for prefill)
            value: Value tensor (for prefill) 
            kv_cache: KV cache tensor
            attn_metadata: vLLM attention metadata
            selected_indices: SwiftKV selected token indices
            scale: Attention scale
            is_prefill: Whether this is prefill or decode
        """
        
        if is_prefill and key is not None and value is not None:
            return self.forward_prefill(query, key, value, kv_cache, attn_metadata, scale)
        elif selected_indices is not None:
            return self.forward_decode(query, kv_cache, attn_metadata, selected_indices, scale)
        else:
            raise ValueError("Must provide either (key, value) for prefill or selected_indices for decode")


def create_swiftkv_flashinfer_attention(vllm_config, layer_idx: int) -> SwiftKVFlashInferAttention:
    """
    Factory function to create SwiftKV FlashInfer attention from vLLM config.
    """
    model_config = vllm_config.model_config.hf_config
    cache_config = vllm_config.cache_config
    
    return SwiftKVFlashInferAttention(
        num_heads=model_config.num_attention_heads,
        num_kv_heads=getattr(model_config, 'num_key_value_heads', model_config.num_attention_heads),
        head_size=model_config.hidden_size // model_config.num_attention_heads,
        page_size=cache_config.block_size,
        dtype=torch.float16  # You may want to get this from config
    ) 