"""
LoRA (Low-Rank Adaptation) for efficient fine-tuning of transformer models.

LoRA freezes pre-trained weights and injects trainable low-rank decomposition
matrices into transformer layers, dramatically reducing trainable parameters
while maintaining performance.

Reference: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""

import torch
import torch.nn as nn
from typing import Optional, List
import math


class LoRALayer(nn.Module):
    """
    LoRA adapter layer that wraps a linear layer.
    
    Instead of fine-tuning W ∈ R^(d×k), we freeze W and learn:
        ΔW = BA, where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
    
    Forward pass: h = Wx + (BA)x = Wx + ΔWx
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of low-rank decomposition (default: 8)
        alpha: Scaling factor (default: 16)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.
        
        Args:
            x: Input tensor. Shape: [..., in_features]
        
        Returns:
            LoRA output. Shape: [..., out_features]
        """
        # x @ A^T @ B^T = (BA)x
        result = x @ self.lora_A.T
        result = self.dropout(result)
        result = result @ self.lora_B.T
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with optional LoRA adapter.
    
    Wraps nn.Linear and adds LoRA adaptation when enabled.
    Exposes weight and bias properties for compatibility with PyTorch modules
    that expect these attributes (e.g., MultiheadAttention).
    
    Args:
        linear: Original linear layer (will be frozen)
        rank: LoRA rank (default: 8)
        alpha: LoRA scaling (default: 16)
        dropout: LoRA dropout (default: 0.0)
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Store dimensions for compatibility
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    @property
    def weight(self):
        """Expose weight from underlying linear layer."""
        return self.linear.weight
    
    @property
    def bias(self):
        """Expose bias from underlying linear layer."""
        return self.linear.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original + LoRA adaptation."""
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: Model to apply LoRA to
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to all attention projection layers.
                       Examples: ['q_proj', 'v_proj'], ['.*proj'], ['encoder.*.linear']
        rank: LoRA rank
        alpha: LoRA scaling
        dropout: LoRA dropout
    
    Returns:
        Model with LoRA adapters applied
    
    Example:
        >>> from ehrsequencing.models.behrt import BEHRT, BEHRTConfig
        >>> config = BEHRTConfig.large(vocab_size=1000)
        >>> model = BEHRT(config)
        >>> # Apply LoRA to attention projections
        >>> model = apply_lora_to_model(model, target_modules=['q_proj', 'k_proj', 'v_proj'])
        >>> # Now only LoRA parameters are trainable
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        >>> print(f"Trainable parameters: {trainable:,}")
    """
    import re
    
    # Default: apply to attention projection layers
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    
    # Compile regex patterns
    patterns = [re.compile(pattern) for pattern in target_modules]
    
    def should_apply_lora(name: str) -> bool:
        """Check if module name matches any pattern."""
        return any(pattern.search(name) for pattern in patterns)
    
    # Find and replace linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            # Get parent module and attribute name
            *parent_path, attr_name = name.split('.')
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            
            # Replace with LoRA version
            lora_module = LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_module)
            
            print(f"Applied LoRA to {name} (in={module.in_features}, out={module.out_features}, rank={rank})")
    
    return model


def apply_lora_to_behrt(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    lora_attention: bool = True,
    lora_feedforward: bool = False,
    freeze_base: bool = True,
    train_embeddings: bool = True,
    train_head: bool = True
) -> nn.Module:
    """
    Apply LoRA to BEHRT model with sensible defaults.
    
    Args:
        model: BEHRT model (can be BEHRT, BEHRTForMLM, etc.)
        rank: LoRA rank (default: 8)
        alpha: LoRA scaling (default: 16)
        dropout: LoRA dropout (default: 0.0)
        lora_attention: Apply LoRA to attention layers (default: True)
        lora_feedforward: Apply LoRA to feedforward layers (default: False)
        freeze_base: Freeze all non-LoRA parameters (default: True)
        train_embeddings: Keep embedding layers trainable (default: True)
            IMPORTANT: Set to True when training from scratch, False when
            fine-tuning a pre-trained model.
        train_head: Keep task head (MLM, classification) trainable (default: True)
    
    Returns:
        Model with LoRA adapters
    
    Example:
        >>> from ehrsequencing.models.behrt import BEHRT, BEHRTConfig
        >>> config = BEHRTConfig.large(vocab_size=1000)
        >>> model = BEHRT(config)
        >>> 
        >>> # Apply LoRA for efficient fine-tuning
        >>> model = apply_lora_to_behrt(model, rank=8, lora_attention=True)
        >>> 
        >>> # Count parameters
        >>> total = sum(p.numel() for p in model.parameters())
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    """
    # First, freeze ALL parameters if requested
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    
    target_modules = []
    
    if lora_attention:
        # Apply to all attention projections in transformer encoder
        target_modules.extend([
            'encoder.*.self_attn.in_proj_weight',  # Q, K, V projections (combined)
            'encoder.*.self_attn.out_proj',         # Output projection
        ])
    
    if lora_feedforward:
        # Apply to feedforward layers
        target_modules.extend([
            'encoder.*.linear1',  # First FFN layer
            'encoder.*.linear2',  # Second FFN layer
        ])
    
    # PyTorch's TransformerEncoderLayer uses different naming
    # We need to target the actual linear layers
    patterns = []
    if lora_attention:
        patterns.extend([
            '.*self_attn.*',  # All attention layers
        ])
    if lora_feedforward:
        patterns.extend([
            '.*linear1',  # First FFN layer
            '.*linear2',  # Second FFN layer
        ])
    
    # Apply LoRA to target modules
    model = apply_lora_to_model(model, target_modules=patterns, rank=rank, alpha=alpha, dropout=dropout)
    
    # Ensure LoRA parameters are trainable
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    
    # Unfreeze embeddings if requested (critical for training from scratch)
    if train_embeddings:
        for name, module in model.named_modules():
            if 'embedding' in name.lower():
                for param in module.parameters():
                    param.requires_grad = True
        # Also handle layer norms in embedding layer
        for name, param in model.named_parameters():
            if 'embeddings' in name or 'embedding' in name:
                param.requires_grad = True
    
    # Unfreeze task head if requested (MLM head, classifier, etc.)
    if train_head:
        for name, module in model.named_modules():
            if any(h in name.lower() for h in ['mlm_head', 'nvp_head', 'classifier', 'head']):
                for param in module.parameters():
                    param.requires_grad = True
        # Also handle by parameter name
        for name, param in model.named_parameters():
            if any(h in name.lower() for h in ['mlm_head', 'nvp_head', 'classifier', 'head']):
                param.requires_grad = True
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.
    
    Args:
        model: Model with LoRA adapters
    
    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    # Count LoRA parameters specifically
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    
    # Count embedding parameters
    embedding_params = 0
    embedding_trainable = 0
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            embedding_params += param.numel()
            if param.requires_grad:
                embedding_trainable += param.numel()
    
    # Count head parameters (MLM, classifier, etc.)
    head_params = 0
    head_trainable = 0
    for name, param in model.named_parameters():
        if any(h in name.lower() for h in ['mlm_head', 'nvp_head', 'classifier', 'head']):
            head_params += param.numel()
            if param.requires_grad:
                head_trainable += param.numel()
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'lora': lora_params,
        'trainable_percent': 100 * trainable / total if total > 0 else 0,
        'lora_percent': 100 * lora_params / total if total > 0 else 0,
        'embedding_total': embedding_params,
        'embedding_trainable': embedding_trainable,
        'head_total': head_params,
        'head_trainable': head_trainable
    }


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only LoRA weights (not full model).
    
    This is much more efficient than saving the full model, especially
    for large pre-trained models.
    
    Args:
        model: Model with LoRA adapters
        path: Path to save LoRA weights
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    
    torch.save(lora_state_dict, path)
    print(f"Saved LoRA weights to {path}")
    print(f"LoRA parameters: {sum(p.numel() for p in lora_state_dict.values()):,}")


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA weights into a model.
    
    Args:
        model: Model with LoRA adapters
        path: Path to LoRA weights
    """
    lora_state_dict = torch.load(path)
    
    # Load weights into LoRA layers
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            if f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in lora_state_dict:
                module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
    
    print(f"Loaded LoRA weights from {path}")
