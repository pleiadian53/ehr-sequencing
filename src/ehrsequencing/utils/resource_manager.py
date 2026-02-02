"""
Resource Manager for automatic platform detection and parameter optimization.

Automatically detects:
- GPU availability and VRAM capacity
- CPU cores and RAM
- Platform type (local laptop, cloud GPU, HPC)

Recommends optimal training parameters based on available resources.
"""

import torch
import psutil
import platform
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PlatformType(Enum):
    """Platform categories for resource management."""
    LOCAL_CPU = "local_cpu"           # No GPU, CPU only
    LOCAL_LAPTOP = "local_laptop"     # Laptop GPU (8-16GB VRAM)
    LOCAL_WORKSTATION = "local_workstation"  # Desktop GPU (16-24GB VRAM)
    CLOUD_A40 = "cloud_a40"           # A40 GPU (48GB VRAM)
    CLOUD_A100 = "cloud_a100"         # A100 GPU (40-80GB VRAM)
    CLOUD_V100 = "cloud_v100"         # V100 GPU (16-32GB VRAM)
    CLOUD_T4 = "cloud_t4"             # T4 GPU (16GB VRAM)
    CLOUD_GENERIC = "cloud_generic"   # Other cloud GPU


@dataclass
class SystemResources:
    """Container for detected system resources."""
    platform_type: PlatformType
    device: str  # 'cuda', 'mps', 'cpu'
    gpu_name: Optional[str]
    vram_gb: Optional[float]
    ram_gb: float
    cpu_cores: int
    is_cloud: bool
    
    def __str__(self):
        if self.device == 'cpu':
            return f"CPU-only: {self.cpu_cores} cores, {self.ram_gb:.1f}GB RAM"
        else:
            return f"{self.gpu_name}: {self.vram_gb:.1f}GB VRAM, {self.ram_gb:.1f}GB RAM, {self.cpu_cores} cores"


@dataclass
class TrainingConfig:
    """Recommended training configuration based on resources."""
    model_size: str
    batch_size: int
    num_patients: int
    epochs: int
    use_lora: bool
    lora_rank: int
    gradient_accumulation_steps: int
    num_workers: int
    mixed_precision: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy parameter passing."""
        return {
            'model_size': self.model_size,
            'batch_size': self.batch_size,
            'num_patients': self.num_patients,
            'epochs': self.epochs,
            'use_lora': self.use_lora,
            'lora_rank': self.lora_rank,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'num_workers': self.num_workers,
            'mixed_precision': self.mixed_precision,
        }


class ResourceManager:
    """
    Automatic resource detection and training parameter optimization.
    
    Usage:
        manager = ResourceManager()
        resources = manager.detect_resources()
        config = manager.recommend_config(resources, task='demo')
        
        print(f"Detected: {resources}")
        print(f"Recommended: batch_size={config.batch_size}, model_size={config.model_size}")
    """
    
    def __init__(self):
        self.resources = None
    
    def detect_resources(self) -> SystemResources:
        """
        Detect available system resources.
        
        Returns:
            SystemResources with detected platform, GPU, RAM, etc.
        """
        # Detect device
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif torch.backends.mps.is_available():
            device = 'mps'
            gpu_name = 'Apple Silicon (MPS)'
            # MPS doesn't expose VRAM directly, estimate based on system
            vram_gb = self._estimate_mps_vram()
        else:
            device = 'cpu'
            gpu_name = None
            vram_gb = None
        
        # Detect RAM and CPU
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        
        # Classify platform type
        platform_type, is_cloud = self._classify_platform(device, gpu_name, vram_gb, ram_gb)
        
        self.resources = SystemResources(
            platform_type=platform_type,
            device=device,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            is_cloud=is_cloud
        )
        
        return self.resources
    
    def _estimate_mps_vram(self) -> float:
        """Estimate MPS VRAM based on system RAM (rough heuristic)."""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        # Apple Silicon typically shares memory, estimate ~60% available for GPU
        return ram_gb * 0.6
    
    def _classify_platform(
        self,
        device: str,
        gpu_name: Optional[str],
        vram_gb: Optional[float],
        ram_gb: float
    ) -> Tuple[PlatformType, bool]:
        """
        Classify platform type based on detected resources.
        
        Returns:
            (PlatformType, is_cloud)
        """
        if device == 'cpu':
            return PlatformType.LOCAL_CPU, False
        
        if device == 'mps':
            # Apple Silicon - always local
            return PlatformType.LOCAL_LAPTOP, False
        
        # CUDA GPU - classify by name and VRAM
        gpu_lower = gpu_name.lower()
        
        # Cloud GPUs (common patterns)
        if 'a40' in gpu_lower:
            return PlatformType.CLOUD_A40, True
        elif 'a100' in gpu_lower:
            return PlatformType.CLOUD_A100, True
        elif 'v100' in gpu_lower:
            return PlatformType.CLOUD_V100, True
        elif 't4' in gpu_lower:
            return PlatformType.CLOUD_T4, True
        
        # Local GPUs (common patterns)
        if any(x in gpu_lower for x in ['rtx', 'gtx', 'geforce', 'titan']):
            if vram_gb >= 20:
                return PlatformType.LOCAL_WORKSTATION, False
            else:
                return PlatformType.LOCAL_LAPTOP, False
        
        # Heuristic: High VRAM + high RAM = likely cloud
        if vram_gb >= 40 and ram_gb >= 100:
            return PlatformType.CLOUD_GENERIC, True
        elif vram_gb >= 20:
            return PlatformType.LOCAL_WORKSTATION, False
        else:
            return PlatformType.LOCAL_LAPTOP, False
    
    def recommend_config(
        self,
        resources: Optional[SystemResources] = None,
        task: str = 'demo',
        model_size_override: Optional[str] = None
    ) -> TrainingConfig:
        """
        Recommend training configuration based on resources.
        
        Args:
            resources: SystemResources (if None, auto-detect)
            task: Task type ('demo', 'realistic', 'benchmark')
            model_size_override: Force specific model size
        
        Returns:
            TrainingConfig with recommended parameters
        """
        if resources is None:
            resources = self.detect_resources()
        
        # Get base config for platform
        config = self._get_platform_config(resources.platform_type)
        
        # Adjust for task type
        config = self._adjust_for_task(config, task)
        
        # Override model size if specified
        if model_size_override:
            config.model_size = model_size_override
            # Adjust other params based on model size
            config = self._adjust_for_model_size(config, model_size_override)
        
        return config
    
    def _get_platform_config(self, platform_type: PlatformType) -> TrainingConfig:
        """Get base configuration for platform type."""
        
        configs = {
            PlatformType.LOCAL_CPU: TrainingConfig(
                model_size='small',
                batch_size=4,
                num_patients=100,
                epochs=10,
                use_lora=True,
                lora_rank=4,
                gradient_accumulation_steps=4,
                num_workers=2,
                mixed_precision=False
            ),
            PlatformType.LOCAL_LAPTOP: TrainingConfig(
                model_size='small',
                batch_size=16,
                num_patients=500,
                epochs=20,
                use_lora=True,
                lora_rank=8,
                gradient_accumulation_steps=2,
                num_workers=4,
                mixed_precision=True
            ),
            PlatformType.LOCAL_WORKSTATION: TrainingConfig(
                model_size='medium',
                batch_size=64,
                num_patients=2000,
                epochs=50,
                use_lora=True,
                lora_rank=16,
                gradient_accumulation_steps=1,
                num_workers=8,
                mixed_precision=True
            ),
            PlatformType.CLOUD_T4: TrainingConfig(
                model_size='medium',
                batch_size=64,
                num_patients=3000,
                epochs=75,
                use_lora=True,
                lora_rank=16,
                gradient_accumulation_steps=1,
                num_workers=4,
                mixed_precision=True
            ),
            PlatformType.CLOUD_V100: TrainingConfig(
                model_size='large',
                batch_size=96,
                num_patients=5000,
                epochs=100,
                use_lora=True,
                lora_rank=16,
                gradient_accumulation_steps=1,
                num_workers=8,
                mixed_precision=True
            ),
            PlatformType.CLOUD_A40: TrainingConfig(
                model_size='large',
                batch_size=128,
                num_patients=5000,
                epochs=100,
                use_lora=True,
                lora_rank=16,
                gradient_accumulation_steps=1,
                num_workers=8,
                mixed_precision=True
            ),
            PlatformType.CLOUD_A100: TrainingConfig(
                model_size='large',
                batch_size=256,
                num_patients=10000,
                epochs=100,
                use_lora=True,
                lora_rank=32,
                gradient_accumulation_steps=1,
                num_workers=16,
                mixed_precision=True
            ),
            PlatformType.CLOUD_GENERIC: TrainingConfig(
                model_size='large',
                batch_size=96,
                num_patients=5000,
                epochs=100,
                use_lora=True,
                lora_rank=16,
                gradient_accumulation_steps=1,
                num_workers=8,
                mixed_precision=True
            ),
        }
        
        return configs.get(platform_type, configs[PlatformType.LOCAL_LAPTOP])
    
    def _adjust_for_task(self, config: TrainingConfig, task: str) -> TrainingConfig:
        """Adjust config based on task type."""
        if task == 'demo':
            # Demo: prioritize speed and clear learning signal
            pass  # Use base config
        elif task == 'realistic':
            # Realistic: may need more epochs for convergence
            config.epochs = int(config.epochs * 1.5)
        elif task == 'benchmark':
            # Benchmark: ensure fair comparison, more epochs
            config.epochs = max(config.epochs, 100)
        
        return config
    
    def _adjust_for_model_size(self, config: TrainingConfig, model_size: str) -> TrainingConfig:
        """Adjust config when model size is overridden."""
        if model_size == 'small':
            config.lora_rank = min(config.lora_rank, 8)
            config.batch_size = int(config.batch_size * 1.5)  # Can fit more
        elif model_size == 'large':
            config.lora_rank = max(config.lora_rank, 16)
            config.batch_size = max(config.batch_size // 2, 16)  # May need less
        
        return config
    
    def print_recommendations(self, config: TrainingConfig, resources: Optional[SystemResources] = None):
        """Print resource detection and recommendations in a user-friendly format."""
        if resources is None:
            resources = self.resources or self.detect_resources()
        
        print("\n" + "="*80)
        print("🔍 Resource Manager - Auto-detected Configuration")
        print("="*80)
        
        print(f"\n📊 Detected Resources:")
        print(f"   Platform: {resources.platform_type.value}")
        print(f"   Device: {resources.device}")
        if resources.gpu_name:
            print(f"   GPU: {resources.gpu_name}")
            print(f"   VRAM: {resources.vram_gb:.1f} GB")
        print(f"   RAM: {resources.ram_gb:.1f} GB")
        print(f"   CPU Cores: {resources.cpu_cores}")
        print(f"   Cloud: {'Yes' if resources.is_cloud else 'No'}")
        
        print(f"\n⚙️  Recommended Configuration:")
        print(f"   Model Size: {config.model_size}")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Num Patients: {config.num_patients}")
        print(f"   Epochs: {config.epochs}")
        print(f"   LoRA: {'Enabled' if config.use_lora else 'Disabled'} (rank={config.lora_rank})")
        print(f"   Mixed Precision: {'Enabled' if config.mixed_precision else 'Disabled'}")
        print(f"   Gradient Accumulation: {config.gradient_accumulation_steps} steps")
        print(f"   Data Workers: {config.num_workers}")
        
        print(f"\n💡 Note: You can override any parameter via command-line arguments")
        print("="*80 + "\n")


def get_recommended_config(
    task: str = 'demo',
    model_size_override: Optional[str] = None,
    verbose: bool = True
) -> Tuple[TrainingConfig, SystemResources]:
    """
    Convenience function to get recommended config.
    
    Args:
        task: Task type ('demo', 'realistic', 'benchmark')
        model_size_override: Force specific model size
        verbose: Print recommendations
    
    Returns:
        (TrainingConfig, SystemResources)
    
    Example:
        config, resources = get_recommended_config(task='demo', verbose=True)
        # Use config.batch_size, config.num_patients, etc.
    """
    manager = ResourceManager()
    resources = manager.detect_resources()
    config = manager.recommend_config(resources, task=task, model_size_override=model_size_override)
    
    if verbose:
        manager.print_recommendations(config, resources)
    
    return config, resources
