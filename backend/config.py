"""Configuration management for Z-Image WebUI."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = "sqlite+aiosqlite:///./zimage.db"


class JWTConfig(BaseModel):
    """JWT authentication configuration."""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440


class ModelsConfig(BaseModel):
    """AI models path configuration."""
    base_model_path: str = "models/Diffusion_Transformer/Z-Image-Turbo/"
    controlnet_path: str = "models/Personalized_Model/"
    lora_path: str = "models/Lora/"
    annotator_path: str = "models/Annotators/"
    outputs_path: str = "outputs/"


class OptimizationConfig(BaseModel):
    """GPU optimization configuration."""
    gpu_memory_mode: str = "model_cpu_offload_and_qfloat8"
    weight_dtype: str = "bfloat16"
    attention_type: str = "FLASH_ATTENTION"
    compile_dit: bool = False


class GenerationDefaultsConfig(BaseModel):
    """Default generation parameters."""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25
    guidance_scale: float = 0.0
    control_context_scale: float = 0.75
    sampler: str = "Flow"


class LLMProviderConfig(BaseModel):
    """LLM provider configuration."""
    api_key: str = ""
    model: str = ""
    base_url: Optional[str] = None


class SystemPromptsConfig(BaseModel):
    """System prompts for LLM."""
    default: str = ""
    custom: str = ""


class LLMConfig(BaseModel):
    """LLM configuration."""
    default_provider: str = "openai"
    providers: Dict[str, LLMProviderConfig] = {}
    system_prompts: Dict[str, SystemPromptsConfig] = {}


class AppConfig(BaseModel):
    """Main application configuration."""
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig = DatabaseConfig()
    jwt: JWTConfig = JWTConfig()
    models: ModelsConfig = ModelsConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    generation_defaults: GenerationDefaultsConfig = GenerationDefaultsConfig()
    llm: LLMConfig = LLMConfig()


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        # Parse LLM providers
        if "llm" in config_data and "providers" in config_data["llm"]:
            providers = {}
            for name, provider_data in config_data["llm"]["providers"].items():
                providers[name] = LLMProviderConfig(**provider_data)
            config_data["llm"]["providers"] = providers
        
        # Parse system prompts
        if "llm" in config_data and "system_prompts" in config_data["llm"]:
            prompts = {}
            for name, prompt_data in config_data["llm"]["system_prompts"].items():
                prompts[name] = SystemPromptsConfig(**prompt_data)
            config_data["llm"]["system_prompts"] = prompts
        
        return AppConfig(**config_data)
    
    return AppConfig()


def save_config(config: AppConfig, config_path: str = "config.yaml") -> None:
    """Save configuration to YAML file."""
    config_file = Path(config_path)
    
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, allow_unicode=True)


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> AppConfig:
    """Reload configuration from file."""
    global _config
    _config = load_config()
    return _config
