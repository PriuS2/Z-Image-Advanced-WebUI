"""LLM service for translation and prompt enhancement."""
import asyncio
from typing import Optional, Literal
from abc import ABC, abstractmethod

from backend.config import get_config


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
    
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate response using OpenAI API."""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
    
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate response using Claude API."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Claude API error: {e}")


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
    
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate response using Gemini API."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt
            )
            response = await asyncio.to_thread(
                model.generate_content, prompt
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider implementation."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model
    
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate response using Ollama API."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["response"]
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


class LLMService:
    """Service for LLM-based translation and prompt enhancement."""
    
    def __init__(self):
        self.config = get_config()
        self._providers = {}
    
    def _get_provider(self, provider_name: Optional[str] = None) -> LLMProvider:
        """Get or create an LLM provider."""
        name = provider_name or self.config.llm.default_provider
        
        if name not in self._providers:
            provider_config = self.config.llm.providers.get(name)
            if provider_config is None:
                raise ValueError(f"Unknown provider: {name}")
            
            if name == "openai":
                self._providers[name] = OpenAIProvider(
                    api_key=provider_config.api_key,
                    model=provider_config.model
                )
            elif name == "claude":
                self._providers[name] = ClaudeProvider(
                    api_key=provider_config.api_key,
                    model=provider_config.model
                )
            elif name == "gemini":
                self._providers[name] = GeminiProvider(
                    api_key=provider_config.api_key,
                    model=provider_config.model
                )
            elif name == "ollama":
                self._providers[name] = OllamaProvider(
                    base_url=provider_config.base_url or "http://localhost:11434",
                    model=provider_config.model
                )
            else:
                raise ValueError(f"Unsupported provider: {name}")
        
        return self._providers[name]
    
    def _get_system_prompt(self, prompt_type: Literal["translate", "enhance"]) -> str:
        """Get the system prompt for a given type."""
        prompts = self.config.llm.system_prompts.get(prompt_type)
        if prompts is None:
            return ""
        
        # Use custom prompt if available, otherwise use default
        return prompts.custom if prompts.custom else prompts.default
    
    async def translate(
        self,
        text: str,
        provider: Optional[str] = None
    ) -> str:
        """Translate Korean text to English for image generation.
        
        Args:
            text: Korean text to translate.
            provider: Optional provider name. Uses default if not specified.
        
        Returns:
            Translated English text.
        """
        llm = self._get_provider(provider)
        system_prompt = self._get_system_prompt("translate")
        return await llm.generate(text, system_prompt)
    
    async def enhance(
        self,
        prompt: str,
        provider: Optional[str] = None
    ) -> str:
        """Enhance a prompt for better image generation results.
        
        Args:
            prompt: Prompt to enhance.
            provider: Optional provider name. Uses default if not specified.
        
        Returns:
            Enhanced prompt.
        """
        llm = self._get_provider(provider)
        system_prompt = self._get_system_prompt("enhance")
        return await llm.generate(prompt, system_prompt)


# Global instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
