from enum import Enum
from types import SimpleNamespace
import json
from pathlib import Path
from typing import Optional

class Tone(Enum):
    EMOJIFY = "emojify"
    PROFESSIONAL = "professional"
    SHORTEN = "shorten"
    WITTY = "witty"
    CASUAL = "casual"
    ELABORATE = "elaborate"
    PROOFREAD = "proofread"
    IMPROVE = "improve"
    KEYPOINTS = "keypoints"

class Prompt:
    """Maintains same interface as current Prompt class"""
    def __init__(self, sys_prompt: str, master_prompt: str, examples=None):
        self.sys_prompt = (
            master_prompt if not sys_prompt 
            else f"{master_prompt}\n\n{sys_prompt}"
        )
        self.examples = examples or []

    def get_messages(self):
        messages = [{"role": "system", "content": self.sys_prompt}]
        for example in self.examples:
            messages.extend([
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}
            ])
        return messages

class PromptConfig:
    """Loads prompts from JSON using SimpleNamespace"""
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self._config = None
        self.commit_hash = None
        self._load()
    
    def _load(self):
        with open(self.json_path) as f:
            self._config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        
        # Extract commit hash from metadata if available
        if hasattr(self._config, '_metadata') and hasattr(self._config._metadata, 'commit_hash'):
            self.commit_hash = self._config._metadata.commit_hash
    
    def get_prompt(self, tone: Tone) -> Prompt:
        """Factory method matching current interface"""
        tone_name = tone.value
        
        # Access tone config from JSON structure
        if not hasattr(self._config, tone_name):
            raise ValueError(f"Unknown tone: {tone}")
        
        tone_config = getattr(self._config, tone_name)
        
        # Convert SimpleNamespace examples back to dicts
        examples = []
        if hasattr(tone_config, 'examples'):
            for ex in tone_config.examples:
                examples.append({
                    "user": ex.user,
                    "assistant": ex.assistant
                })
        
        # Get system prompt from JSON (uses "system" key in your schema)
        master_prompt = self._config.system if hasattr(self._config, 'system') else ""
        tone_prompt = tone_config.prompt if hasattr(tone_config, 'prompt') else ""
        
        return Prompt(
            sys_prompt=tone_prompt,
            master_prompt=master_prompt,
            examples=examples
        )
    
    def get_all_tones(self):
        return [tone.value.lower() for tone in Tone]

# Module-level factory for backward compatibility
_prompt_config: Optional[PromptConfig] = None

def get_prompt(tone: Tone, json_path: str = None, language: str = None, custom_prompts: bool = False) -> Prompt:
    """Load prompt from JSON file
    
    Args:
        tone: The tone to get prompt for
        json_path: Optional explicit path to JSON file
        language: Optional language code (e.g., 'de', 'en_us', 'ja')
        custom_prompts: If True, load from custom_prompts folder, else from actions folder
    """
    global _prompt_config
    
    if json_path:
        # Allow explicit path
        config = PromptConfig(json_path)
    else:
        # Determine base directory
        if custom_prompts:
            base_dir = Path(__file__).parent.parent / "custom_prompts"
        else:
            base_dir = Path(__file__).parent
        
        # Determine which JSON file to load based on language
        if language and custom_prompts:
            # Map language codes to JSON filenames (only for custom prompts)
            lang_map = {
                'de': 'de.json',
                'en': 'en_us.json',
                'en_us': 'en_us.json',
                'en_uk': 'en_uk.json',
                'ja': 'ja.json',
                'jp': 'ja.json',  # alias
            }
            json_file = lang_map.get(language.lower(), 'en_us.json')
            default_path = base_dir / json_file
        else:
            # Use default: prompt_tones.json for actions, en_us.json for custom
            json_file = "en_us.json" if custom_prompts else "prompt_tones.json"
            default_path = base_dir / json_file
        
        # Cache per path
        if _prompt_config is None or getattr(_prompt_config, 'json_path', None) != default_path:
            _prompt_config = PromptConfig(default_path)
        config = _prompt_config
    
    return config.get_prompt(tone)

def get_commit_hash(json_path: str = None, language: str = None, custom_prompts: bool = False) -> Optional[str]:
    """Get commit hash from prompt JSON metadata"""
    global _prompt_config
    
    if json_path:
        config = PromptConfig(json_path)
    else:
        # Determine base directory
        if custom_prompts:
            base_dir = Path(__file__).parent.parent / "custom_prompts"
        else:
            base_dir = Path(__file__).parent
        
        # Determine which JSON file to load based on language
        if language and custom_prompts:
            lang_map = {
                'de': 'de.json',
                'en': 'en_us.json',
                'en_us': 'en_us.json',
                'en_uk': 'en_uk.json',
                'ja': 'ja.json',
                'jp': 'ja.json',
            }
            json_file = lang_map.get(language.lower(), 'en_us.json')
            default_path = base_dir / json_file
        else:
            json_file = "en_us.json" if custom_prompts else "prompt_tones.json"
            default_path = base_dir / json_file
        
        if _prompt_config is None or getattr(_prompt_config, 'json_path', None) != default_path:
            _prompt_config = PromptConfig(default_path)
        config = _prompt_config
    
    return config.commit_hash

def get_all_tones():
    return [tone.value.lower() for tone in Tone]
