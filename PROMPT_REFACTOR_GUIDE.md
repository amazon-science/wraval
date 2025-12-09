# Prompt System Refactoring Guide

## Overview

The prompt system has been refactored from Python classes to JSON-based configuration with `SimpleNamespace` for clean object access.

## What Changed

### Before (Python-based)
```python
# src/wraval/actions/prompt_tones.py
class WittyPrompt(Prompt):
    def __init__(self):
        super().__init__(
            sys_prompt="Make this text witty.",
            examples=[...]
        )
```

### After (JSON-based)
```json
// src/wraval/actions/prompt_tones.json
{
  "system": "You are now a writing assistant...",
  "witty": {
    "prompt": "Make this text witty.",
    "examples": [...]
  }
}
```

## File Structure

```
src/wraval/
├── actions/
│   ├── prompt_loader.py          # NEW: JSON-based loader
│   ├── prompt_tones.json         # NEW: Default prompts (English)
│   └── prompt_tones.py           # DEPRECATED: Keep for reference
└── custom_prompts/
    ├── prompt_loader.py          # NEW: Custom prompts loader
    ├── en_us.json                # EXISTING: Custom prompts
    └── prompt_tones.py           # DEPRECATED: Keep for reference
```

## Usage (No Changes Required!)

The interface remains identical:

```python
from wraval.actions.prompt_loader import get_prompt, Tone

# Get a prompt (same as before)
prompt = get_prompt(Tone.WITTY)

# Get messages (same as before)
messages = prompt.get_messages()
# [
#   {"role": "system", "content": "..."},
#   {"role": "user", "content": "..."},
#   {"role": "assistant", "content": "..."}
# ]
```

## JSON Schema

Your existing schema is preserved:

```json
{
  "system": "Master system prompt for all tones",
  "tone_name": {
    "prompt_id": "identifier",
    "prompt": "Tone-specific instruction",
    "examples": [
      {
        "user": "Input text",
        "assistant": "Output text"
      }
    ],
    "lora_model_id": "optional_model_id"
  },
  "default": "improve"
}
```

## Benefits

1. **Easy Editing**: Non-developers can modify prompts in JSON
2. **Version Control**: Clear diffs for prompt changes
3. **Multi-Language**: Easy to add new language files
4. **Type Safety**: `Tone` enum provides type checking
5. **Clean Access**: SimpleNamespace enables dot notation
6. **Backward Compatible**: Existing code works unchanged

## Adding New Tones

### 1. Add to JSON
```json
{
  "new_tone": {
    "prompt": "Your instruction here",
    "examples": [...]
  }
}
```

### 2. Add to Enum
```python
# src/wraval/actions/prompt_loader.py
class Tone(Enum):
    NEW_TONE = "new_tone"
```

### 3. Use It
```python
prompt = get_prompt(Tone.NEW_TONE)
```

## Adding New Languages

Create a new JSON file with the same structure:

```
src/wraval/custom_prompts/
├── en_us.json    # English
├── fr_fr.json    # French
└── es_es.json    # Spanish
```

Load with custom path:
```python
from wraval.custom_prompts.prompt_loader import get_prompt, Tone

# Load French prompts
prompt = get_prompt(Tone.WITTY, json_path="src/wraval/custom_prompts/fr_fr.json")
```

## Migration Checklist

- [x] Create `prompt_loader.py` modules
- [x] Create JSON files with existing prompts
- [x] Maintain `Tone` enum for type safety
- [x] Keep `Prompt` class interface unchanged
- [x] Preserve `get_prompt()` function signature
- [x] Support custom prompts via `--custom-prompts` flag
- [ ] Test with existing workflows
- [ ] Update documentation
- [ ] Deprecate Python prompt classes

## Testing

Run the test script to validate:

```bash
python test_prompt_loader.py
```

Expected output:
```
Testing default prompts...
✓ Loaded witty prompt
✓ Available tones: emojify, professional, shorten, ...

Testing custom prompts...
✓ Loaded professional prompt
...

🎉 All tests passed!
```

## Rollback Plan

If issues arise, the old Python-based system is still available:

```python
# Temporarily use old system
from wraval.actions.prompt_tones import get_prompt, Tone  # Old import
```

## Notes

- The `summarize` tone was removed (same as `shorten`)
- JSON files use your existing schema (no changes needed)
- `SimpleNamespace` provides clean object access internally
- Caching prevents repeated file reads
