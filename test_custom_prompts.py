#!/usr/bin/env python3
"""Quick test to verify custom prompts are loading correctly"""

from src.wraval.custom_prompts.prompt_loader import get_prompt, Tone

# Test witty prompt
witty_prompt = get_prompt(Tone.WITTY)
messages = witty_prompt.get_messages()

print(f"Witty prompt has {len(witty_prompt.examples)} examples")
print(f"Total messages: {len(messages)}")
print(f"\nSystem prompt:\n{messages[0]['content'][:200]}...\n")

print("Examples:")
for i, example in enumerate(witty_prompt.examples, 1):
    print(f"\n{i}. User: {example['user']}")
    print(f"   Assistant: {example['assistant'][:80]}...")

print(f"\n\nAll messages:")
for i, msg in enumerate(messages):
    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
    print(f"{i}. {msg['role']}: {content}")
