#!/usr/bin/env python3
"""Test script to validate JSON-based prompt loader"""

from src.wraval.actions.prompt_loader import get_prompt, Tone, get_all_tones

def test_default_prompts():
    """Test loading from default prompt_tones.json"""
    print("Testing default prompts (actions/prompt_tones.json)...")
    
    # Test getting a prompt
    witty_prompt = get_prompt(Tone.WITTY)
    messages = witty_prompt.get_messages()
    
    print(f"\n✓ Loaded {Tone.WITTY.value} prompt")
    print(f"  System prompt length: {len(messages[0]['content'])} chars")
    print(f"  Number of examples: {len(witty_prompt.examples)}")
    print(f"  Total messages: {len(messages)}")
    
    # Test all tones
    all_tones = get_all_tones()
    print(f"\n✓ Available tones: {', '.join(all_tones)}")
    
    for tone in Tone:
        try:
            prompt = get_prompt(tone)
            msgs = prompt.get_messages()
            print(f"  ✓ {tone.value}: {len(msgs)} messages")
        except Exception as e:
            print(f"  ✗ {tone.value}: {e}")

def test_custom_prompts():
    """Test loading from custom_prompts/en_us.json"""
    print("\n\nTesting custom prompts (custom_prompts/en_us.json)...")
    
    from src.wraval.custom_prompts.prompt_loader import get_prompt as get_custom_prompt
    from src.wraval.custom_prompts.prompt_loader import Tone as CustomTone
    
    # Test getting a prompt
    professional_prompt = get_custom_prompt(CustomTone.PROFESSIONAL)
    messages = professional_prompt.get_messages()
    
    print(f"\n✓ Loaded {CustomTone.PROFESSIONAL.value} prompt")
    print(f"  System prompt length: {len(messages[0]['content'])} chars")
    print(f"  Number of examples: {len(professional_prompt.examples)}")
    
    # Test all tones
    for tone in CustomTone:
        try:
            prompt = get_custom_prompt(tone)
            msgs = prompt.get_messages()
            print(f"  ✓ {tone.value}: {len(msgs)} messages")
        except Exception as e:
            print(f"  ✗ {tone.value}: {e}")

def test_message_structure():
    """Test that message structure matches expected format"""
    print("\n\nTesting message structure...")
    
    prompt = get_prompt(Tone.ELABORATE)
    messages = prompt.get_messages()
    
    # Should have system + (user + assistant) * num_examples
    expected_count = 1 + (2 * len(prompt.examples))
    assert len(messages) == expected_count, f"Expected {expected_count} messages, got {len(messages)}"
    
    # First message should be system
    assert messages[0]["role"] == "system", "First message should be system"
    
    # Alternating user/assistant
    for i in range(1, len(messages), 2):
        assert messages[i]["role"] == "user", f"Message {i} should be user"
        if i + 1 < len(messages):
            assert messages[i + 1]["role"] == "assistant", f"Message {i+1} should be assistant"
    
    print("✓ Message structure is correct")
    print(f"  System message: {messages[0]['content'][:80]}...")
    if len(messages) > 1:
        print(f"  First example user: {messages[1]['content'][:60]}...")
        print(f"  First example assistant: {messages[2]['content'][:60]}...")

def test_system_prompt_composition():
    """Test that system prompts are composed correctly"""
    print("\n\nTesting system prompt composition...")
    
    # Test with tone that has a prompt
    witty = get_prompt(Tone.WITTY)
    witty_msgs = witty.get_messages()
    
    # Should contain both master and tone-specific prompt
    assert "writing assistant" in witty_msgs[0]["content"].lower()
    assert "witty" in witty_msgs[0]["content"].lower()
    
    print("✓ System prompt composition is correct")
    print(f"  Contains master prompt: ✓")
    print(f"  Contains tone-specific prompt: ✓")

if __name__ == "__main__":
    try:
        test_default_prompts()
        test_custom_prompts()
        test_message_structure()
        test_system_prompt_composition()
        print("\n\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
