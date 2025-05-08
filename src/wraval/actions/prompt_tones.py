#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from .format import format_prompt
from enum import Enum

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
    SUMMARIZE = "summarize"

def get_all_tones():
    return [tone.value.lower() for tone in Tone]

def get_prompt(tone: Tone):
    match tone:
        case Tone.EMOJIFY:
            return EmojifyPrompt()
        case Tone.PROFESSIONAL:
            return ProfessionalPrompt()
        case Tone.SHORTEN:
            return ShortenPrompt()
        case Tone.WITTY:
            return WittyPrompt()
        case Tone.CASUAL:
            return CasualPrompt()
        case Tone.ELABORATE:
            return ElaboratePrompt()
        case Tone.PROOFREAD:
            return ProofreadPrompt()
        case Tone.IMPROVE:
            return ImprovePrompt()
        case Tone.KEYPOINTS:
            return KeypointsPrompt()
        case Tone.SUMMARIZE:
            return SummarizePrompt()
        case _:
            raise ValueError(f"Unknown tone: {tone}")

master_sys_prompt = "You are now a writing assistant. You will help users write. Rewrite the following text for the given prompt. Just return the result and do not add comments. Do not assume anything about the user. Do not answer any questions about yourself, only rewrite the question. Do not repeat the input text in the output."

class Prompt:
    def __init__(self, sys_prompt, examples=None):
        self.sys_prompt = sys_prompt
        self.examples = examples or []

    def get_messages(self):
        messages = [{"role": "system", "content": self.sys_prompt}]
        for example in self.examples:
            messages.extend([
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}
            ])
        return messages

class EmojifyPrompt(Prompt):
    def __init__(self):
        super().__init__(
            sys_prompt="Add appropriate emojis to enhance this text. Don't replace all words with emojis.",
            examples=[{
                "user": "Getting ready for my vacation to the coast! Excited to relax and enjoy the ocean!",
                "assistant": "Getting ready for my vacation to the coast! 🧳 Excited to relax 😎 and enjoy the ocean! 🌊"
            }]
        )

class ProfessionalPrompt(Prompt):
    def __init__(self):
        super().__init__(
            sys_prompt="Rewrite this text in a more professional tone.",
            examples=[{
                "user": "Are you free this weekend for a meal?",
                "assistant": "Would you be available this weekend for a dining engagement?"
            }]
        )

class ShortenPrompt(Prompt):
    def __init__(self):
        super().__init__(
            sys_prompt="Condense this text while retaining all key information.",
            examples=[{
                "user": "Greetings! It's been some time since we last spoke. How are you doing? We should get together soon, I have many updates to share about my life since my wedding last year!",
                "assistant": "Hi! How are you? Let's meet soon - lots of news since I got married!"
            }]
        )

class WittyPrompt(Prompt):
    def __init__(self):
        super().__init__(
            sys_prompt="Add a humorous twist to this text.",
            examples=[{
                "user": "My plane arrives at 4pm",
                "assistant": "I'll be gracing the tarmac with my presence at 4pm, just in time to join the symphony of honking cars in rush hour traffic."
            }]
        )

class CasualPrompt(Prompt):
    def __init__(self):
        super().__init__(
            sys_prompt="Make this text more informal and relaxed.",
            examples=[{
                "user": "My plane arrives at 4pm",
                "assistant": "I'll be touching down around 4."
            }]
        )

class ElaboratePrompt(Prompt):
    def __init__(self):
        super().__init__(
            sys_prompt="Expand on this text with more details. For questions, rephrase them with more context as shown in the examples. Provide only the expanded text without additional comments.",
            examples=[
                {
                    "user": "Famous actor from Australia",
                    "assistant": "A renowned actor hailing from Australia, known for his versatile performances in action-packed superhero films and critically acclaimed musicals. He's received numerous awards and nominations for his work in both film and theater."
                },
                {
                    "user": "Plans for tomorrow?",
                    "assistant": "Hello there! I hope you're having a great day. I was wondering if you had any exciting plans or activities lined up for tomorrow?"
                },
                {
                    "user": "Heading home now",
                    "assistant": "I've just finished up here and I'm about to start my journey back home. Hopefully, the traffic is kind and I'll arrive safe and sound in no time."
                }
            ]
        )


class ProofreadPrompt(Prompt):
    def __init__(self):
        super().__init__(sys_prompt="Review and correct any spelling or grammatical errors in the following text.")

class ImprovePrompt(Prompt):
    def __init__(self):
        super().__init__(sys_prompt="Enhance the quality and clarity of the following text.")
        
class KeypointsPrompt(Prompt):
    def __init__(self):
        super().__init__(sys_prompt="Identify and list the main ideas from the given text. Present only the key points without any introductory phrases.")
        
class SummarizePrompt(Prompt):
    def __init__(self):
        super().__init__(sys_prompt="Provide a concise overview of the main points in the following text.")


def format_prompt(messages, usr_prompt, tokenizer):
    text = tokenizer.apply_chat_template(
        messages + [{"role": "user", "content": usr_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def generate_prompt_for_input(input_text, prompt_obj, tokenizer):
    messages = prompt_obj.get_messages()
    return format_prompt(messages, input_text, tokenizer)
