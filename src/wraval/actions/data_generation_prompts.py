# Meta-prompt template that ensures consistent CSV-compatible output
META_PROMPT_TEMPLATE = """Generate 100 {description} in a format that can be saved as a single-column CSV file.

Requirements:
- Start immediately with the first item (no introductory text or headers)
- Each line should contain exactly one {item_type}
- No quotes, commas, or special characters that would interfere with CSV parsing
- No numbering or bullet points
- No Python list formatting
- Just plain text, one {item_type} per line
- Each {item_type} should be {constraints}

Example format (start directly with content):
I went to the store yesterday
The weather is beautiful today
She completed her project on time

Begin generating 100 {item_type}s (start with first item, no introduction):
"""

# Individual prompts using the meta template
WITTY_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="sentences that someone would want to rewrite in a more witty way",
    item_type="sentence",
    constraints="straightforward and factual, perfect for adding humor",
)

PROFESSIONAL_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="casual or informal sentences that should be rewritten professionally",
    item_type="sentence",
    constraints="informal or casual in tone, suitable for professional reformulation",
)

CASUAL_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="formal sentences that should be rewritten in a casual way",
    item_type="sentence",
    constraints="formal or stiff in tone, ready to be made more casual",
)

ELABORATE_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="simple sentences that should be rewritten more elaborately",
    item_type="sentence",
    constraints="basic and straightforward, ready for detailed expansion",
)

SHORTEN_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="long, wordy sentences that should be shortened",
    item_type="sentence",
    constraints="verbose and detailed, ready to be made concise",
)

IMPROVE_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="poorly written sentences that need improvement",
    item_type="sentence",
    constraints="containing common writing issues like passive voice, redundancy, or unclear meaning",
)

KEYPOINTS_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="detailed paragraphs that need key points extracted",
    item_type="paragraph",
    constraints="containing multiple important points that could be summarized",
)

PROOFREAD_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="sentences with grammatical errors, typos, and style issues",
    item_type="sentence",
    constraints="containing common mistakes that need correction",
)

EMOJIFY_SENTENCES_PROMPT = META_PROMPT_TEMPLATE.format(
    description="plain text sentences that could be enhanced with emojis",
    item_type="sentence",
    constraints="describing emotions, actions, or subjects that could be represented with emojis",
)

# Special case for paragraph-summary pairs since it needs a different format
PARAGRAPH_SUMMARY_PROMPT = """Generate 100 paragraph-summary pairs in a format that can be saved as a CSV file.

Requirements:
- Each line should contain a paragraph and its summary, separated by ||| (triple pipe)
- No quotes, commas, or special characters that would interfere with CSV parsing
- Each paragraph should be 150-300 words about topics like:
  * Scientific discoveries
  * Historical events
  * Technological innovations
  * Cultural phenomena
  * Business cases
  * Environmental issues
  * Social movements
  * Biographical snippets
  * Current events
  * Academic concepts
- Each summary should be 30-50 words
- Paragraphs should be self-contained and factually accurate
- Summaries should capture main points without introducing new information

Example format:
The Industrial Revolution marked a major turning point in human history... ||| The Industrial Revolution transformed manufacturing and society through mechanization and new manufacturing processes
The theory of evolution by natural selection... ||| Darwin's theory explains species adaptation through natural selection over generations

Begin generating 100 paragraph-summary pairs:
"""
