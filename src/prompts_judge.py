#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import re
from typing import Dict, Any

SCORING_RUBRIC = {
    'EMOJIFY': {
        'ACCURACY': {
            'GUIDE': "How well does the emoji version preserve meaning?",
            'LOW': "Meaning significantly changed by emoji choices",
            'MID': "Meaning mostly intact with minor emoji mismatches",
            'HIGH': "Perfect meaning preservation through emoji use"
        },
        'COMPLETENESS': {
            'GUIDE': "Coverage of original content with emojis",
            'LOW': "Major elements lack emoji representation",
            'MID': "Most elements have emoji coverage",
            'HIGH': "Full emoji coverage of all elements"
        },
        'COHERENCE': {
            'GUIDE': "Logical flow of emoji-enhanced text",
            'LOW': "Emojis disrupt readability",
            'MID': "Generally clear with occasional awkward emoji placement",
            'HIGH': "Seamless integration of emojis"
        },
        'CONCISENESS': {
            'GUIDE': "Efficient use of emojis",
            'LOW': "Overuse of emojis creates clutter",
            'MID': "Generally appropriate emoji density",
            'HIGH': "Optimal emoji usage throughout"
        }
    },

    'PROFESSIONAL': {
        'ACCURACY': {
            'GUIDE': "Maintains meaning in professional context",
            'LOW': "Professional conversion alters meaning",
            'MID': "Mostly accurate with minor shifts",
            'HIGH': "Perfect professional translation"
        },
        'COMPLETENESS': {
            'GUIDE': "Coverage of original content professionally",
            'LOW': "Key professional elements missing",
            'MID': "Most professional elements included",
            'HIGH': "Complete professional coverage"
        },
        'COHERENCE': {
            'GUIDE': "Professional flow and structure",
            'LOW': "Disjointed professional presentation",
            'MID': "Generally clear professional flow",
            'HIGH': "Excellent professional structure"
        },
        'CONCISENESS': {
            'GUIDE': "Efficient professional language",
            'LOW': "Overly verbose professional style",
            'MID': "Generally concise professional tone",
            'HIGH': "Optimal professional brevity"
        }
    },

    'SHORTEN': {
        'ACCURACY': {
            'GUIDE': "Meaning preservation in shorter form",
            'LOW': "Key meaning lost in shortening",
            'MID': "Most meaning retained despite cuts",
            'HIGH': "Perfect meaning in shorter form"
        },
        'COMPLETENESS': {
            'GUIDE': "Essential content retention",
            'LOW': "Critical content missing",
            'MID': "Most key content included",
            'HIGH': "All essential content present"
        },
        'COHERENCE': {
            'GUIDE': "Flow of shortened version",
            'LOW': "Choppy or disconnected",
            'MID': "Generally smooth with minor gaps",
            'HIGH': "Perfect flow in short form"
        },
        'CONCISENESS': {
            'GUIDE': "Efficiency of length reduction",
            'LOW': "Insufficient shortening",
            'MID': "Good reduction with room to improve",
            'HIGH': "Optimal length reduction"
        }
    },
    'WITTY': {
        'ACCURACY': {
            'GUIDE': "Maintains meaning while adding humor",
            'LOW': "Humor distorts original message",
            'MID': "Message mostly intact with humor",
            'HIGH': "Perfect balance of wit and accuracy"
        },
        'COMPLETENESS': {
            'GUIDE': "Coverage while maintaining wit",
            'LOW': "Content lost to humor attempts",
            'MID': "Most content preserved with humor",
            'HIGH': "All points enhanced by wit"
        },
        'COHERENCE': {
            'GUIDE': "Flow of humorous content",
            'LOW': "Humor disrupts message flow",
            'MID': "Generally smooth with occasional awkward jokes",
            'HIGH': "Seamless integration of humor"
        },
        'CONCISENESS': {
            'GUIDE': "Efficient use of wit",
            'LOW': "Overwrought or forced humor",
            'MID': "Generally efficient wit usage",
            'HIGH': "Perfectly balanced humor"
        }
    },

    'CASUAL': {
        'ACCURACY': {
            'GUIDE': "Meaning preservation in casual tone",
            'LOW': "Meaning lost in casual conversion",
            'MID': "Mostly accurate in casual form",
            'HIGH': "Perfect casual translation"
        },
        'COMPLETENESS': {
            'GUIDE': "Content coverage in casual style",
            'LOW': "Major points lost in casual tone",
            'MID': "Most points retained casually",
            'HIGH': "Full content in casual style"
        },
        'COHERENCE': {
            'GUIDE': "Flow of casual language",
            'LOW': "Disjointed casual presentation",
            'MID': "Generally smooth casual flow",
            'HIGH': "Perfect casual coherence"
        },
        'CONCISENESS': {
            'GUIDE': "Efficiency of casual expression",
            'LOW': "Rambling or unfocused",
            'MID': "Generally efficient casual style",
            'HIGH': "Optimally casual and brief"
        }
    },

    'ELABORATE': {
        'ACCURACY': {
            'GUIDE': "Accuracy of expanded content",
            'LOW': "Added details are incorrect",
            'MID': "Most additions are accurate",
            'HIGH': "Perfect accuracy in elaboration"
        },
        'COMPLETENESS': {
            'GUIDE': "Thoroughness of elaboration",
            'LOW': "Insufficient expansion",
            'MID': "Good expansion with some gaps",
            'HIGH': "Comprehensive elaboration"
        },
        'COHERENCE': {
            'GUIDE': "Flow of expanded content",
            'LOW': "Disjointed elaboration",
            'MID': "Generally clear expansion",
            'HIGH': "Perfectly integrated details"
        },
        'CONCISENESS': {
            'GUIDE': "Efficiency of elaboration",
            'LOW': "Needlessly wordy expansion",
            'MID': "Generally focused elaboration",
            'HIGH': "Efficient and thorough detail"
        }
    },

    'PROOFREAD': {
        'ACCURACY': {
            'GUIDE': "Error detection and correction",
            'LOW': "Major errors remain",
            'MID': "Minor errors persist",
            'HIGH': "All errors properly fixed"
        },
        'COMPLETENESS': {
            'GUIDE': "Coverage of error types",
            'LOW': "Many error types missed",
            'MID': "Most error types addressed",
            'HIGH': "All error types handled"
        },
        'COHERENCE': {
            'GUIDE': "Consistency of corrections",
            'LOW': "Inconsistent error handling",
            'MID': "Mostly consistent corrections",
            'HIGH': "Perfectly consistent fixes"
        },
        'CONCISENESS': {
            'GUIDE': "Efficiency of corrections",
            'LOW': "Corrections create wordiness",
            'MID': "Generally efficient fixes",
            'HIGH': "Optimal correction style"
        }
    },

    'IMPROVE': {
        'ACCURACY': {
            'GUIDE': "Accuracy of improvements",
            'LOW': "Changes create errors",
            'MID': "Most changes are accurate",
            'HIGH': "All improvements enhance accuracy"
        },
        'COMPLETENESS': {
            'GUIDE': "Coverage of improvements",
            'LOW': "Many areas need improvement",
            'MID': "Most areas improved",
            'HIGH': "Comprehensive enhancement"
        },
        'COHERENCE': {
            'GUIDE': "Integration of improvements",
            'LOW': "Changes disrupt flow",
            'MID': "Changes mostly well-integrated",
            'HIGH': "Perfectly seamless improvements"
        },
        'CONCISENESS': {
            'GUIDE': "Efficiency of improvements",
            'LOW': "Changes create bloat",
            'MID': "Generally efficient changes",
             'HIGH': "Optimally streamlined improvements"
        }
    },

    'KEYPOINTS': {
        'ACCURACY': {
            'GUIDE': "Accuracy of key point selection",
            'LOW': "Major points misidentified",
            'MID': "Most points correctly identified",
            'HIGH': "Perfect point identification"
        },
        'COMPLETENESS': {
            'GUIDE': "Coverage of main points",
            'LOW': "Essential points missing",
            'MID': "Most main points included",
            'HIGH': "All key points captured"
        },
        'COHERENCE': {
            'GUIDE': "Organization of key points",
            'LOW': "Points poorly organized",
            'MID': "Generally logical arrangement",
            'HIGH': "Perfect point organization"
        },
        'CONCISENESS': {
            'GUIDE': "Efficiency of point presentation",
            'LOW': "Points unnecessarily verbose",
            'MID': "Generally concise points",
            'HIGH': "Optimally succinct points"
        }
    }
}


def get_rubric(tone: str) -> Dict[str, Any]:
    if tone not in SCORING_RUBRIC:
        raise KeyError(f"Rubric not found for tone: {tone}")
    return SCORING_RUBRIC[tone]

def generate_system_prompt(rubric: Dict[str, Any]) -> str:
    rubric_text = "\n".join([f"{k}: {v}" for k, v in rubric.items() if k in ['GUIDE', 'LOW', 'MID', 'HIGH']])
    
    return f"""<task>
    You are a Judge evaluating LLM output quality. Inputs:
    1. <OriginalText>: User's original text
    2. <Output>: LLM's transformed response
    3. <Tone>: Specified task (e.g., "emojify", "professional", "proofread")
    4. <score_rubric>: Evaluation criteria
    
    Evaluation steps:
    1. Read the score rubric
    2. Assess output quality based on rubric, Original Text, Output Text, and Tone. Be strict and detailed.
    3. Choose best matching score (1-3) based on your reasoning.
    4. Format: "<reasoning>(observations)</reasoning><score>1, 2, or 3</score>"
    5. Provide only the reasoning and score, no other text.
    </task>
    
    <score_rubric>
    {rubric_text}
    </score_rubric>
    """.strip()

def generate_input_prompt(input_text: str, output_text: str, tone: str) -> str:

    return f"""
    <OriginalText>{input_text}</OriginalText>
    <Output>{output_text}</Output>
    <Tone>{tone}</Tone>
    """

def rewrite_prompt(input_text: str, output_text: str) -> str:
    return f"""
    <User>{input_text}</User>
    <System>{output_text}</System>
    <Instruction>Score the previous user-assistant interaction. If it is a rewrite, score 1, if it is a back-and-forth conversation, score 0. Just return the score.</Instruction>
    """
