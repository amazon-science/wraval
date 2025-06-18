#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import random
import numpy as np
from evaluate import load

# Load BLEU metric
bleu = load("bleu")


def compute_bleu_with_ci(
    predictions, references, num_bootstrap_samples=1000, confidence_level=0.95
):
    # Compute the original BLEU score
    original_bleu = bleu.compute(predictions=predictions, references=references)["bleu"]

    # Bootstrap sampling
    bootstrap_scores = []
    n = len(predictions)

    for _ in range(num_bootstrap_samples):
        # Sample indices with replacement
        indices = [random.randint(0, n - 1) for _ in range(n)]
        sampled_predictions = [predictions[i] for i in indices]
        sampled_references = [references[i] for i in indices]

        # Compute BLEU for the bootstrap sample
        score = bleu.compute(
            predictions=sampled_predictions, references=sampled_references
        )["bleu"]
        bootstrap_scores.append(score)

    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrap_scores, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_scores, (1 + confidence_level) / 2 * 100)

    return {"bleu": original_bleu, "confidence_interval": (lower_bound, upper_bound)}


# Example usage
predictions = ["This is a test", "Another sentence"]
references = [["This is a test"], ["Another sentence"]]

results = compute_bleu_with_ci(predictions, references)

print(f"BLEU Score: {results['bleu']}")
print(f"95% Confidence Interval: {results['confidence_interval']}")
