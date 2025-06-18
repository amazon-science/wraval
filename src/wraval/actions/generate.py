#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
def gen(text, tokenizer, model):
    """
    Generate for a self-hosted model (one query at a time)

    :param text: Full text formatted for the model
    :param tokenizer: The models's Transformer tokenizer.
    :param model: The model in Transformers format
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True).input_ids.cuda()
    input_length = inputs.shape[1]
    outputs = model.handle_generate(
        inputs,
    )[:, input_length:]
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
