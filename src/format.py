#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

def format_prompt(messages, usr_prompt, tokenizer):
    """
    Format prompts according to each model's prompt guidelines (e.g. xml tags for Haiku). 
    
    :param messages: List of messages
    :param usr_prompt: User prompt
    :param tokenizer: The models's Transformer tokenizer.
    """
    text = tokenizer.apply_chat_template(
        messages + [{"role": "user", "content": usr_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    return(text)

def format_prompt_as_xml(usr_prompt, prompt):
    """
    Format a Prompt object as XML.
    
    Args:
        prompt: A Prompt object containing system prompt and optional examples
        
    Returns:
        str: Formatted XML string
    """
    def create_conversation_elements(parent, role, text):
        """Helper function to create conversation elements"""
        element = SubElement(parent, role)
        element.text = text
        return element

    # Create root element
    root = Element("prompt")
    
    # Add system prompt
    create_conversation_elements(root, "system", prompt.sys_prompt)
    
    # Add examples if they exist
    if prompt.examples:
        examples = SubElement(root, "example")
        for example in prompt.examples:
            # example_elem = SubElement(examples, "example")
            create_conversation_elements(examples, "user", example["user"])
            create_conversation_elements(examples, "assistant", example["assistant"])

    # Add the user prompt
    user_text = SubElement(root, "user")
    user_text.text = usr_prompt
    
    # Convert to pretty XML
    dom = xml.dom.minidom.parseString(tostring(root))
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove the XML declaration and temp_root tags
    lines = pretty_xml.split('\n')
    cleaned_xml = '\n'.join(line for line in lines[2:-2] if line.strip())  # Skip the first two and last two lines, and any empty lines
    
    
    # Clean up the output
    # lines = pretty_xml.split('\n')
    # cleaned_xml = '\n'.join(line for line in lines[1:] if line.strip())  # Skip XML declaration
    
    return cleaned_xml
