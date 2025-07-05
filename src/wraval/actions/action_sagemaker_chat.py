import boto3
import json
import re
import sys

def start_sagemaker_chat(endpoint_name="Phi-4-mini-instruct-test", region="us-east-1", max_new_tokens=1000):
    """
    Start an interactive chat with a SageMaker endpoint.
    """
    client = boto3.client("sagemaker-runtime", region_name=region)
    print(f"Chatting with SageMaker endpoint: {endpoint_name} (region: {region})")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Exiting chat.")
            break
        payload = {
            "inputs": user_input,
            "parameters": {"max_new_tokens": max_new_tokens}
        }
        try:
            response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(payload).encode("utf-8"),
                ContentType="application/json",
            )
            result = response["Body"].read().decode("utf-8")
            # Try to parse as JSON, else print raw
            try:
                result_json = json.loads(result)
                # Try to extract the text field (common in HuggingFace-style endpoints)
                if isinstance(result_json, dict) and "generated_text" in result_json:
                    assistant_output = result_json["generated_text"]
                elif isinstance(result_json, list) and len(result_json) > 0 and "generated_text" in result_json[0]:
                    assistant_output = result_json[0]["generated_text"]
                else:
                    assistant_output = result_json
            except Exception:
                assistant_output = result

            # Clean up output: remove repeated prompt and <|endoftext|>
            def clean_output(text, user_input):
                if isinstance(text, list):
                    text = "\n".join(str(x) for x in text)
                elif not isinstance(text, str):
                    text = str(text)
                # Remove repeated prompt at the start
                text = text.strip()
                if text.lower().startswith(user_input.strip().lower()):
                    text = text[len(user_input):].lstrip()
                # Remove <|endoftext|> tokens
                text = text.replace("<|endoftext|>", "").strip()
                return text

            # Format output nicely
            def print_formatted(text):
                # Find Markdown code blocks
                code_block_pattern = r"```([\w]*)\n([\s\S]*?)```"
                last_end = 0
                for match in re.finditer(code_block_pattern, text):
                    # Print text before code block
                    before = text[last_end:match.start()]
                    if before.strip():
                        print(f"\nAssistant: {before.strip()}\n")
                    # Print code block
                    lang = match.group(1)
                    code = match.group(2)
                    print(f"\nAssistant (code block):\n{'-'*30}\n{code}\n{'-'*30}\n")
                    last_end = match.end()
                # Print any remaining text after last code block
                after = text[last_end:]
                if after.strip():
                    print(f"\nAssistant: {after.strip()}\n")

            cleaned_output = clean_output(assistant_output, user_input)
            print_formatted(cleaned_output)
        except Exception as e:
            print(f"Error invoking endpoint: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chat with a SageMaker endpoint interactively.")
    parser.add_argument("--endpoint-name", type=str, default="Phi-4-mini-instruct-test", help="SageMaker endpoint name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--max-new-tokens", type=int, default=1000, help="Max new tokens for generation")
    args = parser.parse_args()
    sagemaker_chat(args.endpoint_name, args.region, args.max_new_tokens) 