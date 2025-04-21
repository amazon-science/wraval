

elif endpoint_type == "ollama":
    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_name, trust_remote_code=True
    )
    prompts = [
        format_prompt(user, sys, tokenizer, type="hf") for text in tqdm(queries)
    ]
    print(f"Sample prompt:\n{prompts[0]}")
    outputs = [
        invoke_ollama_endpoint(prompt)
        for prompt in tqdm(prompts)
    ]