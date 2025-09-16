# from transformers import pipeline
# import torch

# pipe = pipeline(
#     "text-generation",
#     model="google/gemma-3-4b-it",
#     device="cuda",
#     torch_dtype=torch.bfloat16
# )

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who are you?"},
# ]

# output = pipe(text=messages, max_new_tokens=200)
# print(output[0]["generated_text"][-1]["content"])



# import transformers
# import torch

# model_id = "/mount/resources9/models/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = pipeline(
#     messages,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# print(outputs[0]["generated_text"][-1])

import torch
print(torch.cuda.is_available())

print("yes")
