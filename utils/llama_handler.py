from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel,
)
import torch

from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")


def askme(question):
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who acts as a doctor",
        },
        {"role": "user", "content": question},
    ]

    print("here 2")

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("here 3")
    
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print("here 4")
    
    return outputs[0]["generated_text"]


def second_ask_me(question):
    # client = Client("https://umutozdemir-medicalai-clinicalbert.hf.space/")
    # result = client.predict(
    #     question,  # str  in 'Input' Textbox component
    #     api_name="/predict",
    # )
    # print(result)
    pass
