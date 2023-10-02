# example with OSCAR 2201
from datasets import load_dataset
from huggingface_hub import login


login(token="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")

dataset = load_dataset("oscar-corpus/OSCAR-2301",
                        token=True, # required
                        language="mn", 
                        streaming=True, # optional
                        split="train") # optional

iterator = iter(dataset)
# Create a new file for writing with UTF-8 encoding
with open("res.txt", "w", encoding="utf-8") as file:    
    for _ in range(3):
        item = next(iterator)
        text = item["text"]
        file.write(text)

print("Result saved to result.txt")





