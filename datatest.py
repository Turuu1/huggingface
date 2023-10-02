# example with OSCAR 2201
from datasets import load_dataset
from huggingface_hub import login


login(token="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")

dataset = load_dataset("oscar-corpus/OSCAR-2301",
                        token=True, # required
                        language="mn", 
                        streaming=True, # optional
                        split="train",
                        ) # optional


# # Create a new file for writing with UTF-8 encoding
# with open("result.txt", "w", encoding="utf-8") as file:
#     # Iterate through the dataset and write each text property to the file
#     for item in dataset:
#         text = item["text"]
#         file.write(text + "\n")

print("Result saved to result.txt")





