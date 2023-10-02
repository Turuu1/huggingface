from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./models/EsperBERTo-small/session_9",
    tokenizer="./models/EsperBERTo-small"
)

t = fill_mask("Хэтэрхий их <mask>.")

print(t)