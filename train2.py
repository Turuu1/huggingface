from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig
import torch
import os

# Initialize tokenizer and model configuration
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model_config = RobertaConfig(vocab_size=tokenizer.vocab_size, num_hidden_layers=6)
model = RobertaForMaskedLM(config=model_config)

# Initial Pretraining
initial_training_data = "../out/output_1.txt"  # Replace with your data file
tokenized_data = tokenizer(initial_training_data, return_tensors="pt", padding=True, truncation=True)
print("-----------------1------------------------")
model.train()
print("-----------------2------------------------")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):  # Example: 3 epochs for the initial training
    optimizer.zero_grad()
    outputs = model(**tokenized_data)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Save the initial checkpoint
model.save_pretrained("initial_checkpoint")

# Subsequent Incremental Training
for session in range(1, 10):  # Example: 3 incremental training sessions
    incremental_data = f"../out/outut_{session}.txt"  # Replace with your data file
    tokenized_data = tokenizer(incremental_data, return_tensors="pt", padding=True, truncation=True)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Load the checkpoint from the previous session
    model = RobertaForMaskedLM.from_pretrained(f"checkpoint_session_{session - 1}")

    for epoch in range(2):  # Example: 2 epochs for each incremental session
        optimizer.zero_grad()
        outputs = model(**tokenized_data)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Save the checkpoint for this session
    model.save_pretrained(f"checkpoint_session_{session}")
