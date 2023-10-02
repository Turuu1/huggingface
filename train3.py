from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained("./models/EsperBERTo-small", max_len=512)

for session in range(2, 10):
    model = RobertaForMaskedLM.from_pretrained("./models/EsperBERTo-small")
    print(model.num_parameters())
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=f"../out/output_{session}.txt",
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"./models/EsperBERTo-small",  # Update the output_dir
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(f"./models/EsperBERTo-small")  # Update the save_model path
    print(f"Session- {session}")
