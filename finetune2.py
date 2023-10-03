from transformers import AutoTokenizer, AutoModelForQuestionAnswering ,BertTokenizerFast
import torch
import os
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT-1.3B-mongol")

model = AutoModelForQuestionAnswering.from_pretrained("ai-forever/mGPT-1.3B-mongol")

data = load_dataset("json",field="data", data_files= "./data.json")


contexts = []
trainquestion = []
trainanswers  = []
tokenized_data = []
for group in data["train"]:
    for passage in group["paragraphs"]:     
        for qa in passage["qas"]:
            question = qa["question"]
            context = qa["context"]
            for anser in qa["answers"]:
                context_tokens = tokenizer.encode(context, add_special_tokens=True)
                question_tokens = tokenizer.encode(question, add_special_tokens=True)
                tokenized_data.append({
                "context_tokens": context_tokens,
                "question_tokens": question_tokens,
                "answer_start": anser["answer_start"]
    })
context_tokens = [item["context_tokens"] for item in tokenized_data]
question_tokens = [item["question_tokens"] for item in tokenized_data]
answer_start = [item["answer_start"] for item in tokenized_data]

dataset = TensorDataset(context_tokens, question_tokens, answer_start)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    output_dir='./qa_finetuned_model',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataloader,
)

trainer.train()

model.save_pretrained('./qa_finetuned_model')
tokenizer.save_pretrained('./qa_finetuned_model')

