from transformers import AutoTokenizer, AutoModelForQuestionAnswering ,BertTokenizerFast
import torch
import os
from datasets import load_dataset
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT-1.3B-mongol")

model = AutoModelForQuestionAnswering.from_pretrained("ai-forever/mGPT-1.3B-mongol")

data = load_dataset("json",field="data", data_files= "../annotation/answers.json")

# print(data["train"][0])
contexts = []
trainquestion = []
trainanswers  = []

for group in data["train"]:
    for passage in group["paragraphs"]:
        context = passage["context"]
        for qa in passage["qas"]:
            question = qa["question"]
            for anser in qa["answers"]:
                contexts.append(context)
                trainanswers.append(anser)
                trainquestion.append(question)

train_encodings = tokenizer(contexts, trainquestion, truncation=True, padding=True)

def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = tokenizer.model_max_length

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, trainanswers)

class SQuAD_Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)
  
train_dataset = SQuAD_Dataset(train_encodings)

from torch.utils.data import DataLoader

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

from transformers import AdamW

N_EPOCHS = 5
optim = AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

model.to(device)
model.train()

for epoch in range(N_EPOCHS):
  loop = tqdm(train_loader, leave=True)
  for batch in loop:
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
    loss = outputs[0]
    loss.backward()
    optim.step()

    loop.set_description(f'Epoch {epoch+1}')
    loop.set_postfix(loss=loss.item())

model.save_pretrained("./testmodel")
tokenizer.save_pretrained("./testmodel")