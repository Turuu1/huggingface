import json
path = "answers.json"
data = [{"paragraphs": [{"qas":[]}]}]
with open('../annotation/answers.json', 'rb') as f:
  squad = json.load(f)
  data[0]["paragraphs"][0]["document_id"] = squad["data"][0]["paragraphs"][0]["document_id"]
#   print(squad["data"][0]["paragraphs"][0]["qas"])
  for q in squad["data"][0]["paragraphs"][0]["qas"]:
    qas = q
    start = 0
    end = 0
    answer = q["answers"][0]["text"]
    start = q["context"].index(answer[0:10])

    qas["answers"][0]["answer_start"] = start
    qas["answers"][0]["answer_end"] = start + len(answer)
    print(q["context"][start:  start + len(answer)])
    data[0]["paragraphs"][0]["qas"].append(q)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
