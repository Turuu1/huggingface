import json

input_jsonl_path = "D:\OSCAR-2301\mn_meta\mn_meta_part_1.jsonl"
output_text_path = "output_text.txt"

with open(input_jsonl_path, "r", encoding="utf-8") as input_file, \
        open(output_text_path, "w", encoding="utf-8") as output_file:
    for line in input_file:
        json_line = json.loads(line.strip())
        content = json_line.get("content", "")  # Assuming "content" is the field name
        output_file.write(content + "\n")


print("Text content extracted and saved to", output_text_path)