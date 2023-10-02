import os
import re
import neattext as nt
import neattext.functions as nf
import chardet


def remove(text):
    p = re.compile(r'[A-Za-z]')
    return p.sub("", text)
# Specify the input file path
input_file_path = "output_text.txt"  # Replace with your file path

# Specify the output directory
output_dir = "../out"  # Replace with your desired output directory
os.makedirs(output_dir, exist_ok=True)

# Specify the chunk size (in bytes)
chunk_size = 50000  # Adjust the size according to your needs

# Initialize a chunk counter
chunk_counter = 1

# Open the input file in binary mode
with open(input_file_path, "rb") as input_file:
    while True:

        chunk = input_file.read(chunk_size)
        if not chunk:
            break
        
        # Use chardet to detect the encoding of the chunk
        result = chardet.detect(chunk)
        detected_encoding = result["encoding"]
        
        # Decode the chunk using the detected encoding

        decoded_chunk = chunk.decode(detected_encoding, errors="replace")

        decoded_chunk =  decoded_chunk.lower()
        nf.clean_text(decoded_chunk, urls=True, emails= True,  )
        remove(decoded_chunk)
        decoded_chunk = " ".join(decoded_chunk.split())

        output_file_path = os.path.join(output_dir, f"output_{chunk_counter}.txt")
        
        # Write the decoded chunk to the output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(decoded_chunk)
        
        chunk_counter += 1

print(f"{chunk_counter - 1} output files created in '{output_dir}' directory.")

