input_file = "fullqa.jsonl"
output_file = "fullqa_trimmed.jsonl"

start_cut = 19001
end_cut = 340936

with open(input_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

# Remove lines from 19001 to 340936 (index 19000 to 340935)
trimmed_lines = lines[:19000] + lines[340936:]

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.writelines(trimmed_lines)

print(f"✅ Trimmed file saved to {output_file}. Original lines: {len(lines)}, New lines: {len(trimmed_lines)}")
