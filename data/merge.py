import json
import os

def merge_json_parts(input_dir, output_file):
    all_data = []

    files = sorted(f for f in os.listdir(input_dir) if f.endswith('.json'))

    for file in files:
        file_path = os.path.join(input_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"File {file} does not contain a list at the top level.")
            all_data.extend(data)
            print(f"Merged {len(data)} items from {file}")

 
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_data, f_out, indent=2, ensure_ascii=False)


