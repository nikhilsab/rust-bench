import os
import time

import json
import numpy as np
import pandas as pd
from openai import OpenAI

def main():
    # Set DeepInfra API key
    openai = OpenAI(
        api_key="",
        base_url="https://api.deepinfra.com/v1/openai",
    )

    # Load dataset
    dataset = []
    data_file = "RUST-BENCH-FINAL"
    data_path = os.path.join(f"./data/{data_file}.json")
    with open(data_path, "r") as f:
        data = json.load(f)

    for i, dic in enumerate(data):
        if i >= 0:
            caption = dic['table']['name']
            question = dic['question']
            answer_text = dic['answers']
            header = dic['table']['header']
            rows = dic['table']['rows']

            data = {
                "id": i,
                "table": {
                    "header": header,
                    "rows": rows,
                    "page_title": caption
                },
                "question": question,
                "answer_text": answer_text,
            }
            dataset.append(data)

    for idx, i in enumerate(dataset):
        question = i['question']
        metadata = i['metadata']
        answer = i['answer_text']
        table_name = i['table']['page_title']
        idx = i['id']
        g_dict = {'idx': idx}
        df = pd.DataFrame(i['table']['rows'], columns=i['table']['header'])
        string = ''
        col_list = df.columns.values.tolist()
        string += 'col : ' + ' | '.join(df.columns) + '\n'
        for row_id, row in df.iterrows():
            string += f'row {row_id} : '
            for column_id, header in enumerate(df.columns):
                string += str(row[header])
                if column_id != len(df.columns) - 1:
                    string += ' | '
            string += '\n'
        string += '*/\n'

        with open('./prompts/end2end.txt', 'r') as file:
            template = file.read()

        prompt = template.format(table=string, question=question)

        # Call DeepInfra LLaMA 3.3 70B
        response = openai.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048
        )

        result = response.choices[0].message.content
        print(f"Question: {question}")
        print(f"Response: {result}\n")
        g_dict['question'] = question
        g_dict['metadata'] = metadata
        g_dict['answer'] = answer
        g_dict['prediction'] = result

        save_file_name = f'{data_file}_llama_3.3_70b_e2e.json'
        with open(os.path.join('./results', save_file_name), 'a') as f:
            json.dump(g_dict, f, indent=4)
    
    # Calculate and print accuracy
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '_main_':
    main()