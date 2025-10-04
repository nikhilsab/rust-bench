"""
Final Text Reasoning using Gemini
"""

import time
import json
import os
import numpy as np
import pandas as pd
import google.generativeai as genai
import time

# Configure Gemini API
GOOGLE_API_KEY = ""  # Replace with your actual API key

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def execute_gemini(table_string, question, table_name, table_id):
    """
    Execute table question answering using Gemini model with zero-shot prompting
    """
    # Convert dataframe to string representation
    # Load prompt template
    with open('./prompts/fewshot_science.txt', 'r') as file:
        template = file.read()
    
    # Format prompt
    prompt = template.format(table=table_string, question=question)

    # Generate response using Gemini
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        return result, prompt, response.text
            
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error", prompt, str(e)

def main():
    # Load dataset
    start_time = time.time()
    # Load dataset
    dataset = []
    data_file = "RUST-BENCH-FINAL"
    data_path = os.path.join(f"./data/{data_file}.json")
        # lines = f.readlines()
    with open(data_path, "r") as f:
        data = json.load(f) 
    print(len(data))
    # Process examples
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
    
    results= []
    for idx, i in enumerate(dataset):
        question = i['question']
        metadata = i['metadata']
        answer = i['answer_text']
        # answer_string = ", ".join(answer)
        table_name = i['table']['page_title']
        # id = i['table']['id']
        idx = i['id']
        g_dict={'idx': idx}
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
        
        time.sleep(10)
        result, prompt, full_response = execute_gemini(string, question, table_name, id)
        
        # Store results
        results.append({
            'id': i['id'],
            'question': question,
            'predicted': result,
            'actual': i['answer_text'],
            'table_name': table_name,
            'correct': str(result).lower() == str(i['answer_text']).lower()
        })
        g_dict['question'] = question
        g_dict['metadata'] = metadata
        g_dict['answer'] = answer
        g_dict['prediction'] = result
        
        # Print results
        print(f"\nExample {i['id']}:")
        print(f"Question: {question}")
        print(f"Predicted: {result}")
        print(f"Actual: {i['answer_text']}")
        # print(f"Correct: {results[-1]['correct']}")
        # print("-" * 80)

        save_file_name = f'{data_file}_gemini_2.0_flash_fewshot.json'
        with open(os.path.join('./results', save_file_name), 'a') as f:
            json.dump(g_dict, f, indent=4)
    
    # Calculate and print accuracy
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '_main_':
    main()