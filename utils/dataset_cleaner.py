import json
import re

def deep_clean_text(text):
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'\*\*+', '', text)
    text = re.sub(r'\*+', '', text)
    
    text = text.replace('\\\\', '').replace('\\', '')
    
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_and_sort_dataset(input_path, output_path):
    dataset = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    dataset.sort(key=lambda x: int(x.get('id', 0)))

    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for new_id, record in enumerate(dataset):
            cleaned_q = deep_clean_text(record.get('Question', ''))
            cleaned_a = deep_clean_text(record.get('Answer', ''))
            
            final_record = {
                "id": new_id,
                "original_id": record.get('id'),
                "Question": cleaned_q,
                "Answer": cleaned_a
            }
            
            outfile.write(json.dumps(final_record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    INPUT = '/home/shams/Documents/cleaned_medical_dataset.jsonl'
    OUTPUT = '/home/shams/Documents/final_hf_dataset.jsonl'
    
    process_and_sort_dataset(INPUT, OUTPUT)
