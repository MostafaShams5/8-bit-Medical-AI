import os
import mlflow
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
import requests

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
mlflow.set_experiment("Medical_RAG_Evaluation")

API_URL = "http://localhost:8000/api/chat"
DATASET_ID = "Shams03/Ara-Egy-Medical-QA"

def evaluate_system():
    dataset = load_dataset(DATASET_ID, split="train").select(range(50))
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    rouge_scores = []
    hallucination_flags = 0

    with mlflow.start_run(run_name="AWQ_8Bit_vLLM_Eval"):
        mlflow.log_param("model", "Shams03/ArEgy-Medical-Command-R7B-AWQ-8Bit")
        mlflow.log_param("engine", "vllm")
        mlflow.log_param("dataset_samples", len(dataset))

        for item in dataset:
            question = item['Question']
            ground_truth = item['Answer']
            
            try:
                response_obj = requests.post(API_URL, json={"question": question}, timeout=15)
                response_obj.raise_for_status()
                response = response_obj.json()
            except requests.exceptions.RequestException as e:
                print(f"Request failed for question: {question} - Error: {e}")
                rouge_scores.append(0.0)
                continue

            generated_answer = response.get("answer", "")
            retrieved_chunks = response.get("retrieved_chunks", [])
            
            score = scorer.score(ground_truth, generated_answer)['rougeL'].fmeasure
            rouge_scores.append(score)
            
            if not retrieved_chunks and "استشارة طبيب مختص" not in generated_answer:
                hallucination_flags += 1
                
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
        hallucination_rate = hallucination_flags / len(dataset)
        
        mlflow.log_metric("avg_rouge_l", avg_rouge)
        mlflow.log_metric("hallucination_rate", hallucination_rate)
        
        print("\nEvaluation Complete.")
        print(f"Average ROUGE-L: {avg_rouge:.4f}")
        print(f"Hallucination Rate (Proxy): {hallucination_rate:.2%}")

if __name__ == "__main__":
    evaluate_system()
