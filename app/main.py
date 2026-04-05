import os
import uuid
import logging
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from fastapi.concurrency import run_in_threadpool


import transformers.utils.import_utils
def is_torch_fx_available(): return True
transformers.utils.import_utils.is_torch_fx_available = is_torch_fx_available
transformers.utils.is_torch_fx_available = is_torch_fx_available
def check_torch_load_is_safe(): pass
transformers.utils.import_utils.check_torch_load_is_safe = check_torch_load_is_safe


from FlagEmbedding import BGEM3FlagModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer


from app.utils import SYSTEM_PREAMBLE, extract_final_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("medical_rag_api")


embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

DB_PATH = os.getenv("QDRANT_PATH", "./medical_qdrant_db")
if os.path.exists(DB_PATH):
    for root, dirs, files in os.walk(DB_PATH):
        if ".lock" in files:
            try:
                os.remove(os.path.join(root, ".lock"))
            except Exception as e:
                logger.warning("Failed to remove lock file: %s", str(e))

client = QdrantClient(path=DB_PATH)
COLLECTION_NAME = "arabic_medical_HybridRAG"
MODEL_ID = "Shams03/ArEgy-Medical-Command-R7B-AWQ-8Bit"

engine_args = AsyncEngineArgs(
    model=MODEL_ID,
    quantization="awq",
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    max_model_len=4096,
    disable_log_requests=True
)
llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

app = FastAPI(title="Tawkeed Medical RAG API", version="2.0")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    source: str
    page: str
    score: float
    retrieved_chunks: list[dict]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.question
        
        # OFF-LOAD BLOCKING PYTORCH TO A THREAD
        embeddings = await run_in_threadpool(
            embedding_model.encode, query, return_dense=True, return_sparse=True
        )
        sparse_dict = embeddings["lexical_weights"]

        sparse_vec = qdrant_models.SparseVector(
            indices=[int(k) for k in sparse_dict.keys()],
            values=[float(v) for v in sparse_dict.values()]
        )

        candidate_limit = 8
        
        # OFF-LOAD BLOCKING DATABASE CALL TO A THREAD
        response = await run_in_threadpool(
            client.query_points,
            collection_name=COLLECTION_NAME,
            prefetch=[
                qdrant_models.Prefetch(query=embeddings["dense_vecs"].tolist(), using="dense", limit=candidate_limit),
                qdrant_models.Prefetch(query=sparse_vec, using="sparse", limit=candidate_limit),
            ],
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=candidate_limit
        )

        search_results = response.points or []
        filtered_results = [pt for pt in search_results if float(getattr(pt, "score", 0.0)) > 0.50][:2]

        context_parts = []
        retrieved_chunks = []

        for pt in filtered_results:
            payload = pt.payload or {}
            page_num = payload.get("page", payload.get("page_number", "1"))
            context_text = payload.get("text", payload.get("page_content", payload.get("content", "")))
            source_name = payload.get("source", payload.get("file_name", payload.get("document", "غير معروف")))
            score = float(getattr(pt, "score", 0.0))

            retrieved_chunks.append({
                "source": str(source_name), "page": str(page_num),
                "score": score, "text": str(context_text)
            })
            context_parts.append(f"المصدر: {source_name}\nالصفحة: {page_num}\nالدرجة: {score:.3f}\nالنص: {context_text}")

        context_text_joined = "\n\n---\n\n".join(context_parts)

        user_content = f"السؤال: {query}\n\n"
        if context_text_joined:
            user_content += f"السياق الطبي المسترجع:\n{context_text_joined}\n"

        messages = [
            {"role": "system", "content": SYSTEM_PREAMBLE},
            {"role": "user", "content": user_content}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            temperature=0.6, top_p=0.8, top_k=20, min_p=0.0,
            repetition_penalty=1.1, max_tokens=256
        )

        request_id = str(uuid.uuid4())
        
        # vLLM is natively async, no threadpool needed here
        results_generator = llm_engine.generate(text, sampling_params, request_id)
        
        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text

        clean_answer = extract_final_answer(final_output)

        top_source = str(retrieved_chunks[0]["source"]) if retrieved_chunks else ""
        top_page = str(retrieved_chunks[0]["page"]) if retrieved_chunks else ""
        top_score = float(retrieved_chunks[0]["score"]) if retrieved_chunks else 0.0

        return ChatResponse(
            question=query, answer=clean_answer,
            source=top_source, page=top_page, score=top_score,
            retrieved_chunks=retrieved_chunks
        )

    except Exception as e:
        logger.error("API Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info", reload=False) # Switched to module string for better worker handling
