# HakimAI: Arabic Clinical LLM & Retrieval System

## Project Overview
Fine-tuned the Cohere Command R7B model on 50,000 real patient-doctor chats. Because the base model struggled to understand Egyptian users, we used the Gemini API to rewrite and focus the Arabic data specifically on the Egyptian dialect. We also compressed the model using AWQ 8-bit quantization via vLLM, which cut VRAM usage by about 40% while keeping the medical accuracy intact.

Set up a search system (Hybrid RAG) using Qdrant and BGE-M3 to scan over 20 Arabic medical textbooks using OCR. It retrieves answers in under 30 milliseconds on local benchmarks.

Packaged the application with Docker. By using FastAPI and vLLM batching, the system easily handles 150 users at once at 25+ requests per second (P50 <7s / P95 <12s). We optimized GPU memory to successfully downgrade from an AWS g5.2xlarge ($1.212/hr) to a g4dn.2xlarge ($0.752/hr), saving about ~$330/month per instance while keeping a success rate of over 99.5%.

Used MLflow to track performance, improving answer quality (ROUGE-L went up 12% compared to the baseline) and keeping made-up answers (hallucinations) under 3%. The system is designed to always cite its sources and automatically refuses to answer questions outside its medical scope.

## System Architecture & Technical Deep-Dive

### 1. Dataset Generation & Fine-Tuning
* **Data Augmentation:** Used `gemini-3.1-flash-lite-preview` to clean up the raw Q&A pairs. Since the original model wasn't understanding Egyptian well, we prompted Gemini to expand short answers into accurate medical Arabic, heavily focused on the Egyptian dialect. We kept everything strictly formatted as JSON.
* **Supervised Fine-Tuning (SFT):** The Cohere Command R7B model was fine-tuned using PEFT/LoRA (rank=16, alpha=32, dropout=0.07) targeting attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). The training process used a cosine learning rate scheduler and gradient checkpointing to save memory.

### 2. Quantization (AWQ W8A16)
* **Calibration with Noisy OCR Data:** To make sure the compressed model stays accurate when reading messy text from the textbooks, the AWQ calibration dataset (`llmcompressor`) was built using 64 challenging search examples. These included simulated OCR noise, inverted words, and disjointed phrasing.
* **vLLM Integration:** The resulting 8-bit weights were formatted to work directly with vLLM. This enabled dynamic and continuous batching, using a `gpu_memory_utilization` of 0.6 to leave enough room for the embedding model.

### 3. OCR & Hybrid RAG Pipeline
* **Multithreaded OCR:** Medical textbooks (PDFs) were converted to images, cropped to remove headers and footers, and processed using `pytesseract` (ara+eng). A custom Python script using `ThreadPoolExecutor` processed chunks of 120-200 words, applying regex to clean up persistent Arabic encoding issues.
* **Dual Embedding Retrieval:** Text chunks were encoded using BAAI/bge-m3, which naturally supports 1024-dimensional dense vectors and lexical (sparse) weights. Both vector types were stored in Qdrant.
* **Reciprocal Rank Fusion (RRF):** When a user asks a question, the system fetches candidates using both dense and sparse searches, then merges the results using Qdrant's RRF fusion to make sure the retrieved documents are highly accurate and relevant.

### 4. Inference Optimization & Guardrails
* **Async FastAPI Backend:** The `AsyncLLMEngine` handles text generation. Heavy PyTorch operations (like embedding the query) and database I/O (Qdrant queries) are shifted to secondary threads using `run_in_threadpool` so the main application doesn't freeze.
* **Strict Citation Policy:** The system's hidden prompt forces the model to cite its sources explicitly (e.g., "بناءً على [اسم المصدر]، صفحة [رقم الصفحة]").
* **Output Parsing:** A regex-based cleanup tool (`extract_final_answer`) removes internal reasoning tags (like `<think>`) and makes sure the final text only contains Arabic characters for clean API responses.

## Knowledge Base (Medical Textbooks)
The RAG pipeline retrieves context from over 20 Arabic medical sources. Some of the books are:
* أساسيات علم الأعصاب في تفسير التعلم
* الدليل العلاجي في الطب الباطني
* الموسوعة الطبية الشاملة
* دليل الطب النفسي -- نخبة من أساتذة الطب النفسي بالجامعات العربية
* علم الأدوية والعلاج
* مدخل الى الصيدلة 
* *MedlinePlus Arabic Scraped PDFs*

## Example Query & Response

**User Query:**
> "عندي وجع فظيع في ضهري من ورا ونازل على الحوض، ولقيت نقط دم في البول، أعمل إيه؟"

**System Response:**
> بناءً على الدليل العلاجي في الطب الباطني، صفحة 210:
> الأعراض التي تصفها تشير بقوة إلى احتمال وجود حصوات في الكلى. هذه الحصوات تسبب ألماً شديداً يبدأ في الظهر ويمتد إلى أسفل الحوض، وغالباً ما يصاحبه ظهور دم في البول نتيجة احتكاك الحصوة بالمجرى البولي. كإجراء مبدئي، يُنصح بشرب كميات كبيرة من الماء، ويمكن تناول مسكنات للألم ومضادات للتشنج لتسهيل خروج الحصوة. يجب عليك استشارة طبيب مختص في أسرع وقت لإجراء الفحوصات اللازمة.

## How to Use It

### Prerequisites
* Docker and Docker Compose
* NVIDIA Container Toolkit (GPU Support)
* At least 16GB VRAM (24GB recommended for optimal batching)

### Deployment
1. Clone the repository and navigate to the deployment directory.
2. Start the container via Docker Compose. The `entrypoint.sh` script will automatically download the pre-compiled Qdrant vector database (~2.2GB) from HuggingFace if it is not found locally.

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

### API Usage
The service exposes a REST API at `http://localhost:8000`.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "ما هي الأعراض الجانبية لدواء الأوميبرازول؟"}'
```

**Expected JSON Response:**
```json
{
  "question": "ما هي الأعراض الجانبية لدواء الأوميبرازول؟",
  "answer": "بناءً على الدليل العلاجي في الطب الباطني، صفحة 104: يستخدم أوميبرازول كعلاج أساسي لقرحة المعدة. الأعراض الجانبية المحتملة تشمل...",
  "source": "الدليل العلاجي في الطب الباطني",
  "page": "104",
  "score": 0.854,
  "retrieved_chunks":[
    {
      "source": "الدليل العلاجي في الطب الباطني",
      "page": "104",
      "score": 0.854,
      "text": "..."
    }
  ]
}
```

## Dataset & License

* **Dataset:** The fine-tuning dataset containing 50,000 cleaned Arabic medical QA pairs (focused on the Egyptian dialect) is publicly available on HuggingFace:[Shams03/Ara-Egy-Medical-QA](https://huggingface.co/datasets/Shams03/Ara-Egy-Medical-QA).
* **License:** This project, including the code, model adapters, and dataset, is entirely **Free** and open-source.
