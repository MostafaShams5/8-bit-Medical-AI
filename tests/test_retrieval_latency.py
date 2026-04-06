import time
import statistics
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from FlagEmbedding import BGEM3FlagModel


embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 
client = QdrantClient(path="../medical_qdrant_db")
COLLECTION_NAME = "arabic_medical_HybridRAG"

queries = [
    "ما هي الأعراض الجانبية لدواء الأوميبرازول؟",
    "علاج انخفاض السكر في الدم",
    "أسباب طنين الأذن وارتفاع ضغط الدم",
    "جرعة الباراسيتامول للأطفال",
    "الفرق بين القرحة وارتجاع المريء"
]

latencies = []
db_only_latencies = []

# warming it
embedding_model.encode("تجربة", return_dense=True, return_sparse=True)

for query in queries:
    start_total = time.perf_counter()
    
    embeddings = embedding_model.encode(query, return_dense=True, return_sparse=True)
    sparse_dict = embeddings["lexical_weights"]
    sparse_vec = qdrant_models.SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()]
    )
    
    start_db = time.perf_counter()
    
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            qdrant_models.Prefetch(query=embeddings["dense_vecs"].tolist(), using="dense", limit=8),
            qdrant_models.Prefetch(query=sparse_vec, using="sparse", limit=8),
        ],
        query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
        limit=2
    )
    
    end_time = time.perf_counter()
    
    total_ms = (end_time - start_total) * 1000
    db_ms = (end_time - start_db) * 1000
    
    latencies.append(total_ms)
    db_only_latencies.append(db_ms)
    
    print(f"Query: '{query}' -> Total: {total_ms:.2f}ms | DB Only: {db_ms:.2f}ms")

print(f"Average Qdrant DB Latency: {statistics.mean(db_only_latencies):.2f} ms")
print(f"Average Total Retrieval Latency (Embedding + DB): {statistics.mean(latencies):.2f} ms")
