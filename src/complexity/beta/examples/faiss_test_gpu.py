import faiss


import numpy as np
dim = 1024
index = faiss.IndexFlatIP(dim)
print(faiss.get_num_gpus())

embeddings = np.random.random((1000, dim)).astype(np.float32)
faiss.normalize_L2(embeddings)
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print("GPU index created")
else:
    print("CPU index used")

index.add(embeddings)