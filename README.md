# 🧠 Classification Approaches: Non-Parametric (RAG) vs. Fine-Tuned Model

This repository demonstrates **two distinct approaches** for classifying questions as "Simple" vs. "Complex":

1. **🔍 RAG-Style Embedding & Retrieval** (Non-Parametric)  
2. **🤖 Fine-Tuned DistilBERT** (Parametric)

Each approach provides its own **inference script** showing how to classify new questions.

---

## 1. 🔍 RAG-Style (Embedding & Retrieval) Approach

### 📝 Overview

- Uses [BGE Embeddings](https://huggingface.co/BAAI/bge-large-en-v1.5) to **embed** each question (upgraded from ModernBert).
- Stores these **embeddings** (with labels) in [ArangoDB](https://www.arangodb.com/).
- At inference time:
  1. Embed the new query.
  2. Retrieve the top neighbors via `COSINE_SIMILARITY`.
  3. Use **weighted majority vote** to decide the label: "Simple" (0) or "Complex" (1).

### 🛠️ Script: `rag_classifier.py`

1. **⚡ Efficient Embedding**:  
   - Loads the dataset (`wesley7137/question_complexity_classification`), filters invalid ratings.
   - Embeds questions in **batches** for optimal GPU utilization.
   - Implements **caching** to avoid redundant embedding computations.
2. **🗄️ ArangoDB Storage**:  
   - Creates a collection with proper 1024-dimensional vector index.
   - Builds an ArangoSearch view for efficient vector similarity search.
3. **🧪 Inference Optimizations**:
   - **Pre-computes** embeddings before evaluation to reduce latency.
   - Uses **weighted voting** with exponential weighting to prioritize closer matches.
   - Configurable k-values for finding the optimal number of neighbors.

This approach is **adaptive**: if you want new classes or new data, you simply embed and store them. No need to retrain a final classifier head. With the BGE embeddings, this approach can achieve accuracy very close to trained models.

---

## 2. 🤖 Fine-Tuned DistilBERT Approach

### 📝 Overview

- Uses **DistilBertForSequenceClassification** with 2 output logits ("Simple" vs. "Complex").
- **Trains** on your dataset, splitting into train/val/test (80/10/10).
- Evaluates on the test set and **saves** the best model to `OUTPUT_DIR`.

### 🛠️ Script: `train_model.py`

1. **📥 Data Loading**:  
   - Pulls the same dataset from Hugging Face.  
   - Tokenizes and assigns labels (1 if rating > 0.5, else 0).
2. **🏋️ Trainer Setup**:  
   - Standard Hugging Face `Trainer` with early stopping and logging to TensorBoard.
3. **🔮 Inference**:  
   - Loads the best model from `OUTPUT_DIR`.  
   - Classifies new questions by a straightforward `argmax` of the output logits.

This approach traditionally yields **higher accuracy** for stable sets of classes and can be **faster** at runtime (one forward pass per query). But adding new classes typically requires **re-training**.

---

## 3. ⚖️ Comparing the Two Approaches

| Aspect                    | RAG-Style Embedding (Non-Parametric)                               | Fine-Tuned DistilBERT (Parametric)                       |
|---------------------------|---------------------------------------------------------------------|-----------------------------------------------------------|
| **Accuracy**              | Nearly comparable to fine-tuned models with BGE embeddings           | Slightly higher accuracy but requires retraining          |
| **Adaptability**          | Highly adaptive; just insert new embeddings and labels              | Fixed set of classes; must retrain if labels change       |
| **Inference**             | Optimized with batching and caching                                 | Single forward pass; consistent inference time            |
| **Memory**                | Stores all embeddings in a DB                                       | Only the learned weights; no per-sample embedding storage |
| **Explainability**        | Transparent: nearest neighbors show "why"                           | Less transparent "black box" logits                       |
| **Maintenance**           | No retraining needed - just add new examples                        | Requires periodic retraining for new data                 |
| **Typical Use**           | Evolving classes, easy updates, smaller/medium DB                   | Stable classes, large data, desire for maximum accuracy   |

---

## 4. 🚀 How to Run Each Approach

### A) RAG Approach (Embedding + Retrieval)

1. **📦 Install requirements** (including `arango`, `tqdm`, `transformers`, `datasets`, etc.).
2. **🐳 Have ArangoDB Running** on the configured host (default `http://localhost:8529`).  
   - Adjust credentials in `config.py`.
3. **▶️ Run**:
   ```bash
   python evaluation.py
   ```
   It will:
   - Load the dataset, embed it using the BGE model, store in ArangoDB with proper vector index.
   - Run evaluation comparing semantic search performance at different k-values.
   - Generate a detailed report comparing RAG and fine-tuned model approaches.

### C) Standalone Module Validation (New)

Most core modules now include a self-validation check within an `if __name__ == "__main__":` block. This allows you to quickly verify the basic functionality of a module independently.

1. **📦 Ensure requirements are installed**: `uv sync`
2. **▶️ Run validation for a specific module**:
   Use the `uv run python -m <module_path>` pattern. Replace `<module_path>` with the Python path to the module (e.g., `src.complexity.beta.utils.arango_setup`).

   ```bash
   # Example: Validate the ArangoDB setup module
   uv run python -m src.complexity.beta.utils.arango_setup

   # Example: Validate the RAG classifier module
   uv run python -m src.complexity.beta.rag.rag_classifier

   # Example: Validate the relationship builder module
   uv run python -m src.complexity.beta.utils.relationship_builder
   ```
   The script will print `✅ VALIDATION COMPLETE` and exit with code 0 on success, or `❌ VALIDATION FAILED` with details and exit code 1 on failure. Some validations require ArangoDB to be running and potentially populated with data.

### D) Fine-Tuned DistilBERT

1. **📦 Install requirements** (including `transformers`, `datasets`).
2. **▶️ Run**:
   ```bash
   python train_model.py
   ```
   It will:
   - Download the dataset, split/filter it.
   - Fine-tune DistilBERT (`num_labels=2`) for N epochs.
   - Evaluate and save the best model to `OUTPUT_DIR`.
3. **🔮 Inference**:
   - Load the saved model:
     ```python
     from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
     model = DistilBertForSequenceClassification.from_pretrained("./model")
     tokenizer = DistilBertTokenizerFast.from_pretrained("./model")
     
     inputs = tokenizer("Is splitting an atom simple?", return_tensors="pt")
     outputs = model(**inputs)
     label_id = outputs.logits.argmax(-1).item()
     label = "Complex" if label_id == 1 else "Simple"
     print(label)
     ```
   - You can adapt the example code in the script or your own script for batch inference.

---

## 5. 🚀 Performance Optimizations

The RAG approach has been significantly optimized:

1. **🔄 Upgraded Embeddings**: Switched from ModernBert to BGE embeddings for better semantic representation.
2. **📦 Batch Processing**: All embeddings are generated in batches for optimal GPU utilization.
3. **💾 Embedding Cache**: Implemented caching to avoid regenerating embeddings for the same text.
4. **⏱️ Pre-computation**: All test embeddings are pre-computed before evaluation begins.
5. **🧮 Weighted Voting**: Exponential weighting gives higher importance to closer matches.

These optimizations significantly improve both accuracy and performance, making the RAG approach very competitive with fine-tuned models while maintaining its adaptability advantages.

---

## 6. 🏁 Conclusion

Both approaches offer strong solutions, with different tradeoffs:

1. **🔍 Non-Parametric (RAG)**  
   - Excellent for dynamic, evolving label sets or datasets that grow over time.
   - With BGE embeddings and optimizations, accuracy is now very close to fine-tuned models.
   - No retraining required - simply add new examples to improve performance.

2. **🤖 Parametric (Fine-Tuned Model)**  
   - Good for stable label sets and requirements for maximum accuracy.
   - Single artifact, fast inference, but less flexible for new classes.

Our evaluation shows that the accuracy gap between these approaches has narrowed significantly with the BGE embedding model, making the RAG approach an excellent choice for many real-world applications.