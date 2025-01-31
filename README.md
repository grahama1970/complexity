# 🧠 Classification Approaches: Non-Parametric (RAG) vs. Fine-Tuned Model

This repository demonstrates **two distinct approaches** for classifying questions as “Simple” vs. “Complex”:

1. **🔍 RAG-Style Embedding & Retrieval** (Non-Parametric)  
2. **🤖 Fine-Tuned DistilBERT** (Parametric)

Each approach provides its own **inference script** showing how to classify new questions.

---

## 1. 🔍 RAG-Style (Embedding & Retrieval) Approach

### 📝 Overview

- Uses [ModernBert](https://huggingface.co/nomic-ai/modernbert-embed-base) to **embed** each question.
- Stores these **embeddings** (with labels) in [ArangoDB](https://www.arangodb.com/).
- At inference time:
  1. Embed the new query (`search_query: <text>`).
  2. Retrieve the top neighbors via BM25 + `COSINE_SIMILARITY`.
  3. **Majority vote** to decide the label: “Simple” (0) or “Complex” (1).

### 🛠️ Script: `rag_classifier.py`

1. **⚡ Concurrent Embedding**:  
   - Loads the dataset (`wesley7137/question_complexity_classification`), filters invalid ratings.  
   - Embeds questions in **parallel** (threads + `tqdm`).  
2. **🗄️ ArangoDB Storage**:  
   - Creates a collection, inserts the `(embedding, label, question)`.  
   - Builds an ArangoSearch view to support text + vector queries.  
3. **🔮 Inference**:  
   - The script includes a **loop** over sample questions, measuring classification time and printing results in a table.

This approach is **adaptive**: if you want new classes or new data, you simply embed and store them. No need to retrain a final classifier head.

---

## 2. 🤖 Fine-Tuned DistilBERT Approach

### 📝 Overview

- Uses **DistilBertForSequenceClassification** with 2 output logits (“Simple” vs. “Complex”).
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

This approach can yield **higher accuracy** for stable sets of classes and can be **faster** at runtime (one forward pass per query). But adding new classes typically requires **re-training**.

---

## 3. ⚖️ Comparing the Two Approaches

| Aspect                    | RAG-Style Embedding (Non-Parametric)                               | Fine-Tuned DistilBERT (Parametric)                       |
|---------------------------|---------------------------------------------------------------------|-----------------------------------------------------------|
| **Adaptability**          | Highly adaptive; just insert new embeddings and labels              | Fixed set of classes; must retrain if labels change       |
| **Inference**             | Nearest-neighbor retrieval; can become slower with huge data        | Single forward pass; consistent inference time            |
| **Memory**                | Stores all embeddings in a DB                                       | Only the learned weights; no per-sample embedding storage |
| **Explainability**        | Transparent: nearest neighbors show “why”                           | Less transparent “black box” logits                       |
| **Typical Use**           | Evolving classes, easy updates, smaller/medium DB                   | Stable classes, large data, desire for maximum accuracy   |

---

## 4. 🚀 How to Run Each Approach

### A) RAG Approach (Embedding + Retrieval)

1. **📦 Install requirements** (including `arango`, `tqdm`, `transformers`, `datasets`, etc.).
2. **🐳 Have ArangoDB Running** on the configured host (default `http://localhost:8529`).  
   - Adjust credentials (`arango_username`, `arango_password`) in `rag.py`.
3. **▶️ Run**:
   ```bash
   python rag.py
   ```
   It will:
   - Load the dataset, embed it (concurrently), store in ArangoDB, and build a search view.
   - Then perform a **sample inference** loop on a few questions.

### B) Fine-Tuned DistilBERT

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

## 5. 🏁 Conclusion

This repository provides two **inference scripts** demonstrating two classification paradigms:

1. **🔍 Non-Parametric (RAG)**  
   - Great for dynamic, evolving label sets or smaller data.  
   - Explanatory with nearest neighbors.  

2. **🤖 Parametric (Fine-Tuned Model)**  
   - Stable label sets, large data, potentially higher accuracy.  
   - Single artifact, fast inference, but less flexible for new classes.

Feel free to experiment with both to see which fits your **speed**, **adaptability**, and **explainability** needs!


