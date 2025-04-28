# Plan: RAG Complexity Classifier with Local Embeddings & ArangoDB

**Goal:** Transition complexity classification from model training to a RAG approach using `modernbert` local embeddings, cosine similarity, majority vote, and storing data in the `memory_bank` ArangoDB database.

**Confirmed Details:**

*   Target ArangoDB Database: `memory_bank`
*   New ArangoDB Collection: `complexity`
*   New ArangoDB View: `complexity_view` (derived from collection name)

**Important Note:** All code modifications and creations must adhere to the project's Memory Bank rules and documentation standards found in `docs/memory_bank/`.
**Steps:**

1.  [X] **Environment Setup:**
    *   Add `sentence-transformers` to `requirements.txt`.
    *   Ensure necessary environment variables (like `ARANGO_HOST`, `ARANGO_USER`, `ARANGO_PASSWORD`, `ARANGO_DB_NAME=memory_bank`) are set, potentially in a `.env` file.

2.  [X] **Refactor `embedding_utils.py` (`server/src/complexity/utils/embedding_utils.py`):**
    *   Add a new function `get_local_embedding(text: str, model_name: str = "nomic-ai/modernbert-embed-base") -> List[float]` using `sentence-transformers`.
    *   Modify the `__main__` block to test `get_local_embedding` and `cosine_similarity`. Ensure it follows validation rules.

3.  [X] **Refactor `arango_setup.py` (`server/src/complexity/utils/arango_setup.py`):**
    *   Update configuration constants (or load from env vars) to use `ARANGO_DB_NAME='memory_bank'`, `COLLECTION_NAME='complexity'`, `VIEW_NAME='complexity_view'`.
    *   Remove `pdf_extractor` specific imports/logic if no longer needed.
    *   Verify `EMBEDDING_DIMENSIONS` matches `modernbert-embed-base` (768).
    *   Ensure `ensure_vector_index` uses the correct collection name (`complexity`), field (`embedding`), dimension (768), and metric (`cosine`).
    *   Ensure `ensure_arangosearch_view` uses the correct view (`complexity_view`) / collection (`complexity`) names and indexes the `question` (for BM25) and `embedding` (for vector search) fields appropriately.
    *   Update the `__main__` block to test setup for the `complexity` collection and `complexity_view`. Ensure validation.

4.  [ ] **Refactor `rag_classifer.py` (`server/src/complexity/rag/rag_classifer.py`):**
    *   Update configuration loading to use environment variables or a central config mechanism, matching the details used in `arango_setup.py` (`memory_bank`, `complexity`, `complexity_view`).
    *   Modify `ModernBertEmbedder` or its usage:
        *   Instantiate it using the correct model name (`nomic-ai/modernbert-embed-base`).
        *   Ensure its `embed_text` and `embed_batch` methods correctly apply the `search_document:` and `search_query:` prefixes.
    *   Remove the calls to `store_docs_in_arango` and `create_arangosearch_view` from `rag/database/arango_utils.py`. Assume `arango_setup.py` is run separately to prepare the DB.
    *   Keep the call to `initialize_database` from `rag/database/arango_utils.py` for connecting to the already set up DB.
    *   Ensure the AQL query loaded via `load_aql_query` uses the correct view name (`complexity_view`).

5.  [ ] **Update AQL Query (`server/src/complexity/rag/database/aql/rag_classifier.aql`):**
    *   Verify the AQL query references the correct view name (`complexity_view`) and field names (`question`, `embedding`).

6.  [ ] **Refactor/Cleanup `rag/database/arango_utils.py`:**
    *   Keep `initialize_database` (for connection).
    *   Remove `store_docs_in_arango` and `create_arangosearch_view` as their functionality is covered by the main `arango_setup.py`.

7.  [ ] **Testing and Validation:**
    *   Run `uv run server/src/complexity/utils/arango_setup.py` to ensure the database, collection (`complexity`), view (`complexity_view`), and index are created correctly in the `memory_bank` database. Check logs for success/validation messages.
    *   Run `uv run server/src/complexity/rag/rag_classifer.py`. Verify:
        *   It connects to the `memory_bank` DB.
        *   It embeds the dataset and stores documents in `complexity`.
        *   It successfully classifies the test questions using retrieval from `complexity_view`.
        *   The output table shows reasonable classifications and timings.

---

**Workflow Diagram:**

```mermaid
graph TD
    subgraph Setup Phase (Run arango_setup.py)
        direction LR
        S1[Load Config (memory_bank, complexity)] --> S2(Ensure DB: memory_bank);
        S2 --> S3(Ensure Collection: complexity);
        S3 --> S4(Ensure View: complexity_view);
        S4 --> S5(Ensure Vector Index on embedding);
        S5 --> S6[Setup Complete];
    end

    subgraph Indexing & Inference Phase (Run rag_classifer.py)
        direction TB
        I1[Load Config (memory_bank, complexity)] --> I2(Connect to DB: memory_bank);
        I3[Load Dataset (HF)] --> I4(Filter Dataset);
        I5[Init ModernBertEmbedder] --> I6{Embed Dataset (doc: prefix)};
        I4 --> I6;
        I6 -- Embedded Docs --> I7(Store Docs in complexity);
        I2 --> I7;
        I7 --> I8[Indexing Done];

        I9[Test Questions] --> I10{Embed Question (query: prefix)};
        I5 --> I10;
        I10 -- Query Embedding --> I11{Retrieve Neighbors (AQL on complexity_view)};
        I2 --> I11;
        I11 -- Top K Docs --> I12{Majority Vote};
        I12 --> I13[Classify: Simple/Complex];
        I13 --> I14[Log Results Table];
    end

    S6 --> I2;