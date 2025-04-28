

# ARANGODB TROUBLESHOOTING

_Last updated: 2025-04-28_

This guide consolidates common issues, error messages, and solutions when working with ArangoDB, especially focused on vector search, AQL queries, indexing, and search API scripting.

---

## Table of Contents

- [1. General Troubleshooting](#1-general-troubleshooting)
- [2. Vector Search Issues](#2-vector-search-issues)
  - [2.1. Vector Index Creation](#21-vector-index-creation)
  - [2.2. Query Structure: FILTER Placement](#22-query-structure-filter-placement)
  - [2.3. Data and Index Mismatch](#23-data-and-index-mismatch)
  - [2.4. nLists and Resource Limits](#24-nlists-and-resource-limits)
  - [2.5. Error 1554: "AQL: failed vector search"](#25-error-1554-aql-failed-vector-search)
  - [2.6. Version and Feature Readiness](#26-version-and-feature-readiness)
- [3. AQL and Indexing](#3-aql-and-indexing)
- [4. Debugging and Logs](#4-debugging-and-logs)
- [5. Minimal Dataset Testing](#5-minimal-dataset-testing)
- [6. Reporting Bugs](#6-reporting-bugs)
- [7. ArangoSearch BM25 and Glossary Search Scripts](#7-arangosearch-bm25-and-glossary-search-scripts)
- [8. Standard python-arango CRUD Operations](#8-standard-python-arango-crud-operations)
- [9. Additional Resources](#9-additional-resources)

---

## 1. General Troubleshooting

- **Check server logs** (`arangod.log`) for detailed error messages and stack traces.
- **Validate configuration** and ensure your ArangoDB deployment is healthy.
- **Cluster health:** Verify all components are up and consistent if using a cluster.

---

## 2. Vector Search Issues

### 2.1. Vector Index Creation

- Always **create the vector index after data is loaded** to ensure all documents are indexed.
- Index parameters must exactly match your data:
  - `dimension` = embedding vector length (e.g., 768).
  - `metric` = similarity metric (`cosine`, `l2`, etc.).
- **Index name is arbitrary and does not affect query execution.**

### 2.2. Query Structure: FILTER Placement

- **FILTER conditions on non-embedding fields must come *after* the vector search, SORT, and LIMIT.**
- Example of correct post-filtering:
  ```aql
  FOR doc IN collection
    LET score = APPROX_NEAR_COSINE(doc.embedding, @query_emb)
    SORT score DESC
    LIMIT 100
    FILTER doc.category == @category
    RETURN doc
  ```
- Placing FILTER before vector search disables index usage and may cause errors.

### 2.3. Data and Index Mismatch

- Ensure **all documents have valid embeddings**:
  - Embeddings must be numeric arrays of the correct length.
  - No `null`, `NaN`, or `Infinity` values.
- Use AQL to check:
  ```aql
  FOR doc IN collection
    FILTER !HAS(doc, "embedding") OR LENGTH(doc.embedding) != 768
    RETURN doc._key
  ```
  ```aql
  FOR doc IN collection
    FILTER LENGTH(
      FOR v IN doc.embedding
        FILTER v == null OR !IS_NUMBER(v) OR v > 1e10 OR v  0
    RETURN doc._key
  ```

### 2.4. nLists and Resource Limits

- Choose `nLists` based on dataset size:
  -  @min_age
  RETURN doc
"""
cursor = db.aql.execute(aql, bind_vars={"min_age": 20})
for doc in cursor:
    print(doc)

# Bulk insert documents
docs = [
    {"_key": "doc2", "name": "Bob", "age": 25},
    {"_key": "doc3", "name": "Carol", "age": 27},
]
collection.insert_many(docs)
```

---

## 9. Additional Resources

- [ArangoDB Official Documentation](https://www.arangodb.com/docs/)
- [ArangoDB Vector Search Guide](https://arangodb.com/2024/11/vector-search-in-arangodb-practical-insights-and-hands-on-examples/)
- [ArangoDB GitHub Issues](https://github.com/arangodb/arangodb/issues)
- [ArangoDB Community Forum](https://community.arangodb.com/)
- [python-arango Driver Documentation](https://python-arango.readthedocs.io/en/latest/)

---

# Summary

- Vector search in ArangoDB is powerful but still experimental; expect rough edges.
- Proper query structure, especially **post-filtering**, is critical.
- Data and index consistency is essential.
- Use minimal datasets and logs to isolate issues.
- Use the python-arango driver for robust CRUD and query operations.
- Consider dedicated vector DBs if production stability is paramount.

---

If you want, I can help you generate a minimal reproducible example or assist with bug report drafting for ArangoDB.

---

