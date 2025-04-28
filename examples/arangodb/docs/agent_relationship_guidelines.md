# Agent Relationship Guidelines

## Overview
This guide explains how to create useful relationships between documents in the PDF Extractor to improve query answering.

## When to Create Relationships
Create relationships when:
1. Search results are poor (<3 results or low RRF scores).
2. Query needs information from multiple documents.
3. Relationships will help future queries (e.g., connecting problems to solutions).

## Decision Process
1. Run hybrid search (`hybrid_search`).
2. If results are poor, use existing relationships via graph traversal.
3. If still poor, create new relationships.

## Good Relationships
- **SIMILAR**: Documents with similar content (e.g., both discuss query optimization).
- **SHARED_TOPIC**: Documents with shared tags (e.g., both tagged "database").
- **REFERENCES**: Doc A cites Doc B (e.g., mentions its key or title).
- **PREREQUISITE**: Doc A is needed to understand Doc B (e.g., basic vs. advanced).
- **CAUSAL**: Doc A's concept causes Doc B's outcome (e.g., configuration causes performance).

## Bad Relationships
- Based only on shared keywords without deeper connection.
- Redundant with search results.
- Too vague or speculative.

## Assessing Relationships
- **Rationale**: Provide a clear explanation (≥50 characters) of why the documents are related. Use Perplexity (https://perplexity.ai) to research connections if needed. Example: "Doc1 discusses query optimization, and Doc2 provides advanced indexing techniques, both relevant to performance."
- **Confidence Score**:
  - 1: Essential connection (e.g., problem-solution pair).
  - 2: Strong connection (e.g., clear shared topic).
  - 3: Helpful connection (e.g., likely related).
  - 4: Weak connection (e.g., possible relation).
  - 5: Uncertain connection (e.g., speculative).
- **Process**:
  1. Review document content and tags.
  2. Check query context.
  3. Use Perplexity to verify connections if unsure.
  4. Write a specific rationale and assign a score.

## Testing
- Verify relationships improve query results (more results or better relevance).
- Ensure rationales are clear and ≥50 characters.
- Confirm confidence scores reflect connection strength (1=best, 5=worst).
