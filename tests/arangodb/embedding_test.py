#!/usr/bin/env python3
"""
Test script for verifying document creation with embeddings.

This script tests document creation with proper embedding generation in the CLI:
- Creates documents with text that should be embedded
- Verifies the embedding field is present and has the right format
- Updates documents and checks that embeddings are updated
"""

import sys
import os
import json
import subprocess
import uuid
from datetime import datetime, timezone
import numpy as np

# Test document content
def get_test_doc():
    """Generate a test document with unique key and content for embedding"""
    return {
        "_key": f"embed_test_{uuid.uuid4().hex[:8]}",
        "title": "Embedding Test Document",
        "content": f"This is a test document for embedding verification. It contains enough text to generate a meaningful embedding vector. This text should be processed by the embedding model to create a vector representation. Generated at {datetime.now().isoformat()}",
        "tags": ["test", "embedding", "vector"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Output formatting
def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_command(cmd):
    print(f"\n> {cmd}")

def print_result(result, success=True):
    if success:
        print(f"\n✅ SUCCESS: {result}")
    else:
        print(f"\n❌ ERROR: {result}")

def run_command(cmd):
    """Run a command and return its output"""
    print_command(cmd)
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip(), True
    except subprocess.CalledProcessError as e:
        return e.stderr.strip() or e.stdout.strip(), False

# Test functions
def test_init():
    """Test database initialization"""
    print_header("Testing Database Initialization")
    # Use the cli_with_embeddings script which has both the uuid import and embedding support
    cmd = "PYTHONPATH=$PYTHONPATH:/home/graham/workspace/experiments python /home/graham/workspace/experiments/complexity/src/complexity/cli_with_embeddings.py init"
    output, success = run_command(cmd)
    if success:
        print_result("Database initialized successfully")
    else:
        print_result(f"Database initialization failed: {output}", False)
        sys.exit(1)
    return success

def test_create_with_embedding():
    """Test document creation with embedding generation"""
    print_header("Testing Document Creation with Embedding")
    
    # Create a test document
    test_doc = get_test_doc()
    doc_key = test_doc["_key"]
    
    # Write document to temp file
    temp_file = f"/tmp/embed_doc_{uuid.uuid4().hex[:8]}.json"
    with open(temp_file, "w") as f:
        json.dump(test_doc, f)
    
    # Run create command with the embedding-enhanced CLI script
    # For collections that should have embeddings
    cmd = f"PYTHONPATH=$PYTHONPATH:/home/graham/workspace/experiments python /home/graham/workspace/experiments/complexity/src/complexity/cli_with_embeddings.py db create --collection messages --data-file {temp_file} --json-output"
    output, success = run_command(cmd)
    
    # Clean up temp file
    os.unlink(temp_file)
    
    if success:
        try:
            result = json.loads(output)
            created_key = result.get("_key")
            print_result(f"Document created with key: {created_key}")
            
            # Now check if the document has an embedding
            read_cmd = f"PYTHONPATH=$PYTHONPATH:/home/graham/workspace/experiments python /home/graham/workspace/experiments/complexity/src/complexity/cli_with_embeddings.py db read {created_key} --collection messages"
            read_output, read_success = run_command(read_cmd)
            
            if read_success:
                try:
                    doc_result = json.loads(read_output)
                    # Check for embedding field
                    embedding = doc_result.get("embedding")
                    if embedding is not None:
                        # Verify this is a properly formatted embedding
                        if isinstance(embedding, list) and len(embedding) > 0:
                            print_result(f"Embedding generated successfully with {len(embedding)} dimensions")
                            return doc_key, created_key, True
                        else:
                            print_result(f"Embedding has incorrect format: {type(embedding)}", False)
                            return doc_key, created_key, False
                    else:
                        print_result("Document does not contain an embedding field", False)
                        return doc_key, created_key, False
                except json.JSONDecodeError:
                    print_result(f"Invalid JSON in document read response: {read_output}", False)
                    return doc_key, created_key, False
            else:
                print_result(f"Failed to read created document: {read_output}", False)
                return doc_key, created_key, False
        except json.JSONDecodeError:
            print_result(f"Invalid JSON in creation response: {output}", False)
            return None, None, False
    else:
        print_result(f"Document creation failed: {output}", False)
        return None, None, False

def test_update_with_embedding(doc_key):
    """Test document update with embedding update"""
    print_header("Testing Document Update with Embedding")
    
    if not doc_key:
        print_result("Skipping update test: No valid document key", False)
        return False
    
    # Create update data with new text content that should trigger a new embedding
    update_data = {
        "title": "Updated Embedding Test Document",
        "content": f"This content has been modified and should generate a new embedding vector. The embedding model should process this text and create a different vector representation. Updated at {datetime.now().isoformat()}",
        "updated": True,
        "update_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Write update to temp file
    temp_file = f"/tmp/update_embed_{uuid.uuid4().hex[:8]}.json"
    with open(temp_file, "w") as f:
        json.dump(update_data, f)
    
    # Run update command with the embedding-enhanced CLI script
    cmd = f"PYTHONPATH=$PYTHONPATH:/home/graham/workspace/experiments python /home/graham/workspace/experiments/complexity/src/complexity/cli_with_embeddings.py db update {doc_key} --collection messages --data-file {temp_file} --json-output"
    output, success = run_command(cmd)
    
    # Clean up temp file
    os.unlink(temp_file)
    
    if success:
        # Read the document to get the embedding
        read_cmd = f"PYTHONPATH=$PYTHONPATH:/home/graham/workspace/experiments python /home/graham/workspace/experiments/complexity/src/complexity/cli_with_embeddings.py db read {doc_key} --collection messages"
        read_output, read_success = run_command(read_cmd)
        
        if read_success:
            try:
                doc_result = json.loads(read_output)
                # Check for embedding field
                embedding = doc_result.get("embedding")
                if embedding is not None:
                    # Verify this is a properly formatted embedding
                    if isinstance(embedding, list) and len(embedding) > 0:
                        print_result(f"Embedding updated successfully with {len(embedding)} dimensions")
                        return True
                    else:
                        print_result(f"Updated embedding has incorrect format: {type(embedding)}", False)
                        return False
                else:
                    print_result("Updated document does not contain an embedding field", False)
                    return False
            except json.JSONDecodeError:
                print_result(f"Invalid JSON in document read response after update: {read_output}", False)
                return False
        else:
            print_result(f"Failed to read updated document: {read_output}", False)
            return False
    else:
        print_result(f"Document update failed: {output}", False)
        return False

def test_document_cleanup(doc_key):
    """Clean up by deleting the test document"""
    print_header("Cleaning Up Test Document")
    
    if not doc_key:
        print_result("Skipping cleanup: No valid document key", False)
        return False
    
    # Run delete command with --yes to skip confirmation
    cmd = f"PYTHONPATH=$PYTHONPATH:/home/graham/workspace/experiments python /home/graham/workspace/experiments/complexity/src/complexity/cli_with_embeddings.py db delete {doc_key} --collection messages --yes --json-output"
    output, success = run_command(cmd)
    
    if success:
        print_result(f"Test document deleted successfully: {doc_key}")
        return True
    else:
        print_result(f"Test document deletion failed: {output}", False)
        return False

# Main test sequence
def run_tests():
    """Run all embedding tests in sequence"""
    test_init()
    original_key, created_key, has_embedding = test_create_with_embedding()
    
    if has_embedding:
        test_update_with_embedding(created_key)
        test_document_cleanup(created_key)

if __name__ == "__main__":
    print_header("Document Embedding Test Script")
    run_tests()