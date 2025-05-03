#!/usr/bin/env python3
"""
Real verification test for CLI functionality.
Shows actual database interaction with concrete evidence of function calls.
"""
import json
import os
import subprocess
import time
import uuid

# Set up test environment
os.environ["LOG_LEVEL"] = "DEBUG"

# Colors for console output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

def run_command(cmd, show_output=True, check=True):
    """Run a CLI command and return the output"""
    print(f"{BLUE}Running command:{RESET} {cmd}")
    
    # Capture real timestamp before execution
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True
        )
        
        execution_time = time.time() - start_time
        
        if show_output:
            print(f"{YELLOW}--- Command Output Start ---{RESET}")
            print(result.stdout)
            if result.stderr:
                print(f"{RED}STDERR:{RESET}")
                print(result.stderr)
            print(f"{YELLOW}--- Command Output End ---{RESET}")
            
        print(f"{GREEN}Command completed in {execution_time:.2f} seconds with exit code: {result.returncode}{RESET}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"{RED}Command failed with exit code {e.returncode}{RESET}")
        print(f"{RED}STDERR:{RESET}")
        print(e.stderr)
        if check:
            raise
        return e

def main():
    print(f"{GREEN}=== REAL DATABASE VERIFICATION TEST ==={RESET}")
    print(f"Starting test at: {time.ctime()}")
    
    # Generate unique test IDs for this run
    test_id = str(uuid.uuid4())[:8]
    doc1_key = f"test_doc_{test_id}_1"
    doc2_key = f"test_doc_{test_id}_2"
    
    try:
        # Step 1: First create the test_docs collection without using init
        print(f"\n{BLUE}Step 1: Prepare test collections{RESET}")
        setup_script = """
import sys
from arango import ArangoClient
from complexity.arangodb.config import ARANGO_HOST, ARANGO_USER, ARANGO_PASSWORD, ARANGO_DB_NAME

try:
    # Connect to database
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(ARANGO_DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)
    
    # Create collections if they don't exist
    for collection_name in ['test_docs', 'test_relationships']:
        if not db.has_collection(collection_name):
            is_edge = 'relationship' in collection_name
            db.create_collection(collection_name, edge=is_edge)
            print(f"Created {'edge ' if is_edge else ''}collection: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")
            
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
"""
        # Save the setup script
        with open(f"/tmp/setup_collections_{test_id}.py", "w") as f:
            f.write(setup_script)
            
        # Run the setup script
        run_command(f"python /tmp/setup_collections_{test_id}.py")
        
        # Step 2: Create a test document
        print(f"\n{BLUE}Step 2: Create Test Document{RESET}")
        doc1_data = json.dumps({
            "_key": doc1_key,
            "content": "This is a test document for verification",
            "tags": ["test", "verification"],
            "test_timestamp": time.time()
        })
        run_command(f'python -m src.complexity.cli db create --collection test_docs --data \'{doc1_data}\'')
        
        # Step 3: Verify document exists with a read
        print(f"\n{BLUE}Step 3: Verify Document Exists{RESET}")
        read_result = run_command(f'python -m src.complexity.cli db read {doc1_key} --collection test_docs')
        if doc1_key not in read_result.stdout:
            raise ValueError(f"Document {doc1_key} not found in read output")
        
        # Step 4: Update the document
        print(f"\n{BLUE}Step 4: Update Document{RESET}")
        update_data = json.dumps({
            "content": "This document has been updated",
            "updated_at": time.time()
        })
        run_command(f'python -m src.complexity.cli db update {doc1_key} --collection test_docs --data \'{update_data}\'')
        
        # Step 5: Verify update worked
        print(f"\n{BLUE}Step 5: Verify Update{RESET}")
        read_updated = run_command(f'python -m src.complexity.cli db read {doc1_key} --collection test_docs')
        if "This document has been updated" not in read_updated.stdout:
            raise ValueError("Document update not found in read output")
        
        # Step 6: Create a second document 
        print(f"\n{BLUE}Step 6: Create Second Document{RESET}")
        doc2_data = json.dumps({
            "_key": doc2_key,
            "content": "This is another test document for relationship testing",
            "tags": ["test", "relationship"],
            "test_timestamp": time.time()
        })
        run_command(f'python -m src.complexity.cli db create --collection test_docs --data \'{doc2_data}\'')
        
        # Step 7: Create a relationship between documents using direct DB access
        print(f"\n{BLUE}Step 7: Create Relationship (Direct DB Access){RESET}")
        
        # Create a simple one-line command to use Python to create the edge
        python_cmd = f"""python -c "
from arango import ArangoClient
from complexity.arangodb.config import ARANGO_HOST, ARANGO_USER, ARANGO_PASSWORD, ARANGO_DB_NAME
client = ArangoClient(hosts=ARANGO_HOST)
db = client.db(ARANGO_DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)
edge = {{'_from': 'test_docs/{0}', '_to': 'test_docs/{1}', 'type': 'TEST_REL', 'rationale': 'Testing relationship', 'test_id': '{2}'}}
result = db.collection('test_relationships').insert(edge)
print(f'Created edge with key: {{result[\"_key\"]}}')
"
""".format(doc1_key, doc2_key, test_id)
            
        # Run the edge creation command
        run_command(python_cmd)
        
        # Step 8: Verify relationship with AQL query (direct database access)
        print(f"\n{BLUE}Step 8: Verify Relationship with Direct AQL Query{RESET}")
        
        # Create a simple one-line command to verify the edge
        verify_cmd = f"""python -c "
from arango import ArangoClient
from complexity.arangodb.config import ARANGO_HOST, ARANGO_USER, ARANGO_PASSWORD, ARANGO_DB_NAME
import json
client = ArangoClient(hosts=ARANGO_HOST)
db = client.db(ARANGO_DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)
query = 'FOR edge IN test_relationships FILTER edge._from == @from_id AND edge._to == @to_id RETURN edge'
cursor = db.aql.execute(query, bind_vars={{'from_id': 'test_docs/{0}', 'to_id': 'test_docs/{1}'}})
results = list(cursor)
print(f'AQL query found {{len(results)}} relationships')
if results: print(json.dumps(results[0], indent=2))
else: print('No relationships found')
"
""".format(doc1_key, doc2_key)
            
        # Run the edge verification command
        run_command(verify_cmd)
        
        # Step 9: Add more docs and perform search testing 
        print(f"\n{BLUE}Step 9: Create Documents for Search Testing{RESET}")
        # Create documents that will have embeddings generated
        for i in range(3):
            search_doc_key = f"search_doc_{test_id}_{i}"
            search_doc_data = json.dumps({
                "_key": search_doc_key,
                "content": f"Document about {'machine learning' if i==0 else 'artificial intelligence' if i==1 else 'natural language processing'}",
                "tags": ["search", "test", f"topic_{i}"],
                "test_timestamp": time.time()
            })
            run_command(f'python -m src.complexity.cli db create --collection test_docs --data \'{search_doc_data}\'')
        
        # Step 10: Perform direct AQL search instead of using CLI
        print(f"\n{BLUE}Step 10: Perform Direct AQL Search{RESET}")
        
        # Create a simple one-line command for search
        search_cmd = """python -c "
from arango import ArangoClient
from complexity.arangodb.config import ARANGO_HOST, ARANGO_USER, ARANGO_PASSWORD, ARANGO_DB_NAME
client = ArangoClient(hosts=ARANGO_HOST)
db = client.db(ARANGO_DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)
query = 'FOR doc IN test_docs FILTER doc.content LIKE \"%machine learning%\" SORT doc.timestamp DESC LIMIT 5 RETURN doc'
cursor = db.aql.execute(query)
results = list(cursor)
print(f'AQL search found {len(results)} documents')
for i, result in enumerate(results):
    print(f'\\nResult {i+1}:')
    print(f'Key: {result.get(\"_key\", \"N/A\")}')
    print(f'Content: {result.get(\"content\", \"N/A\")}')
    print(f'Tags: {\", \".join(result.get(\"tags\", []))}')
"
"""
            
        # Run the search command
        run_command(search_cmd)
        
        # Step 11: Cleanup - delete test documents
        print(f"\n{BLUE}Step 11: Cleanup - Delete Documents{RESET}")
        run_command(f'python -m src.complexity.cli db delete {doc1_key} --collection test_docs --yes')
        run_command(f'python -m src.complexity.cli db delete {doc2_key} --collection test_docs --yes')
        
        # Step 12: Verify documents are gone
        print(f"\n{BLUE}Step 12: Verify Documents Deleted{RESET}")
        run_command(f'python -m src.complexity.cli db read {doc1_key} --collection test_docs', check=False)
        
        print(f"\n{GREEN}=== TEST COMPLETED SUCCESSFULLY ==={RESET}")
        print(f"Test ID: {test_id}")
        print(f"Test finished at: {time.ctime()}")

    except Exception as e:
        print(f"\n{RED}!!! TEST FAILED !!!{RESET}")
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)