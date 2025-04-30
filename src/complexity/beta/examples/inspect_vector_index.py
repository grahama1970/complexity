from arango import ArangoClient


def inspect_vector_configuration(db_name, collection_name, embedding_field):
    """Check if vector search is properly configured in ArangoDB."""
    # Initialize the client
    client = ArangoClient(hosts='http://localhost:8529')

    # Connect to the database
    db = client.db('_system', username='root', password='openSesame')
    if not db.has_database(db_name):
        print(f"Database '{db_name}' does not exist.")
        return
    db = client.db(db_name, username='root', password='openSesame')
 
    print(f"ArangoDB Version: {db.version()}")
    
    # Check collection
    collection = db.collection(collection_name)
    print(f"Collection exists: {collection.name == collection_name}")
    
    # Check indexes
    indexes = list(collection.indexes())
    print(f"Found {len(indexes)} indexes on collection")
    
    vector_indexes = [idx for idx in indexes if idx.get("type") == "vector"]
    print(f"Found {len(vector_indexes)} vector indexes")
    
    for idx in vector_indexes:
        print(f"Vector index details:")
        print(f"  - Fields: {idx.get('fields', [])}")
        print(f"  - Parameters: {idx.get('params', {})}")
    
    # Check if vectors exist and have the right format
    sample_doc = db.aql.execute(f"FOR doc IN {collection_name} LIMIT 1 RETURN doc").next()
    vector_exists = embedding_field in sample_doc
    print(f"Vector field exists in documents: {vector_exists}")
    
    if vector_exists:
        vector = sample_doc[embedding_field]
        print(f"Vector type: {type(vector)}")
        print(f"Vector length: {len(vector)}")
        print(f"First few elements: {vector[:5]}")

if __name__ == "__main__": 
    inspect_vector_configuration(
        "memory_bank",
        "complexity",
        "embedding"
    )