def test_basic_cosine_similarity(db, collection_name, embedding_field):
    """Test a basic cosine similarity query between two documents."""
    try:
        # Get two documents
        aql = f"""
        FOR doc IN {collection_name}
            LIMIT 2
            RETURN {{ _id: doc._id, question: doc.question, embedding: doc.{embedding_field} }}
        """
        cursor = db.aql.execute(aql)
        docs = list(cursor)
        
        if len(docs) < 2:
            print("Not enough documents for test")
            return
            
        # Test cosine similarity between them
        aql = f"""
        RETURN COSINE_SIMILARITY(@vec1, @vec2)
        """
        cursor = db.aql.execute(aql, bind_vars={
            "vec1": docs[0]["embedding"], 
            "vec2": docs[1]["embedding"]
        })
        similarity = cursor.next()
        print(f"Cosine similarity between two docs: {similarity}")
        
        # Now test it in a more structured query
        aql = f"""
        LET doc1 = DOCUMENT("{docs[0]['_id']}")
        LET doc2 = DOCUMENT("{docs[1]['_id']}")
        RETURN {{
            doc1_id: doc1._id,
            doc2_id: doc2._id,
            similarity: COSINE_SIMILARITY(doc1.{embedding_field}, doc2.{embedding_field})
        }}
        """
        cursor = db.aql.execute(aql)
        result = cursor.next()
        print(f"Structured query result: {result}")
        
        return True
    except Exception as e:
        print(f"Error in test: {str(e)}")
        return False