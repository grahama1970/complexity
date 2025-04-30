from arango import ArangoClient

def check_collection_exists(db, collection_name: str) -> bool:
  """
  Checks if a collection exists in the given ArangoDB database.

  Args:
    db: An initialized ArangoDB database connection object 
        (from ArangoClient().db(...)).
    collection_name: The name of the collection to check.

  Returns:
    True if the collection exists, False otherwise.
  """
  return db.has_collection(collection_name)

# --- Example Usage ---
# Replace with your actual connection details
ARANGO_HOST = "http://localhost:8529" 
ARANGO_USER = "root"
ARANGO_PASSWORD = "openSesame" # Replace with your root password if set
DB_NAME = "memory_bank" # Or your target database
COLLECTION_TO_CHECK = "complexity" 

# Initialize the client
client = ArangoClient(hosts=ARANGO_HOST)

# Connect to the database
try:
  db = client.db(DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)

  # Check if the collection exists
  exists = check_collection_exists(db, COLLECTION_TO_CHECK)

  if exists:
    print(f"Collection '{COLLECTION_TO_CHECK}' exists in database '{DB_NAME}'.")
  else:
    print(f"Collection '{COLLECTION_TO_CHECK}' does NOT exist in database '{DB_NAME}'.")

except Exception as e:
  print(f"An error occurred: {e}")