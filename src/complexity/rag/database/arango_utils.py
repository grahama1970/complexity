from typing import Dict, Any
from loguru import logger
from arango import ArangoClient
# Import specific exceptions used in the except block
from arango.exceptions import ArangoError, ServerConnectionError, ArangoClientError, ArangoServerError
from arango.database import StandardDatabase # Import for type hinting
from typing import List

def initialize_database(config: Dict[str, Any]):
    """
    Sets up and connects to the ArangoDB client, ensuring the database is created if it doesn't exist.

    Args:
        config (dict): Either a standalone `arango_config` dictionary or a larger `config` dictionary
                    containing `arango_config` as a nested field.

    Returns:
        db: The connected ArangoDB database instance or None if an error occurs.
    """
    db_name_for_log = config.get("db_name", "UNKNOWN") # For logging in except blocks
    try:
        # Use the passed config directly, assuming it contains the necessary keys
        # Default values are handled by the caller (rag_classifier.py using os.getenv)
        # or can be added here if preferred.
        hosts = config.get("hosts", ["http://localhost:8529"]) # Keep default host if not provided
        db_name = config.get("db_name") # Get db_name from config
        username = config.get("username") # Get username from config
        password = config.get("password") # Get password from config

        if not db_name:
            logger.error("Database name ('db_name') not provided in config.")
            return None
        db_name_for_log = db_name # Update for logging if db_name is valid

        # Initialize the ArangoDB client
        client = ArangoClient(hosts=hosts)

        # Connect to the database
        db = client.db(db_name, username=username, password=password)

        # Verify connection by fetching database properties
        db.properties() # This will raise error if DB doesn't exist or credentials fail

        logger.success(f"Successfully connected to and verified database '{db_name}'.")
        return db

    # Catch specific Arango errors during connection/verification
    except (ArangoClientError, ArangoServerError, ArangoError) as e:
        logger.error(f"Failed to connect/verify database '{db_name_for_log}': {e}")
        return None
    # Catch other unexpected errors
    except Exception as e:
        logger.error(f"Unexpected error during database initialization for '{db_name_for_log}': {e}")
        return None


# Functions store_docs_in_arango and create_arangosearch_view removed as per Task 4.
# Their functionality is handled by src/complexity/utils/arango_setup.py
