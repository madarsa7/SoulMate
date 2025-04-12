from src.utils.database_manager import DatabaseManager
import pymongo

def check_mongo_status():
    print("Checking MongoDB connection...")
    try:
        # Check if our DatabaseManager can connect
        db_manager = DatabaseManager()
        print(f"DatabaseManager connection: {'Connected' if db_manager.db is not None else 'Not connected'}")
        
        if db_manager.use_fallback:
            print("WARNING: Application is using fallback in-memory storage instead of MongoDB")
        else:
            print(f"Connected to database: {db_manager.db.name}")
            print(f"Collections: {db_manager.db.list_collection_names()}")
        
        # Direct check with pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("Direct MongoDB connection: Connected")
        
        # List all databases
        db_list = client.list_database_names()
        print(f"Available databases: {db_list}")
        
        if "soulmate_agi" in db_list:
            print("The 'soulmate_agi' database exists")
            db = client["soulmate_agi"]
            collection_list = db.list_collection_names()
            print(f"Collections in soulmate_agi: {collection_list}")
        else:
            print("The 'soulmate_agi' database does not exist yet")
            
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        print("You need to start MongoDB server before running your application")

if __name__ == "__main__":
    check_mongo_status()