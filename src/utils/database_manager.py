import os
import logging
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
import json
import time

# Import pymongo conditionally to avoid errors if not installed
try:
    import pymongo
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages database operations for SoulMate.AGI using a MongoDB database for metadata
    and FAISS for vector embeddings.
    """
    
    def __init__(self, mongo_uri: str = None, vector_dim: int = 384, use_mongo: bool = False):
        """
        Initialize the database manager with MongoDB and FAISS
        
        Args:
            mongo_uri: MongoDB connection string (defaults to localhost if not provided)
            vector_dim: Dimension of the embedding vectors
            use_mongo: Whether to try using MongoDB or go straight to fallback
        """
        # Connect to MongoDB
        self.mongo_uri = mongo_uri or "mongodb://localhost:27017/"
        self.use_fallback = True  # Default to fallback
        
        if use_mongo and PYMONGO_AVAILABLE:
            try:
                self.mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
                # Test connection with a ping
                self.mongo_client.admin.command('ping')
                self.db = self.mongo_client["soulmate_agi"]
                
                # Main collections
                self.interactions = self.db["interactions"]
                self.preferences = self.db["preferences"]
                self.user_profiles = self.db["user_profiles"]
                self.chat_history = self.db["chat_history"]
                self.memory_vault = self.db["memory_vault"]
                self.journal_entries = self.db["journal_entries"]
                
                # Create indexes for faster retrieval
                self.interactions.create_index([("user_id", pymongo.ASCENDING)])
                self.preferences.create_index([("user_id", pymongo.ASCENDING), ("key", pymongo.ASCENDING)], unique=True)
                self.chat_history.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
                self.memory_vault.create_index([("user_id", pymongo.ASCENDING), ("category", pymongo.ASCENDING)])
                self.journal_entries.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
                self.user_profiles.create_index([("user_id", pymongo.ASCENDING)], unique=True)

                logger.info("Connected to MongoDB database")
                self.use_fallback = False
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}. Using in-memory fallback storage.")
                self.mongo_client = None
                self.db = None
                self.interactions = self.preferences = self.user_profiles = None 
                self.chat_history = self.memory_vault = self.journal_entries = None
        else:
            logger.info("Using in-memory fallback storage (MongoDB not requested or not available)")
            self.mongo_client = self.db = self.interactions = self.preferences = None
            self.user_profiles = self.chat_history = self.memory_vault = self.journal_entries = None
        
        # In-memory fallback storage
        self.fallback_interactions = {}  # user_id -> list of interactions
        self.fallback_preferences = {}   # user_id -> dict of preferences
        self.fallback_profiles = {}      # user_id -> user profile
        self.fallback_chat_history = {}  # user_id -> list of chat history
        self.fallback_memory_vault = {}  # user_id -> list of memory vault entries
        self.fallback_journal = {}       # user_id -> list of journal entries
        
        # Initialize FAISS indexes per user
        self.vector_dim = vector_dim
        self.faiss_indexes = {}
        self.vector_ids = {}  # To map FAISS IDs back to MongoDB IDs
    
    def _get_user_faiss_index(self, user_id: str) -> Optional[faiss.Index]:
        """
        Get or create a FAISS index for a specific user
        
        Args:
            user_id: The user identifier
            
        Returns:
            FAISS index for the user
        """
        if user_id not in self.faiss_indexes:
            try:
                # Create a new FAISS index for the user
                index = faiss.IndexFlatL2(self.vector_dim)
                self.faiss_indexes[user_id] = index
                self.vector_ids[user_id] = []
                
                if not self.use_fallback and self.interactions is not None:
                    # Populate it with existing vectors from MongoDB
                    interactions = list(self.interactions.find({"user_id": user_id}))
                    vectors = []
                    ids = []
                    
                    for interaction in interactions:
                        if "vector" in interaction and interaction["vector"]:
                            vectors.append(interaction["vector"])
                            ids.append(str(interaction["_id"]))
                    
                    if vectors:
                        index.add(np.array(vectors, dtype=np.float32))
                        self.vector_ids[user_id] = ids
                elif self.use_fallback and user_id in self.fallback_interactions:
                    # Populate from in-memory fallback storage
                    interactions = self.fallback_interactions[user_id]
                    vectors = []
                    ids = []
                    
                    for i, interaction in enumerate(interactions):
                        if "vector" in interaction and interaction["vector"]:
                            vectors.append(interaction["vector"])
                            ids.append(str(i))  # Use index as ID
                    
                    if vectors:
                        index.add(np.array(vectors, dtype=np.float32))
                        self.vector_ids[user_id] = ids
                
                logger.info(f"Created FAISS index for user {user_id}")
            except Exception as e:
                logger.error(f"Error creating FAISS index for user {user_id}: {e}")
                return None
        
        return self.faiss_indexes.get(user_id)
    
    def store_interaction(self, user_id: str, user_input: str, ai_response: str, 
                          vector: List[float], context: Dict[str, Any] = None) -> bool:
        """
        Store an interaction in MongoDB and its vector in FAISS
        
        Args:
            user_id: The user identifier
            user_input: The user's input text
            ai_response: The AI's response text
            vector: The embedding vector for this interaction
            context: Additional context data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create interaction object
            interaction = {
                "user_id": user_id,
                "user_input": user_input,
                "ai_response": ai_response,
                "vector": vector,
                "timestamp": time.time(),
                "context": context or {}
            }
            
            # Store in MongoDB or fallback
            if not self.use_fallback and self.interactions is not None:
                result = self.interactions.insert_one(interaction)
                mongo_id = str(result.inserted_id)
            else:
                # Use in-memory fallback storage
                if user_id not in self.fallback_interactions:
                    self.fallback_interactions[user_id] = []
                
                self.fallback_interactions[user_id].append(interaction)
                mongo_id = str(len(self.fallback_interactions[user_id]) - 1)
            
            # Store vector in FAISS
            index = self._get_user_faiss_index(user_id)
            if index:
                vector_np = np.array([vector], dtype=np.float32)
                index.add(vector_np)
                if user_id in self.vector_ids:
                    self.vector_ids[user_id].append(mongo_id)
            
            return True
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            return False
    
    def find_similar_interactions(self, user_id: str, query_vector: List[float], 
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar interactions based on vector similarity
        
        Args:
            user_id: The user identifier
            query_vector: The query embedding vector
            top_k: Maximum number of results to return
            
        Returns:
            List of similar interactions with metadata
        """
        try:
            # Get FAISS index for user
            index = self._get_user_faiss_index(user_id)
            if not index or index.ntotal == 0:
                return []
            
            # Search for similar vectors
            query_np = np.array([query_vector], dtype=np.float32)
            distances, indices = index.search(query_np, min(top_k, index.ntotal))
            
            # Retrieve matched documents
            results = []
            if user_id in self.vector_ids and len(indices) > 0:
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.vector_ids[user_id]):
                        mongo_id = self.vector_ids[user_id][idx]
                        
                        if not self.use_fallback and self.interactions is not None:
                            # Get from MongoDB
                            doc = self.interactions.find_one({"_id": pymongo.ObjectId(mongo_id)})
                            if doc:
                                doc_dict = {
                                    "user_input": doc["user_input"],
                                    "ai_response": doc["ai_response"],
                                    "timestamp": doc.get("timestamp", 0),
                                    "similarity_score": float(distances[0][i]),
                                }
                                results.append(doc_dict)
                        else:
                            # Get from fallback storage
                            try:
                                idx = int(mongo_id)
                                if user_id in self.fallback_interactions and idx < len(self.fallback_interactions[user_id]):
                                    doc = self.fallback_interactions[user_id][idx]
                                    doc_dict = {
                                        "user_input": doc["user_input"],
                                        "ai_response": doc["ai_response"],
                                        "timestamp": doc.get("timestamp", 0),
                                        "similarity_score": float(distances[0][i]),
                                    }
                                    results.append(doc_dict)
                            except (ValueError, IndexError):
                                continue
            
            return results
        except Exception as e:
            logger.error(f"Error finding similar interactions: {e}")
            return []
    
    def store_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        Store or update a user preference
        
        Args:
            user_id: The user identifier
            key: Preference key
            value: Preference value (must be JSON serializable)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert value to JSON-safe format
            value_json = json.dumps(value)
            
            if not self.use_fallback and self.preferences is not None:
                # Upsert to MongoDB
                self.preferences.update_one(
                    {"user_id": user_id, "key": key},
                    {"$set": {"value": value_json}},
                    upsert=True
                )
            else:
                # Use in-memory fallback
                if user_id not in self.fallback_preferences:
                    self.fallback_preferences[user_id] = {}
                
                self.fallback_preferences[user_id][key] = value_json
            
            return True
        except Exception as e:
            logger.error(f"Error storing preference: {e}")
            return False
    
    def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a user preference
        
        Args:
            user_id: The user identifier
            key: Preference key
            default: Default value to return if preference doesn't exist
            
        Returns:
            The preference value or default if not found
        """
        try:
            if not self.use_fallback and self.preferences is not None:
                # Get from MongoDB
                doc = self.preferences.find_one({"user_id": user_id, "key": key})
                if doc and "value" in doc:
                    return json.loads(doc["value"])
            else:
                # Get from fallback storage
                if user_id in self.fallback_preferences and key in self.fallback_preferences[user_id]:
                    return json.loads(self.fallback_preferences[user_id][key])
            
            return default
        except Exception as e:
            logger.error(f"Error retrieving preference: {e}")
            return default
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get all preferences for a user
        
        Args:
            user_id: The user identifier
            
        Returns:
            Dictionary of user preferences
        """
        prefs = {}
        try:
            if not self.use_fallback and self.preferences is not None:
                # Get from MongoDB
                for doc in self.preferences.find({"user_id": user_id}):
                    try:
                        value = json.loads(doc["value"])
                        prefs[doc["key"]] = value
                    except json.JSONDecodeError:
                        prefs[doc["key"]] = doc["value"]
            elif user_id in self.fallback_preferences:
                # Get from fallback storage
                for key, value_json in self.fallback_preferences[user_id].items():
                    try:
                        value = json.loads(value_json)
                        prefs[key] = value
                    except json.JSONDecodeError:
                        prefs[key] = value_json
            
            return prefs
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    def delete_interaction(self, user_id: str, interaction_id: str) -> bool:
        """
        Delete a specific interaction
        
        Args:
            user_id: The user identifier
            interaction_id: The interaction identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.use_fallback and self.interactions is not None:
                # Delete from MongoDB
                result = self.interactions.delete_one({
                    "_id": pymongo.ObjectId(interaction_id), 
                    "user_id": user_id
                })
                success = result.deleted_count > 0
                
                # Note: We don't delete from FAISS here as it would require rebuilding the index
                # The vector will remain but become "orphaned" and won't be returned in results
                
                if success:
                    logger.info(f"Deleted interaction {interaction_id} for user {user_id}")
                    return True
                return False
            else:
                # Not implemented for fallback storage as it requires index management
                logger.warning("Delete interaction not supported in fallback mode")
                return False
        except Exception as e:
            logger.error(f"Error deleting interaction: {e}")
            return False
    
    def delete_preference(self, user_id: str, key: str) -> bool:
        """
        Delete a user preference
        
        Args:
            user_id: The user identifier
            key: Preference key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.use_fallback and self.preferences is not None:
                # Delete from MongoDB
                result = self.preferences.delete_one({"user_id": user_id, "key": key})
                success = result.deleted_count > 0
            else:
                # Delete from fallback storage
                success = False
                if user_id in self.fallback_preferences and key in self.fallback_preferences[user_id]:
                    del self.fallback_preferences[user_id][key]
                    success = True
            
            if success:
                logger.info(f"Deleted preference {key} for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting preference: {e}")
            return False
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data for a specific user (GDPR compliance)
        
        Args:
            user_id: The user identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = True
            
            # Delete from all collections
            if not self.use_fallback:
                # Delete from interactions
                if self.interactions is not None:
                    result = self.interactions.delete_many({"user_id": user_id})
                    success = success and result.acknowledged
                
                # Delete from preferences
                if self.preferences is not None:
                    result = self.preferences.delete_many({"user_id": user_id})
                    success = success and result.acknowledged
                
                # Delete from chat history
                if self.chat_history is not None:
                    result = self.chat_history.delete_many({"user_id": user_id})
                    success = success and result.acknowledged
                
                # Delete from memory vault
                if self.memory_vault is not None:
                    result = self.memory_vault.delete_many({"user_id": user_id})
                    success = success and result.acknowledged
                
                # Delete from journal entries
                if self.journal_entries is not None:
                    result = self.journal_entries.delete_many({"user_id": user_id})
                    success = success and result.acknowledged
                
                # Delete user profile
                if self.user_profiles is not None:
                    result = self.user_profiles.delete_one({"user_id": user_id})
                    # Don't affect success status if profile doesn't exist
            else:
                # Clear from fallback storage
                if user_id in self.fallback_interactions:
                    del self.fallback_interactions[user_id]
                if user_id in self.fallback_preferences:
                    del self.fallback_preferences[user_id]
                if user_id in self.fallback_chat_history:
                    del self.fallback_chat_history[user_id]
                if user_id in self.fallback_memory_vault:
                    del self.fallback_memory_vault[user_id]
                if user_id in self.fallback_journal:
                    del self.fallback_journal[user_id]
                if user_id in self.fallback_profiles:
                    del self.fallback_profiles[user_id]
            
            # Reset FAISS index if it exists
            if user_id in self.faiss_indexes:
                del self.faiss_indexes[user_id]
            if user_id in self.vector_ids:
                del self.vector_ids[user_id]
            
            logger.info(f"Deleted all data for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting user data: {e}")
            return False
    
    def close(self) -> None:
        """
        Close database connections and clean up resources
        """
        try:
            if self.mongo_client and not self.use_fallback:
                self.mongo_client.close()
                logger.info("MongoDB connections closed")
            
            # Clear FAISS indexes
            self.faiss_indexes.clear()
            self.vector_ids.clear()
            
            if self.use_fallback:
                # Clear fallback storage
                self.fallback_interactions.clear()
                self.fallback_preferences.clear()
                
            logger.info("Database manager resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

    def search_interactions_by_text(self, user_id: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for interactions based on text content rather than vector similarity
        
        Args:
            user_id: The user identifier
            query_text: The text to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of matching interactions
        """
        try:
            results = []
            
            if not self.use_fallback and self.interactions is not None:
                # Get from MongoDB using text search
                # First, try a regex search on user_input and ai_response fields
                pipeline = [
                    {"$match": {"user_id": user_id}},
                    {"$match": {
                        "$or": [
                            {"user_input": {"$regex": query_text, "$options": "i"}},
                            {"ai_response": {"$regex": query_text, "$options": "i"}}
                        ]
                    }},
                    {"$sort": {"timestamp": -1}},
                    {"$limit": top_k}
                ]
                
                matches = list(self.interactions.aggregate(pipeline))
                
                for doc in matches:
                    doc_dict = {
                        "user_input": doc["user_input"],
                        "ai_response": doc["ai_response"],
                        "timestamp": doc.get("timestamp", 0),
                        "similarity_score": 1.0,  # Default similarity score for text match
                    }
                    if "context" in doc:
                        doc_dict["context"] = doc["context"]
                    results.append(doc_dict)
                
                return results
            
            elif user_id in self.fallback_interactions:
                # Search in fallback storage
                fallback_results = []
                
                for interaction in self.fallback_interactions[user_id]:
                    score = 0
                    
                    # Check if query appears in user input
                    if query_text.lower() in interaction["user_input"].lower():
                        score += 1
                    
                    # Check if query appears in AI response
                    if query_text.lower() in interaction["ai_response"].lower():
                        score += 0.5
                    
                    if score > 0:
                        result = {
                            "user_input": interaction["user_input"],
                            "ai_response": interaction["ai_response"],
                            "timestamp": interaction.get("timestamp", 0),
                            "similarity_score": score
                        }
                        if "context" in interaction:
                            result["context"] = interaction["context"]
                        fallback_results.append(result)
                
                # Sort by score and limit
                fallback_results.sort(key=lambda x: x["similarity_score"], reverse=True)
                return fallback_results[:top_k]
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching interactions by text: {e}")
            return []

    def store_chat_message(self, user_id: str, message: str, is_user: bool, timestamp: float = None) -> bool:
        """Store a chat message with timestamp"""
        try:
            chat_entry = {
                "user_id": user_id,
                "message": message,
                "is_user": is_user,
                "timestamp": timestamp or time.time()
            }
            
            if not self.use_fallback and self.chat_history is not None:
                self.chat_history.insert_one(chat_entry)
            else:
                if user_id not in self.fallback_chat_history:
                    self.fallback_chat_history[user_id] = []
                self.fallback_chat_history[user_id].append(chat_entry)
            return True
        except Exception as e:
            logger.error(f"Error storing chat message: {e}")
            return False

    def get_chat_history(self, user_id: str, limit: int = 50, before_timestamp: float = None) -> List[Dict[str, Any]]:
        """Get chat history for a user with pagination"""
        try:
            if not self.use_fallback and self.chat_history is not None:
                query = {"user_id": user_id}
                if before_timestamp:
                    query["timestamp"] = {"$lt": before_timestamp}
                    
                return list(self.chat_history.find(query)
                    .sort("timestamp", pymongo.DESCENDING)
                    .limit(limit))
            else:
                if user_id in self.fallback_chat_history:
                    history = self.fallback_chat_history[user_id]
                    if before_timestamp:
                        history = [msg for msg in history if msg["timestamp"] < before_timestamp]
                    return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
            return []
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []

    def store_memory(self, user_id: str, category: str, content: Dict[str, Any]) -> bool:
        """Store a memory in the user's memory vault"""
        try:
            memory_entry = {
                "user_id": user_id,
                "category": category,
                "content": content,
                "timestamp": time.time()
            }
            
            if not self.use_fallback and self.memory_vault is not None:
                self.memory_vault.insert_one(memory_entry)
            else:
                if user_id not in self.fallback_memory_vault:
                    self.fallback_memory_vault[user_id] = []
                self.fallback_memory_vault[user_id].append(memory_entry)
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False

    def get_memories(self, user_id: str, category: str = None) -> List[Dict[str, Any]]:
        """Retrieve memories from the user's memory vault"""
        try:
            query = {"user_id": user_id}
            if category:
                query["category"] = category
                
            if not self.use_fallback and self.memory_vault is not None:
                return list(self.memory_vault.find(query).sort("timestamp", pymongo.DESCENDING))
            else:
                if user_id in self.fallback_memory_vault:
                    memories = self.fallback_memory_vault[user_id]
                    if category:
                        memories = [m for m in memories if m["category"] == category]
                    return sorted(memories, key=lambda x: x["timestamp"], reverse=True)
            return []
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def add_journal_entry(self, user_id: str, content: str, mood: str = None, tags: List[str] = None) -> bool:
        """Add a journal entry for the user"""
        try:
            entry = {
                "user_id": user_id,
                "content": content,
                "mood": mood,
                "tags": tags or [],
                "timestamp": time.time()
            }
            
            if not self.use_fallback and self.journal_entries is not None:
                self.journal_entries.insert_one(entry)
            else:
                if user_id not in self.fallback_journal:
                    self.fallback_journal[user_id] = []
                self.fallback_journal[user_id].append(entry)
            return True
        except Exception as e:
            logger.error(f"Error adding journal entry: {e}")
            return False

    def get_journal_entries(self, user_id: str, limit: int = 50, before_timestamp: float = None) -> List[Dict[str, Any]]:
        """Get journal entries for a user with pagination"""
        try:
            query = {"user_id": user_id}
            if before_timestamp:
                query["timestamp"] = {"$lt": before_timestamp}
                
            if not self.use_fallback and self.journal_entries is not None:
                return list(self.journal_entries.find(query)
                    .sort("timestamp", pymongo.DESCENDING)
                    .limit(limit))
            else:
                if user_id in self.fallback_journal:
                    entries = self.fallback_journal[user_id]
                    if before_timestamp:
                        entries = [e for e in entries if e["timestamp"] < before_timestamp]
                    return sorted(entries, key=lambda x: x["timestamp"], reverse=True)[:limit]
            return []
        except Exception as e:
            logger.error(f"Error retrieving journal entries: {e}")
            return []

    def create_or_update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Create or update a user's profile"""
        try:
            profile_data["last_updated"] = time.time()
            
            if not self.use_fallback and self.user_profiles is not None:
                self.user_profiles.update_one(
                    {"user_id": user_id},
                    {"$set": profile_data},
                    upsert=True
                )
            else:
                self.fallback_profiles[user_id] = profile_data
            return True
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's profile"""
        try:
            if not self.use_fallback and self.user_profiles is not None:
                profile = self.user_profiles.find_one({"user_id": user_id})
                return profile if profile else None
            else:
                return self.fallback_profiles.get(user_id)
        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
            return None