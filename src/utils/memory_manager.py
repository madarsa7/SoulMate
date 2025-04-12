import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import hashlib
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages user memories and preferences
    Provides persistence across sessions using MongoDB
    """
    
    def __init__(self, user_id=None, data_dir="data", db_manager=None):
        self.data_dir = data_dir
        self.user_id = user_id
        self.db_manager = db_manager
        
        # Import database manager if not provided
        if self.db_manager is None:
            try:
                from src.utils.database_manager import DatabaseManager
                # Use environment variables for MongoDB connection
                import os
                mongo_uri = os.getenv('MONGODB_URI')
                use_mongo = os.getenv('USE_MONGODB', 'true').lower() == 'true'
                self.db_manager = DatabaseManager(mongo_uri=mongo_uri, use_mongo=use_mongo)
                logger.info(f"Initialized database manager for MemoryManager")
            except ImportError:
                logger.error("DatabaseManager could not be imported. Make sure the class is available.")
                raise
            except Exception as e:
                logger.error(f"Error initializing DatabaseManager: {e}")
                raise
    
    def get_memory_filename(self, user_id: str) -> str:
        """Get the filename for a user's memory file"""
        # Sanitize user_id to be safe for filenames
        safe_user_id = re.sub(r'[^\w\-\.]', '_', user_id)
        return os.path.join(self.data_dir, f"user_memory_{safe_user_id}.pkl")
    
    def load_user_memory(self, user_id: str) -> Dict[str, Any]:
        """Load a user's memory from database or fallback to disk if it exists"""
        try:
            # Get all user preferences from database
            user_preferences = self.db_manager.get_user_preferences(user_id)
            
            # If there's a "memory_structure" preference, use that as the base memory structure
            if "memory_structure" in user_preferences:
                return user_preferences["memory_structure"]
            
            # Check if we need to migrate from pickle file
            filename = self.get_memory_filename(user_id)
            if os.path.exists(filename):
                logger.info(f"Found file-based memory for user {user_id}. Migrating to database.")
                try:
                    with open(filename, 'rb') as f:
                        file_memory = pickle.load(f)
                        
                    # Store the entire memory structure in the database
                    self.db_manager.store_preference(user_id, "memory_structure", file_memory)
                    
                    # Also store individual memories as separate preferences for better query options
                    for idx, memory in enumerate(file_memory.get("memories", [])):
                        memory_key = f"memory_{idx}"
                        self.db_manager.store_preference(user_id, memory_key, memory)
                    
                    # Store memory vault entries separately
                    for vault_type, entries in file_memory.get("memory_vault", {}).items():
                        for entry in entries:
                            vault_id = entry.get('id', f"vault_{datetime.now().timestamp()}")
                            vault_key = f"vault_{vault_type}_{vault_id}"
                            self.db_manager.store_preference(user_id, vault_key, entry)
                    
                    logger.info(f"Successfully migrated memory for user {user_id} to database")
                    return file_memory
                except Exception as e:
                    logger.error(f"Error migrating file memory to database for user {user_id}: {e}")
            
            # Create a new memory structure if nothing exists
            return {"preferences": {}, "memories": [], "memory_vault": {}}
            
        except Exception as e:
            logger.error(f"Error loading memory for user {user_id}: {e}")
            return {"preferences": {}, "memories": [], "memory_vault": {}}
    
    def save_user_memory(self, user_id: str, memory_data: Dict[str, Any]) -> bool:
        """Save a user's memory to the database"""
        try:
            # Store the entire memory structure in the database
            success = self.db_manager.store_preference(user_id, "memory_structure", memory_data)
            
            # Also store individual memories as separate entries for better querying
            # First, clear existing memory entries that might have changed
            # (In a production system, you would use proper transactions and delta updates)
            
            # We keep the memories as part of memory_structure but also store them individually
            # for better query capabilities
            for idx, memory in enumerate(memory_data.get("memories", [])):
                memory_key = f"memory_{idx}"
                self.db_manager.store_preference(user_id, memory_key, memory)
            
            logger.info(f"Memory saved in database for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"Error saving memory for user {user_id} to database: {e}")
            return False
    
    def store_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Store a user preference directly in the database"""
        try:
            # Add timestamp to preference
            if isinstance(value, dict) and 'stored_at' not in value:
                value['stored_at'] = datetime.now().isoformat()
            
            # Store directly in database
            success = self.db_manager.store_preference(user_id, key, value)
            
            # Also update the memory structure for consistency
            try:
                memory_data = self.load_user_memory(user_id)
                memory_data["preferences"][key] = value
                self.db_manager.store_preference(user_id, "memory_structure", memory_data)
            except Exception as e:
                logger.warning(f"Could not update memory structure for preference {key}: {e}")
            
            return success
        except Exception as e:
            logger.error(f"Error storing preference {key} for user {user_id}: {e}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all preferences for a user directly from the database"""
        try:
            # Get preferences directly from the database
            return self.db_manager.get_user_preferences(user_id)
        except Exception as e:
            logger.error(f"Error getting preferences for user {user_id}: {e}")
            # Fall back to memory structure if database lookup fails
            memory_data = self.load_user_memory(user_id)
            return memory_data.get("preferences", {})
    
    def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get a specific user preference"""
        preferences = self.get_user_preferences(user_id)
        return preferences.get(key, default)
    
    def store_memory(self, user_id: str, memory_data: Dict[str, Any]) -> bool:
        """Store a conversation memory directly in the database"""
        try:
            # Ensure memory has timestamp
            if 'timestamp' not in memory_data:
                memory_data['timestamp'] = datetime.now().isoformat()
                
            # Generate a unique ID for this memory if it doesn't have one
            if 'id' not in memory_data:
                memory_data['id'] = f"memory_{datetime.now().timestamp()}_{hash(str(memory_data))}"
            
            # Store this memory as an individual preference with its ID as part of the key
            memory_key = f"memory_{memory_data['id']}"
            direct_success = self.db_manager.store_preference(user_id, memory_key, memory_data)
            
            # Also add to the consolidated memory structure
            try:
                # Get current memory structure
                memory = self.load_user_memory(user_id)
                
                # Add memory to list
                memory["memories"].append(memory_data)
                
                # Limit size of memory list to prevent it from growing too large
                if len(memory["memories"]) > 1000:  # Arbitrary limit, adjust as needed
                    memory["memories"] = memory["memories"][-1000:]
                
                # Save updated memory structure
                structure_success = self.db_manager.store_preference(user_id, "memory_structure", memory)
                
                return direct_success and structure_success
            except Exception as e:
                logger.error(f"Error updating memory structure: {e}")
                # If updating the structure fails, we still saved the individual memory
                return direct_success
                
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    def get_memories(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories for a user directly from database"""
        try:
            # Get all preferences that start with 'memory_' (not memory_vault or memory_structure)
            all_prefs = self.db_manager.get_user_preferences(user_id)
            memories = []
            
            # Find memory entries - these will have keys starting with 'memory_'
            memory_pattern = re.compile(r'^memory_(?!structure|vault)')
            for key, value in all_prefs.items():
                if memory_pattern.match(key) and isinstance(value, dict):
                    memories.append(value)
            
            # Sort by timestamp (newest first)
            try:
                sorted_memories = sorted(
                    memories, 
                    key=lambda x: x.get('timestamp', ''), 
                    reverse=True
                )
                return sorted_memories[:limit]
            except Exception as e:
                logger.error(f"Error sorting memories: {e}")
                # Return unsorted if sorting fails
                return memories[:limit]
                
        except Exception as e:
            logger.error(f"Error retrieving memories for user {user_id}: {e}")
            # Fall back to memory structure if direct lookup fails
            memory = self.load_user_memory(user_id)
            memories = memory.get("memories", [])
            
            # Sort and return
            try:
                sorted_memories = sorted(
                    memories,
                    key=lambda x: x.get('timestamp', ''),
                    reverse=True
                )
                return sorted_memories[:limit]
            except Exception as sorting_error:
                logger.error(f"Error sorting fallback memories: {sorting_error}")
                return memories[:limit]
    
    def search_memories(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Search through user memories using simple keyword matching
        In a production system, this would use a vector database or more sophisticated search
        """
        memory = self.load_user_memory(user_id)
        memories = memory.get("memories", [])
        
        results = []
        for mem in memories:
            # Check message text for keyword
            if 'text' in mem and query.lower() in mem['text'].lower():
                results.append(mem)
                continue
                
            # Check memory tags
            if 'tags' in mem and isinstance(mem['tags'], list):
                for tag in mem['tags']:
                    if query.lower() in tag.lower():
                        results.append(mem)
                        break
        
        return results
    
    def add_to_memory_vault(self, user_id: str, memory_type: str, content: Dict[str, Any]) -> bool:
        """
        Add content to the user's memory vault
        
        Args:
            user_id: The user's ID
            memory_type: Type of memory (e.g., 'journal', 'letter', 'reflection')
            content: The memory content
            
        Returns:
            Success status
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in content:
                content['timestamp'] = datetime.now().isoformat()
                
            # Add a unique ID
            if 'id' not in content:
                content_str = json.dumps(content, sort_keys=True)
                content['id'] = hashlib.md5(f"{content_str}_{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Store in the memory_vault collection directly
            return self.db_manager.store_memory(user_id, memory_type, content)
        except Exception as e:
            logger.error(f"Error adding to memory vault: {e}")
            return False
    
    def get_from_memory_vault(self, user_id: str, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get content from the user's memory vault
        
        Args:
            user_id: The user's ID
            memory_type: Type of memory to retrieve (if None, returns all types)
            
        Returns:
            The memory vault content
        """
        try:
            # Use the dedicated memory_vault collection
            memories = self.db_manager.get_memories(user_id, memory_type)
            
            if not memories:
                return {} if memory_type is None else {memory_type: []}
                
            # Organize by category
            result = {}
            for memory in memories:
                category = memory.get('category', 'general')
                if category not in result:
                    result[category] = []
                result[category].append(memory.get('content', {}))
                
            # If a specific type was requested, just return that
            if memory_type:
                return {memory_type: result.get(memory_type, [])}
            
            return result
        except Exception as e:
            logger.error(f"Error retrieving from memory vault: {e}")
            # Fall back to the old method if necessary
            memory = self.load_user_memory(user_id)
            vault = memory.get("memory_vault", {})
            
            if memory_type:
                return {memory_type: vault.get(memory_type, [])}
            else:
                return vault
            
    def delete_from_memory_vault(self, user_id: str, memory_type: str, memory_id: str) -> bool:
        """
        Delete a specific memory from the vault
        
        Args:
            user_id: The user's ID
            memory_type: The memory type
            memory_id: The ID of the memory to delete
            
        Returns:
            Success status
        """
        try:
            # Implement deletion using a new method in database_manager
            # For now we'll use a fallback approach
            # In a real implementation, you'd add a delete_memory method to DatabaseManager
            
            # First try to delete from the dedicated memory_vault collection
            deleted = False
            if hasattr(self.db_manager, 'memory_vault') and self.db_manager.memory_vault is not None:
                # Use MongoDB directly if available
                result = self.db_manager.memory_vault.delete_one({
                    "user_id": user_id,
                    "category": memory_type,
                    "content.id": memory_id
                })
                deleted = result.deleted_count > 0
            
            # If that didn't work or isn't available, fall back to the old method
            if not deleted:
                memory = self.load_user_memory(user_id)
                vault = memory.get("memory_vault", {})
                
                # Check if memory type exists
                if memory_type not in vault:
                    return False
                    
                # Find and remove the memory
                memories = vault[memory_type]
                updated_memories = [m for m in memories if m.get('id') != memory_id]
                
                # If nothing was removed
                if len(memories) == len(updated_memories):
                    return False
                    
                # Update and save
                vault[memory_type] = updated_memories
                memory["memory_vault"] = vault
                return self.save_user_memory(user_id, memory)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting from memory vault: {e}")
            return False
        
    def clear_user_data(self, user_id: str) -> bool:
        """Delete all data for a user from both database and any existing files"""
        try:
            success = True
            
            # Delete data from database
            if self.db_manager:
                db_success = self.db_manager.delete_user_data(user_id)
                if not db_success:
                    logger.warning(f"Could not fully clear database data for user {user_id}")
                    success = False
            
            # Also delete any existing pickle file for completeness
            filename = self.get_memory_filename(user_id)
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    logger.info(f"Deleted pickle file for user {user_id}")
                except Exception as e:
                    logger.error(f"Error deleting pickle file for user {user_id}: {e}")
                    success = False
            
            if success:
                logger.info(f"Successfully cleared all data for user {user_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error clearing data for user {user_id}: {e}")
            return False

    def search_similar_interactions(self, user_id: str, query: str = None, limit: int = 5, k: int = None) -> List[Dict[str, Any]]:
        """
        Search for interactions similar to the query using vector search if possible
        
        Args:
            user_id: The user's ID
            query: The query to search for similar interactions (optional)
            limit: Maximum number of results to return
            k: Alternative parameter for limit (for compatibility)
            
        Returns:
            List of similar interactions
        """
        try:
            # Use k parameter if provided (for backwards compatibility)
            if k is not None:
                limit = k
                
            # If no query or query is "*" (wildcard), return most recent memories
            if not query or query == "*":
                return self.get_memories(user_id, limit=limit)
            
            # Try to use the database manager's vector search if available
            # This requires that user interactions have been stored with embeddings
            # using the DatabaseManager.store_interaction method
            try:
                if hasattr(self.db_manager, 'find_similar_interactions') and query != "*":
                    # Get vector for query text
                    # In a real implementation, you would compute the embedding vector for the query
                    # using the same method used when storing interactions
                    
                    # Since we don't have direct access to the embedding model here,
                    # we'll try to use the database's keyword-based search as a fallback
                    if hasattr(self.db_manager, 'search_interactions_by_text'):
                        db_results = self.db_manager.search_interactions_by_text(user_id, query, top_k=limit)
                        if db_results and len(db_results) > 0:
                            return db_results
            except Exception as e:
                logger.warning(f"Error using vector search: {e}. Falling back to keyword search.")
            
            # Fallback to keyword-based search
            # Load memory structure for searching
            all_prefs = self.db_manager.get_user_preferences(user_id)
            memories = []
            
            # Find memory entries
            memory_pattern = re.compile(r'^memory_(?!structure|vault)')
            for key, value in all_prefs.items():
                if memory_pattern.match(key) and isinstance(value, dict):
                    memories.append(value)
            
            # Simple keyword matching
            matches = []
            for mem in memories:
                score = 0
                
                # Check in user_input field
                if 'user_input' in mem and query.lower() in mem['user_input'].lower():
                    score += 1
                
                # Check in ai_response field
                if 'ai_response' in mem and query.lower() in mem['ai_response'].lower():
                    score += 0.5
                
                # Check in any other text fields
                for field in ['text', 'content', 'message']:
                    if field in mem and isinstance(mem[field], str):
                        if query.lower() in mem[field].lower():
                            score += 1
                
                # Check tags
                if 'tags' in mem and isinstance(mem['tags'], list):
                    for tag in mem['tags']:
                        if query.lower() in tag.lower():
                            score += 0.5
                
                # If score > 0, add to matches with score
                if score > 0:
                    mem_with_score = mem.copy()
                    mem_with_score['similarity_score'] = score
                    matches.append(mem_with_score)
            
            # Sort by score
            matches.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Return limited number of matches
            return matches[:limit]
        
        except Exception as e:
            logger.error(f"Error searching similar interactions: {e}")
            return []

    def get_user_preference(self, key, user_id=None, default: Any = None) -> Any:
        """
        Alias for get_preference for backwards compatibility
        
        Args:
            key: The preference key to retrieve
            user_id: The user's ID (optional if set during initialization)
            default: Default value if preference not found
            
        Returns:
            The preference value or default
        """
        # Use the instance user_id if not provided
        if user_id is None:
            if self.user_id is None:
                logger.error("No user_id provided and none set during initialization")
                return default
            user_id = self.user_id
            
        return self.get_preference(user_id, key, default)
        
    def store_user_preference(self, key, value, user_id=None):
        """
        Store a user preference
        
        Args:
            key: The preference key
            value: The preference value
            user_id: The user's ID (optional if set during initialization)
            
        Returns:
            Success status
        """
        # Use the instance user_id if not provided
        if user_id is None:
            if self.user_id is None:
                logger.error("No user_id provided and none set during initialization")
                return False
            user_id = self.user_id
            
        return self.store_preference(user_id, key, value)

    def get_context_for_conversation(self, message=None, user_id=None, limit: int = 5) -> Dict[str, Any]:
        """
        Get context for a conversation including relevant memories and preferences
        
        Args:
            message: The current message to find relevant context for (optional)
            user_id: The user's ID (optional if set during initialization)
            limit: Maximum number of memories to include
            
        Returns:
            Dictionary with context information
        """
        try:
            # Use the instance user_id if not provided
            if user_id is None:
                if self.user_id is None:
                    logger.error("No user_id provided and none set during initialization")
                    return {
                        "recent_interactions": [],
                        "relevant_memories": [],
                        "preferences": {},
                        "timestamp": datetime.now().isoformat(),
                        "error": "No user_id provided"
                    }
                user_id = self.user_id
                
            context = {
                "recent_interactions": [],
                "relevant_memories": [],
                "preferences": self.get_user_preferences(user_id),
                "timestamp": datetime.now().isoformat()
            }
            
            # Get recent interactions
            recent = self.get_memories(user_id, limit=limit)
            if recent:
                context["recent_interactions"] = recent
            
            # Get message-specific relevant memories if message provided
            if message:
                relevant = self.search_similar_interactions(user_id, query=message, limit=limit)
                if relevant:
                    # Remove duplicates that might already be in recent interactions
                    recent_ids = [m.get('id') for m in recent if 'id' in m]
                    context["relevant_memories"] = [m for m in relevant if m.get('id') not in recent_ids]
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {
                "recent_interactions": [],
                "relevant_memories": [],
                "preferences": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def add_memory(self, user_input: str, ai_response: str, context: Dict[str, Any] = None) -> bool:
        """
        Add a chat memory to the database with proper vector embedding
        
        Args:
            user_input: The user's message
            ai_response: The AI's response
            context: Additional context information
            
        Returns:
            Success status
        """
        try:
            # Create a simple vector representation (this would use a proper embeddings model in production)
            # For now, we'll create a placeholder vector of the right dimension
            vector_dim = 384  # Default dimension for the vector db
            placeholder_vector = [0.0] * vector_dim
            
            # Get the timestamp for consistent recording
            timestamp = context.get('timestamp', datetime.now().timestamp())
            
            # Get user_id
            user_id = self.user_id if self.user_id else (context.get('user_id') if context else None)
            if not user_id:
                logger.error("No user_id provided for storing memory")
                return False
            
            # Create memory data object    
            memory_data = {
                "user_input": user_input,
                "ai_response": ai_response,
                "timestamp": timestamp,
                "context": context
            }
            
            # Store messages in chat_history collection
            chat_success = True
            if hasattr(self.db_manager, 'store_chat_message'):
                # Store user message
                user_msg_success = self.db_manager.store_chat_message(
                    user_id=user_id,
                    message=user_input,
                    is_user=True,
                    timestamp=timestamp
                )
                
                # Store AI response
                ai_msg_success = self.db_manager.store_chat_message(
                    user_id=user_id,
                    message=ai_response,
                    is_user=False,
                    timestamp=timestamp + 0.001  # Slightly later to preserve order
                )
                
                chat_success = user_msg_success and ai_msg_success
                
                if not chat_success:
                    logger.warning(f"Failed to store chat message in dedicated collection for user {user_id}")
            
            # Also store in interactions for semantic search capabilities
            interaction_success = self.db_manager.store_interaction(
                user_id=user_id,
                user_input=user_input,
                ai_response=ai_response,
                vector=placeholder_vector,
                context=context
            )
            
            if interaction_success:
                logger.info(f"Successfully stored chat memory in interactions for user {user_id}")
            else:
                logger.error(f"Failed to store chat memory in interactions for user {user_id}")
                
            # Additionally, store in memory structure for backward compatibility
            try:
                self.store_memory(user_id, memory_data)
            except Exception as e:
                logger.warning(f"Error storing in memory structure: {e}")
                
            return chat_success and interaction_success
            
        except Exception as e:
            logger.error(f"Error adding memory with vector: {e}")
            return False