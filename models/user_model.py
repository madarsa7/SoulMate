import os
import random
from datetime import datetime
from src.utils.database_manager import DatabaseManager
import google.generativeai as genai
from typing import Dict, List, Optional, Any
from src.utils.memory_manager import MemoryManager

# Initialize the database manager with MongoDB connection
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/soulmate_agi')
use_mongo = os.getenv('USE_MONGODB', 'true').lower() == 'true'
db_manager = DatabaseManager(mongo_uri=mongo_uri, use_mongo=use_mongo)

# Initialize Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

class UserModel:
    def __init__(self, user_id):
        self.user_id = user_id
        self.adaptation_level = 0
        self.training_iterations = 0
        self.db_manager = db_manager
        
        # Initialize memory manager for this user
        self.memory_manager = MemoryManager(user_id=user_id, db_manager=db_manager)
        
        # Load user preferences if available
        self.user_preferences = self._load_user_preferences()
        
        # Configure Gemini model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Load chat history from database
        self.chat_history = self._load_chat_history() or []
        self.max_history_length = 10  # Keep last 10 exchanges for context
        
        print(f"Loaded {len(self.chat_history)} chat history items for user {user_id}")
    
    def generate_response(self, message: str) -> str:
        """Generate a personalized response using Gemini LLM"""
        try:
            # Reload user preferences to get the latest settings
            self.user_preferences = self._load_user_preferences()
            
            # Add user message to history
            self.chat_history.append({"role": "user", "content": message})
            
            # Truncate history if it gets too long
            if len(self.chat_history) > self.max_history_length * 2:
                self.chat_history = self.chat_history[-self.max_history_length * 2:]
            
            # Detect message type for better prompt crafting
            message_type = self._detect_message_type(message)
            emotion = self._detect_emotion(message.lower())
            
            # Create context-aware prompt
            prompt = self._create_prompt(message, message_type, emotion)
            
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            
            # Clean up response text
            response_text = response.text.strip()
            
            # Add response to history
            self.chat_history.append({"role": "assistant", "content": response_text})
            
            # Save chat history to database
            self._save_chat_history()
            
            return response_text
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble connecting right now. Can we try again in a moment?"
    
    def _create_prompt(self, message: str, message_type: str, emotion: str) -> str:
        """Create a contextual prompt for Gemini based on message type and user preferences"""
        # Get user preferences
        communication_style = self.user_preferences.get('communication_style', 'balanced')
        interests = self.user_preferences.get('interests', [])
        
        # Base system prompt
        system_prompt = (
            "You are SoulMate, an empathetic AI companion. "
            "Your responses should be warm, supportive, and thoughtful. "
            f"Communication style preference: {communication_style}. "
        )
        
        # Add message type specific instructions
        if message_type == "greeting":
            system_prompt += "Respond to the user's greeting warmly and ask about their wellbeing."
        elif message_type == "question":
            system_prompt += "Answer the user's question thoughtfully but acknowledge limitations of your knowledge when appropriate."
        elif message_type == "feeling":
            system_prompt += f"The user seems to be expressing {emotion} emotions. Respond with empathy and appropriate support."
        
        # Add user interests context if available
        if interests:
            system_prompt += f" User interests: {', '.join(interests[:3])}."
        
        # Add conversation history context
        history_context = ""
        if len(self.chat_history) > 2:
            for entry in self.chat_history[-6:-2]:  # Get a few recent exchanges excluding the current one
                role = "User" if entry["role"] == "user" else "SoulMate"
                history_context += f"{role}: {entry['content']}\n"
        
        # Get relevant memories from memory vault
        memory_vault_context = self._get_memory_vault_context(message)
        if memory_vault_context:
            system_prompt += f"\n\nRelevant memories: {memory_vault_context}"
            
        # Construct final prompt
        full_prompt = f"{system_prompt}\n\nRecent conversation:\n{history_context}\nUser: {message}\nSoulMate:"
        
        return full_prompt
    
    def _get_memory_vault_context(self, message: str) -> str:
        """Get relevant memory vault items to provide context for the response"""
        try:
            # Get all memory vault entries
            vault_memories = self.memory_manager.get_from_memory_vault(self.user_id)
            
            # Extract key memories that might be relevant to this message
            context_items = []
            
            # First extract important memories like journal entries
            if 'journal' in vault_memories:
                journals = vault_memories['journal'][:2]  # Get most recent
                for entry in journals:
                    content = entry.get('content', '')
                    if isinstance(content, str) and content:
                        context_items.append(f"Journal: {content[:100]}...")
            
            # Extract important memories based on keywords in the message
            keywords = [word.lower() for word in message.split() if len(word) > 3]
            for category, memories in vault_memories.items():
                for memory in memories[:5]:  # Check up to 5 recent memories per category
                    content = memory.get('content', '')
                    if isinstance(content, dict):
                        content = str(content.get('text', ''))
                    
                    # Check if any keywords match
                    if any(keyword in content.lower() for keyword in keywords):
                        context_items.append(f"{category.capitalize()}: {content[:100]}...")
                        
                    # Limit to 3 memories maximum for context
                    if len(context_items) >= 3:
                        break
            
            return "\n".join(context_items)
        except Exception as e:
            print(f"Error getting memory vault context: {e}")
            return ""
    
    def _detect_message_type(self, message: str) -> str:
        """Detect the type of message from the user"""
        message_lower = message.lower()
        
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        elif "?" in message:
            return "question"
        elif any(feeling in message_lower for feeling in ["feel", "sad", "happy", "angry", "excited", "worried"]):
            return "feeling"
        else:
            return "default"
    
    def _load_user_preferences(self):
        """Load user preferences from the database"""
        try:
            return self.db_manager.get_preference(self.user_id, "user_preferences") or {}
        except:
            return {}
    
    def _detect_emotion(self, message):
        """Simple emotion detection"""
        emotions = {
            "happy": ["happy", "glad", "joy", "excited", "great"],
            "sad": ["sad", "unhappy", "depressed", "down", "upset"],
            "worried": ["worried", "anxious", "nervous", "fear", "scared"],
            "angry": ["angry", "mad", "frustrated", "annoyed"]
        }
        
        for emotion, keywords in emotions.items():
            if any(keyword in message for keyword in keywords):
                return emotion
        
        return "neutral"
    
    def should_train(self):
        return self.training_iterations < 5  # Simple training condition
    
    # Memory Vault Operations
    def add_to_memory_vault(self, memory_type: str, content: Dict[str, Any]) -> bool:
        """
        Add a memory to the user's memory vault
        
        Args:
            memory_type: Type of memory (e.g., 'journal', 'letter', 'reflection')
            content: The memory content
            
        Returns:
            Success status
        """
        return self.memory_manager.add_to_memory_vault(self.user_id, memory_type, content)
    
    def get_from_memory_vault(self, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memories from the user's memory vault
        
        Args:
            memory_type: Type of memory to retrieve (if None, returns all types)
            
        Returns:
            Dictionary of memory vault entries organized by category
        """
        return self.memory_manager.get_from_memory_vault(self.user_id, memory_type)
    
    def delete_from_memory_vault(self, memory_type: str, memory_id: str) -> bool:
        """
        Delete a specific memory from the vault
        
        Args:
            memory_type: The memory type
            memory_id: The ID of the memory to delete
            
        Returns:
            Success status
        """
        return self.memory_manager.delete_from_memory_vault(self.user_id, memory_type, memory_id)
        
    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for memories based on text content
        
        Args:
            query: Text to search for in memories
            
        Returns:
            List of matching memories
        """
        return self.memory_manager.search_similar_interactions(self.user_id, query)
    
    @staticmethod
    def get_user_by_username(username):
        """
        Get a user by username from the MongoDB database
        """
        user_data = db_manager.get_preference(username, "auth_data")
        return user_data
    
    @staticmethod
    def save_user(username, user_data):
        """
        Save user data to the MongoDB database
        """
        return db_manager.store_preference(username, "auth_data", user_data)
    
    def _load_chat_history(self):
        """Load chat history from the database"""
        try:
            # Check if we have chat history in the database
            history = self.db_manager.get_preference(self.user_id, "chat_history")
            if history:
                return history
            return []
        except Exception as e:
            print(f"Error loading chat history: {e}")
            return []
    
    def _save_chat_history(self):
        """Save chat history to the database"""
        try:
            # Keep only the last 20 messages to avoid overly large history
            history_to_save = self.chat_history[-20:] if len(self.chat_history) > 20 else self.chat_history
            self.db_manager.store_preference(self.user_id, "chat_history", history_to_save)
            return True
        except Exception as e:
            print(f"Error saving chat history: {e}")
            return False