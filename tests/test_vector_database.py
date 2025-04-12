import sys
import os
import logging
from datetime import datetime

# Add project root to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.memory_manager import MemoryManager
from src.models.language_model import SoulMateLanguageModel

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vector_database():
    """Test the vector database implementation with the SoulMate.AGI app"""
    logger.info("Starting vector database test")
    
    # Create a test user
    test_user_id = f"test_user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.info(f"Created test user: {test_user_id}")
    
    # Initialize the memory manager directly
    memory_manager = MemoryManager(test_user_id)
    
    # Store some sample conversations
    logger.info("Storing sample conversations in vector database")
    
    # List of sample conversations (user input, AI response)
    sample_conversations = [
        ("Hello, how are you today?", "I'm doing well, thank you for asking! How about you?"),
        ("I'm feeling sad today", "I'm sorry to hear that. Would you like to talk about what's bothering you?"),
        ("What's your favorite color?", "As an AI, I don't have personal preferences, but I appreciate all colors!"),
        ("Tell me about machine learning", "Machine learning is a subset of AI focused on building systems that learn from data."),
        ("I had a great day at work today!", "That's wonderful! Would you like to share what made your day so good?")
    ]
    
    # Add each conversation to the memory manager
    for user_input, ai_response in sample_conversations:
        success = memory_manager.add_memory(
            user_input=user_input,
            ai_response=ai_response,
            context={"timestamp": datetime.now().isoformat()}
        )
        if success:
            logger.info(f"Successfully stored: {user_input}")
        else:
            logger.error(f"Failed to store: {user_input}")
    
    # Test semantic search
    logger.info("Testing semantic search for similar conversations")
    
    # List of test queries
    test_queries = [
        "I'm not feeling good today",
        "Do you have any preferences?",
        "Tell me about artificial intelligence",
        "I had a positive experience"
    ]
    
    # Search for similar conversations for each query
    for query in test_queries:
        logger.info(f"Searching for conversations similar to: '{query}'")
        similar = memory_manager.search_similar_interactions(query, k=2)
        
        if similar:
            logger.info(f"Found {len(similar)} similar conversations:")
            for i, item in enumerate(similar):
                logger.info(f"  {i+1}. User: {item['user_input']}")
                logger.info(f"     AI: {item['ai_response']}")
                logger.info(f"     Similarity score: {item['similarity_score']:.4f}")
        else:
            logger.warning(f"No similar conversations found for: '{query}'")
    
    # Test preference storage
    logger.info("Testing preference storage")
    memory_manager.store_user_preference("favorite_topics", ["technology", "science", "art"])
    memory_manager.store_user_preference("chat_frequency", "daily")
    
    # Retrieve preferences
    preferences = memory_manager.get_user_preferences()
    logger.info(f"Retrieved preferences: {preferences}")
    
    # Test the language model with vector database integration
    logger.info("Testing SoulMateLanguageModel with vector database")
    try:
        language_model = SoulMateLanguageModel(test_user_id, use_api=False)
        
        # Generate a response using the context from vector database
        test_input = "I'm having a bad day"
        logger.info(f"Testing model response to: '{test_input}'")
        
        response = language_model.generate_personalized_response(test_input)
        logger.info(f"Model response: {response}")
        
        # Check if context was retrieved
        similar = language_model.find_similar_interactions(test_input)
        logger.info(f"Found {len(similar)} similar conversations for context")
        
    except Exception as e:
        logger.error(f"Error testing language model: {e}")
    
    logger.info("Vector database test completed")

if __name__ == "__main__":
    test_vector_database()