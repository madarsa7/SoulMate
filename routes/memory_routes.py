from flask import jsonify, session, Blueprint, request
from middleware import login_required
from src.utils.database_manager import DatabaseManager
from datetime import datetime
import json
import logging
import uuid

memory_bp = Blueprint('memory', __name__)
db_manager = DatabaseManager()
logger = logging.getLogger(__name__)

# Add an additional route for /memories that redirects to memory-vault for compatibility
@memory_bp.route('/memories', methods=['GET'])
@login_required
def get_memories_redirect():
    """Redirect /memories to /memory-vault"""
    return get_memory_vault()

@memory_bp.route('/memory-vault', methods=['POST'])
@login_required
def save_to_memory_vault():
    """Save an encrypted memory to the user's private vault"""
    user_id = session.get('user_id')
    data = request.json
    memory_content = data.get('content')
    memory_type = data.get('type', 'general')
    
    if not user_id or not memory_content:
        return jsonify({'error': 'User ID and memory content are required'}), 400
    
    try:
        # In a production app, this would be encrypted with a user-specific key
        memory_entry = {
            "user_id": user_id,
            "content": memory_content,
            "type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "private": True
        }
        
        memory_id = None
        
        # Try to store in MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                result = db_manager.db.memory_vault.insert_one(memory_entry)
                memory_id = str(result.inserted_id)
                logger.info(f"Memory stored in MongoDB for user: {user_id}")
            except Exception as e:
                logger.error(f"MongoDB storage error: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available
        if db_manager.use_fallback:
            # Generate a unique ID for the memory
            memory_id = str(uuid.uuid4())
            memory_entry['_id'] = memory_id
            
            # Initialize fallback structure if needed
            if 'fallback_memory_vault' not in dir(db_manager) or db_manager.fallback_memory_vault is None:
                db_manager.fallback_memory_vault = {}
                
            if user_id not in db_manager.fallback_memory_vault:
                db_manager.fallback_memory_vault[user_id] = []
                
            db_manager.fallback_memory_vault[user_id].append(memory_entry)
            logger.info(f"Memory stored in fallback storage for user: {user_id}")
            
        return jsonify({
            'success': True,
            'message': 'Memory saved to your private vault',
            'memory_id': memory_id
        })
        
    except Exception as e:
        logger.error(f"Error saving to memory vault: {str(e)}")
        return jsonify({'error': f'Error saving to memory vault: {str(e)}'}), 500

@memory_bp.route('/memory-vault', methods=['GET'])
@login_required
def get_memory_vault():
    """Retrieve memories from the user's private vault"""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        memories = []
        
        # Try to get memories from MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                memories = list(db_manager.db.memory_vault.find({"user_id": user_id}))
                
                # Convert ObjectId to string for JSON serialization
                for memory in memories:
                    memory['_id'] = str(memory['_id'])
                    
                logger.info(f"Memories retrieved from MongoDB for user: {user_id}")
            except Exception as e:
                logger.error(f"MongoDB retrieval error: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available
        if db_manager.use_fallback:
            # Initialize container if needed
            if 'fallback_memory_vault' not in dir(db_manager) or db_manager.fallback_memory_vault is None:
                db_manager.fallback_memory_vault = {}
                
            # Get memories from fallback storage
            memories = db_manager.fallback_memory_vault.get(user_id, [])
            logger.info(f"Memories retrieved from fallback storage for user: {user_id}")
        
        return jsonify({
            'success': True,
            'memories': memories
        })
        
    except Exception as e:
        logger.error(f"Error retrieving memory vault: {str(e)}")
        return jsonify({'error': f'Error retrieving memory vault: {str(e)}'}), 500

@memory_bp.route('/memory-vault/<memory_id>', methods=['DELETE'])
@login_required
def delete_memory(memory_id):
    """Delete a memory from the vault"""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        deleted = False
        
        # Try to delete from MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                from bson.objectid import ObjectId
                
                result = db_manager.db.memory_vault.delete_one({
                    "_id": ObjectId(memory_id), 
                    "user_id": user_id
                })
                
                if result.deleted_count > 0:
                    deleted = True
                    logger.info(f"Memory deleted from MongoDB for user: {user_id}")
            except Exception as e:
                logger.error(f"MongoDB delete error: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available or delete failed
        if db_manager.use_fallback or not deleted:
            # Initialize container if needed
            if 'fallback_memory_vault' not in dir(db_manager) or db_manager.fallback_memory_vault is None:
                db_manager.fallback_memory_vault = {}
                
            # Get memories from fallback storage
            memories = db_manager.fallback_memory_vault.get(user_id, [])
            for i, memory in enumerate(memories):
                if str(memory.get('_id')) == memory_id:
                    memories.pop(i)
                    deleted = True
                    logger.info(f"Memory deleted from fallback storage for user: {user_id}")
                    break
        
        if not deleted:
            return jsonify({
                'error': 'Memory not found'
            }), 404
            
        return jsonify({
            'success': True,
            'message': 'Memory deleted successfully'
        })
            
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        return jsonify({'error': f'Error deleting memory: {str(e)}'}), 500