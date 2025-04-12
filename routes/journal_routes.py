from flask import jsonify, session, Blueprint, request
from middleware import login_required
from src.utils.database_manager import DatabaseManager
from datetime import datetime
from bson.objectid import ObjectId
import json
import logging
import uuid

journal_bp = Blueprint('journal', __name__)
db_manager = DatabaseManager()
logger = logging.getLogger(__name__)

@journal_bp.route('/journal', methods=['POST'])
@login_required
def save_journal_entry():
    """Save a journal entry to the database"""
    user_id = session.get('user_id')
    data = request.json
    
    if not user_id or not data:
        return jsonify({'error': 'Missing data'}), 400
    
    entry_content = data.get('content')
    entry_title = data.get('title', 'Untitled Entry')
    entry_tags = data.get('tags', [])
    
    if not entry_content:
        return jsonify({'error': 'Journal content is required'}), 400
    
    try:
        # Create the journal entry document
        journal_entry = {
            "user_id": user_id,
            "title": entry_title,
            "content": entry_content,
            "tags": entry_tags,
            "timestamp": datetime.now().isoformat(),
            "mood": data.get('mood'),
            "edited": False
        }
        
        entry_id = None
        
        # Try to store in MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                result = db_manager.db.journal_entries.insert_one(journal_entry)
                entry_id = str(result.inserted_id)
                logger.info(f"Journal entry stored in MongoDB for user: {user_id}")
            except Exception as e:
                logger.error(f"MongoDB storage error: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available
        if db_manager.use_fallback:
            # Generate a unique ID for the entry
            entry_id = str(uuid.uuid4())
            journal_entry['_id'] = entry_id
            
            # Initialize user's journal entries if not exists
            if 'fallback_journal' not in dir(db_manager) or db_manager.fallback_journal is None:
                db_manager.fallback_journal = {}
                
            if user_id not in db_manager.fallback_journal:
                db_manager.fallback_journal[user_id] = []
                
            db_manager.fallback_journal[user_id].append(journal_entry)
            logger.info(f"Journal entry stored in fallback storage for user: {user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Journal entry saved',
            'entry_id': entry_id
        })
        
    except Exception as e:
        logger.error(f"Error saving journal entry: {str(e)}")
        return jsonify({'error': f'Error saving journal entry: {str(e)}'}), 500

@journal_bp.route('/journal', methods=['GET'])
@login_required
def get_journal_entries():
    """Get all journal entries for the current user"""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        entries = []
        
        # Try to get entries from MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                entries = list(db_manager.db.journal_entries.find(
                    {"user_id": user_id}
                ).sort("timestamp", -1))
                
                # Convert ObjectId to string for JSON serialization
                for entry in entries:
                    entry['_id'] = str(entry['_id'])
                    
                logger.info(f"Journal entries retrieved from MongoDB for user: {user_id}")
            except Exception as e:
                logger.error(f"MongoDB retrieval error: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available
        if db_manager.use_fallback:
            # Initialize container if needed
            if 'fallback_journal' not in dir(db_manager) or db_manager.fallback_journal is None:
                db_manager.fallback_journal = {}
                
            # Get entries from fallback storage
            entries = db_manager.fallback_journal.get(user_id, [])
            # Sort by timestamp in descending order
            entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            logger.info(f"Journal entries retrieved from fallback storage for user: {user_id}")
        
        return jsonify({
            'success': True,
            'entries': entries
        })
        
    except Exception as e:
        logger.error(f"Error retrieving journal entries: {str(e)}")
        return jsonify({'error': f'Error retrieving journal entries: {str(e)}'}), 500

# Handle the rest of the journal operations with similar fallback logic
@journal_bp.route('/journal/<entry_id>', methods=['GET'])
@login_required
def get_journal_entry(entry_id):
    """Get a specific journal entry"""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        entry = None
        
        # Try to get entry from MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                entry = db_manager.db.journal_entries.find_one({
                    "_id": ObjectId(entry_id),
                    "user_id": user_id
                })
                
                if entry:
                    # Convert ObjectId to string for JSON serialization
                    entry['_id'] = str(entry['_id'])
            except Exception as e:
                logger.error(f"MongoDB retrieval error: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available or entry not found
        if db_manager.use_fallback or not entry:
            # Initialize container if needed
            if 'fallback_journal' not in dir(db_manager) or db_manager.fallback_journal is None:
                db_manager.fallback_journal = {}
                
            # Get entries from fallback storage
            entries = db_manager.fallback_journal.get(user_id, [])
            for e in entries:
                if str(e.get('_id')) == entry_id:
                    entry = e
                    break
        
        if not entry:
            return jsonify({'error': 'Journal entry not found'}), 404
        
        return jsonify({
            'success': True,
            'entry': entry
        })
        
    except Exception as e:
        logger.error(f"Error retrieving journal entry: {str(e)}")
        return jsonify({'error': f'Error retrieving journal entry: {str(e)}'}), 500

@journal_bp.route('/journal/<entry_id>', methods=['PUT'])
@login_required
def update_journal_entry(entry_id):
    """Update a journal entry"""
    user_id = session.get('user_id')
    data = request.json
    
    if not user_id or not data:
        return jsonify({'error': 'Missing data'}), 400
    
    try:
        updated = False
        
        # Try to update in MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                result = db_manager.db.journal_entries.update_one(
                    {"_id": ObjectId(entry_id), "user_id": user_id},
                    {"$set": {
                        "title": data.get('title'),
                        "content": data.get('content'),
                        "tags": data.get('tags', []),
                        "mood": data.get('mood'),
                        "edited": True,
                        "edited_at": datetime.now().isoformat()
                    }}
                )
                
                if result.matched_count > 0:
                    updated = True
            except Exception as e:
                logger.error(f"MongoDB update error: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available or update failed
        if db_manager.use_fallback or not updated:
            # Initialize container if needed
            if 'fallback_journal' not in dir(db_manager) or db_manager.fallback_journal is None:
                db_manager.fallback_journal = {}
                
            # Get entries from fallback storage
            entries = db_manager.fallback_journal.get(user_id, [])
            for i, e in enumerate(entries):
                if str(e.get('_id')) == entry_id:
                    entries[i].update({
                        "title": data.get('title'),
                        "content": data.get('content'),
                        "tags": data.get('tags', []),
                        "mood": data.get('mood'),
                        "edited": True,
                        "edited_at": datetime.now().isoformat()
                    })
                    updated = True
                    break
        
        if not updated:
            return jsonify({'error': 'Journal entry not found'}), 404
        
        return jsonify({
            'success': True,
            'message': 'Journal entry updated'
        })
        
    except Exception as e:
        logger.error(f"Error updating journal entry: {str(e)}")
        return jsonify({'error': f'Error updating journal entry: {str(e)}'}), 500

@journal_bp.route('/journal/<entry_id>', methods=['DELETE'])
@login_required
def delete_journal_entry(entry_id):
    """Delete a journal entry"""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        deleted = False
        
        # Try to delete from MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                result = db_manager.db.journal_entries.delete_one({
                    "_id": ObjectId(entry_id),
                    "user_id": user_id
                })
                
                if result.deleted_count > 0:
                    deleted = True
            except Exception as e:
                logger.error(f"MongoDB delete error: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available or delete failed
        if db_manager.use_fallback or not deleted:
            # Initialize container if needed
            if 'fallback_journal' not in dir(db_manager) or db_manager.fallback_journal is None:
                db_manager.fallback_journal = {}
                
            # Get entries from fallback storage
            entries = db_manager.fallback_journal.get(user_id, [])
            for i, e in enumerate(entries):
                if str(e.get('_id')) == entry_id:
                    entries.pop(i)
                    deleted = True
                    break
        
        if not deleted:
            return jsonify({'error': 'Journal entry not found'}), 404
        
        return jsonify({
            'success': True,
            'message': 'Journal entry deleted'
        })
        
    except Exception as e:
        logger.error(f"Error deleting journal entry: {str(e)}")
        return jsonify({'error': f'Error deleting journal entry: {str(e)}'}), 500