import logging
from flask import Blueprint, request, jsonify, session
from middleware import login_required
from src.utils.database_manager import DatabaseManager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize blueprint
preferences_bp = Blueprint('preferences', __name__)

# Initialize database manager
mongo_uri = os.getenv('MONGODB_URI')
use_mongo = os.getenv('USE_MONGODB', 'true').lower() == 'true'
db_manager = DatabaseManager(mongo_uri=mongo_uri, use_mongo=use_mongo)

@preferences_bp.route('/preferences', methods=['POST'])
@login_required
def update_preferences():
    """
    Update user preferences
    """
    # Get user ID from session
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": True, "message": "Not authenticated"}), 401
    
    # Get preferences from request JSON
    try:
        preferences = request.json
        if not preferences or not isinstance(preferences, dict):
            return jsonify({"error": True, "message": "Invalid preferences data"}), 400
        
        # Log the preferences being saved
        logger.info(f"Saving preferences for user {user_id}: {preferences}")
        
        # Store preferences in database
        success = db_manager.store_preference(user_id, "user_preferences", preferences)
        
        if success:
            logger.info(f"Updated preferences for user {user_id}")
            return jsonify({"error": False, "message": "Preferences updated successfully"})
        else:
            logger.error(f"Failed to update preferences for user {user_id}")
            return jsonify({"error": True, "message": "Failed to update preferences"}), 500
            
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        return jsonify({"error": True, "message": f"Server error: {str(e)}"}), 500

@preferences_bp.route('/preferences', methods=['GET'])
@login_required
def get_preferences():
    """
    Get user preferences
    """
    # Get user ID from session
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": True, "message": "Not authenticated"}), 401
    
    try:
        # Get preferences from database
        preferences = db_manager.get_preference(user_id, "user_preferences", {})
        
        logger.info(f"Retrieved preferences for user {user_id}")
        return jsonify({"error": False, "preferences": preferences})
            
    except Exception as e:
        logger.error(f"Error retrieving preferences: {e}")
        return jsonify({"error": True, "message": f"Server error: {str(e)}"}), 500