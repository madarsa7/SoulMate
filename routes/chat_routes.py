from flask import request, jsonify, session, Blueprint
from middleware import login_required
from models.user_model import UserModel
from datetime import datetime
from src.utils.database_manager import DatabaseManager
import logging

chat_bp = Blueprint('chat', __name__)
db_manager = DatabaseManager()
logger = logging.getLogger(__name__)

@chat_bp.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    user_id = session['user_id']
    
    # Create user model instance for this user
    model = UserModel(user_id)
    
    # Log debugging information about preferences and history
    logger.info(f"User preferences: {model.user_preferences}")
    logger.info(f"Chat history length before: {len(model.chat_history)}")
    
    # Generate response (this will also save the chat history)
    response = model.generate_response(message)
    
    logger.info(f"Generated response using preferences: communication_style={model.user_preferences.get('communication_style', 'default')}")
    logger.info(f"Chat history length after: {len(model.chat_history)}")
    
    # No need to store chat separately as UserModel now handles this
    return jsonify({
        'response': response,
        'preferences_applied': {
            'communication_style': model.user_preferences.get('communication_style', 'default'),
            'interests': model.user_preferences.get('interests', [])
        }
    })