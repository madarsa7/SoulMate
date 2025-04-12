from flask import jsonify, session, Blueprint
from middleware import login_required
from src.utils.database_manager import DatabaseManager

analytics_bp = Blueprint('analytics', __name__)
db_manager = DatabaseManager()

@analytics_bp.route('/summary', methods=['GET'])
@login_required
def summary():
    user_id = session['user_id']
    
    # Get chat data from MongoDB
    chats = list(db_manager.db.chat_history.find({"user_id": user_id}).sort("timestamp", -1).limit(100))
    
    # Simple summary logic
    summary = f"You've had {len(chats)} conversations recently."
    if chats:
        last_chat = chats[0]['message'][:50] + '...' if len(chats[0]['message']) > 50 else chats[0]['message']
        summary += f" Last chat: {last_chat}"
    
    return jsonify({
        'summary': summary,
        'chat_count': len(chats),
        'last_chat': chats[0]['message'] if chats else None
    })