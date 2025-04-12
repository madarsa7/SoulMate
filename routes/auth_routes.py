from flask import request, jsonify, session, Blueprint
from middleware import login_required
from models.user_model import UserModel
from src.utils.auth_manager import AuthManager
from src.utils.database_manager import DatabaseManager
import logging

auth_bp = Blueprint('auth', __name__)
db_manager = DatabaseManager(use_mongo=True)
auth_manager = AuthManager(db_manager)

logger = logging.getLogger(__name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    # Use AuthManager for secure registration
    success, message, user_info = auth_manager.register_user(username, password, email)
    
    if not success:
        return jsonify({'error': message}), 400
    
    return jsonify({'success': True, 'user_id': user_info.get('user_id')})

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    # Debug database connection
    logger.info(f"Login attempt for user: {username}")
    logger.info(f"MongoDB fallback status: {db_manager.use_fallback}")
    logger.info(f"Database connection: {'Connected' if db_manager.db is not None else 'Not connected'}")
    
    # Check if user exists in preferences
    existing_user = db_manager.get_preference(username, "auth_data")
    logger.info(f"User found in database: {existing_user is not None}")
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    # Use AuthManager for secure authentication
    success, message, user_info = auth_manager.authenticate(username, password)
    
    if not success:
        logger.error(f"Authentication failed: {message}")
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Store user_id in session
    session['user_id'] = user_info.get('user_id')
    logger.info(f"User authenticated successfully: {username}")
    
    return jsonify({
        'success': True, 
        'user_id': session['user_id'],
        'token': user_info.get('token')
    })

@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@auth_bp.route('/user', methods=['GET'])
@login_required
def get_user_info():
    """Get information about the currently logged in user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    user_info = auth_manager.get_user_info(user_id)
    
    if not user_info:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'success': True,
        'user': {
            'user_id': user_id,
            'username': user_info.get('username'),
            'email': user_info.get('email')
        }
    })