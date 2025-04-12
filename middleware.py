from flask import request, jsonify, session
from functools import wraps

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Login required'}), 401
        return f(*args, **kwargs)
    return wrapper