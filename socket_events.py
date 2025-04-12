import logging
from flask import session, request
from flask_socketio import emit, join_room

logger = logging.getLogger(__name__)

def register_socket_events(socketio):
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('message')
    def handle_message(data):
        user_id = session.get('user_id')
        if user_id:
            emit('response', {'data': f"Echo: {data}"}, room=request.sid)
    
    @socketio.on('join')
    def handle_join(data):
        user_id = data.get('user_id')
        if user_id:
            # Join a room specific to this user for targeted events
            join_room(user_id)
            logger.info(f"User {user_id} joined their room")
            
    @socketio.on('stream_message')
    def handle_stream_message(data):
        """Handle streaming message for real-time typing effect"""
        user_id = session.get('user_id') or data.get('user_id')
        message = data.get('message', '')
        
        if not user_id or not message:
            return
        
        # Start typing indicator
        emit('typing_indicator', {'status': 'started'}, room=user_id)
        
        # In a full implementation, this would stream tokens one by one
        # For now, we'll send the complete response after a brief delay
        from src.models.language_model import SoulMateLanguageModel
        
        try:
            # Get model for this user
            model = SoulMateLanguageModel(user_id)
            response = model.generate_personalized_response(message, {'type': 'stream'})
            
            # Send the complete response
            emit('ai_response', {'response': response}, room=user_id)
            
            # Stop typing indicator
            emit('typing_indicator', {'status': 'stopped'}, room=user_id)
        except Exception as e:
            logger.error(f"Error in stream_message: {e}")
            emit('error', {'message': 'Error processing your message'}, room=user_id)
    
    @socketio.on('training_status')
    def check_training_status(data):
        """Check if user model is ready for training"""
        user_id = session.get('user_id') or data.get('user_id')
        
        if not user_id:
            return
        
        try:
            from src.models.language_model import SoulMateLanguageModel
            model = SoulMateLanguageModel(user_id)
            is_ready = model.should_train()
            
            emit('training_status', {
                'ready': is_ready,
                'user_id': user_id,
                'adaptation_level': model.adaptation_level,
                'persona_divergence': model.persona_divergence,
                'training_iterations': model.training_iterations
            }, room=user_id)
        except Exception as e:
            logger.error(f"Error checking training status: {e}")
            emit('error', {'message': 'Error checking training status'}, room=user_id)