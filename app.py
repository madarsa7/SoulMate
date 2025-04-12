import os
import logging
from datetime import timedelta
from flask import Flask, render_template
from flask_socketio import SocketIO
from dotenv import load_dotenv

# Import route blueprints
from routes.auth_routes import auth_bp
from routes.chat_routes import chat_bp
from routes.analytics_routes import analytics_bp
from routes.memory_routes import memory_bp
from routes.wellness_routes import wellness_bp
from routes.journal_routes import journal_bp
from routes.insights_routes import insights_bp
from routes.preferences_routes import preferences_bp
from socket_events import register_socket_events

# Import utils
from src.utils.incremental_trainer import IncrementalTrainer
from src.utils.database_manager import DatabaseManager

# Basic configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')
app.permanent_session_lifetime = timedelta(days=30)

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# Register socket events
register_socket_events(socketio)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api')
app.register_blueprint(chat_bp, url_prefix='/api')
app.register_blueprint(analytics_bp, url_prefix='/api')
app.register_blueprint(memory_bp, url_prefix='/api')
app.register_blueprint(wellness_bp, url_prefix='/api')
app.register_blueprint(journal_bp, url_prefix='/api')
app.register_blueprint(insights_bp, url_prefix='/api')
app.register_blueprint(preferences_bp, url_prefix='/api')

# Global incremental trainer instance
incremental_trainer = None

# Initialize incremental trainer
def initialize_incremental_trainer():
    global incremental_trainer
    
    try:
        # Initialize database manager for the trainer
        mongo_uri = os.getenv('MONGODB_URI')
        use_mongo = os.getenv('USE_MONGODB', 'true').lower() == 'true'
        db_manager = DatabaseManager(mongo_uri=mongo_uri, use_mongo=use_mongo)
        
        # Initialize and start the incremental trainer
        incremental_trainer = IncrementalTrainer(db_manager=db_manager)
        incremental_trainer.start_scheduler()
        
        logger.info("Incremental trainer initialized and scheduler started")
    except Exception as e:
        logger.error(f"Error initializing incremental trainer: {e}")

# Main routes for rendering HTML templates
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

# API route for model training status
@app.route('/api/training/status')
def training_status():
    global incremental_trainer
    
    if incremental_trainer:
        return incremental_trainer.get_training_status()
    else:
        return {"error": "Incremental trainer not initialized"}, 500

# Main entry point
if __name__ == '__main__':
    # Initialize incremental trainer
    initialize_incremental_trainer()
    
    # Start the Flask app
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port)