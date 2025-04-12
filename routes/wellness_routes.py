import logging
from flask import Blueprint, request, jsonify, session
from src.utils.wellness_manager import WellnessManager
from src.utils.emotion_analyzer import EmotionAnalyzer
from src.utils.database_manager import DatabaseManager
from src.models.language_model import SoulMateLanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize blueprint
wellness_bp = Blueprint('wellness', __name__)

# Initialize managers/services (will be properly initialized during request)
wellness_manager = None
emotion_analyzer = None
db_manager = None

@wellness_bp.before_request
def before_request():
    """Initialize required components before handling request"""
    global wellness_manager, emotion_analyzer, db_manager
    
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    # Initialize database manager if needed
    if db_manager is None:
        try:
            import os
            mongo_uri = os.getenv('MONGODB_URI')
            use_mongo = os.getenv('USE_MONGODB', 'true').lower() == 'true'
            db_manager = DatabaseManager(mongo_uri=mongo_uri, use_mongo=use_mongo)
            logger.info("Database manager initialized for wellness routes")
        except Exception as e:
            logger.error(f"Error initializing database manager: {e}")
            return jsonify({'error': 'Service temporarily unavailable'}), 503
    
    # Initialize emotion analyzer if needed
    if emotion_analyzer is None:
        try:
            emotion_analyzer = EmotionAnalyzer(db_manager)
            logger.info("Emotion analyzer initialized for wellness routes")
        except Exception as e:
            logger.error(f"Error initializing emotion analyzer: {e}")
            return jsonify({'error': 'Service temporarily unavailable'}), 503
    
    # Initialize wellness manager if needed
    if wellness_manager is None:
        try:
            # Get language model for story generation
            try:
                user_id = session.get('user_id')
                language_model = SoulMateLanguageModel(user_id=user_id)
            except Exception as e:
                logger.warning(f"Could not initialize language model for wellness manager: {e}")
                language_model = None
                
            wellness_manager = WellnessManager(language_model=language_model, db_manager=db_manager)
            logger.info("Wellness manager initialized for wellness routes")
        except Exception as e:
            logger.error(f"Error initializing wellness manager: {e}")
            return jsonify({'error': 'Service temporarily unavailable'}), 503

@wellness_bp.route('/activity', methods=['GET'])
def get_wellness_activity():
    """Get a personalized wellness activity"""
    try:
        user_id = session.get('user_id')
        
        # Get emotion from request or detect from recent interactions
        emotion = request.args.get('emotion', None)
        
        # Get activity from wellness manager
        activity = wellness_manager.get_personalized_wellness_activity(user_id, emotion)
        
        return jsonify({
            'success': True,
            'activity': activity
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting wellness activity: {e}")
        return jsonify({'error': 'Could not generate wellness activity'}), 500

@wellness_bp.route('/activity/feedback', methods=['POST'])
def record_activity_feedback():
    """Record feedback on a wellness activity"""
    try:
        user_id = session.get('user_id')
        data = request.json
        
        if not data:
            return jsonify({'error': 'Missing required data'}), 400
        
        activity_type = data.get('activity_type')
        rating = data.get('rating')
        feedback = data.get('feedback')
        
        if not activity_type or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Invalid activity type or rating'}), 400
        
        # Record feedback
        success = wellness_manager.record_activity_feedback(
            user_id=user_id,
            activity_type=activity_type,
            rating=rating,
            feedback=feedback
        )
        
        if success:
            return jsonify({'success': True, 'message': 'Feedback recorded'}), 200
        else:
            return jsonify({'error': 'Could not record feedback'}), 500
        
    except Exception as e:
        logger.error(f"Error recording activity feedback: {e}")
        return jsonify({'error': 'Could not record feedback'}), 500

@wellness_bp.route('/loneliness-support', methods=['GET'])
def get_loneliness_support():
    """Get personalized loneliness support strategies"""
    try:
        user_id = session.get('user_id')
        
        # Get loneliness score from request or analyze
        loneliness_score = request.args.get('score')
        if loneliness_score is not None:
            try:
                loneliness_score = float(loneliness_score)
                if loneliness_score < 0 or loneliness_score > 1:
                    loneliness_score = None  # Will be calculated by the manager
            except ValueError:
                loneliness_score = None
        
        # Get support from wellness manager
        support = wellness_manager.get_loneliness_support(
            user_id=user_id,
            loneliness_score=loneliness_score
        )
        
        return jsonify({
            'success': True,
            'support': support
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting loneliness support: {e}")
        return jsonify({'error': 'Could not generate loneliness support'}), 500

@wellness_bp.route('/daily-plan', methods=['GET'])
def get_daily_wellness_plan():
    """Get a personalized daily wellness plan"""
    try:
        user_id = session.get('user_id')
        
        # Get plan from wellness manager
        plan = wellness_manager.get_daily_wellness_plan(user_id)
        
        return jsonify({
            'success': True,
            'plan': plan
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting daily wellness plan: {e}")
        return jsonify({'error': 'Could not generate daily wellness plan'}), 500

@wellness_bp.route('/emotional-insight', methods=['GET'])
def get_emotional_insight():
    """Get insights about the user's emotional patterns"""
    try:
        user_id = session.get('user_id')
        
        # Get insight from emotion analyzer
        insight = emotion_analyzer.generate_emotional_insight(user_id)
        
        return jsonify({
            'success': True,
            'insight': insight
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting emotional insight: {e}")
        return jsonify({'error': 'Could not generate emotional insight'}), 500

@wellness_bp.route('/emotion-trend', methods=['GET'])
def get_emotion_trend():
    """Get the user's emotion trend over a specified time window"""
    try:
        user_id = session.get('user_id')
        
        # Get time window from request
        time_window_hours = request.args.get('hours', 24)
        try:
            time_window_hours = int(time_window_hours)
            if time_window_hours < 1:
                time_window_hours = 24
        except ValueError:
            time_window_hours = 24
        
        # Get trend from emotion analyzer
        trend = emotion_analyzer.get_emotion_trend(
            user_id=user_id,
            time_window_hours=time_window_hours
        )
        
        return jsonify({
            'success': True,
            'trend': trend
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting emotion trend: {e}")
        return jsonify({'error': 'Could not generate emotion trend'}), 500

@wellness_bp.route('/loneliness-analysis', methods=['GET'])
def get_loneliness_analysis():
    """Get analysis of user's loneliness risk"""
    try:
        user_id = session.get('user_id')
        
        # Get analysis from emotion analyzer
        analysis = emotion_analyzer.analyze_loneliness_risk(user_id)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting loneliness analysis: {e}")
        return jsonify({'error': 'Could not generate loneliness analysis'}), 500

@wellness_bp.route('/mood-activities', methods=['GET'])
def get_mood_support_activities():
    """Get personalized mood support activities"""
    try:
        user_id = session.get('user_id')
        
        # Get activities from emotion analyzer
        activities = emotion_analyzer.get_mood_support_activities(user_id)
        
        return jsonify({
            'success': True,
            'activities': activities
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting mood support activities: {e}")
        return jsonify({'error': 'Could not generate mood support activities'}), 500