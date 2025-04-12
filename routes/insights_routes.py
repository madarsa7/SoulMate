from flask import jsonify, session, Blueprint, request
from middleware import login_required
from src.utils.database_manager import DatabaseManager
from src.utils.emotion_analyzer import EmotionAnalyzer
from src.utils.memory_manager import MemoryManager
from datetime import datetime, timedelta
import logging
import random
import collections
import re
import statistics
import nltk
import os
import json

# Setup logger first
logger = logging.getLogger(__name__)

# Initialize NLTK resources if not already present
nltk_available = True
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True, raise_on_error=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK punkt: {str(e)}")
        nltk_available = False

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True, raise_on_error=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK stopwords: {str(e)}")
        nltk_available = False

# Define a fallback tokenizer function in case NLTK is not available
def safe_tokenize(text):
    if nltk_available:
        try:
            return word_tokenize(text)
        except:
            pass
    # Simple fallback tokenization
    return re.findall(r'\b\w+\b', text)

def safe_stopwords():
    if nltk_available:
        try:
            return set(stopwords.words('english'))
        except:
            pass
    # Basic English stopwords list as fallback
    return {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

insights_bp = Blueprint('insights', __name__)
db_manager = DatabaseManager()

@insights_bp.route('/insights', methods=['GET'])
@login_required
def get_insights():
    """Get insights based on user's chat history and interactions"""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        # Initialize memory manager and emotion analyzer
        memory_manager = MemoryManager(user_id=user_id, db_manager=db_manager)
        emotion_analyzer = EmotionAnalyzer()
        
        # Get data for analysis
        chat_history = get_user_chat_history(user_id)
        journal_entries = get_user_journal_entries(user_id)
        user_preferences = memory_manager.get_user_preferences(user_id)
        
        # Get time range parameter (optional)
        time_range = request.args.get('timeRange', 'all')  # all, week, month
        
        # Filter data by time range if needed
        if time_range != 'all':
            chat_history = filter_by_time_range(chat_history, time_range)
            journal_entries = filter_by_time_range(journal_entries, time_range)
        
        # Generate insights
        insights = []
        metrics = {}
        
        # Get previous period data for comparison if relevant
        previous_chat_history = None
        if time_range == 'week':
            previous_chat_history = get_previous_period_data(user_id, 'week')
        elif time_range == 'month':
            previous_chat_history = get_previous_period_data(user_id, 'month')
        
        # Add interaction pattern insights
        interaction_insights = analyze_interaction_patterns(user_id, chat_history)
        insights.extend(interaction_insights)
        
        # Add emotional insights if we have journal entries
        if journal_entries:
            emotional_insights = analyze_emotional_patterns(user_id, journal_entries, emotion_analyzer)
            insights.extend(emotional_insights)
        
        # Add topic insights
        topic_insights = analyze_conversation_topics(user_id, chat_history)
        insights.extend(topic_insights)
        
        # Add response time insights
        response_time_insights = analyze_response_times(user_id, chat_history)
        insights.extend(response_time_insights)
        metrics['responseTime'] = get_response_time_metrics(chat_history)
        
        # Add conversation depth insights
        depth_insights = analyze_conversation_depth(user_id, chat_history)
        insights.extend(depth_insights)
        metrics['conversationDepth'] = get_conversation_depth_metrics(chat_history)
        
        # Add word usage pattern insights
        word_usage_insights = analyze_word_usage(user_id, chat_history)
        insights.extend(word_usage_insights)
        
        # Add comparative insights if previous data available
        if previous_chat_history:
            comparative_insights = generate_comparative_insights(user_id, chat_history, previous_chat_history, time_range)
            insights.extend(comparative_insights)
        
        # Add emotional progression tracking
        if journal_entries and len(journal_entries) >= 3:
            progression_insights = track_emotional_progression(user_id, journal_entries, emotion_analyzer)
            insights.extend(progression_insights)
        
        # Add personalized recommendations
        recommendation_insights = generate_recommendations(user_id, chat_history, journal_entries, user_preferences)
        insights.extend(recommendation_insights)
        
        # Add growth insights
        growth_insights = generate_growth_insights(user_id, chat_history, user_preferences)
        insights.extend(growth_insights)
        
        # Sort insights by creation date (newest first)
        insights = sorted(insights, key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Limit to a reasonable number
        insights = insights[:10]  # Take up to 10 insights
        
        # If no insights are available, provide a message
        if not insights:
            return jsonify({
                'success': True,
                'message': 'Not enough data to generate insights yet. Continue using the app to get personalized insights.',
                'insights': [],
                'metrics': {}
            })
        
        return jsonify({
            'success': True,
            'insights': insights,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return jsonify({'error': f'Error generating insights: {str(e)}')}), 500

def filter_by_time_range(data_list, time_range):
    """Filter data by time range"""
    if not data_list:
        return []
    
    now = datetime.now()
    cutoff_date = None
    
    if time_range == 'week':
        cutoff_date = now - timedelta(days=7)
    elif time_range == 'month':
        cutoff_date = now - timedelta(days=30)
    else:
        return data_list  # No filtering
    
    filtered_data = []
    for item in data_list:
        timestamp = None
        if isinstance(item.get('timestamp'), str):
            try:
                timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                try:
                    timestamp = datetime.fromisoformat(item['timestamp'])
                except ValueError:
                    continue
        elif isinstance(item.get('timestamp'), (int, float)):
            timestamp = datetime.fromtimestamp(item['timestamp'])
        
        if timestamp and timestamp >= cutoff_date:
            filtered_data.append(item)
    
    return filtered_data

def get_previous_period_data(user_id, period_type):
    """Get data from the previous period for comparison"""
    now = datetime.now()
    start_date = None
    end_date = None
    
    if period_type == 'week':
        start_date = now - timedelta(days=14)
        end_date = now - timedelta(days=7)
    elif period_type == 'month':
        start_date = now - timedelta(days=60)
        end_date = now - timedelta(days=30)
    else:
        return []  # Unsupported period type
    
    
    try:
        # Try to get data from MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                # Convert dates to timestamps for MongoDB query
                start_timestamp = start_date.timestamp()
                end_timestamp = end_date.timestamp()
                
                # Get chats from previous period
                previous_chat_history = list(db_manager.db.chat_history.find({
                    "user_id": user_id,
                    "timestamp": {"$gte": start_timestamp, "$lt": end_timestamp}
                }).sort("timestamp", -1))
                
                logger.info(f"Retrieved {len(previous_chat_history)} previous period chats from MongoDB")
            except Exception as e:
                logger.error(f"MongoDB retrieval error for previous period: {e}")
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available
        if db_manager.use_fallback:
            if hasattr(db_manager, 'fallback_chat_history') and user_id in db_manager.fallback_chat_history:
                all_chats = db_manager.fallback_chat_history.get(user_id, [])
                
                # Filter by date range
                for chat in all_chats:
                    timestamp = None
                    if isinstance(chat.get('timestamp'), str):
                        try:
                            timestamp = datetime.fromisoformat(chat['timestamp'].replace('Z', '+00:00'))
                        except ValueError:
                            try:
                                timestamp = datetime.fromisoformat(chat['timestamp'])
                            except ValueError:
                                continue
                    elif isinstance(chat.get('timestamp'), (int, float)):
                        timestamp = datetime.fromtimestamp(chat['timestamp'])
                    
                    if timestamp and start_date <= timestamp < end_date:
                        previous_chat_history.append(chat)
                
                logger.info(f"Retrieved {len(previous_chat_history)} previous period chats from fallback storage")
    except Exception as e:
        logger.error(f"Error retrieving previous period data: {e}")
    
    return previous_chat_history

def get_user_chat_history(user_id, limit=100):
    """Get recent chat history for the user"""
    chat_history = []
    
    try:
        # Try to get data from MongoDB if available
        if db_manager.db and not db_manager.use_fallback:
            try:
                # Get recent chats
                chat_history = list(db_manager.db.chat_history.find(
                    {"user_id": user_id}
                ).sort("timestamp", -1).limit(limit))
                
                logger.info(f"Retrieved {len(chat_history)} chats from MongoDB for insights")
            except Exception as e:
                logger.error(f"MongoDB retrieval error for chat history: {e}")
                # Fall back to in-memory storage
                db_manager.use_fallback = True
        
        # Use fallback storage if MongoDB is not available
        if db_manager.use_fallback:
            if hasattr(db_manager, 'fallback_chat_history') and user_id in db_manager.fallback_chat_history:
                chat_history = db_manager.fallback_chat_history.get(user_id, [])
                logger.info(f"Retrieved {len(chat_history)} chats from fallback storage for insights")
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
    
    return chat_history

def get_user_journal_entries(user_id, limit=20):
    """Get recent journal entries for the user"""
    journal_entries = []
    
    try:
        if db_manager.db and not db_manager.use_fallback:
            try:
                journal_entries = list(db_manager.db.journal_entries.find(
                    {"user_id": user_id}
                ).sort("timestamp", -1).limit(limit))
                
                logger.info(f"Retrieved {len(journal_entries)} journal entries from MongoDB")
            except Exception as e:
                logger.error(f"MongoDB retrieval error for journal entries: {e}")
                db_manager.use_fallback = True
        
        if db_manager.use_fallback:
            if hasattr(db_manager, 'fallback_journal') and user_id in db_manager.fallback_journal:
                journal_entries = db_manager.fallback_journal.get(user_id, [])
                logger.info(f"Retrieved {len(journal_entries)} journal entries from fallback storage")
    except Exception as e:
        logger.error(f"Error retrieving journal entries: {e}")
    
    return journal_entries

def analyze_interaction_patterns(user_id, chat_history):
    """Analyze interaction patterns from chat history"""
    insights = []
    
    if not chat_history:
        return insights
    
    # Convert timestamps to datetime objects
    chats_with_time = []
    for chat in chat_history:
        try:
            if isinstance(chat.get('timestamp'), str):
                timestamp = datetime.fromisoformat(chat.get('timestamp').replace('Z', '+00:00'))
            else:
                timestamp = datetime.fromtimestamp(chat.get('timestamp', 0))
            
            chats_with_time.append((timestamp, chat))
        except Exception as e:
            logger.warning(f"Error parsing timestamp: {e}")
    
    if not chats_with_time:
        return insights
    
    # Sort by timestamp
    chats_with_time.sort(key=lambda x: x[0])
    
    # Analyze time of day patterns
    hour_counts = collections.Counter([t.hour for t, _ in chats_with_time])
    
    # Determine when user is most active
    if hour_counts:
        most_active_hour, _ = hour_counts.most_common(1)[0]
        time_category = "mornings" if 5 <= most_active_hour < 12 else \
                        "afternoons" if 12 <= most_active_hour < 17 else \
                        "evenings" if 17 <= most_active_hour < 22 else "nights"
        
        insights.append({
            "id": f"time-pattern-{user_id[:8]}",
            "title": "Your Active Time",
            "description": f"You tend to engage more with your companion during {time_category}.",
            "type": "usage",
            "created_at": datetime.now().isoformat()
        })
    
    # Analyze conversation frequency
    if len(chats_with_time) >= 2:
        # Get first and last chat timestamps
        first_time, _ = chats_with_time[0]
        last_time, _ = chats_with_time[-1]
        
        # Calculate days between
        days_span = (last_time - first_time).days + 1
        
        if days_span > 0:
            chats_per_day = len(chats_with_time) / days_span
            
            if chats_per_day > 10:
                frequency_desc = "very frequently"
            elif chats_per_day > 5:
                frequency_desc = "frequently"
            elif chats_per_day > 1:
                frequency_desc = "regularly"
            else:
                frequency_desc = "occasionally"
            
            insights.append({
                "id": f"freq-pattern-{user_id[:8]}",
                "title": "Conversation Frequency",
                "description": f"You communicate with your SoulMate {frequency_desc}, with an average of {chats_per_day:.1f} messages per day.",
                "type": "communication",
                "created_at": datetime.now().isoformat()
            })
    
    return insights

def analyze_emotional_patterns(user_id, journal_entries, emotion_analyzer):
    """Analyze emotional patterns from journal entries"""
    insights = []
    
    if not journal_entries:
        return insights
    
    # Extract moods from journal entries
    moods = []
    for entry in journal_entries:
        if 'mood' in entry and entry['mood']:
            moods.append(entry['mood'])
    
    # Analyze mood trends
    if moods:
        mood_counts = collections.Counter(moods)
        total_moods = len(moods)
        
        # Get the most common mood
        most_common_mood, most_common_count = mood_counts.most_common(1)[0]
        
        # Calculate percentage for most common mood
        percentage = (most_common_count / total_moods) * 100
        
        insights.append({
            "id": f"mood-pattern-{user_id[:8]}",
            "title": "Mood Insights",
            "description": f"In your journal entries, you express feeling '{most_common_mood}' most frequently ({percentage:.0f}% of entries).",
            "type": "emotional",
            "created_at": datetime.now().isoformat()
        })
    
    # Analyze content of journal entries for sentiment
    if hasattr(emotion_analyzer, 'analyze_text'):
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for entry in journal_entries:
            if 'content' in entry and entry['content']:
                try:
                    result = emotion_analyzer.analyze_text(entry['content'])
                    sentiment = result.get('sentiment', 'neutral')
                    
                    if sentiment == 'positive':
                        positive_count += 1
                    elif sentiment == 'negative':
                        negative_count += 1
                    else:
                        neutral_count += 1
                except Exception as e:
                    logger.warning(f"Error analyzing journal entry: {e}")
                    neutral_count += 1
        
        total_analyzed = positive_count + negative_count + neutral_count
        
        if total_analyzed > 0:
            # Determine predominant sentiment
            if positive_count > negative_count and positive_count > neutral_count:
                predominant = "positive"
                percentage = (positive_count / total_analyzed) * 100
            elif negative_count > positive_count and negative_count > neutral_count:
                predominant = "negative"
                percentage = (negative_count / total_analyzed) * 100
            else:
                predominant = "balanced"
                percentage = (neutral_count / total_analyzed) * 100
            
            insights.append({
                "id": f"sentiment-pattern-{user_id[:8]}",
                "title": "Journal Sentiment",
                "description": f"Your journal writing tends to have a {predominant} tone ({percentage:.0f}% of entries).",
                "type": "emotional",
                "created_at": datetime.now().isoformat()
            })
    
    return insights

def analyze_conversation_topics(user_id, chat_history):
    """Analyze conversation topics from chat history"""
    insights = []
    
    if not chat_history:
        return insights
    
    # Extract messages where the user is speaking
    user_messages = [chat.get('message', '') for chat in chat_history if chat.get('is_user', False)]
    
    if not user_messages:
        return insights
    
    # Simple topic detection based on keywords
    topics = {
        "work": ["work", "job", "career", "office", "colleague", "business", "meeting", "project"],
        "relationships": ["friend", "relationship", "family", "partner", "spouse", "marriage", "date", "love"],
        "health": ["health", "exercise", "workout", "gym", "diet", "nutrition", "doctor", "medical"],
        "entertainment": ["movie", "music", "book", "show", "game", "play", "concert", "entertainment"],
        "personal growth": ["goal", "learn", "skill", "improve", "growth", "develop", "progress", "achievement"],
        "technology": ["tech", "computer", "app", "software", "hardware", "device", "digital", "online"]
    }
    
    # Count mentions of each topic
    topic_counts = {topic: 0 for topic in topics}
    
    for message in user_messages:
        message = message.lower()
        for topic, keywords in topics.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', message):
                    topic_counts[topic] += 1
                    break
    
    # Get most discussed topics
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Generate insight for top topic if any topics were detected
    if sorted_topics and sorted_topics[0][1] > 0:
        top_topic = sorted_topics[0][0]
        
        insights.append({
            "id": f"topic-pattern-{user_id[:8]}",
            "title": "Conversation Focus",
            "description": f"You frequently discuss topics related to {top_topic} with your SoulMate.",
            "type": "communication",
            "created_at": datetime.now().isoformat()
        })
    
    return insights

def analyze_response_times(user_id, chat_history):
    """Analyze response times from chat history (NEW)"""
    insights = []
    
    if not chat_history or len(chat_history) < 4:  # Need enough messages for meaningful analysis
        return insights
    
    # Parse chat history to calculate response times
    response_times = []
    last_timestamp = None
    last_is_user = None
    
    # Sort by timestamp to ensure proper ordering
    sorted_chats = sorted(chat_history, key=lambda x: x.get('timestamp', 0))
    
    for chat in sorted_chats:
        timestamp = None
        if isinstance(chat.get('timestamp'), str):
            try:
                timestamp = datetime.fromisoformat(chat.get('timestamp').replace('Z', '+00:00'))
            except ValueError:
                try:
                    timestamp = datetime.fromisoformat(chat.get('timestamp'))
                except ValueError:
                    continue
        elif isinstance(chat.get('timestamp'), (int, float)):
            timestamp = datetime.fromtimestamp(chat.get('timestamp', 0))
        else:
            continue
        
        is_user = chat.get('is_user', False)
        
        # If we have a non-user message after a user message, calculate response time
        if last_timestamp and last_is_user and not is_user:
            # Calculate time difference in seconds
            time_diff = (timestamp - last_timestamp).total_seconds()
            
            # Only consider reasonable response times (less than 5 minutes)
            if 0 <= time_diff < 300:
                response_times.append(time_diff)
        
        last_timestamp = timestamp
        last_is_user = is_user
    
    if not response_times:
        return insights
    
    # Calculate statistics
    avg_response_time = sum(response_times) / len(response_times)
    median_response_time = statistics.median(response_times) if len(response_times) > 1 else response_times[0]
    
    # Format time nicely
    if avg_response_time < 3:
        avg_time_str = "almost instantly"
    elif avg_response_time < 10:
        avg_time_str = "in just a few seconds"
    else:
        avg_time_str = f"in about {int(avg_response_time)} seconds"
    
    insights.append({
        "id": f"response-time-{user_id[:8]}",
        "title": "Quick Conversations",
        "description": f"Your SoulMate typically responds to your messages {avg_time_str}, allowing for smooth conversations.",
        "type": "interaction",
        "created_at": datetime.now().isoformat(),
        "data": {
            "average_seconds": round(avg_response_time, 1),
            "median_seconds": round(median_response_time, 1)
        },
        "actionable": "You'll get the best experience by continuing the conversation flow rather than sending multiple messages at once."
    })
    
    return insights

def get_response_time_metrics(chat_history):
    """Calculate response time metrics for visualization (NEW)"""
    response_times_by_hour = {h: [] for h in range(24)}
    
    if not chat_history or len(chat_history) < 4:
        return {"by_hour": []}
    
    # Parse chat history to calculate response times by hour
    last_timestamp = None
    last_is_user = None
    
    # Sort by timestamp
    sorted_chats = sorted(chat_history, key=lambda x: x.get('timestamp', 0))
    
    for chat in sorted_chats:
        timestamp = None
        if isinstance(chat.get('timestamp'), str):
            try:
                timestamp = datetime.fromisoformat(chat.get('timestamp').replace('Z', '+00:00'))
            except ValueError:
                try:
                    timestamp = datetime.fromisoformat(chat.get('timestamp'))
                except ValueError:
                    continue
        elif isinstance(chat.get('timestamp'), (int, float)):
            timestamp = datetime.fromtimestamp(chat.get('timestamp', 0))
        else:
            continue
        
        is_user = chat.get('is_user', False)
        
        # If we have a non-user message after a user message, calculate response time
        if last_timestamp and last_is_user and not is_user:
            # Calculate time difference in seconds
            time_diff = (timestamp - last_timestamp).total_seconds()
            
            # Only consider reasonable response times (less than 5 minutes)
            if 0 <= time_diff < 300:
                # Add to hour bucket
                hour = last_timestamp.hour
                response_times_by_hour[hour].append(time_diff)
        
        last_timestamp = timestamp
        last_is_user = is_user
    
    # Calculate average by hour (only for hours with data)
    response_by_hour = []
    for hour in range(24):
        times = response_times_by_hour[hour]
        if times:
            response_by_hour.append({
                "hour": hour,
                "avg_response_time": round(sum(times) / len(times), 1),
                "count": len(times)
            })
    
    return {
        "by_hour": response_by_hour
    }

def analyze_conversation_depth(user_id, chat_history):
    """Analyze conversation depth from chat history (NEW)"""
    insights = []
    
    if not chat_history or len(chat_history) < 5:
        return insights
    
    # Extract message content and length
    user_messages = []
    ai_messages = []
    
    for chat in chat_history:
        message = chat.get('message', '')
        if not message:
            continue
        
        is_user = chat.get('is_user', False)
        words = len(re.findall(r'\b\w+\b', message))
        
        if is_user:
            user_messages.append({
                'text': message,
                'words': words
            })
        else:
            ai_messages.append({
                'text': message,
                'words': words
            })
    
    if not user_messages or not ai_messages:
        return insights
    
    # Calculate average message lengths
    avg_user_words = sum(m['words'] for m in user_messages) / len(user_messages)
    avg_ai_words = sum(m['words'] for m in ai_messages) / len(ai_messages)
    
    # Calculate conversation depth metrics
    conversation_depth = 0
    
    # Factor 1: Message length ratio (longer messages = deeper conversations)
    if avg_user_words > 15:
        conversation_depth += 1
    if avg_ai_words > 30:
        conversation_depth += 1
    
    # Factor 2: Question frequency (more questions = deeper engagement)
    question_count = 0
    for message in user_messages:
        if '?' in message['text']:
            question_count += 1
    
    question_ratio = question_count / len(user_messages) if user_messages else 0
    if question_ratio > 0.3:
        conversation_depth += 1
    
    # Factor 3: Back-and-forth exchanges
    exchanges = min(len(user_messages), len(ai_messages))
    if exchanges > 10:
        conversation_depth += 1
    
    # Create insight based on depth
    depth_description = ""
    if conversation_depth == 0:
        depth_description = "Your conversations tend to be brief exchanges. Try asking more follow-up questions to explore topics more deeply."
    elif conversation_depth == 1:
        depth_description = "Your conversations show some depth in certain areas. Continuing dialogs on topics of interest could lead to more meaningful exchanges."
    elif conversation_depth == 2:
        depth_description = "You engage in moderately deep conversations with thoughtful exchanges. Your SoulMate responds well to your level of engagement."
    elif conversation_depth == 3:
        depth_description = "Your conversations show substantial depth with detailed exchanges. You're maximizing the value of your AI companion."
    else:
        depth_description = "Your conversations are exceptionally deep and meaningful, with rich back-and-forth exchanges. You're getting the full benefit of your AI companion."
    
    insights.append({
        "id": f"conversation-depth-{user_id[:8]}",
        "title": "Conversation Depth",
        "description": depth_description,
        "type": "communication",
        "created_at": datetime.now().isoformat(),
        "data": {
            "depth_score": conversation_depth,
            "avg_user_words": round(avg_user_words, 1),
            "avg_ai_words": round(avg_ai_words, 1),
            "question_ratio": round(question_ratio, 2)
        },
        "actionable": "Try asking open-ended 'why' and 'how' questions to encourage deeper responses from your SoulMate."
    })
    
    return insights

def get_conversation_depth_metrics(chat_history):
    """Calculate conversation depth metrics for visualization (NEW)"""
    if not chat_history:
        return {}
    
    # Group chats by day
    chats_by_day = {}
    
    for chat in chat_history:
        timestamp = None
        if isinstance(chat.get('timestamp'), str):
            try:
                timestamp = datetime.fromisoformat(chat['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                try:
                    timestamp = datetime.fromisoformat(chat['timestamp'])
                except ValueError:
                    continue
        elif isinstance(chat.get('timestamp'), (int, float)):
            timestamp = datetime.fromtimestamp(chat.get('timestamp'))
        else:
            continue
        
        day_str = timestamp.strftime('%Y-%m-%d')
        if day_str not in chats_by_day:
            chats_by_day[day_str] = []
        
        chats_by_day[day_str].append(chat)
    
    # Calculate metrics for each day
    depth_by_day = []
    
    for day, day_chats in sorted(chats_by_day.items()):
        user_messages = [c for c in day_chats if c.get('is_user', False)]
        ai_messages = [c for c in day_chats if not c.get('is_user', False)]
        
        if not user_messages or not ai_messages:
            continue
        
        # Average words per message
        avg_user_words = sum(len(re.findall(r'\b\w+\b', m.get('message', ''))) for m in user_messages) / len(user_messages)
        avg_ai_words = sum(len(re.findall(r'\b\w+\b', m.get('message', ''))) for m in ai_messages) / len(ai_messages)
        
        # Question count
        question_count = sum(1 for m in user_messages if '?' in m.get('message', ''))
        
        depth_by_day.append({
            "date": day,
            "message_count": len(day_chats),
            "avg_user_words": round(avg_user_words, 1),
            "avg_ai_words": round(avg_ai_words, 1),
            "question_count": question_count
        })
    
    return {
        "by_day": depth_by_day
    }

def analyze_word_usage(user_id, chat_history):
    """Analyze word usage patterns in conversations (NEW)"""
    insights = []
    
    if not chat_history or len(chat_history) < 10:
        return insights
    
    # Extract messages where the user is speaking
    user_messages = [chat.get('message', '') for chat in chat_history if chat.get('is_user', False)]
    
    if not user_messages:
        return insights
    
    # Concatenate all messages and tokenize
    all_text = ' '.join(user_messages).lower()
    try:
        words = safe_tokenize(all_text)
    except Exception as e:
        logger.warning(f"Error tokenizing text: {str(e)}")
        words = re.findall(r'\b\w+\b', all_text)
    
    # Filter out stopwords and very short words
    stop_words = safe_stopwords()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_counts = collections.Counter(words)
    
    # Get most frequent words
    most_common = word_counts.most_common(5)
    
    if not most_common:
        return insights
    
    # Create word cloud description
    word_cloud = ', '.join([f'"{word}"' for word, count in most_common])
    
    insights.append({
        "id": f"word-usage-{user_id[:8]}",
        "title": "Your Vocabulary Patterns",
        "description": f"Your most frequently used words include {word_cloud}. These reflect the themes and topics that matter most to you in conversations.",
        "type": "language",
        "created_at": datetime.now().isoformat(),
        "data": {
            "top_words": [{"word": word, "count": count} for word, count in most_common]
        },
        "actionable": "Try exploring new topics and vocabulary to expand the range of your conversations with your SoulMate."
    })
    
    return insights

def generate_comparative_insights(user_id, current_data, previous_data, time_range):
    """Generate comparative insights between current and previous periods (NEW)"""
    insights = []
    
    if not current_data or not previous_data:
        return insights
    
    period_name = "week" if time_range == "week" else "month" if time_range == "month" else "period"
    
    # Compare message frequency
    current_message_count = len(current_data)
    previous_message_count = len(previous_data)
    
    if current_message_count > 0 and previous_message_count > 0:
        percent_change = ((current_message_count - previous_message_count) / previous_message_count) * 100
        
        if abs(percent_change) >= 20:  # Only show significant changes
            trend_direction = "more" if percent_change > 0 else "fewer"
            
            insights.append({
                "id": f"trend-message-count-{user_id[:8]}",
                "title": "Conversation Trend",
                "description": f"You're having {trend_direction} conversations this {period_name} compared to the previous {period_name} ({abs(percent_change):.0f}% {trend_direction}).",
                "type": "trend",
                "created_at": datetime.now().isoformat(),
                "data": {
                    "current_count": current_message_count,
                    "previous_count": previous_message_count,
                    "percent_change": round(percent_change, 1)
                }
            })
    
    # Compare conversation depth
    current_user_messages = [c for c in current_data if c.get('is_user', False)]
    previous_user_messages = [c for c in previous_data if c.get('is_user', False)]
    
    if current_user_messages and previous_user_messages:
        # Average message length comparison
        current_avg_length = sum(len(re.findall(r'\b\w+\b', m.get('message', ''))) for m in current_user_messages) / len(current_user_messages)
        previous_avg_length = sum(len(re.findall(r'\b\w+\b', m.get('message', ''))) for m in previous_user_messages) / len(previous_user_messages)
        
        if abs(current_avg_length - previous_avg_length) >= 3:  # Only show significant changes
            trend_direction = "longer" if current_avg_length > previous_avg_length else "shorter"
            
            insights.append({
                "id": f"trend-message-length-{user_id[:8]}",
                "title": "Message Length Trend",
                "description": f"Your messages this {period_name} are {trend_direction} than in the previous {period_name}, showing a change in your conversation style.",
                "type": "trend",
                "created_at": datetime.now().isoformat(),
                "data": {
                    "current_avg_length": round(current_avg_length, 1),
                    "previous_avg_length": round(previous_avg_length, 1)
                }
            })
    
    return insights

def track_emotional_progression(user_id, journal_entries, emotion_analyzer):
    """Track emotional progression over time from journal entries (NEW)"""
    insights = []
    
    if not journal_entries or len(journal_entries) < 3:
        return insights
    
    # Sort journal entries by timestamp
    sorted_entries = sorted(journal_entries, key=lambda x: x.get('timestamp', 0))
    
    # Extract mood progression
    moods_over_time = []
    for entry in sorted_entries:
        if 'mood' in entry and entry['mood'] and 'timestamp' in entry:
            timestamp = None
            if isinstance(entry['timestamp'], str):
                try:
                    timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                except ValueError:
                    try:
                        timestamp = datetime.fromisoformat(entry['timestamp'])
                    except ValueError:
                        continue
            elif isinstance(entry['timestamp'], (int, float)):
                timestamp = datetime.fromtimestamp(entry['timestamp'])
            else:
                continue
            
            moods_over_time.append({
                'mood': entry['mood'],
                'date': timestamp.strftime('%Y-%m-%d')
            })
    
    if len(moods_over_time) < 3:
        return insights
    
    # Analyze emotional progression
    mood_categories = {
        'positive': ['happy', 'excited', 'content', 'peaceful', 'joyful', 'grateful', 'optimistic'],
        'negative': ['sad', 'angry', 'anxious', 'frustrated', 'depressed', 'upset', 'stressed'],
        'neutral': ['neutral', 'calm', 'okay', 'fine', 'moderate']
    }
    
    # Categorize moods
    categorized_moods = []
    for mood_data in moods_over_time:
        mood = mood_data['mood'].lower()
        category = 'neutral'
        
        for cat, terms in mood_categories.items():
            if any(term in mood for term in terms):
                category = cat
                break
        
        categorized_moods.append({
            'category': category,
            'original': mood,
            'date': mood_data['date']
        })
    
    # Detect trend
    recent_moods = categorized_moods[-3:]  # Last 3 moods
    positive_count = sum(1 for m in recent_moods if m['category'] == 'positive')
    negative_count = sum(1 for m in recent_moods if m['category'] == 'negative')
    neutral_count = sum(1 for m in recent_moods if m['category'] == 'neutral')
    
    trend = None
    if positive_count >= 2:
        trend = 'improving'
    elif negative_count >= 2:
        trend = 'declining'
    else:
        trend = 'stable'
    
    # Look for change from previous state
    earlier_moods = categorized_moods[:-3] if len(categorized_moods) > 3 else []
    
    if earlier_moods:
        earlier_positive = sum(1 for m in earlier_moods if m['category'] == 'positive')
        earlier_negative = sum(1 for m in earlier_moods if m['category'] == 'negative')
        
        earlier_trend = 'positive' if earlier_positive > earlier_negative else 'negative' if earlier_negative > earlier_positive else 'neutral'
        recent_trend = 'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'neutral'
        
        if earlier_trend != recent_trend:
            if recent_trend == 'positive' and earlier_trend == 'negative':
                recent_moods_text = ", ".join([f'"{m["original"]}"' for m in recent_moods])
                insights.append({
                    "id": f"mood-improvement-{user_id[:8]}",
                    "title": "Mood Improvement",
                    "description": f"Your recent journal entries show a positive trend in your mood, with entries marked as {recent_moods_text}. This is an improvement compared to your earlier entries.",
                    "type": "emotional-progression",
                    "created_at": datetime.now().isoformat(),
                    "data": {
                        "recent_moods": [m["original"] for m in recent_moods],
                        "trend": "improving"
                    },
                    "actionable": "Take note of what's been working well for you recently. These positive influences could be helpful to remember during challenging times."
                })
            elif recent_trend == 'negative' and earlier_trend == 'positive':
                recent_moods_text = ", ".join([f'"{m["original"]}"' for m in recent_moods])
                insights.append({
                    "id": f"mood-decline-{user_id[:8]}",
                    "title": "Emotional Well-being Check",
                    "description": f"Your recent journal entries suggest a shift in your mood with entries marked as {recent_moods_text}. This is different from your more positive earlier entries.",
                    "type": "emotional-progression",
                    "created_at": datetime.now().isoformat(),
                    "data": {
                        "recent_moods": [m["original"] for m in recent_moods],
                        "trend": "declining"
                    },
                    "actionable": "Consider what factors might be affecting your mood lately. Reflecting on past positive experiences or reaching out to supportive friends might help."
                })
    
    # If no change insight was added, add a general mood stability insight
    if len(insights) == 0 and trend:
        mood_state = (
            "consistently positive" if trend == "improving" and positive_count >= 2 else
            "rather mixed" if trend == "stable" else
            "somewhat challenging" if trend == "declining" and negative_count >= 2 else
            "varying"
        )
        
        insights.append({
            "id": f"mood-stability-{user_id[:8]}",
            "title": "Emotional Consistency",
            "description": f"Your journal entries reveal a {mood_state} emotional state over time. Awareness of your emotional patterns can help build emotional resilience.",
            "type": "emotional-progression",
            "created_at": datetime.now().isoformat(),
            "data": {
                "recent_moods": [m["original"] for m in recent_moods],
                "trend": trend
            },
            "actionable": "Regular journaling helps track emotional patterns. Try adding more detail about what influences your moods for even deeper insights."
        })
    
    return insights

def generate_recommendations(user_id, chat_history, journal_entries, user_preferences):
    """Generate personalized recommendations based on insights (NEW)"""
    insights = []
    
    # Analyze chat data
    if not chat_history:
        return insights
    
    # Extract user messages and topics of interest
    user_messages = [chat.get('message', '') for chat in chat_history if chat.get('is_user', False)]
    all_user_text = ' '.join(user_messages).lower()
    
    # Detect potential interests based on message content
    interest_categories = {
        "reading": ["book", "novel", "reading", "author", "story", "literature"],
        "fitness": ["exercise", "workout", "fitness", "gym", "run", "train", "health"],
        "mindfulness": ["meditate", "mindful", "relax", "stress", "anxiety", "calm", "peace"],
        "learning": ["learn", "course", "study", "knowledge", "skill", "education"],
        "creativity": ["art", "create", "write", "paint", "creative", "music", "draw"],
        "social": ["friend", "family", "social", "people", "relationship", "connect"]
    }
    
    # Check which interests appear in messages
    interest_matches = {}
    for category, keywords in interest_categories.items():
        matches = sum(1 for keyword in keywords if keyword in all_user_text)
        if matches > 0:
            interest_matches[category] = matches
    
    # Sort by number of matches
    sorted_interests = sorted(interest_matches.items(), key=lambda x: x[1], reverse=True)
    
    # Generate recommendations based on top interests
    if sorted_interests:
        top_interest = sorted_interests[0][0]
        
        recommendations = {
            "reading": {
                "title": "Book Recommendation",
                "description": "Based on your conversations, you might enjoy exploring more books. Setting aside 20 minutes each day for reading could be both enjoyable and beneficial.",
                "actionable": "Try starting with short stories or articles if you don't have time for a full novel."
            },
            "fitness": {
                "title": "Wellness Activity",
                "description": "Your messages show an interest in physical well-being. Even a short 10-minute daily activity can improve both physical and mental health.",
                "actionable": "Consider trying a brief morning stretch routine to start your day with energy."
            },
            "mindfulness": {
                "title": "Mindfulness Practice",
                "description": "Your conversations suggest you might benefit from mindfulness practices. Even 5 minutes of daily meditation can reduce stress and improve focus.",
                "actionable": "Try a simple breathing exercise: breathe in for 4 counts, hold for 7, and exhale for 8."
            },
            "learning": {
                "title": "Learning Opportunity",
                "description": "You seem interested in expanding your knowledge. Setting aside time for learning new skills or subjects can be deeply fulfilling.",
                "actionable": "Consider spending 15 minutes daily on a topic that fascinates you."
            },
            "creativity": {
                "title": "Creative Expression",
                "description": "Your messages reveal a creative side. Making time for creative expression can be both enjoyable and therapeutic.",
                "actionable": "Try a 10-minute daily freewriting or sketching session with no judgment or expectations."
            },
            "social": {
                "title": "Connection Ritual",
                "description": "Your conversations highlight the importance of social connections in your life. Regular meaningful interactions can significantly boost well-being.",
                "actionable": "Consider setting aside time each week to reach out to someone you care about."
            }
        }
        
        if top_interest in recommendations:
            rec = recommendations[top_interest]
            insights.append({
                "id": f"recommendation-{top_interest}-{user_id[:8]}",
                "title": rec["title"],
                "description": rec["description"],
                "type": "recommendation",
                "created_at": datetime.now().isoformat(),
                "actionable": rec["actionable"]
            })
    
    # Add a wellbeing recommendation if journal entries show emotional patterns
    if journal_entries and len(journal_entries) >= 3:
        # Extract moods
        moods = [entry.get('mood', '').lower() for entry in journal_entries if 'mood' in entry]
        
        # Check for negative mood indicators
        negative_indicators = ["sad", "angry", "anxious", "stressed", "tired", "frustrated", "overwhelmed"]
        negative_count = sum(1 for mood in moods if any(indicator in mood for indicator in negative_indicators))
        
        if negative_count >= 2:
            insights.append({
                "id": f"wellbeing-recommendation-{user_id[:8]}",
                "title": "Wellbeing Check-In",
                "description": "Your journal entries suggest you might be experiencing some challenging emotions. Taking small steps to care for your wellbeing can make a meaningful difference.",
                "type": "recommendation",
                "created_at": datetime.now().isoformat(),
                "actionable": "Try the '3-3-3 Rule' when feeling overwhelmed: name 3 things you see, 3 sounds you hear, and move 3 parts of your body. This can help bring you back to the present moment."
            })
    
    return insights

def generate_growth_insights(user_id, chat_history, user_preferences):
    """Generate growth-oriented insights based on user data"""
    insights = []
    
    # Check chat history length
    if len(chat_history) < 10:
        insights.append({
            "id": f"growth-chat-{user_id[:8]}",
            "title": "Deepen Your Connection",
            "description": "Have more conversations with your SoulMate to help it understand you better and provide more personalized insights.",
            "type": "growth",
            "created_at": datetime.now().isoformat()
        })
    
    # Check if user has set preferences
    communication_style = user_preferences.get('communication_style', None)
    interests = user_preferences.get('interests', [])
    
    if not communication_style:
        insights.append({
            "id": f"growth-style-{user_id[:8]}",
            "title": "Communication Style",
            "description": "Set your preferred communication style in preferences to make interactions more natural and aligned with your expectations.",
            "type": "growth",
            "created_at": datetime.now().isoformat()
        })
    
    if not interests:
        insights.append({
            "id": f"growth-interests-{user_id[:8]}",
            "title": "Share Your Interests",
            "description": "Add your interests in the preferences section to help your SoulMate provide more relevant conversations and recommendations.",
            "type": "growth",
            "created_at": datetime.now().isoformat()
        })
    
    return insights

@insights_bp.route('/insights/export', methods=['GET'])
@login_required
def export_insights():
    """Export insights as a JSON file"""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        # Get insights data
        memory_manager = MemoryManager(user_id=user_id, db_manager=db_manager)
        emotion_analyzer = EmotionAnalyzer()
        
        chat_history = get_user_chat_history(user_id)
        journal_entries = get_user_journal_entries(user_id)
        user_preferences = memory_manager.get_user_preferences(user_id)
        
        # Generate all insights
        insights = []
        
        # Add all insight types
        insights.extend(analyze_interaction_patterns(user_id, chat_history))
        if journal_entries:
            insights.extend(analyze_emotional_patterns(user_id, journal_entries, emotion_analyzer))
        insights.extend(analyze_conversation_topics(user_id, chat_history))
        insights.extend(analyze_response_times(user_id, chat_history))
        insights.extend(analyze_conversation_depth(user_id, chat_history))
        insights.extend(analyze_word_usage(user_id, chat_history))
        if journal_entries and len(journal_entries) >= 3:
            insights.extend(track_emotional_progression(user_id, journal_entries, emotion_analyzer))
        insights.extend(generate_recommendations(user_id, chat_history, journal_entries, user_preferences))
        insights.extend(generate_growth_insights(user_id, chat_history, user_preferences))
        
        # Sort insights by creation date (newest first)
        insights = sorted(insights, key=lambda x: x.get('created_at', ''), reverse=True)
        
        # If no insights are available, provide a message
        if not insights:
            return jsonify({
                'success': True,
                'message': 'Not enough data to generate insights for export. Continue using the app to get personalized insights.',
                'export_data': {
                    'user_id': user_id,
                    'export_date': datetime.now().isoformat(),
                    'insights': [],
                    'summary': {
                        'total_insights': 0,
                        'chat_count': len(chat_history),
                        'journal_count': len(journal_entries),
                        'insight_types': []
                    }
                }
            })
        
        # Create export data
        export_data = {
            'user_id': user_id,
            'export_date': datetime.now().isoformat(),
            'insights': insights,
            'summary': {
                'total_insights': len(insights),
                'chat_count': len(chat_history),
                'journal_count': len(journal_entries),
                'insight_types': list(set(insight.get('type') for insight in insights))
            }
        }
        
        # Return as downloadable JSON
        return jsonify({
            'success': True,
            'export_data': export_data
        })
        
    except Exception as e:
        logger.error(f"Error exporting insights: {str(e)}")
        return jsonify({'error': f'Error exporting insights: {str(e)}')}), 500

@insights_bp.route('/insights/actions', methods=['POST'])
@login_required
def log_insight_action():
    """Log user actions taken on insights (NEW)"""
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        data = request.json
        insight_id = data.get('insight_id')
        action = data.get('action')  # 'viewed', 'dismissed', 'implemented', etc.
        
        if not insight_id or not action:
            return jsonify({'error': 'Insight ID and action are required'}), 400
        
        # Log the action
        memory_manager = MemoryManager(user_id=user_id, db_manager=db_manager)
        insight_actions = memory_manager.get_preference(user_id, 'insight_actions', {})
        
        if insight_id not in insight_actions:
            insight_actions[insight_id] = []
        
        insight_actions[insight_id].append({
            'action': action,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save updated actions
        memory_manager.store_preference(user_id, 'insight_actions', insight_actions)
        
        return jsonify({
            'success': True,
            'message': 'Action logged successfully'
        })
        
    except Exception as e:
        logger.error(f"Error logging insight action: {str(e)}")
        return jsonify({'error': f'Error logging insight action: {str(e)}')}), 500