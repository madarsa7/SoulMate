import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """Analyzes and tracks user emotions over time"""
    
    # Basic emotion categories
    EMOTION_CATEGORIES = [
        "joy", "sadness", "anger", "fear", "surprise", 
        "disgust", "trust", "anticipation", "neutral"
    ]
    
    # Simple emotion-related word mappings
    EMOTION_KEYWORDS = {
        "joy": ["happy", "excited", "glad", "delighted", "pleased", "content", "thrilled", "elated"],
        "sadness": ["sad", "unhappy", "depressed", "down", "blue", "gloomy", "melancholy", "heartbroken"],
        "anger": ["angry", "mad", "furious", "outraged", "irritated", "annoyed", "frustrated", "enraged"],
        "fear": ["afraid", "scared", "terrified", "anxious", "worried", "nervous", "frightened", "panicked"],
        "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned", "startled", "unexpected"],
        "disgust": ["disgusted", "revolted", "appalled", "nauseated", "repulsed", "gross", "yuck"],
        "trust": ["trust", "believe", "confident", "faith", "rely", "depend", "assured", "certain"],
        "anticipation": ["anticipate", "expect", "look forward", "await", "excited about", "hopeful"],
        "neutral": ["fine", "okay", "alright", "so-so", "neutral", "indifferent", "meh"]
    }
    
    # Loneliness indicators - words and phrases that might suggest loneliness
    LONELINESS_INDICATORS = [
        "alone", "lonely", "no one", "nobody", "by myself", "no friends", "isolated", "abandoned",
        "forgotten", "left out", "excluded", "rejected", "solitary", "companionship", "empty",
        "miss", "missing", "disconnected", "distant", "not close", "no connection", "separated", 
        "invisible", "unnoticed", "unwanted", "unloved", "neglected"
    ]
    
    def __init__(self, db_manager=None):
        # Initialize basic emotion detection
        # In a production system, this could be replaced with a more sophisticated ML model
        self.history = []
        self.db_manager = db_manager
        logger.info("Emotion analyzer initialized successfully")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text for emotional content
        Returns a dictionary with emotion scores
        """
        # Convert to lowercase for easier matching
        text_lower = text.lower()
        
        # Initialize scores
        emotion_scores = {emotion: 0.0 for emotion in self.EMOTION_CATEGORIES}
        
        # Simple keyword-based matching
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                # Count occurrences of each keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Increment score based on number of matches
                    emotion_scores[emotion] += len(matches) * 0.1
        
        # Apply some normalization
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        else:
            # If no emotions detected, slightly increase neutral
            emotion_scores["neutral"] = 0.6
            
        return emotion_scores
    
    def analyze_loneliness(self, text: str) -> float:
        """
        Analyze text for indicators of loneliness
        Returns a score from 0.0 to 1.0
        """
        text_lower = text.lower()
        loneliness_score = 0.0
        
        # Check for loneliness indicators
        for indicator in self.LONELINESS_INDICATORS:
            pattern = r'\b' + re.escape(indicator) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                loneliness_score += len(matches) * 0.15
        
        # Check for intensity modifiers
        intensifiers = ["very", "really", "extremely", "so", "deeply", "completely", "terribly"]
        for intensifier in intensifiers:
            for indicator in self.LONELINESS_INDICATORS:
                pattern = r'\b' + re.escape(intensifier) + r'\s+' + re.escape(indicator) + r'\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    loneliness_score += len(matches) * 0.1
        
        # Cap at 1.0
        return min(1.0, loneliness_score)
    
    def record_emotion(self, text: str, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Record an emotion analysis entry using text
        
        Args:
            text: User's text input
            user_id: User identifier
            context: Additional context information
            
        Returns:
            Dictionary with emotion analysis results
        """
        # Analyze text
        emotion_scores = self.analyze_text(text)
        
        # Analyze loneliness
        loneliness_score = self.analyze_loneliness(text)
        timestamp = datetime.now().isoformat()
        
        entry = {
            'user_id': user_id,
            'timestamp': timestamp,
            'text': text,
            'emotion_scores': emotion_scores,
            'loneliness_score': loneliness_score,
            'context': context or {}
        }
        
        self.history.append(entry)
        
        # Store in database if available
        if self.db_manager:
            try:
                emotion_key = f"emotion_record_{timestamp}"
                self.db_manager.store_preference(user_id, emotion_key, entry)
            except Exception as e:
                logger.error(f"Error storing emotion record: {e}")
        
        return entry
    
    def get_dominant_emotion(self, text: str) -> Tuple[str, float]:
        """
        Get the dominant emotion from text
        
        Args:
            text: User's text input
            
        Returns:
            Tuple of (dominant_emotion, score)
        """
        # Analyze text
        emotion_scores = self.analyze_text(text)
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def get_recent_emotions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent emotion records for a user
        
        Args:
            user_id: The user's ID
            limit: Maximum number of records to return
            
        Returns:
            List of emotion records, ordered by recency (newest first)
        """
        # First, try to get from database if available
        if self.db_manager:
            try:
                all_prefs = self.db_manager.get_user_preferences(user_id)
                emotion_records = []
                
                for key, value in all_prefs.items():
                    if key.startswith('emotion_record_'):
                        emotion_records.append(value)
                
                # Sort by timestamp, newest first
                emotion_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                return emotion_records[:limit]
            except Exception as e:
                logger.error(f"Error retrieving emotion records from database: {e}")
        
        # Fall back to in-memory history
        user_records = [entry for entry in self.history if entry['user_id'] == user_id]
        user_records.sort(key=lambda x: x['timestamp'], reverse=True)
        return user_records[:limit]
    
    def get_emotion_trend(self, user_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze emotion trends over a specified time window
        Returns aggregated emotion data
        """
        # Try to get records from database first if available
        emotion_records = []
        if self.db_manager:
            try:
                all_prefs = self.db_manager.get_user_preferences(user_id)
                for key, value in all_prefs.items():
                    if key.startswith('emotion_record_'):
                        emotion_records.append(value)
            except Exception as e:
                logger.error(f"Error retrieving emotion records from database: {e}")
        
        # Fall back to in-memory history if needed
        if not emotion_records:
            emotion_records = [entry for entry in self.history if entry['user_id'] == user_id]
        
        if not emotion_records:
            return {
                'dominant_emotion': 'neutral',
                'emotion_distribution': {emotion: 0.0 for emotion in self.EMOTION_CATEGORIES},
                'change_rate': {},
                'stability': 0.0,
                'loneliness_trend': 0.0
            }
        
        # Filter entries by time window
        now = datetime.now()
        cutoff = now - timedelta(hours=time_window_hours)
        filtered_entries = []
        
        for entry in emotion_records:
            try:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff:
                    filtered_entries.append(entry)
            except (ValueError, TypeError, KeyError):
                continue  # Skip entries with invalid timestamps
        
        if not filtered_entries:
            return {
                'dominant_emotion': 'neutral',
                'emotion_distribution': {emotion: 0.0 for emotion in self.EMOTION_CATEGORIES},
                'change_rate': {},
                'stability': 0.0,
                'loneliness_trend': 0.0
            }
        
        # Aggregate emotion scores
        aggregated_scores = {emotion: 0.0 for emotion in self.EMOTION_CATEGORIES}
        loneliness_scores = []
        
        for entry in filtered_entries:
            for emotion, score in entry['emotion_scores'].items():
                aggregated_scores[emotion] += score
            
            loneliness_score = entry.get('loneliness_score', 0.0)
            loneliness_scores.append(loneliness_score)
        
        # Normalize
        total = sum(aggregated_scores.values())
        if total > 0:
            aggregated_scores = {k: v/total for k, v in aggregated_scores.items()}
            
        # Calculate dominant emotion
        dominant_emotion = max(aggregated_scores.items(), key=lambda x: x[1])
        
        # Calculate stability (how consistent emotions have been)
        stability = 0.0
        change_rate = {}
        
        if len(filtered_entries) > 1:
            # Sort by timestamp
            sorted_entries = sorted(filtered_entries, key=lambda x: x['timestamp'])
            
            # Calculate changes between consecutive entries
            changes = []
            for i in range(1, len(sorted_entries)):
                prev = sorted_entries[i-1]['emotion_scores']
                curr = sorted_entries[i]['emotion_scores']
                
                # Euclidean distance between emotion vectors
                distance = sum((prev[e] - curr[e])**2 for e in self.EMOTION_CATEGORIES) ** 0.5
                changes.append(distance)
            
            # Stability is inverse of average change (1.0 = very stable, 0.0 = unstable)
            avg_change = sum(changes) / len(changes)
            stability = max(0.0, 1.0 - min(1.0, avg_change))
            
            # Calculate rate of change for each emotion
            for emotion in self.EMOTION_CATEGORIES:
                first_score = sorted_entries[0]['emotion_scores'][emotion]
                last_score = sorted_entries[-1]['emotion_scores'][emotion]
                time_diff = (datetime.fromisoformat(sorted_entries[-1]['timestamp']) - 
                             datetime.fromisoformat(sorted_entries[0]['timestamp'])).total_seconds() / 3600
                
                if time_diff > 0:
                    rate = (last_score - first_score) / time_diff
                    change_rate[emotion] = rate
        
        # Calculate loneliness trend
        loneliness_trend = sum(loneliness_scores) / len(loneliness_scores) if loneliness_scores else 0.0
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'dominant_score': dominant_emotion[1],
            'emotion_distribution': aggregated_scores,
            'change_rate': change_rate,
            'stability': stability,
            'loneliness_trend': loneliness_trend,
            'data_points': len(filtered_entries)
        }
    
    def get_emotional_triggers(self, user_id: str, emotion: str, min_correlation: float = 0.5) -> List[str]:
        """
        Identify potential triggers for a specific emotion
        Uses simple correlation between words and emotion intensity
        """
        # Get emotion records
        emotion_records = []
        if self.db_manager:
            try:
                all_prefs = self.db_manager.get_user_preferences(user_id)
                for key, value in all_prefs.items():
                    if key.startswith('emotion_record_'):
                        emotion_records.append(value)
            except Exception as e:
                logger.error(f"Error retrieving emotion records from database: {e}")
        
        # Fall back to in-memory if needed
        if not emotion_records:
            emotion_records = [entry for entry in self.history if entry['user_id'] == user_id]
            
        if not emotion_records or len(emotion_records) < 5:  # Need sufficient data
            return []
        
        # Extract all words and their correlation with the emotion
        word_scores = {}
        
        for entry in emotion_records:
            emotion_score = entry['emotion_scores'].get(emotion, 0)
            words = re.findall(r'\b\w+\b', entry['text'].lower())
            
            for word in words:
                if len(word) < 3 or word in ['the', 'and', 'for', 'that', 'this']:
                    continue  # Skip very common words
                    
                if word not in word_scores:
                    word_scores[word] = {'scores': [], 'occurrences': 0}
                
                word_scores[word]['scores'].append(emotion_score)
                word_scores[word]['occurrences'] += 1
        
        # Calculate average emotion score when each word is present
        triggers = []
        for word, data in word_scores.items():
            if data['occurrences'] >= 3:  # Only consider words with sufficient occurrences
                avg_score = sum(data['scores']) / len(data['scores'])
                
                # If average emotion score is high enough when this word is present
                if avg_score >= min_correlation:
                    triggers.append((word, avg_score))
        
        # Sort by correlation strength and return words
        triggers.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in triggers[:10]]
    
    def analyze_loneliness_risk(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze a user's loneliness risk based on recent interactions
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dictionary with loneliness assessment
        """
        # Get recent emotion records (last 7 days)
        emotion_records = []
        if self.db_manager:
            try:
                all_prefs = self.db_manager.get_user_preferences(user_id)
                for key, value in all_prefs.items():
                    if key.startswith('emotion_record_'):
                        emotion_records.append(value)
            except Exception as e:
                logger.error(f"Error retrieving emotion records from database: {e}")
        
        # Fall back to in-memory if needed
        if not emotion_records:
            emotion_records = [entry for entry in self.history if entry['user_id'] == user_id]
        
        if not emotion_records:
            return {
                'risk_level': 'unknown',
                'risk_score': 0.0,
                'recommendation': 'Not enough data to assess loneliness risk.'
            }
        
        # Filter to last 7 days
        now = datetime.now()
        cutoff = now - timedelta(days=7)
        recent_records = []
        
        for entry in emotion_records:
            try:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff:
                    recent_records.append(entry)
            except (ValueError, TypeError, KeyError):
                continue
        
        if not recent_records:
            return {
                'risk_level': 'unknown',
                'risk_score': 0.0,
                'recommendation': 'Not enough recent data to assess loneliness risk.'
            }
        
        # Calculate loneliness risk score
        loneliness_scores = [entry.get('loneliness_score', 0.0) for entry in recent_records]
        avg_loneliness = sum(loneliness_scores) / len(loneliness_scores)
        
        # Check sadness trend
        sadness_scores = [entry['emotion_scores'].get('sadness', 0.0) for entry in recent_records]
        avg_sadness = sum(sadness_scores) / len(sadness_scores)
        
        # Combined risk score (weighted average of loneliness and sadness)
        risk_score = (avg_loneliness * 0.7) + (avg_sadness * 0.3)
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = 'low'
            recommendation = 'Your recent interactions show low risk of loneliness.'
        elif risk_score < 0.6:
            risk_level = 'moderate'
            recommendation = 'Consider reaching out to friends or family for connection.'
        else:
            risk_level = 'high'
            recommendation = 'Your patterns suggest you may be experiencing loneliness. Consider speaking with a mental health professional or trusted friend.'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'avg_loneliness': avg_loneliness,
            'avg_sadness': avg_sadness,
            'data_points': len(recent_records),
            'recommendation': recommendation
        }
    
    def generate_emotional_insight(self, user_id: str) -> str:
        """
        Generate an insightful analysis about the user's emotional patterns
        Combines recent and historical data to provide meaningful patterns and suggestions
        """
        # Get emotion trend for different time periods
        recent_trend = self.get_emotion_trend(user_id, time_window_hours=24)  # Last 24 hours
        weekly_trend = self.get_emotion_trend(user_id, time_window_hours=168)  # Last week
        
        # Get loneliness risk analysis
        loneliness_risk = self.analyze_loneliness_risk(user_id)
        
        # Get recent emotion records to analyze patterns
        recent_records = self.get_recent_emotions(user_id, limit=50)
        
        # Not enough data
        if not recent_records:
            return "I don't have enough data yet to provide meaningful emotional insights. As we continue to chat, I'll be able to offer more personalized observations."
        
        # Extract dominant emotions for different time periods
        recent_dominant = recent_trend['dominant_emotion']
        weekly_dominant = weekly_trend['dominant_emotion']
        stability_recent = recent_trend['stability']
        stability_weekly = weekly_trend['stability']
        
        # Check for emotional shifts
        emotional_shift = None
        if recent_dominant != weekly_dominant and weekly_trend['data_points'] > 5:
            emotional_shift = f"shift from {weekly_dominant} to {recent_dominant}"
        
        # Analyze time patterns - group records by time of day
        morning_emotions = {}
        afternoon_emotions = {}
        evening_emotions = {}
        night_emotions = {}
        
        for record in recent_records:
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(record['timestamp'])
                hour = timestamp.hour
                
                # Group by time of day
                emotion_scores = record['emotion_scores']
                dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]
                
                if 5 <= hour < 12:  # Morning
                    morning_emotions[dominant] = morning_emotions.get(dominant, 0) + 1
                elif 12 <= hour < 17:  # Afternoon
                    afternoon_emotions[dominant] = afternoon_emotions.get(dominant, 0) + 1
                elif 17 <= hour < 22:  # Evening
                    evening_emotions[dominant] = evening_emotions.get(dominant, 0) + 1
                else:  # Night
                    night_emotions[dominant] = night_emotions.get(dominant, 0) + 1
            except:
                continue
        
        # Get dominant emotions by time of day (if enough data points)
        time_pattern = None
        if morning_emotions and afternoon_emotions and evening_emotions:
            morning_dominant = max(morning_emotions.items(), key=lambda x: x[1])[0] if morning_emotions else None
            afternoon_dominant = max(afternoon_emotions.items(), key=lambda x: x[1])[0] if afternoon_emotions else None
            evening_dominant = max(evening_emotions.items(), key=lambda x: x[1])[0] if evening_emotions else None
            night_dominant = max(night_emotions.items(), key=lambda x: x[1])[0] if night_emotions else None
            
            # Check if emotions vary by time of day
            dominants = [d for d in [morning_dominant, afternoon_dominant, evening_dominant, night_dominant] if d]
            if len(set(dominants)) > 1:
                time_pattern = True
        
        # Find patterns in emotional fluctuations
        emotional_cycle = None
        if len(recent_records) >= 10:
            # Convert to numpy array for analysis
            emotions_sequence = []
            timestamps = []
            
            for record in sorted(recent_records, key=lambda x: x['timestamp']):
                try:
                    timestamps.append(datetime.fromisoformat(record['timestamp']))
                    dominant = max(record['emotion_scores'].items(), key=lambda x: x[1])[0]
                    emotions_sequence.append(dominant)
                except:
                    continue
            
            if emotions_sequence and len(set(emotions_sequence)) > 1:
                # Simple pattern detection - check for alternating patterns
                alternating = True
                for i in range(2, len(emotions_sequence)):
                    if emotions_sequence[i] != emotions_sequence[i-2]:
                        alternating = False
                        break
                
                if alternating:
                    emotional_cycle = "alternating"
                else:
                    # Check for daily patterns (if data spans multiple days)
                    if len(timestamps) > 5 and (timestamps[-1] - timestamps[0]).days >= 2:
                        emotional_cycle = "daily variations"
        
        # Generate the insight text
        insight = ""
        
        # Start with overall dominant emotion
        if recent_trend['data_points'] >= 5:
            insight += f"Recently, your conversations have primarily reflected {recent_dominant} emotions. "
            
            # Add emotional shift insight if present
            if emotional_shift:
                insight += f"I've noticed a {emotional_shift} compared to last week. "
        else:
            insight += f"In our most recent conversations, I've picked up primarily {recent_dominant} emotions. "
        
        # Add stability insight
        if stability_recent > 0.8:
            insight += "Your emotional state has been very stable lately. "
        elif stability_recent > 0.5:
            insight += "Your emotions have been moderately stable. "
        else:
            insight += "Your emotions have been quite variable recently. "
        
        # Add time-of-day patterns if detected
        if time_pattern:
            insight += "\n\nInterestingly, I notice your emotions tend to vary throughout the day. "
            
            if morning_dominant:
                insight += f"In the mornings, you often express {morning_dominant}. "
            if afternoon_dominant and afternoon_dominant != morning_dominant:
                insight += f"During afternoons, you tend toward {afternoon_dominant}. "
            if evening_dominant and evening_dominant not in [morning_dominant, afternoon_dominant]:
                insight += f"Evenings often bring {evening_dominant} emotions. "
        
        # Add emotional cycle insight if detected
        if emotional_cycle:
            if emotional_cycle == "alternating":
                insight += "\n\nI've noticed your emotions tend to alternate in a pattern. This can be normal and often reflects how we process different experiences. "
            elif emotional_cycle == "daily variations":
                insight += "\n\nYour emotional patterns seem to follow daily rhythms, which is quite common and often connected to daily activities and sleep patterns. "
        
        # Add trigger insights if available
        triggers = self.get_emotional_triggers(user_id, recent_dominant)
        if triggers:
            trigger_text = ', '.join(triggers[:3])
            insight += f"\n\nWords like '{trigger_text}' often appear when you're feeling {recent_dominant}. "
        
        # Add loneliness insight if relevant
        if loneliness_risk['risk_score'] > 0.4:
            insight += "\n\nI've noticed some indicators of social disconnection in our conversations. "
            
            if loneliness_risk['risk_level'] == 'high':
                insight += "Maintaining social connections is important for wellbeing. Consider reaching out to friends or participating in group activities that align with your interests. "
            elif loneliness_risk['risk_level'] == 'moderate':
                insight += "If you're feeling a bit disconnected, even brief social interactions can boost your mood. "
        
        # Add a supportive comment based on dominant emotion
        insight += "\n\n"
        if recent_dominant == "joy":
            insight += "It's wonderful to see your positive emotions! Savoring these moments can help extend their benefits."
        elif recent_dominant == "sadness":
            insight += "Remember that sadness is a natural emotion that everyone experiences. I'm here to listen whenever you need support."
        elif recent_dominant == "anger":
            insight += "Anger often points to things we deeply care about. Taking time to identify the underlying needs can be helpful."
        elif recent_dominant == "fear":
            insight += "Fear is our mind's way of trying to protect us. Sometimes stepping back to evaluate situations objectively can help provide perspective."
        elif recent_dominant == "surprise":
            insight += "Your sense of curiosity and openness to surprise makes our conversations engaging and dynamic."
        else:
            insight += "I value our conversations and am here to support you through all your emotional experiences."
        
        return insight
        
    def get_mood_support_activities(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Generate personalized mood support activities based on user's emotional state
        
        Args:
            user_id: The user's ID
            
        Returns:
            List of suggested activities
        """
        # Get current emotional state
        trend = self.get_emotion_trend(user_id, time_window_hours=48)
        dominant_emotion = trend['dominant_emotion']
        loneliness_trend = trend.get('loneliness_trend', 0.0)
        
        activities = []
        
        # Breathing/meditation (good for almost any emotional state)
        activities.append({
            'type': 'breathing',
            'title': '4-7-8 Breathing Exercise',
            'description': 'A simple breathing technique to calm your mind and body',
            'steps': [
                'Find a comfortable position',
                'Breathe in through your nose for 4 seconds',
                'Hold your breath for 7 seconds',
                'Exhale completely through your mouth for 8 seconds',
                'Repeat 4 times'
            ],
            'duration': '2 minutes'
        })
        
        # Add activities based on emotion
        if dominant_emotion in ['sadness', 'fear']:
            activities.append({
                'type': 'grounding',
                'title': '5-4-3-2-1 Grounding Exercise',
                'description': 'A technique to ground yourself in the present moment',
                'steps': [
                    'Name 5 things you can see',
                    'Name 4 things you can touch/feel',
                    'Name 3 things you can hear',
                    'Name 2 things you can smell',
                    'Name 1 thing you can taste'
                ],
                'duration': '3 minutes'
            })
            
            activities.append({
                'type': 'affirmations',
                'title': 'Positive Affirmations',
                'description': 'Affirmations to help shift your perspective',
                'affirmations': [
                    'This feeling is temporary and will pass',
                    'I am stronger than I think',
                    'I\'ve overcome difficult times before',
                    'It\'s okay to ask for help when I need it',
                    'I am worthy of peace and happiness'
                ]
            })
            
        elif dominant_emotion == 'anger':
            activities.append({
                'type': 'physical',
                'title': 'Physical Release',
                'description': 'Activities to release physical tension',
                'suggestions': [
                    'Go for a brisk walk or jog',
                    'Do 20 jumping jacks',
                    'Squeeze a stress ball',
                    'Stretch your body',
                    'Write down what\'s bothering you, then tear up the paper'
                ]
            })
            
        elif dominant_emotion == 'joy':
            activities.append({
                'type': 'gratitude',
                'title': 'Gratitude Practice',
                'description': 'Enhance your positive emotions through gratitude',
                'steps': [
                    'Take a moment to savor this feeling',
                    'Write down 3 things you\'re grateful for right now',
                    'Consider sharing your joy with someone else'
                ]
            })
        
        # Add social connection if loneliness is detected
        if loneliness_trend > 0.4:
            activities.append({
                'type': 'connection',
                'title': 'Social Connection',
                'description': 'Ideas to help you feel more connected',
                'suggestions': [
                    'Reach out to an old friend via text or call',
                    'Join an online community related to your interests',
                    'Schedule a video chat with someone you care about',
                    'Attend a local event or meetup',
                    'Volunteer for a cause you believe in'
                ]
            })
        
        return activities