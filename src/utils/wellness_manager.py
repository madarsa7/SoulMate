import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WellnessManager:
    """
    Manages wellness activities and interventions for SoulMate.AGI
    Provides personalized wellness suggestions based on user's emotional state
    """
    
    def __init__(self, language_model=None, db_manager=None):
        self.language_model = language_model
        self.db_manager = db_manager
        logger.info("Wellness Manager initialized")
        
        # Initialize wellness activities
        self._init_activities()
    
    def _init_activities(self):
        """Initialize base wellness activities"""
        
        # Positive affirmations by category
        self.affirmations = {
            "confidence": [
                "You are capable of amazing things.",
                "Your potential is limitless.",
                "You have the power to create change.",
                "You are getting stronger every day.",
                "You've survived all your difficult days so far."
            ],
            "peace": [
                "You deserve peace and happiness.",
                "It's okay to take time for yourself.",
                "You are allowed to set boundaries.",
                "This moment is your life, and it's enough.",
                "Peace begins with you and your breath."
            ],
            "gratitude": [
                "There is always something to be grateful for.",
                "The little things in life are actually the big things.",
                "Each day brings new gifts to unwrap.",
                "Gratitude turns what we have into enough.",
                "Even in difficulty, there are moments of beauty."
            ],
            "resilience": [
                "You can handle whatever comes your way.",
                "Challenges help you grow stronger.",
                "This difficult time is temporary.",
                "You've overcome challenges before, and you will again.",
                "Your resilience is inspiring."
            ],
            "self_love": [
                "You are worthy of love and respect.",
                "You are enough exactly as you are.",
                "Your value doesn't depend on your productivity.",
                "Self-care isn't selfish, it's necessary.",
                "You matter. Your feelings matter."
            ]
        }
        
        # Breathing exercises
        self.breathing_exercises = [
            {
                "name": "4-7-8 Breathing",
                "description": "Calms the nervous system and helps with anxiety",
                "steps": [
                    "Inhale quietly through your nose for 4 seconds",
                    "Hold your breath for 7 seconds",
                    "Exhale completely through your mouth for 8 seconds",
                    "Repeat 4-6 times"
                ],
                "duration": "2-3 minutes"
            },
            {
                "name": "Box Breathing",
                "description": "Reduces stress and improves concentration",
                "steps": [
                    "Inhale through your nose for 4 seconds",
                    "Hold your breath for 4 seconds",
                    "Exhale through your mouth for 4 seconds",
                    "Hold your breath for 4 seconds",
                    "Repeat 5-10 times"
                ],
                "duration": "3-5 minutes"
            },
            {
                "name": "Alternate Nostril Breathing",
                "description": "Balances the mind and reduces anxiety",
                "steps": [
                    "Close your right nostril with your right thumb and inhale through your left nostril",
                    "Close your left nostril with your right ring finger, release your thumb, and exhale through your right nostril",
                    "Inhale through your right nostril",
                    "Close your right nostril, release your ring finger, and exhale through your left nostril",
                    "Repeat 5-10 times"
                ],
                "duration": "3-5 minutes"
            },
            {
                "name": "Diaphragmatic Breathing",
                "description": "Reduces stress and promotes relaxation",
                "steps": [
                    "Place one hand on your chest and the other on your abdomen",
                    "Inhale deeply through your nose, feeling your abdomen expand (not your chest)",
                    "Exhale slowly through pursed lips",
                    "Repeat 5-10 times"
                ],
                "duration": "3-5 minutes"
            }
        ]
        
        # Music suggestions by mood
        self.music_suggestions = {
            "calm": [
                "Classical piano pieces by Ludovico Einaudi",
                "Ambient music by Brian Eno",
                "Acoustic guitar instrumentals",
                "'Weightless' by Marconi Union (scientifically designed to reduce anxiety)",
                "Nature sounds with gentle piano"
            ],
            "joy": [
                "Upbeat pop songs with positive lyrics",
                "Feel-good classics from the 80s",
                "Jazz standards with uplifting tempos",
                "Cheerful instrumental film soundtracks",
                "Dance music with positive messages"
            ],
            "focus": [
                "Lo-fi beats for studying/working",
                "Instrumental post-rock (like Explosions in the Sky)",
                "Minimal electronic music without lyrics",
                "Baroque classical music (Bach, Vivaldi)",
                "Ambient soundscapes"
            ],
            "energy": [
                "Upbeat electronic dance music",
                "Motivational workout playlists",
                "Rhythmic drum-heavy tracks",
                "High-energy rock songs",
                "Uplifting orchestral film scores"
            ],
            "reflection": [
                "Acoustic singer-songwriter ballads",
                "Atmospheric ambient music",
                "Solo piano pieces",
                "Gentle classical music (like Debussy)",
                "Instrumental folk music"
            ]
        }
        
        # Grounding exercises
        self.grounding_exercises = [
            {
                "name": "5-4-3-2-1 Technique",
                "description": "Uses your five senses to bring you back to the present",
                "steps": [
                    "Name 5 things you can see",
                    "Name 4 things you can feel/touch",
                    "Name 3 things you can hear",
                    "Name 2 things you can smell (or like the smell of)",
                    "Name 1 thing you can taste (or like the taste of)"
                ]
            },
            {
                "name": "Body Scan",
                "description": "Helps you reconnect with your physical body",
                "steps": [
                    "Close your eyes and take three deep breaths",
                    "Starting from your toes, slowly bring awareness to each part of your body",
                    "Notice any sensations without judgment",
                    "Gradually move up through your entire body",
                    "End with three more deep breaths"
                ]
            },
            {
                "name": "Object Focus",
                "description": "Directs attention to a physical object",
                "steps": [
                    "Find any object near you",
                    "Hold it in your hands and examine it carefully",
                    "Note its color, texture, weight, temperature, and any other physical properties",
                    "Focus completely on the object for 1-2 minutes"
                ]
            }
        ]
        
        # Simple guided visualization themes
        self.visualization_themes = [
            {
                "name": "Peaceful Beach",
                "description": "A calming beach visualization",
                "prompt": "Imagine yourself walking along a peaceful beach at sunset. Feel the warm sand beneath your feet, hear the gentle waves, and breathe in the fresh ocean air."
            },
            {
                "name": "Forest Sanctuary",
                "description": "A serene forest setting",
                "prompt": "Picture yourself in a lush green forest. Sunlight filters through the leaves, birds sing softly, and a gentle breeze carries the earthy scent of trees."
            },
            {
                "name": "Mountain Viewpoint",
                "description": "A perspective-giving mountain scene",
                "prompt": "Visualize standing on a mountain overlook. The air is clear and cool, and you can see for miles in every direction. Feel the vastness and your place in it."
            },
            {
                "name": "Safe Haven",
                "description": "A personalized safe place",
                "prompt": "Create in your mind a space that feels completely safe and comfortable. This can be real or imagined. Fill it with things that bring you peace."
            }
        ]
    
    def get_personalized_wellness_activity(self, user_id: str, emotion: str = None) -> Dict[str, Any]:
        """
        Get a personalized wellness activity based on the user's current emotional state
        
        Args:
            user_id: The user's ID
            emotion: The current dominant emotion (optional)
            
        Returns:
            A wellness activity suggestion
        """
        try:
            # If no emotion provided, try to get from recent history
            if not emotion and self.db_manager:
                try:
                    from src.utils.emotion_analyzer import EmotionAnalyzer
                    analyzer = EmotionAnalyzer(self.db_manager)
                    trend = analyzer.get_emotion_trend(user_id, time_window_hours=12)
                    emotion = trend.get('dominant_emotion', 'neutral')
                except Exception as e:
                    logger.warning(f"Error getting emotion trend: {e}")
                    emotion = 'neutral'
            
            # Get user preferences if available
            activity_preferences = {}
            if self.db_manager:
                try:
                    prefs = self.db_manager.get_preference(user_id, "wellness_preferences", {})
                    activity_preferences = prefs
                except Exception as e:
                    logger.warning(f"Error getting wellness preferences: {e}")
            
            # Map emotions to activity types
            emotion_to_activity = {
                "joy": ["affirmation", "gratitude", "music"],
                "sadness": ["breathing", "affirmation", "visualization", "music", "story"],
                "anger": ["breathing", "grounding", "music"],
                "fear": ["breathing", "grounding", "visualization"],
                "anxiety": ["breathing", "grounding", "visualization"],
                "stress": ["breathing", "music", "visualization"],
                "neutral": ["affirmation", "breathing", "music", "story"],
                "loneliness": ["story", "affirmation", "music"]
            }
            
            # Get appropriate activity types for the emotion
            activity_types = emotion_to_activity.get(emotion, ["affirmation", "breathing", "music"])
            
            # Check user preferences
            preferred_types = activity_preferences.get("preferred_activities", [])
            if preferred_types:
                # Prioritize preferred activities but keep some variety
                activity_types = [t for t in activity_types if t in preferred_types]
                if not activity_types:  # If filtering removed all options
                    activity_types = preferred_types
            
            # Select a random activity type
            activity_type = random.choice(activity_types)
            
            # Generate the activity based on type
            if activity_type == "affirmation":
                return self._generate_affirmation(emotion)
            elif activity_type == "breathing":
                return self._generate_breathing_exercise()
            elif activity_type == "grounding":
                return self._generate_grounding_exercise()
            elif activity_type == "visualization":
                return self._generate_visualization()
            elif activity_type == "music":
                return self._generate_music_suggestion(emotion)
            elif activity_type == "story":
                return self._generate_story(user_id, emotion)
            else:
                # Fallback
                return self._generate_affirmation(emotion)
        
        except Exception as e:
            logger.error(f"Error generating wellness activity: {e}")
            # Return a simple affirmation as fallback
            return {
                "type": "affirmation",
                "content": "You are doing your best, and that is enough.",
                "title": "Simple Affirmation"
            }
    
    def _generate_affirmation(self, emotion: str) -> Dict[str, Any]:
        """Generate a positive affirmation based on emotion"""
        
        # Map emotions to affirmation categories
        emotion_map = {
            "joy": "gratitude",
            "sadness": "resilience",
            "anger": "peace",
            "fear": "confidence",
            "anxiety": "confidence",
            "stress": "peace",
            "neutral": "self_love",
            "loneliness": "self_love"
        }
        
        category = emotion_map.get(emotion, "self_love")
        affirmations = self.affirmations.get(category, self.affirmations["self_love"])
        
        return {
            "type": "affirmation",
            "content": random.choice(affirmations),
            "title": f"{category.replace('_', ' ').title()} Affirmation",
            "category": category
        }
    
    def _generate_breathing_exercise(self) -> Dict[str, Any]:
        """Generate a breathing exercise"""
        exercise = random.choice(self.breathing_exercises)
        
        return {
            "type": "breathing",
            "title": exercise["name"],
            "description": exercise["description"],
            "steps": exercise["steps"],
            "duration": exercise["duration"]
        }
    
    def _generate_grounding_exercise(self) -> Dict[str, Any]:
        """Generate a grounding exercise"""
        exercise = random.choice(self.grounding_exercises)
        
        return {
            "type": "grounding",
            "title": exercise["name"],
            "description": exercise["description"],
            "steps": exercise["steps"]
        }
    
    def _generate_visualization(self) -> Dict[str, Any]:
        """Generate a guided visualization"""
        theme = random.choice(self.visualization_themes)
        
        return {
            "type": "visualization",
            "title": theme["name"],
            "description": theme["description"],
            "prompt": theme["prompt"]
        }
    
    def _generate_music_suggestion(self, emotion: str) -> Dict[str, Any]:
        """Generate a music suggestion based on emotion"""
        
        # Map emotions to music categories
        emotion_map = {
            "joy": "joy",
            "sadness": "reflection",
            "anger": "calm",
            "fear": "calm",
            "anxiety": "calm",
            "stress": "calm",
            "neutral": "focus",
            "loneliness": "reflection"
        }
        
        category = emotion_map.get(emotion, "calm")
        suggestions = self.music_suggestions.get(category, self.music_suggestions["calm"])
        
        return {
            "type": "music",
            "title": f"Music for {category.title()}",
            "suggestions": random.sample(suggestions, min(3, len(suggestions))),
            "mood": category
        }
    
    def _generate_story(self, user_id: str, emotion: str) -> Dict[str, Any]:
        """
        Generate an AI story based on user preferences and emotional state
        
        This requires the language_model to be set during initialization
        """
        if not self.language_model:
            return {
                "type": "story",
                "title": "Moment of Reflection",
                "content": "Take a moment to reflect on a positive memory. Close your eyes and revisit that experience in as much detail as you can.",
            }
        
        # Get user preferences if available
        story_preferences = {}
        if self.db_manager:
            try:
                prefs = self.db_manager.get_preference(user_id, "story_preferences", {})
                story_preferences = prefs
            except Exception as e:
                logger.warning(f"Error getting story preferences: {e}")
        
        # Default story themes by emotion
        emotion_themes = {
            "joy": ["uplifting", "adventure", "achievement"],
            "sadness": ["hope", "transformation", "comfort"],
            "anger": ["justice", "understanding", "resolution"],
            "fear": ["courage", "safety", "overcoming"],
            "anxiety": ["calm", "clarity", "control"],
            "stress": ["simplicity", "nature", "peace"],
            "neutral": ["curiosity", "discovery", "mystery"],
            "loneliness": ["connection", "friendship", "belonging"]
        }
        
        # Get appropriate themes for the emotion
        themes = emotion_themes.get(emotion, ["hope", "wonder", "connection"])
        
        # Incorporate user preferences if available
        preferred_themes = story_preferences.get("preferred_themes", [])
        if preferred_themes:
            themes.extend(preferred_themes)
        
        preferred_length = story_preferences.get("preferred_length", "medium")
        length_words = {"short": 150, "medium": 300, "long": 500}.get(preferred_length, 300)
        
        # Generate story prompt
        prompt = f"""
        Create a brief, uplifting story of around {length_words} words with themes of {', '.join(themes[:3])}.
        The story should be emotionally resonant and appropriate for someone feeling {emotion}.
        It should be personal and intimate in tone, as if sharing a meaningful story with a friend.
        The story should end on a hopeful or meaningful note.
        Just provide the story text without any introduction or explanation.
        """
        
        try:
            # Generate the story using the language model
            story_text = self.language_model.base_model.generate_response(prompt)
            
            # Generate a title for the story
            title_prompt = f"Create a short, engaging title for this story (maximum 6 words):\n\n{story_text[:100]}..."
            title = self.language_model.base_model.generate_response(title_prompt)
            
            # Clean up title (remove quotes, periods, etc.)
            title = title.strip('"\'.,!?').strip()
            
            return {
                "type": "story",
                "title": title,
                "content": story_text,
                "themes": themes[:3]
            }
        except Exception as e:
            logger.error(f"Error generating story: {e}")
            return {
                "type": "story",
                "title": "A Moment of Connection",
                "content": "Sometimes, the most important stories are the ones we create in our own lives. Take a moment to reflect on a time when you felt truly connected to someone or something. What made that moment special?",
            }
    
    def record_activity_feedback(self, user_id: str, activity_type: str, rating: int, feedback: str = None) -> bool:
        """
        Record user feedback on a wellness activity
        
        Args:
            user_id: The user's ID
            activity_type: The type of activity (affirmation, breathing, etc.)
            rating: User rating (1-5)
            feedback: Optional text feedback
            
        Returns:
            Success status
        """
        if not self.db_manager:
            logger.warning("No database manager available to store feedback")
            return False
        
        try:
            # Get existing preferences
            wellness_prefs = self.db_manager.get_preference(user_id, "wellness_preferences", {})
            
            # Initialize if needed
            if "activity_ratings" not in wellness_prefs:
                wellness_prefs["activity_ratings"] = {}
            if "preferred_activities" not in wellness_prefs:
                wellness_prefs["preferred_activities"] = []
            
            # Update ratings
            if activity_type not in wellness_prefs["activity_ratings"]:
                wellness_prefs["activity_ratings"][activity_type] = []
                
            # Add new rating
            rating_entry = {
                "timestamp": datetime.now().isoformat(),
                "rating": rating,
                "feedback": feedback
            }
            wellness_prefs["activity_ratings"][activity_type].append(rating_entry)
            
            # Keep only the most recent 10 ratings per activity
            if len(wellness_prefs["activity_ratings"][activity_type]) > 10:
                wellness_prefs["activity_ratings"][activity_type] = wellness_prefs["activity_ratings"][activity_type][-10:]
            
            # Update preferred activities based on ratings
            preferred = []
            for act_type, ratings in wellness_prefs["activity_ratings"].items():
                if not ratings:
                    continue
                
                # Calculate average rating
                avg_rating = sum(r["rating"] for r in ratings) / len(ratings)
                if avg_rating >= 4.0:  # If average rating is 4 or higher
                    preferred.append(act_type)
            
            wellness_prefs["preferred_activities"] = preferred
            
            # Store updated preferences
            success = self.db_manager.store_preference(user_id, "wellness_preferences", wellness_prefs)
            
            return success
        except Exception as e:
            logger.error(f"Error recording activity feedback: {e}")
            return False
    
    def get_loneliness_support(self, user_id: str, loneliness_score: float = None) -> Dict[str, Any]:
        """
        Get personalized support for loneliness
        
        Args:
            user_id: The user's ID
            loneliness_score: Loneliness score if available (0-1 scale)
            
        Returns:
            Dictionary with support strategies
        """
        try:
            # If no loneliness score provided, try to get from emotion analyzer
            if loneliness_score is None and self.db_manager:
                try:
                    from src.utils.emotion_analyzer import EmotionAnalyzer
                    analyzer = EmotionAnalyzer(self.db_manager)
                    loneliness_risk = analyzer.analyze_loneliness_risk(user_id)
                    loneliness_score = loneliness_risk.get('risk_score', 0.5)
                except Exception as e:
                    logger.warning(f"Error getting loneliness risk: {e}")
                    loneliness_score = 0.5
            
            # Default to moderate if no score available
            if loneliness_score is None:
                loneliness_score = 0.5
            
            # Determine level of support needed
            if loneliness_score < 0.3:
                level = "light"
            elif loneliness_score < 0.7:
                level = "moderate"
            else:
                level = "significant"
            
            # Generate support strategies
            strategies = []
            
            if level in ["moderate", "significant"]:
                strategies.append({
                    "title": "Reach Out",
                    "description": "Send a message to someone you haven't spoken to in a while, just to check in.",
                    "difficulty": "medium"
                })
                
                strategies.append({
                    "title": "Community Engagement",
                    "description": "Find an online or local community related to your interests. Even brief interactions can help.",
                    "difficulty": "medium"
                })
            
            if level == "significant":
                strategies.append({
                    "title": "Professional Support",
                    "description": "Consider speaking with a mental health professional who can provide personalized strategies.",
                    "difficulty": "high"
                })
            
            # These are helpful for all levels
            strategies.append({
                "title": "Nature Connection",
                "description": "Spend 15 minutes in nature. The natural world can provide a sense of connection beyond human relationships.",
                "difficulty": "low"
            })
            
            strategies.append({
                "title": "Mindful Solitude",
                "description": "Transform alone time into quality 'me time' with a purposeful activity you enjoy.",
                "difficulty": "low"
            })
            
            if self.language_model:
                # Generate a personalized reflection using the language model
                prompt = f"""
                Create a brief, supportive message (about 100 words) for someone experiencing {"mild" if level == "light" else "moderate" if level == "moderate" else "significant"} feelings of loneliness.
                The message should be compassionate but not pitying, and include 1-2 practical insights about loneliness that normalize the experience.
                End with an uplifting thought that encourages gentle self-compassion.
                """
                
                try:
                    reflection = self.language_model.base_model.generate_response(prompt)
                except Exception:
                    reflection = "Remember that loneliness is a universal human experience. It signals our deep need for connection, which is a strength, not a weakness. Be as kind to yourself in these moments as you would be to a good friend."
            else:
                reflection = "Remember that loneliness is a universal human experience. It signals our deep need for connection, which is a strength, not a weakness. Be as kind to yourself in these moments as you would be to a good friend."
            
            return {
                "level": level,
                "strategies": strategies,
                "reflection": reflection,
                "score": loneliness_score
            }
            
        except Exception as e:
            logger.error(f"Error generating loneliness support: {e}")
            return {
                "level": "moderate",
                "strategies": [
                    {
                        "title": "Reach Out",
                        "description": "Send a message to someone you care about, even if it's just to say hello.",
                        "difficulty": "medium"
                    },
                    {
                        "title": "Self-Compassion Break",
                        "description": "Take a moment to place your hand on your heart and speak to yourself with kindness.",
                        "difficulty": "low"
                    }
                ],
                "reflection": "Remember that loneliness is a universal human experience. It signals our deep need for connection, which is a strength, not a weakness."
            }
    
    def get_daily_wellness_plan(self, user_id: str) -> Dict[str, Any]:
        """
        Generate a personalized daily wellness plan
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dictionary with wellness plan components
        """
        try:
            # Get emotional state if available
            emotion = "neutral"
            if self.db_manager:
                try:
                    from src.utils.emotion_analyzer import EmotionAnalyzer
                    analyzer = EmotionAnalyzer(self.db_manager)
                    trend = analyzer.get_emotion_trend(user_id, time_window_hours=48)
                    emotion = trend.get('dominant_emotion', 'neutral')
                except Exception as e:
                    logger.warning(f"Error getting emotion trend: {e}")
            
            # Generate plan components
            morning = self._generate_breathing_exercise()
            
            if emotion in ["sadness", "fear", "anxiety"]:
                midday = self._generate_affirmation(emotion)
            else:
                midday = self._generate_music_suggestion(emotion)
                
            evening = self._generate_grounding_exercise()
            
            # Generate a personalized message if language model available
            if self.language_model:
                prompt = f"""
                Create a brief, encouraging message (about 80 words) for a daily wellness plan.
                The message should acknowledge that self-care can sometimes be challenging but is worth the effort.
                Make it warm, supportive, and genuine - like a caring friend would write.
                The person's recent emotional state has generally been {emotion}.
                """
                
                try:
                    message = self.language_model.base_model.generate_response(prompt)
                except Exception:
                    message = "Taking care of yourself isn't always easy, but it's always worth it. Small acts of self-care add up to significant changes in how we feel. Be patient and gentle with yourself today."
            else:
                message = "Taking care of yourself isn't always easy, but it's always worth it. Small acts of self-care add up to significant changes in how we feel. Be patient and gentle with yourself today."
            
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "message": message,
                "activities": {
                    "morning": morning,
                    "midday": midday,
                    "evening": evening
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating wellness plan: {e}")
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "message": "Small acts of self-care can make a big difference in your day. Remember that taking care of yourself is never selfish—it's necessary.",
                "activities": {
                    "morning": {
                        "type": "breathing",
                        "title": "Morning Breathing",
                        "description": "Start your day with mindful breathing",
                        "steps": [
                            "Take 5 deep breaths, focusing on the sensation of breathing",
                            "With each exhale, imagine releasing any tension"
                        ]
                    },
                    "midday": {
                        "type": "affirmation",
                        "title": "Midday Affirmation",
                        "content": "I am doing my best, and that is enough."
                    },
                    "evening": {
                        "type": "grounding",
                        "title": "Evening Wind-Down",
                        "description": "A simple practice to transition to evening",
                        "steps": [
                            "Notice 5 things you can see",
                            "Notice 4 things you can touch",
                            "Notice 3 things you can hear",
                            "Notice 2 things you can smell",
                            "Notice 1 thing you can taste"
                        ]
                    }
                }
            }