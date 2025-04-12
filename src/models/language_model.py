import os
import numpy as np
from datetime import datetime, timedelta
import pickle
from typing import List, Dict, Any, Optional, Tuple
import logging
import google.generativeai as genai
import json
import time
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLanguageModel:
    """Base class for language model implementation"""
    def __init__(self):
        self.model = None
        
    def load_model(self):
        """Load the model - to be implemented by child classes"""
        raise NotImplementedError
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response based on the prompt"""
        raise NotImplementedError

class GeminiLanguageModel(BaseLanguageModel):
    """Implementation for Google's Gemini 1.5 Flash API"""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Set GOOGLE_API_KEY environment variable.")
        self.model_name = "gemini-1.5-flash"
        self.temperature = 0.7
        self.max_tokens = 1024
        self.load_model()
    
    def load_model(self):
        """Initialize Google Generative AI client"""
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Google Gemini model {self.model_name} initialized")
        except ImportError:
            logger.error("Google Generative AI library not installed. Run 'pip install google-generativeai'")
            raise
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the Gemini API"""
        try:
            # Create a proper chat format for Gemini
            chat_parts = []
            
            # Extract any system message from the prompt
            if "You are SoulMate.AGI" in prompt:
                system_part = prompt.split("Current conversation:")[0].strip()
                chat_parts.append({"role": "user", "parts": [system_part]})
                chat_parts.append({"role": "model", "parts": ["I understand my role as SoulMate.AGI, a personalized AI companion."]})
                
                # Get the actual user message
                if "Current conversation:" in prompt:
                    current_part = prompt.split("Current conversation:")[1].strip()
                    user_msg = current_part.split("User:")[1].split("You:")[0].strip()
                    chat_parts.append({"role": "user", "parts": [user_msg]})
                else:
                    # Fallback if we can't parse properly
                    chat_parts.append({"role": "user", "parts": [prompt]})
            else:
                # Simpler approach if the prompt doesn't match expected format
                chat_parts.append({"role": "user", "parts": [prompt]})

            # Generate response
            response = self.model.generate_content(
                chat_parts,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.95,
                }
            )
            
            # Extract response text
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                return response.parts[0].text
            else:
                # Handle other response formats if needed
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating response from Gemini API: {e}")
            return "I'm having trouble connecting to my services right now. Let's try again shortly."

class SoulMateLanguageModel:
    """SoulMate.AGI's personalized language model that evolves over time"""
    def __init__(self, user_id: str, use_api: bool = True):
        self.user_id = user_id
        self.use_api = use_api
        self.initialized = False
        self.adaptation_level = 0.0  # 0.0-1.0 scale of how much it has adapted to the user
        self.persona_divergence = 0.0  # How much the model has diverged from base model
        self.training_iterations = 0  # Count of training iterations
        self.emotion_mimic_strength = 0.3  # Starting value for emotional mimicry
        
        try:
            # Initialize with Gemini for language and embeddings
            logger.info(f"Initializing SoulMateLanguageModel for user {user_id}")
            
            # Initialize the memory manager with Google's embedding capabilities
            from src.utils.memory_manager import MemoryManager
            self.memory_manager = MemoryManager(user_id)
            logger.info(f"Initialized memory manager for user {user_id}")
            
            # Initialize the Gemini language model
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                logger.warning("No GOOGLE_API_KEY found. Please set GOOGLE_API_KEY environment variable.")
                raise ValueError("GOOGLE_API_KEY not found. Gemini API key is required.")
            
            self.base_model = GeminiLanguageModel(gemini_api_key)
            logger.info("Using Gemini model for language generation")
            
            # Initialize learning rate parameters
            self.learning_params = {
                'tone_adaptation_rate': 0.05,
                'content_adaptation_rate': 0.03,
                'persona_stability': 0.9,    # Higher value means more stable (changes less)
                'emotional_tracking': 0.7,    # How much to track emotional state changes
                'memory_persistence': 14      # Days to strongly weight memories
            }
            
            # Initialize basic memory structure
            self.user_memory = {
                'emotional_state': {},
                'emotional_history': [],
                'last_training': None,
                'tone_preferences': {},
                'subject_preferences': {},
                'interaction_patterns': {},
                'user_satisfaction': [],
                'learning_progress': []
            }
            
            # Load existing learning parameters if available
            self._load_learning_state()
            
            self.initialized = True
            logger.info(f"Successfully initialized SoulMateLanguageModel for user {user_id}")
        except Exception as e:
            logger.error(f"Error initializing SoulMateLanguageModel: {e}")
            # Still create the basic memory structure even if model loading fails
            self.user_memory = {
                'emotional_state': {},
                'emotional_history': [],
                'last_training': None
            }
            # Re-raise the exception to be handled by the caller
            raise
    
    def _load_learning_state(self):
        """Load learning state from persistent storage"""
        try:
            # Try to get learning parameters from memory manager
            learn_params = self.memory_manager.get_user_preference("learning_parameters")
            if learn_params:
                self.learning_params.update(learn_params)
                logger.info(f"Loaded learning parameters for user {self.user_id}")
            
            # Get adaptation level
            adaptation = self.memory_manager.get_user_preference("adaptation_level")
            if adaptation is not None:
                self.adaptation_level = float(adaptation)
            
            # Get persona divergence
            divergence = self.memory_manager.get_user_preference("persona_divergence")
            if divergence is not None:
                self.persona_divergence = float(divergence)
            
            # Get training iterations
            iterations = self.memory_manager.get_user_preference("training_iterations") 
            if iterations is not None:
                self.training_iterations = int(iterations)
            
            # Get emotional mimicry strength
            mimicry = self.memory_manager.get_user_preference("emotion_mimic_strength")
            if mimicry is not None:
                self.emotion_mimic_strength = float(mimicry)
            
            # Get last training time
            last_training = self.memory_manager.get_user_preference("last_training")
            if last_training:
                self.user_memory['last_training'] = last_training
            
            # Get emotional state
            emotional_state = self.memory_manager.get_user_preference("emotional_state")
            if emotional_state:
                self.user_memory['emotional_state'] = emotional_state
            
            logger.info(f"Learning state loaded for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")
            # Continue with defaults if loading fails
    
    def _save_learning_state(self):
        """Save learning state to persistent storage"""
        try:
            # Save all learning parameters
            self.memory_manager.store_user_preference("learning_parameters", self.learning_params)
            self.memory_manager.store_user_preference("adaptation_level", self.adaptation_level)
            self.memory_manager.store_user_preference("persona_divergence", self.persona_divergence)
            self.memory_manager.store_user_preference("training_iterations", self.training_iterations)
            self.memory_manager.store_user_preference("emotion_mimic_strength", self.emotion_mimic_strength)
            self.memory_manager.store_user_preference("emotional_state", self.user_memory['emotional_state'])
            
            logger.info(f"Learning state saved for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
    
    def record_interaction(self, user_input: str, ai_response: str, context: Dict[str, Any] = None):
        """Record a user interaction for personalization learning using vector database"""
        try:
            # Analyze basic emotional content of the interaction
            from src.utils.emotion_analyzer import EmotionAnalyzer
            analyzer = EmotionAnalyzer()
            
            # Add emotion analysis to context
            if not context:
                context = {}
            
            # Add basic time-context information
            now = datetime.now()
            context['timestamp'] = now.timestamp()
            context['time_of_day'] = now.hour
            context['day_of_week'] = now.weekday()
            
            # Add emotional analysis if not already present
            if 'emotion' not in context:
                emotion, score = analyzer.get_dominant_emotion(user_input)
                context['emotion'] = emotion
                context['emotion_score'] = score
            
            # Update user's emotional state
            self._update_emotional_state(context.get('emotion', 'neutral'), 
                                        context.get('emotion_score', 0.5))
            
            # Store the interaction in the vector database
            self.memory_manager.add_memory(user_input, ai_response, context)
            logger.info(f"Recorded interaction for user {self.user_id} in vector database")
            
            # Track interaction for learning patterns (store in memory temporarily)
            self._track_interaction_pattern(user_input, context)
            
        except Exception as e:
            logger.error(f"Error recording interaction in vector database: {e}")
            logger.error(traceback.format_exc())
    
    def _track_interaction_pattern(self, user_input: str, context: Dict[str, Any]):
        """Track patterns in user interactions for adaptive learning"""
        try:
            # Extract basic metrics about the interaction
            word_count = len(user_input.split())
            contains_question = '?' in user_input
            
            # Bin the time of day
            hour = context.get('time_of_day', 0)
            time_bin = 'morning' if 5 <= hour < 12 else 'afternoon' if 12 <= hour < 17 else 'evening' if 17 <= hour < 22 else 'night'
            
            # Initialize pattern tracking
            if 'interaction_patterns' not in self.user_memory:
                self.user_memory['interaction_patterns'] = {
                    'time_of_day': {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0},
                    'avg_word_count': [],
                    'question_ratio': [0, 0],  # [questions, total]
                    'emotion_frequency': {}
                }
            
            patterns = self.user_memory['interaction_patterns']
            
            # Update time of day
            patterns['time_of_day'][time_bin] += 1
            
            # Update word count tracking
            patterns['avg_word_count'].append(word_count)
            if len(patterns['avg_word_count']) > 100:  # Keep only recent 100
                patterns['avg_word_count'].pop(0)
            
            # Update question tracking
            patterns['question_ratio'][1] += 1  # Total
            if contains_question:
                patterns['question_ratio'][0] += 1  # Questions
            
            # Update emotion frequency
            emotion = context.get('emotion', 'neutral')
            patterns['emotion_frequency'][emotion] = patterns['emotion_frequency'].get(emotion, 0) + 1
            
            # Save updated patterns
            self.memory_manager.store_user_preference("interaction_patterns", patterns)
            
        except Exception as e:
            logger.error(f"Error tracking interaction pattern: {e}")
    
    def _update_emotional_state(self, emotion: str, score: float):
        """Update the model's tracking of user emotional state"""
        try:
            if 'emotional_state' not in self.user_memory:
                self.user_memory['emotional_state'] = {}
            
            # Update the current emotional state with new evidence
            current = self.user_memory['emotional_state']
            
            # Initialize if not present
            if emotion not in current:
                current[emotion] = 0.0
            
            # Update with new evidence, applying emotional tracking rate
            # This creates a weighted moving average of emotional state
            tracking_rate = self.learning_params['emotional_tracking']
            current[emotion] = (current[emotion] * (1-tracking_rate)) + (score * tracking_rate)
            
            # Decay other emotions slightly to ensure emotional state shifts over time
            for e in current:
                if e != emotion:
                    current[e] = current[e] * 0.95
            
            # Add to history with timestamp
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion,
                'score': score
            }
            
            if 'emotional_history' not in self.user_memory:
                self.user_memory['emotional_history'] = []
            
            self.user_memory['emotional_history'].append(history_entry)
            
            # Keep only recent history (last 100 entries)
            if len(self.user_memory['emotional_history']) > 100:
                self.user_memory['emotional_history'] = self.user_memory['emotional_history'][-100:]
            
            # Save to persistent storage periodically (every 5 interactions)
            if len(self.user_memory['emotional_history']) % 5 == 0:
                self.memory_manager.store_user_preference("emotional_state", self.user_memory['emotional_state'])
                self.memory_manager.store_user_preference("emotional_history", self.user_memory['emotional_history'])
            
        except Exception as e:
            logger.error(f"Error updating emotional state: {e}")
    
    def find_similar_interactions(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar previous interactions based on semantic similarity using vector database"""
        try:
            return self.memory_manager.search_similar_interactions(query, k=top_k)
        except Exception as e:
            logger.error(f"Error finding similar interactions: {e}")
            return []
    
    def generate_personalized_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Generate a personalized response based on the user's history and preferences"""
        # Get context from memory manager
        conversation_context = self.memory_manager.get_context_for_conversation(user_input)
        similar_interactions = conversation_context.get("similar_interactions", [])
        preferences = conversation_context.get("preferences", {})
        
        # Initialize context if None
        if context is None:
            context = {}
        
        # Add emotion analysis to context if not present
        if 'emotion' not in context:
            from src.utils.emotion_analyzer import EmotionAnalyzer
            analyzer = EmotionAnalyzer()
            emotion, score = analyzer.get_dominant_emotion(user_input)
            context['emotion'] = emotion
            context['emotion_score'] = score
        
        # Determine if we should mimic the user's emotion based on settings
        should_mimic = self._should_mimic_emotion(context.get('emotion', 'neutral'))
        
        # Check if the message contains a reference to past conversations
        references_past = any(word in user_input.lower() for word in [
            "remember", "mentioned", "said", "told", "talked", "discussed", "earlier", 
            "before", "previously", "last time", "yesterday", "recall"
        ])
        
        # If user is referencing past conversations, try to find more relevant history
        if references_past:
            additional_history = self.memory_manager.search_similar_interactions(user_input, k=7)
            # Add only interactions that aren't already in similar_interactions
            existing_ids = {i.get('id', ''): True for i in similar_interactions}
            for interaction in additional_history:
                if interaction.get('id', '') not in existing_ids:
                    similar_interactions.append(interaction)
        
        # Sort similar_interactions by relevance and timestamp
        if similar_interactions:
            similar_interactions.sort(
                key=lambda x: (
                    x.get('similarity_score', 0) * 0.7 + 
                    (1.0 / (abs(datetime.now().timestamp() - x.get('timestamp', 0)) + 1)) * 0.3
                ),
                reverse=True
            )
        
        # Construct a personalized prompt that includes relevant context and past interactions
        personalized_prompt = f"You are SoulMate.AGI, a personalized AI companion for user {self.user_id}.\n\n"
        
        # Add personalization level indicators
        personalized_prompt += f"Adaptation level: {self.adaptation_level:.2f} (0.0-1.0 scale)\n"
        personalized_prompt += f"Persona development: {self.persona_divergence:.2f} (0.0-1.0 scale)\n\n"
        
        # Add emotional guidance based on detected emotion and mimicry settings
        if should_mimic:
            personalized_prompt += f"The user's message shows {context.get('emotion', 'neutral')} emotion. "
            personalized_prompt += f"Match their emotional tone at a strength of {self.emotion_mimic_strength:.1f} (0-1 scale).\n\n"
        
        # Add user's current emotional state context
        if self.user_memory.get('emotional_state'):
            emotional_state = self.user_memory['emotional_state']
            dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])
            personalized_prompt += f"The user's recent emotional pattern primarily shows {dominant_emotion[0]}.\n\n"
        
        # Add context about user preferences if available
        if preferences:
            # Filter out system preferences and just show relevant user preferences
            user_preferences = {k: v for k, v in preferences.items() 
                               if k not in ["learning_parameters", "emotional_state", "emotional_history", 
                                           "memory_structure", "adaptation_level", "persona_divergence",
                                           "training_iterations", "emotion_mimic_strength"]}
            
            if user_preferences:
                prefs_text = "\n".join([f"- {k}: {v}" for k, v in user_preferences.items()])
                personalized_prompt += f"User preferences:\n{prefs_text}\n\n"
        
        # Add instructions for continuity and coherence
        personalized_prompt += "Important: Maintain continuity with previous conversations. Reference prior topics when relevant.\n\n"
        
        # Get any detected topics the user likes to discuss
        detected_topics = preferences.get("detected_topics", {})
        if detected_topics:
            top_topics = sorted(detected_topics.items(), key=lambda x: x[1], reverse=True)[:5]
            topics_text = ", ".join([f"{topic}" for topic, _ in top_topics])
            personalized_prompt += f"User frequently discusses these topics: {topics_text}\n\n"
        
        # Add relevant past interactions for context, with better contextual markers
        if similar_interactions:
            personalized_prompt += "Relevant past conversations:\n"
            
            # Special handling for conversation threads - group by time proximity
            conversations = []
            current_group = []
            last_timestamp = None
            
            for idx, interaction in enumerate(similar_interactions[:8]):  # Include more history
                timestamp = interaction.get('timestamp', 0)
                
                # Start a new group if timestamps are more than 1 hour apart
                if last_timestamp and (last_timestamp - timestamp) > 3600:
                    if current_group:
                        conversations.append(current_group)
                        current_group = [interaction]
                else:
                    current_group.append(interaction)
                
                last_timestamp = timestamp
            
            # Add the last group
            if current_group:
                conversations.append(current_group)
            
            # Add the conversation threads in reverse order (oldest first in each thread)
            for thread in conversations:
                thread_time = datetime.fromtimestamp(thread[0].get('timestamp', 0))
                days_ago = max(0, (datetime.now() - thread_time).days)
                
                if days_ago == 0:
                    time_marker = "Earlier today:"
                elif days_ago == 1:
                    time_marker = "Yesterday:"
                else:
                    time_marker = f"{days_ago} days ago:"
                
                personalized_prompt += f"{time_marker}\n"
                
                # Add the thread as a continuous conversation
                for msg in reversed(thread):  # Display oldest first
                    personalized_prompt += f"User: {msg['user_input']}\nYou: {msg['ai_response']}\n\n"
        
        # Add information about the interaction patterns
        patterns = self.memory_manager.get_user_preference("interaction_patterns")
        if patterns:
            personalized_prompt += "User interaction patterns:\n"
            
            # Add preferred time of day
            time_prefs = patterns.get('time_of_day', {})
            if time_prefs:
                preferred_time = max(time_prefs.items(), key=lambda x: x[1])[0]
                personalized_prompt += f"- Usually talks to you during the {preferred_time}\n"
            
            # Add average message length
            avg_length = patterns.get('avg_word_count', [])
            if avg_length:
                avg = sum(avg_length) / len(avg_length)
                length_style = "short" if avg < 10 else "moderate" if avg < 25 else "detailed"
                personalized_prompt += f"- Typically writes {length_style} messages (avg {avg:.1f} words)\n"
            
            # Add question frequency
            question_ratio = patterns.get('question_ratio', [0, 0])
            if question_ratio[1] > 0:
                q_ratio = question_ratio[0] / question_ratio[1]
                freq = "frequently" if q_ratio > 0.6 else "sometimes" if q_ratio > 0.3 else "rarely"
                personalized_prompt += f"- {freq.capitalize()} asks questions ({q_ratio*100:.1f}% of messages)\n"
            
            personalized_prompt += "\n"
        
        # Add specific guidance for responding to the current message
        if references_past:
            personalized_prompt += "The user is referring to past conversations. Make sure to acknowledge shared history and provide continuity.\n\n"
        
        if "?" in user_input:
            personalized_prompt += "The user is asking a question. Provide a thoughtful, helpful response.\n\n"
        
        if len(user_input.split()) > 30:
            personalized_prompt += "The user has shared a detailed message. Match their depth in your response.\n\n"
        
        # Add the current input
        personalized_prompt += f"Current conversation:\nUser: {user_input}\nYou:"
        
        # Generate response using the base model
        response = self.base_model.generate_response(personalized_prompt)
        
        # Record this interaction for future learning
        self.record_interaction(user_input, response, context)
        
        return response
    
    def _should_mimic_emotion(self, emotion: str) -> bool:
        """Determine if we should mimic the user's current emotion"""
        # Don't mimic negative emotions too strongly
        if emotion in ['anger', 'sadness', 'fear', 'disgust'] and self.emotion_mimic_strength > 0.7:
            return False
            
        # Base chance of mimicry on the adaptation level - more adapted = more mimicry
        mimic_threshold = 0.3 + (self.adaptation_level * 0.4)  # 0.3-0.7 range
        return np.random.random() < mimic_threshold
    
    def update_preferences(self, preference_key: str, preference_value: Any):
        """Update user preferences in the vector database"""
        try:
            self.memory_manager.store_user_preference(preference_key, preference_value)
            logger.info(f"Updated preference for user {self.user_id}: {preference_key}")
            
            # Track preference updates for learning
            if preference_key.startswith('communication_style'):
                if 'tone_preferences' not in self.user_memory:
                    self.user_memory['tone_preferences'] = {}
                self.user_memory['tone_preferences'][preference_key] = preference_value
                
            elif preference_key.startswith('topics_of_interest'):
                if 'subject_preferences' not in self.user_memory:
                    self.user_memory['subject_preferences'] = {}
                self.user_memory['subject_preferences'][preference_key] = preference_value
                
        except Exception as e:
            logger.error(f"Error updating preference: {e}")
    
    def should_train(self) -> bool:
        """Check if it's time to perform incremental training"""
        # First check if we've had enough interactions since the last training
        similar = self.memory_manager.search_similar_interactions("*", k=1)
        total_interactions = len(similar)
        
        if not self.user_memory.get('last_training'):
            # Initial training threshold - after 10+ interactions
            return total_interactions >= 10
        
        # Get last training time
        last_training = datetime.fromisoformat(self.user_memory['last_training'])
        time_since_training = datetime.now() - last_training
        
        # Train if more than a day has passed AND at least 5 new interactions
        interactions_since_training = total_interactions - self.training_iterations
        
        return (time_since_training.days >= 1 and interactions_since_training >= 5)
    
    def perform_incremental_training(self):
        """Perform incremental learning based on recent interactions"""
        logger.info(f"Starting incremental training for user {self.user_id}")
        
        try:
            # Get recent interactions
            recent_interactions = self.memory_manager.search_similar_interactions("*", k=100)
            
            if not recent_interactions:
                logger.info(f"No interactions found for training user {self.user_id}")
                return
            
            # Step 1: Analyze conversation patterns
            self._analyze_conversation_patterns(recent_interactions)
            
            # Step 2: Update adaptation parameters
            self._update_adaptation_parameters()
            
            # Step 3: Track learning progress
            self._track_learning_progress()
            
            # Update last training timestamp
            self.user_memory['last_training'] = datetime.now().isoformat()
            self.memory_manager.store_user_preference("last_training", self.user_memory['last_training'])
            
            # Increment training iterations counter
            self.training_iterations += 1
            self.memory_manager.store_user_preference("training_iterations", self.training_iterations)
            
            # Save all learning state
            self._save_learning_state()
            
            logger.info(f"Completed incremental training for user {self.user_id}")
            logger.info(f"New adaptation level: {self.adaptation_level:.3f}")
            logger.info(f"New persona divergence: {self.persona_divergence:.3f}")
            
        except Exception as e:
            logger.error(f"Error during incremental training: {e}")
            logger.error(traceback.format_exc())
    
    def _analyze_conversation_patterns(self, interactions: List[Dict[str, Any]]):
        """Analyze conversation patterns for personalization"""
        try:
            # Skip if not enough interactions
            if len(interactions) < 5:
                return
                
            # Extract patterns from recent interactions
            topic_keywords = {}
            response_lengths = []
            emotional_responses = {}
            
            for interaction in interactions:
                # Extract simple keywords (this would be more sophisticated in production)
                words = interaction['user_input'].lower().split()
                for word in words:
                    if len(word) > 3 and word not in ['this', 'that', 'what', 'when', 'where', 'which', 'with']:
                        topic_keywords[word] = topic_keywords.get(word, 0) + 1
                
                # Track response lengths
                if 'ai_response' in interaction:
                    response_lengths.append(len(interaction['ai_response'].split()))
                
                # Track emotional responses
                if 'context' in interaction and 'emotion' in interaction['context']:
                    emotion = interaction['context']['emotion']
                    emotional_responses[emotion] = emotional_responses.get(emotion, 0) + 1
            
            # Update topic preferences based on frequency
            top_topics = sorted(topic_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Update subject preferences
            subject_prefs = {topic: score for topic, score in top_topics}
            self.memory_manager.store_user_preference("detected_topics", subject_prefs)
            
            # Calculate average response length preference
            if response_lengths:
                avg_length = sum(response_lengths) / len(response_lengths)
                self.memory_manager.store_user_preference("average_response_length", avg_length)
            
            # Update emotional response patterns
            if emotional_responses:
                self.memory_manager.store_user_preference("emotional_response_patterns", emotional_responses)
        
        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")
    
    def _update_adaptation_parameters(self):
        """Update the adaptation parameters based on learning progress"""
        try:
            # Calculate base adaptation increment based on interactions and time
            base_increment = 0.01  # Small base increment
            
            # Get interaction count factor - more interactions = more adaptation
            interactions = self.memory_manager.search_similar_interactions("*", k=1)
            interaction_factor = min(len(interactions) / 200, 1.0)  # Cap at 1.0
            
            # Days since first interaction
            first_timestamp = None
            if interactions:
                timestamps = [i.get('timestamp', 0) for i in interactions]
                if timestamps:
                    first_timestamp = min(timestamps)
            
            time_factor = 0.0
            if first_timestamp:
                days_since_first = (datetime.now().timestamp() - first_timestamp) / (24 * 3600)
                time_factor = min(days_since_first / 30, 1.0)  # Cap at 1.0 after 30 days
            
            # Calculate adaptation increment
            adaptation_increment = base_increment * (0.4 * interaction_factor + 0.6 * time_factor)
            
            # Apply increment with diminishing returns (sigmoid-like curve)
            # This ensures it grows quickly at first, then slows down as it approaches 1.0
            current = self.adaptation_level
            max_adapt = 0.95  # Never quite reaches 1.0
            self.adaptation_level = current + (adaptation_increment * (max_adapt - current))
            
            # Update persona divergence (more conservative growth)
            divergence_increment = adaptation_increment * 0.7
            current_divergence = self.persona_divergence
            max_divergence = 0.8  # Cap divergence to maintain some core personality
            self.persona_divergence = current_divergence + (divergence_increment * (max_divergence - current_divergence))
            
            # Increase mimicry strength gradually
            mimicry_increment = adaptation_increment * 0.5
            max_mimicry = 0.85
            self.emotion_mimic_strength = min(self.emotion_mimic_strength + mimicry_increment, max_mimicry)
            
            # Persist the updated values
            self.memory_manager.store_user_preference("adaptation_level", self.adaptation_level)
            self.memory_manager.store_user_preference("persona_divergence", self.persona_divergence)
            self.memory_manager.store_user_preference("emotion_mimic_strength", self.emotion_mimic_strength)
            
        except Exception as e:
            logger.error(f"Error updating adaptation parameters: {e}")
    
    def _track_learning_progress(self):
        """Track the learning progress for analytics"""
        try:
            # Create learning progress entry
            progress_entry = {
                'timestamp': datetime.now().isoformat(),
                'training_iteration': self.training_iterations,
                'adaptation_level': self.adaptation_level,
                'persona_divergence': self.persona_divergence,
                'emotion_mimic_strength': self.emotion_mimic_strength
            }
            
            # Initialize learning progress array if needed
            if 'learning_progress' not in self.user_memory:
                self.user_memory['learning_progress'] = []
            
            # Add the entry
            self.user_memory['learning_progress'].append(progress_entry)
            
            # Keep only the last 50 entries
            if len(self.user_memory['learning_progress']) > 50:
                self.user_memory['learning_progress'] = self.user_memory['learning_progress'][-50:]
            
            # Persist to storage
            self.memory_manager.store_user_preference("learning_progress", self.user_memory['learning_progress'])
            
        except Exception as e:
            logger.error(f"Error tracking learning progress: {e}")
    
    def summarize_daily_thoughts(self) -> str:
        """Generate a summary of the user's recent interactions and emotional patterns"""
        # Get interactions from the last 24 hours for recent analysis
        recent_interactions = self.memory_manager.search_similar_interactions("*", k=30)
        
        # Filter to interactions from the last 24 hours
        now = datetime.now()
        yesterday = now.timestamp() - (24 * 60 * 60)
        
        today_interactions = [
            i for i in recent_interactions 
            if i.get("timestamp", 0) >= yesterday
        ]
        
        # Also get older interactions for comparison (from 2-7 days ago)
        one_week_ago = now.timestamp() - (7 * 24 * 60 * 60)
        older_interactions = self.memory_manager.search_similar_interactions("*", k=50)
        older_interactions = [
            i for i in older_interactions 
            if yesterday > i.get("timestamp", 0) >= one_week_ago
        ]
        
        if not today_interactions:
            if older_interactions:
                return "We haven't chatted today, but based on our previous conversations, I'd love to know how you're feeling today."
            return "We haven't chatted today. How are you feeling?"
        
        # Get emotional patterns for today
        emotional_data_today = []
        for interaction in today_interactions:
            if 'context' in interaction and 'emotion' in interaction['context']:
                emotional_data_today.append({
                    'emotion': interaction['context']['emotion'],
                    'score': interaction['context'].get('emotion_score', 0.5),
                    'timestamp': interaction.get('timestamp', 0)
                })
        
        # Get emotional patterns from older interactions
        emotional_data_older = []
        for interaction in older_interactions:
            if 'context' in interaction and 'emotion' in interaction['context']:
                emotional_data_older.append({
                    'emotion': interaction['context']['emotion'],
                    'score': interaction['context'].get('emotion_score', 0.5),
                    'timestamp': interaction.get('timestamp', 0)
                })
        
        # Calculate emotional distribution for today
        emotion_counts_today = {}
        for entry in emotional_data_today:
            emotion = entry['emotion']
            emotion_counts_today[emotion] = emotion_counts_today.get(emotion, 0) + 1
        
        # Calculate emotional distribution for older interactions
        emotion_counts_older = {}
        for entry in emotional_data_older:
            emotion = entry['emotion']
            emotion_counts_older[emotion] = emotion_counts_older.get(emotion, 0) + 1
        
        # Get dominant emotion for today
        dominant_emotion_today = "neutral"
        if emotion_counts_today:
            dominant_emotion_today = max(emotion_counts_today.items(), key=lambda x: x[1])[0]
        
        # Get dominant emotion for older interactions
        dominant_emotion_older = "neutral"
        if emotion_counts_older:
            dominant_emotion_older = max(emotion_counts_older.items(), key=lambda x: x[1])[0]
        
        # Determine if there's an emotional trend/shift
        emotional_shift = None
        if dominant_emotion_today != dominant_emotion_older and emotion_counts_older:
            emotional_shift = f"shift from {dominant_emotion_older} to {dominant_emotion_today}"
        
        # Extract topics from today's conversations
        topics_today = []
        for interaction in today_interactions:
            words = interaction['user_input'].lower().split()
            topics_today.extend([word for word in words if len(word) > 3 and word not in ['this', 'that', 'what', 'when', 'where', 'which', 'with', 'would', 'could', 'should', 'have', 'been', 'were', 'will', 'about']])
        
        # Extract topics from older conversations
        topics_older = []
        for interaction in older_interactions:
            words = interaction['user_input'].lower().split()
            topics_older.extend([word for word in words if len(word) > 3 and word not in ['this', 'that', 'what', 'when', 'where', 'which', 'with', 'would', 'could', 'should', 'have', 'been', 'were', 'will', 'about']])
        
        # Get top topics for today
        topic_counts_today = {}
        for topic in topics_today:
            topic_counts_today[topic] = topic_counts_today.get(topic, 0) + 1
        
        top_topics_today = sorted(topic_counts_today.items(), key=lambda x: x[1], reverse=True)[:5]
        topics_text_today = ", ".join([topic for topic, _ in top_topics_today]) if top_topics_today else "various subjects"
        
        # Get top topics for older conversations
        topic_counts_older = {}
        for topic in topics_older:
            topic_counts_older[topic] = topic_counts_older.get(topic, 0) + 1
        
        top_topics_older = sorted(topic_counts_older.items(), key=lambda x: x[1], reverse=True)[:5]
        topics_text_older = ", ".join([topic for topic, _ in top_topics_older]) if top_topics_older else "various subjects"
        
        # Identify new topics (topics discussed today but not in older conversations)
        new_topics = []
        if top_topics_today:
            older_topics_set = set([t[0] for t in top_topics_older])
            new_topics = [t[0] for t in top_topics_today if t[0] not in older_topics_set]
        
        # Analyze message patterns
        avg_message_length_today = sum(len(i.get('user_input', '').split()) for i in today_interactions) / len(today_interactions) if today_interactions else 0
        avg_message_length_older = sum(len(i.get('user_input', '').split()) for i in older_interactions) / len(older_interactions) if older_interactions else 0
        
        message_length_change = None
        if avg_message_length_older > 0:
            percent_change = ((avg_message_length_today - avg_message_length_older) / avg_message_length_older) * 100
            if percent_change > 20:
                message_length_change = "significantly longer"
            elif percent_change > 10:
                message_length_change = "somewhat longer"
            elif percent_change < -20:
                message_length_change = "significantly shorter"
            elif percent_change < -10:
                message_length_change = "somewhat shorter"
        
        # Check message frequency
        msgs_per_hour_today = len(today_interactions) / 24 if today_interactions else 0
        
        # Generate a personalized summary prompt based on the analysis
        summary_prompt = f"""
        Based on our {len(today_interactions)} conversations today, create a thoughtful summary that includes:
        
        1. The main themes we discussed today (primarily about {topics_text_today})
        """
        
        # Add emotional insight
        if emotional_shift:
            summary_prompt += f"2. Emotional patterns (your emotions appear to have shifted from {dominant_emotion_older} to {dominant_emotion_today})\n"
        else:
            summary_prompt += f"2. Emotional patterns (your dominant emotion today appears to be {dominant_emotion_today})\n"
        
        # Add topic comparison if there are older conversations
        if older_interactions:
            if new_topics:
                summary_prompt += f"3. New topics that emerged today: {', '.join(new_topics)}\n"
            
            if message_length_change:
                summary_prompt += f"4. Your messages today were {message_length_change} than usual\n"
        
        # Add supportive reflection
        summary_prompt += """
        5. A supportive reflection that connects with your current emotional state
        6. A gentle insight about patterns I've noticed that might be helpful
        
        Make this summary personal, empathetic, and insightful, avoiding generic statements.
        Focus on continuity between our recent conversations and what I know about you from our chat history.
        """
        
        return self.base_model.generate_response(summary_prompt)
    
    def analyze_user_satisfaction(self) -> Dict[str, Any]:
        """Analyze user satisfaction based on interaction patterns"""
        try:
            # Get recent interactions
            recent = self.memory_manager.search_similar_interactions("*", k=50)
            
            if not recent:
                return {
                    "satisfaction_level": "unknown",
                    "suggestion": "Not enough interactions to analyze satisfaction."
                }
            
            # Basic heuristics for satisfaction (would be more sophisticated in production)
            # 1. Frequency of interaction
            now = datetime.now().timestamp()
            timestamps = [i.get('timestamp', now) for i in recent if 'timestamp' in i]
            
            frequency_score = 0.5  # Neutral default
            if timestamps:
                # Sort by recency
                timestamps.sort(reverse=True)
                
                # Get average time between interactions
                time_diffs = []
                for i in range(1, len(timestamps)):
                    diff_hours = (timestamps[i-1] - timestamps[i]) / 3600
                    if diff_hours < 72:  # Ignore gaps longer than 3 days
                        time_diffs.append(diff_hours)
                
                avg_hours_between = sum(time_diffs) / len(time_diffs) if time_diffs else 24
                
                # More frequent = higher satisfaction (generally)
                if avg_hours_between < 2:
                    frequency_score = 0.9  # Very frequent
                elif avg_hours_between < 6:
                    frequency_score = 0.8  # Several times daily
                elif avg_hours_between < 24:
                    frequency_score = 0.7  # Daily
                elif avg_hours_between < 48:
                    frequency_score = 0.6  # Every other day
                elif avg_hours_between >= 48:
                    frequency_score = 0.4  # Less frequent
            
            # 2. Message length trend (increasing = more engaged)
            message_lengths = []
            for interaction in recent:
                message_lengths.append(len(interaction.get('user_input', '').split()))
            
            length_score = 0.5  # Neutral default
            if len(message_lengths) > 5:
                # Compare recent average to earlier average
                midpoint = len(message_lengths) // 2
                recent_avg = sum(message_lengths[:midpoint]) / midpoint
                earlier_avg = sum(message_lengths[midpoint:]) / (len(message_lengths) - midpoint)
                
                if recent_avg > earlier_avg * 1.2:
                    length_score = 0.7  # Increasing significantly
                elif recent_avg > earlier_avg * 1.05:
                    length_score = 0.6  # Increasing slightly
                elif recent_avg < earlier_avg * 0.8:
                    length_score = 0.3  # Decreasing significantly
                elif recent_avg < earlier_avg * 0.95:
                    length_score = 0.4  # Decreasing slightly
            
            # 3. Explicit feedback (would parse for expressions of satisfaction/dissatisfaction)
            feedback_score = 0.5  # Neutral default
            positive_indicators = ['thank', 'thanks', 'helpful', 'good', 'great', 'excellent', 'appreciate']
            negative_indicators = ['useless', 'unhelpful', 'bad', 'wrong', 'incorrect', 'frustrated']
            
            positive_count = 0
            negative_count = 0
            
            for interaction in recent:
                text = interaction.get('user_input', '').lower()
                for word in positive_indicators:
                    if word in text:
                        positive_count += 1
                for word in negative_indicators:
                    if word in text:
                        negative_count += 1
            
            total_indicators = positive_count + negative_count
            if total_indicators > 0:
                feedback_score = 0.5 + (0.5 * (positive_count - negative_count) / total_indicators)
            
            # Combine scores with weights
            satisfaction_score = (frequency_score * 0.3) + (length_score * 0.3) + (feedback_score * 0.4)
            
            # Map to qualitative levels
            if satisfaction_score > 0.8:
                level = "very_satisfied"
                suggestion = "User appears very satisfied. Continue current approach."
            elif satisfaction_score > 0.6:
                level = "satisfied"
                suggestion = "User appears satisfied. Consider small improvements in personalization."
            elif satisfaction_score > 0.4:
                level = "neutral"
                suggestion = "User satisfaction appears neutral. Consider more personalized responses."
            elif satisfaction_score > 0.2:
                level = "dissatisfied"
                suggestion = "User may be dissatisfied. Review interaction patterns and adjust tone/style."
            else:
                level = "very_dissatisfied"
                suggestion = "User appears very dissatisfied. Significant changes needed to approach."
            
            return {
                "satisfaction_score": satisfaction_score,
                "satisfaction_level": level,
                "frequency_score": frequency_score,
                "length_score": length_score,
                "feedback_score": feedback_score,
                "suggestion": suggestion
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user satisfaction: {e}")
            return {
                "satisfaction_level": "error",
                "suggestion": "Error analyzing satisfaction."
            }