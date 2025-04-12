import os
import logging
import time
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalTrainer:
    """
    Manages the incremental training of SoulMate.AGI models
    Handles scheduling and executing nightly training jobs
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.running = False
        self.scheduler_thread = None
        self.active_training_jobs = {}  # user_id -> status dict
        
        # Initialize configurations
        self.training_hour = int(os.getenv('TRAINING_HOUR', '2'))  # 2 AM default
        self.training_interval_days = int(os.getenv('TRAINING_INTERVAL_DAYS', '1'))
        self.min_interactions_for_training = int(os.getenv('MIN_INTERACTIONS_FOR_TRAINING', '5'))
        
        logger.info(f"Incremental trainer initialized with training hour: {self.training_hour}")
    
    def start_scheduler(self):
        """Start the scheduling thread for periodic training jobs"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        
        # Schedule the daily check at the specified hour
        schedule.every().day.at(f"{self.training_hour:02d}:00").do(self.check_and_train_all_models)
        
        # Start the scheduler in a separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"Incremental training scheduler started, will run daily at {self.training_hour:02d}:00")
    
    def _run_scheduler(self):
        """Run the scheduler in a loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """Stop the scheduling thread"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            self.scheduler_thread = None
        logger.info("Incremental training scheduler stopped")
    
    def check_and_train_all_models(self):
        """Check all user models and train those that are eligible"""
        if not self.db_manager:
            logger.error("No database manager available for incremental training")
            return
        
        try:
            # Get all user profiles
            user_profiles = self._get_all_user_profiles()
            
            if not user_profiles:
                logger.info("No user profiles found for incremental training")
                return
            
            logger.info(f"Found {len(user_profiles)} user profiles for incremental training check")
            
            # Check each user for training eligibility
            trained_count = 0
            for user_id in user_profiles:
                try:
                    if self._check_training_eligibility(user_id):
                        # Start training in a separate thread to avoid blocking
                        threading.Thread(
                            target=self._train_user_model,
                            args=(user_id,),
                            daemon=True
                        ).start()
                        trained_count += 1
                except Exception as e:
                    logger.error(f"Error checking training eligibility for user {user_id}: {e}")
            
            logger.info(f"Started incremental training for {trained_count} user models")
            
        except Exception as e:
            logger.error(f"Error during incremental training check: {e}")
    
    def _get_all_user_profiles(self) -> List[str]:
        """Get all user IDs from user profiles"""
        try:
            # In a production system, you would query the database directly
            # Here we'll use the profiles in memory
            if hasattr(self.db_manager, 'user_profiles') and self.db_manager.user_profiles is not None:
                profiles = list(self.db_manager.user_profiles.find())
                return [p['user_id'] for p in profiles]
            elif hasattr(self.db_manager, 'fallback_profiles'):
                return list(self.db_manager.fallback_profiles.keys())
            else:
                logger.warning("No user profiles available")
                return []
        except Exception as e:
            logger.error(f"Error getting user profiles: {e}")
            return []
    
    def _check_training_eligibility(self, user_id: str) -> bool:
        """
        Check if a user's model is eligible for incremental training
        
        Args:
            user_id: The user ID to check
            
        Returns:
            True if eligible for training, False otherwise
        """
        try:
            # Check if the user is already being trained
            if user_id in self.active_training_jobs:
                return False
            
            # Import SoulMateLanguageModel here to avoid circular imports
            from src.models.language_model import SoulMateLanguageModel
            
            # Create a temporary model instance to check training eligibility
            model = SoulMateLanguageModel(user_id=user_id)
            
            # Check if training is recommended based on time and interaction count
            return model.should_train()
            
        except Exception as e:
            logger.error(f"Error checking training eligibility for user {user_id}: {e}")
            return False
    
    def _train_user_model(self, user_id: str):
        """
        Perform incremental training for a user's model
        
        Args:
            user_id: The user ID to train
        """
        try:
            # Mark this user as being trained
            self.active_training_jobs[user_id] = {
                'start_time': datetime.now().isoformat(),
                'status': 'started'
            }
            
            # Import SoulMateLanguageModel here to avoid circular imports
            from src.models.language_model import SoulMateLanguageModel
            
            # Create model instance
            model = SoulMateLanguageModel(user_id=user_id)
            
            # Log training start
            logger.info(f"Starting incremental training for user {user_id}")
            self.active_training_jobs[user_id]['status'] = 'training'
            
            # Perform the training
            model.perform_incremental_training()
            
            # Update status
            self.active_training_jobs[user_id]['status'] = 'completed'
            self.active_training_jobs[user_id]['end_time'] = datetime.now().isoformat()
            
            # Log completion
            logger.info(f"Completed incremental training for user {user_id}")
            
            # Remove from active jobs after some time
            def _cleanup_job():
                if user_id in self.active_training_jobs:
                    del self.active_training_jobs[user_id]
            
            # Schedule cleanup after 1 hour
            threading.Timer(3600, _cleanup_job).start()
            
        except Exception as e:
            logger.error(f"Error during incremental training for user {user_id}: {e}")
            logger.error(traceback.format_exc())
            
            if user_id in self.active_training_jobs:
                self.active_training_jobs[user_id]['status'] = 'failed'
                self.active_training_jobs[user_id]['error'] = str(e)
                self.active_training_jobs[user_id]['end_time'] = datetime.now().isoformat()
    
    def get_training_status(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get the status of training jobs
        
        Args:
            user_id: Optional user ID to get status for a specific user
            
        Returns:
            Dictionary with training status information
        """
        if user_id:
            # Return status for specific user
            if user_id in self.active_training_jobs:
                return {
                    'user_id': user_id,
                    **self.active_training_jobs[user_id]
                }
            else:
                # Check if the user has been trained recently
                try:
                    from src.utils.memory_manager import MemoryManager
                    memory_manager = MemoryManager(user_id)
                    last_training = memory_manager.get_user_preference("last_training")
                    
                    if last_training:
                        # Parse the ISO timestamp
                        last_time = datetime.fromisoformat(last_training)
                        return {
                            'user_id': user_id,
                            'status': 'idle',
                            'last_training': last_training,
                            'last_training_days_ago': (datetime.now() - last_time).days
                        }
                except Exception as e:
                    logger.error(f"Error checking last training: {e}")
                
                return {
                    'user_id': user_id,
                    'status': 'never_trained'
                }
        else:
            # Return status for all active jobs
            return {
                'active_jobs': len(self.active_training_jobs),
                'jobs': [{'user_id': uid, **status} for uid, status in self.active_training_jobs.items()]
            }
    
    def manually_trigger_training(self, user_id: str) -> bool:
        """
        Manually trigger training for a specific user
        
        Args:
            user_id: The user ID to train
            
        Returns:
            True if training was started, False otherwise
        """
        try:
            # Check if the user is already being trained
            if user_id in self.active_training_jobs:
                return False
            
            # Start training in a separate thread
            threading.Thread(
                target=self._train_user_model,
                args=(user_id,),
                daemon=True
            ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error triggering manual training for user {user_id}: {e}")
            return False