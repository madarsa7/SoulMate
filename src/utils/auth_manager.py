import os
import logging
import bcrypt
# Fix the import to use the lowercase 'jwt' module name
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class AuthManager:
    """
    Manages user authentication, sessions, and security for SoulMate.AGI
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize the authentication manager
        
        Args:
            db_manager: Database manager instance for storing user credentials
        """
        self.db_manager = db_manager
        self.jwt_secret = os.getenv('JWT_SECRET', 'soulmate-jwt-secret-development')
        self.token_expiry = int(os.getenv('TOKEN_EXPIRY_DAYS', '30'))
        
        logger.info("Authentication manager initialized")
    
    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Register a new user
        
        Args:
            username: Username for the new account
            password: Password for the new account
            email: Optional email address
            
        Returns:
            (success, message, user_data) tuple
        """
        try:
            # Check if username already exists
            if self.db_manager:
                existing = self.db_manager.get_preference(username, "auth_data")
                if existing:
                    return False, "Username already exists", None
            
            # Hash the password with bcrypt
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Create user data
            user_id = str(uuid.uuid4())
            user_data = {
                "user_id": user_id,
                "username": username,
                "password_hash": password_hash,
                "email": email,
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
            
            # Store in database
            if self.db_manager:
                success = self.db_manager.store_preference(username, "auth_data", user_data)
                if not success:
                    return False, "Failed to save user data", None
            
            # Return user data without password hash
            user_info = user_data.copy()
            user_info.pop("password_hash")
            
            return True, "User registered successfully", user_info
        
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return False, f"Registration error: {str(e)}", None
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Authenticate a user with username and password
        
        Args:
            username: Username for authentication
            password: Password for authentication
            
        Returns:
            (success, message, token_data) tuple
        """
        try:
            # Retrieve user data
            if not self.db_manager:
                return False, "Database manager not available", None
            
            user_data = self.db_manager.get_preference(username, "auth_data")
            if not user_data:
                return False, "Invalid username or password", None
            
            # Check password
            stored_hash = user_data.get("password_hash", "").encode('utf-8')
            if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                return False, "Invalid username or password", None
            
            # Update last login
            user_data["last_login"] = datetime.now().isoformat()
            self.db_manager.store_preference(username, "auth_data", user_data)
            
            # Generate JWT token
            token_data = self.generate_token(user_data["user_id"], username)
            
            # Return user info with token
            user_info = user_data.copy()
            user_info.pop("password_hash")
            user_info.update(token_data)
            
            return True, "Authentication successful", user_info
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, f"Authentication error: {str(e)}", None
    
    def generate_token(self, user_id: str, username: str) -> Dict[str, Any]:
        """
        Generate a JWT token for the user
        
        Args:
            user_id: User ID to encode in the token
            username: Username to encode in the token
            
        Returns:
            Dictionary with token and expiry information
        """
        exp = datetime.utcnow() + timedelta(days=self.token_expiry)
        payload = {
            "sub": user_id,
            "username": username,
            "exp": exp,
            "iat": datetime.utcnow()
        }
        
        # Use PyJWT to encode the token
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        # If token is bytes (PyJWT < 2.0.0), decode to string
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        
        return {
            "token": token,
            "expires": exp.isoformat()
        }
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify a JWT token and extract the payload
        
        Args:
            token: JWT token to verify
            
        Returns:
            (is_valid, payload) tuple
        """
        try:
            # Try PyJWT method first
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                return True, payload
            except AttributeError:
                # Fallback for python-jwt
                import python_jwt as pyjwt
                header, payload = pyjwt.verify_jwt(token, self.jwt_secret, ['HS256'])
                return True, payload
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return False, {"error": "Invalid token"}
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False, {"error": f"Verification error: {str(e)}"}
    
    def change_password(self, username: str, current_password: str, new_password: str) -> Tuple[bool, str]:
        """
        Change a user's password
        
        Args:
            username: Username for authentication
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            (success, message) tuple
        """
        try:
            # First authenticate with current password
            auth_success, _, _ = self.authenticate(username, current_password)
            if not auth_success:
                return False, "Current password is incorrect"
            
            # Get user data
            user_data = self.db_manager.get_preference(username, "auth_data")
            if not user_data:
                return False, "User not found"
            
            # Hash the new password
            new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Update user data
            user_data["password_hash"] = new_hash
            user_data["password_changed_at"] = datetime.now().isoformat()
            
            # Save updated user data
            success = self.db_manager.store_preference(username, "auth_data", user_data)
            if not success:
                return False, "Failed to update password"
            
            return True, "Password changed successfully"
        
        except Exception as e:
            logger.error(f"Password change error: {e}")
            return False, f"Password change error: {str(e)}"
    
    def delete_account(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Delete a user account completely (GDPR compliance)
        
        Args:
            username: Username for authentication
            password: Password for verification
            
        Returns:
            (success, message) tuple
        """
        try:
            # First authenticate
            auth_success, _, _ = self.authenticate(username, password)
            if not auth_success:
                return False, "Authentication failed"
            
            # Get user data
            user_data = self.db_manager.get_preference(username, "auth_data")
            if not user_data:
                return False, "User not found"
            
            user_id = user_data.get("user_id")
            
            # Delete all user data
            if self.db_manager:
                # Delete authentication data
                self.db_manager.delete_preference(username, "auth_data")
                
                # Delete all user content data
                if user_id:
                    self.db_manager.delete_user_data(user_id)
            
            return True, "Account deleted successfully"
        
        except Exception as e:
            logger.error(f"Account deletion error: {e}")
            return False, f"Account deletion error: {str(e)}"
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information by user ID
        
        Args:
            user_id: User ID to look up
            
        Returns:
            User information dictionary or None if not found
        """
        try:
            if not self.db_manager:
                return None
            
            # We need to search through users to find by user_id
            # This is inefficient but works for our current structure
            # In a production system, this would use a proper user table
            
            # For now, let's get user data from preferences
            # This assumes we store a mapping of user_id to username
            username = self.db_manager.get_preference(user_id, "username_mapping")
            if not username:
                return None
            
            user_data = self.db_manager.get_preference(username, "auth_data")
            if not user_data:
                return None
            
            # Don't return the password hash
            user_info = user_data.copy()
            user_info.pop("password_hash", None)
            
            return user_info
            
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None