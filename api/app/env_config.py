import os
from typing import Optional

class EnvConfig:
    """Environment configuration for the API"""
    
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv('SUPABASE_URL', '')
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')
    
    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8080'))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # CORS Configuration
    ALLOWED_ORIGINS: str = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000')
    
    # Model and Data Configuration
    DEFAULT_MODEL_NAME: str = os.getenv('DEFAULT_MODEL_NAME', 'model_nfp.pth')
    DEFAULT_DATA_SPLIT: str = os.getenv('DEFAULT_DATA_SPLIT', 'test')
    
    @classmethod
    def validate_supabase_config(cls) -> bool:
        """Validate that Supabase configuration is complete"""
        return bool(cls.SUPABASE_URL and cls.SUPABASE_SERVICE_ROLE_KEY)
    
    @classmethod
    def get_allowed_origins_list(cls) -> list:
        """Get list of allowed origins for CORS"""
        return [origin.strip() for origin in cls.ALLOWED_ORIGINS.split(',') if origin.strip()]
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)"""
        print("=== Environment Configuration ===")
        print(f"SUPABASE_URL: {'Set' if cls.SUPABASE_URL else 'Not set'}")
        print(f"SUPABASE_SERVICE_ROLE_KEY: {'Set' if cls.SUPABASE_SERVICE_ROLE_KEY else 'Not set'}")
        print(f"API_HOST: {cls.API_HOST}")
        print(f"API_PORT: {cls.API_PORT}")
        print(f"DEBUG: {cls.DEBUG}")
        print(f"ALLOWED_ORIGINS: {cls.ALLOWED_ORIGINS}")
        print(f"DEFAULT_MODEL_NAME: {cls.DEFAULT_MODEL_NAME}")
        print(f"DEFAULT_DATA_SPLIT: {cls.DEFAULT_DATA_SPLIT}")
        print("=================================") 