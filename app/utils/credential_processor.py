"""
Credential processing utilities for handling vault-stored credentials.
This module handles decoding of Base64-encoded sensitive fields from vault storage.
"""

import base64
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class CredentialProcessor:
    """Handles processing of credentials retrieved from vault storage."""
    
    # Complete list of sensitive fields that are Base64 encoded in vault
    SENSITIVE_FIELDS = {
        'password', 'secret', 'key', 'token', 'auth', 'credential', 'cert', 'certificate',
        'private', 'public', 'passphrase', 'pin', 'code', 'hash', 'signature', 'access',
        'refresh', 'bearer', 'jwt', 'oauth', 'api', 'client', 'session', 'security',
        'encryption', 'decryption', 'ssh', 'ssl', 'tls', 'pkcs', 'pem', 'der', 'x509',
        'hmac', 'rsa', 'aes'
    }
    
    @classmethod
    def _normalize_field_name(cls, field_name: str) -> str:
        """Normalize field name for sensitive field detection."""
        if not field_name:
            return ""
        # Remove non-alphabetic characters and convert to lowercase
        return re.sub(r'[^a-zA-Z]', '', field_name.lower())
    
    @classmethod
    def _is_sensitive_field(cls, field_name: str) -> bool:
        """Check if a field name is considered sensitive."""
        normalized = cls._normalize_field_name(field_name)
        return normalized in cls.SENSITIVE_FIELDS
    
    @classmethod
    def _is_json_string(cls, value: str) -> bool:
        """Check if a decoded string appears to be JSON."""
        if not value:
            return False
        
        value = value.strip()
        return (
            (value.startswith('{') and value.endswith('}')) or
            (value.startswith('[') and value.endswith(']'))
        )
    
    @classmethod
    def _decode_base64_value(cls, value: str) -> Optional[str]:
        """Safely decode a Base64 encoded value."""
        if not value:
            return None
        
        try:
            # Handle both standard and URL-safe Base64 encoding
            decoded_bytes = base64.b64decode(value)
            return decoded_bytes.decode('utf-8')
        except Exception as e:
            logger.debug(f"Failed to decode Base64 value: {e}")
            # Return original value if decoding fails (backward compatibility)
            return value
    
    @classmethod
    def _parse_json_value(cls, value: str) -> Any:
        """Safely parse a JSON string."""
        try:
            return json.loads(value)
        except Exception as e:
            logger.debug(f"Failed to parse JSON value: {e}")
            return value
    
    @classmethod
    def process_credentials_from_storage(cls, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process credentials retrieved from vault storage.
        Decodes Base64-encoded sensitive fields and parses JSON when appropriate.
        
        Args:
            credentials: Raw credentials dictionary from vault
            
        Returns:
            Processed credentials with decoded sensitive fields
        """
        if not credentials:
            return {}
        
        processed = {}
        
        for field_name, field_value in credentials.items():
            if field_value is None:
                processed[field_name] = None
                continue
            
            # Convert to string for processing
            str_value = str(field_value)
            
            # Check if this is a sensitive field that needs decoding
            if cls._is_sensitive_field(field_name):
                # Decode Base64
                decoded_value = cls._decode_base64_value(str_value)
                
                if decoded_value is not None:
                    # Check if decoded value is JSON and parse it
                    if cls._is_json_string(decoded_value):
                        processed[field_name] = cls._parse_json_value(decoded_value)
                    else:
                        processed[field_name] = decoded_value
                else:
                    processed[field_name] = str_value
            else:
                # Non-sensitive fields are stored as-is
                processed[field_name] = field_value
        
        return processed
    
    @classmethod
    def process_single_credential(cls, credential_value: Any) -> Any:
        """
        Process a single credential value that might be Base64 encoded.
        
        Args:
            credential_value: The credential value to process
            
        Returns:
            Processed credential value
        """
        if credential_value is None:
            return None
        
        # Convert to string for processing
        str_value = str(credential_value)
        
        # Try to decode as Base64
        decoded_value = cls._decode_base64_value(str_value)
        
        if decoded_value is not None and decoded_value != str_value:
            # Successfully decoded, check if it's JSON
            if cls._is_json_string(decoded_value):
                return cls._parse_json_value(decoded_value)
            else:
                return decoded_value
        
        # Return original value if no decoding was needed
        return credential_value
