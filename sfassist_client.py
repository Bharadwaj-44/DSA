"""
SF Assist Client for Elevance Health
Adapted from snowflake_cortex_client.py to send messages like horizon_client.py

Key Changes:
- Sends messages as array (like Horizon API) instead of single combined string
- Maintains conversation history with proper role/content structure
- Simplified _build_payload method
"""

import requests
import json
from typing import List, Dict, Iterator, Optional
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
#                         RESPONSE CLASSES
# ============================================================================

class UsageStats:
    """Token usage statistics"""
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class Choice:
    """Response choice"""
    def __init__(self, message: Dict[str, str], finish_reason: str = "stop"):
        self.message = type('Message', (), message)()
        self.finish_reason = finish_reason
        self.delta = type('Delta', (), {'content': message.get('content', '')})()


class CompletionResponse:
    """Response from completion API"""
    def __init__(self, content: str, usage: UsageStats):
        self.choices = [Choice({"role": "assistant", "content": content})]
        self.usage = usage


class StreamingChunk:
    """Streaming chunk response"""
    def __init__(self, content: str):
        delta = type('Delta', (), {'content': content})()
        choice = type('Choice', (), {'delta': delta})()
        self.choices = [choice]


# ============================================================================
#                         SF ASSIST CLIENT
# ============================================================================

class SFAssistClient:
    """
    SF Assist Client - sends messages like Horizon API (array format)
    
    Key Difference from original snowflake_cortex_client:
    - Sends messages as array: [{"role": "system", ...}, {"role": "user", ...}, ...]
    - NOT as single combined string
    """
    
    def __init__(self, config_or_api_key, base_url: str = None, model: str = None):
        """
        Initialize SF Assist client
        
        Args:
            config_or_api_key: Either a config dict/object OR api_key string
            base_url: Base URL for the SF Assist endpoint (optional if config provided)
            model: Model name to use (optional, defaults from config)
        """
        # Handle both config object and individual parameters
        if isinstance(config_or_api_key, dict):
            # Config dict provided
            config = config_or_api_key
            
            # Check if sfassist section exists
            if 'sfassist' in config:
                sfassist_config = config['sfassist']
                self.api_key = sfassist_config.get('api_key')
                self.base_url = sfassist_config.get('base_url', '').rstrip('/')
                self.model = sfassist_config.get('model', 'claude-4-sonnet')
                self.app_id = sfassist_config.get('app_id', 'edadip')
                self.aplctn_cd = sfassist_config.get('aplctn_cd', 'edagnai')
            else:
                # Fallback to root level
                self.api_key = config.get('api_key')
                self.base_url = config.get('base_url', '').rstrip('/')
                self.model = config.get('model', 'claude-4-sonnet')
                self.app_id = config.get('app_id', 'edadip')
                self.aplctn_cd = config.get('aplctn_cd', 'edagnai')
                
        elif hasattr(config_or_api_key, 'api_key') or hasattr(config_or_api_key, 'sfassist'):
            # Config object provided
            config = config_or_api_key
            
            # Check if sfassist attribute exists
            if hasattr(config, 'sfassist'):
                sfassist_config = config.sfassist
                self.api_key = getattr(sfassist_config, 'api_key', None)
                self.base_url = getattr(sfassist_config, 'base_url', '').rstrip('/')
                self.model = getattr(sfassist_config, 'model', 'claude-4-sonnet')
                self.app_id = getattr(sfassist_config, 'app_id', 'edadip')
                self.aplctn_cd = getattr(sfassist_config, 'aplctn_cd', 'edagnai')
            else:
                # Fallback to root level attributes
                self.api_key = getattr(config, 'api_key', None)
                self.base_url = getattr(config, 'base_url', '').rstrip('/')
                self.model = getattr(config, 'model', 'claude-4-sonnet')
                self.app_id = getattr(config, 'app_id', 'edadip')
                self.aplctn_cd = getattr(config, 'aplctn_cd', 'edagnai')
        else:
            # Individual parameters provided
            self.api_key = config_or_api_key
            self.base_url = base_url.rstrip('/') if base_url else ''
            self.model = model if model else 'claude-4-sonnet'
            self.app_id = 'edadip'
            self.aplctn_cd = 'edagnai'
        
        # Debug output
        print(f"DEBUG INIT: api_key={'[SET]' if self.api_key else '[EMPTY]'}")
        print(f"DEBUG INIT: base_url={self.base_url if self.base_url else '[EMPTY]'}")
        print(f"DEBUG INIT: model={self.model}")
        
        self.chat = self.ChatCompletion(self)
    
    def _build_payload(self, messages: List[Dict[str, str]], system_message: str = None) -> Dict:
        """
        Build request payload - HORIZON STYLE (messages as array)
        
        This is the KEY CHANGE from the original snowflake client:
        - Sends messages as array: [{"role": "system", ...}, {"role": "user", ...}]
        - NOT as single combined string
        
        Args:
            messages: List of message dicts with role and content
            system_message: Optional system message to prepend
            
        Returns:
            Payload dict for SF Assist API
        """
        print(f"\n{'='*80}")
        print(f"🔍 BUILDING PAYLOAD - HORIZON STYLE (messages as array)")
        print(f"{'='*80}")
        
        # Extract system message
        sys_msg = system_message
        filtered_messages = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                if not sys_msg:
                    sys_msg = msg['content']
            else:
                filtered_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Default system message
        if not sys_msg:
            sys_msg = "You are a helpful AI assistant for data analysis and Python programming."
        
        # Build messages array with system first
        all_messages = [{"role": "system", "content": sys_msg}] + filtered_messages
        
        print(f"✅ Messages array: {len(all_messages)} messages")
        print(f"   • System message: {len(sys_msg)} chars")
        print(f"   • User/Assistant messages: {len(filtered_messages)}")
        
        # Build payload - SF Assist format with messages array
        payload = {
            "query": {
                "aplctn_cd": self.aplctn_cd,
                "app_id": self.app_id,
                "api_key": self.api_key,
                "method": "cortex",
                "model": self.model,
                "sys_msg": sys_msg,
                "limit_convs": "0",
                "prompt": {
                    "messages": all_messages  # 🔥 KEY CHANGE: Send as array!
                },
                "app_lvl_prefix": "edadip",
                "user_id": "",
                "session_id": "dsa_session"
            }
        }
        
        print(f"✅ Payload built with {len(all_messages)} messages in array format")
        print(f"{'='*80}\n")
        
        return payload
    
    def _make_request(self, payload: Dict) -> requests.Response:
        """Make HTTP request to SF Assist API"""
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "api-key": self.api_key,
            "Authorization": f'Snowflake Token="{self.api_key}"'
        }
        
        print(f"DEBUG: Making request to {self.base_url}")
        print(f"DEBUG: Headers: {list(headers.keys())}")
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            verify=False,
            timeout=120
        )
        
        return response
    
    class ChatCompletion:
        """Chat completion interface for SF Assist"""
        
        def __init__(self, client):
            self.client = client
            self.completions = self  # Support OpenAI-style API
        
        def create(self, model: str, messages: List[Dict[str, str]], 
                   stream: bool = False, **kwargs):
            """
            Create chat completion using SF Assist
            
            Args:
                model: Model name (will use client's configured model)
                messages: List of messages
                stream: Whether to stream the response
                **kwargs: Additional parameters
                
            Returns:
                CompletionResponse object or Iterator[StreamingChunk] for streaming
            """
            # Build payload (messages as array - like Horizon!)
            payload = self.client._build_payload(messages)
            
            # Make request
            response = self.client._make_request(payload)
            
            # Handle response
            if response.status_code == 200:
                try:
                    # Try parsing as JSON first
                    data = response.json()
                    
                    # Extract content from various possible response formats
                    content = None
                    if 'text' in data:
                        content = data['text']
                    elif 'response' in data:
                        content = data['response']
                    elif 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0].get('message', {}).get('content', '')
                    elif 'message' in data:
                        # Horizon-style response
                        content = data['message'].get('content', '')
                    else:
                        content = str(data)
                    
                    # Handle streaming vs non-streaming
                    if stream:
                        # Simulate streaming by yielding chunks
                        return self._simulate_streaming(content)
                    else:
                        # Create usage stats (estimate if not provided)
                        usage_data = data.get('usage', {})
                        usage = UsageStats(
                            prompt_tokens=usage_data.get('prompt_tokens', 0),
                            completion_tokens=usage_data.get('completion_tokens', 0),
                            total_tokens=usage_data.get('total_tokens', 0)
                        )
                        return CompletionResponse(content, usage)
                
                except (json.JSONDecodeError, ValueError) as e:
                    # If not JSON, treat as plain text
                    print(f"DEBUG: JSON decode error, treating as plain text: {e}")
                    content = response.text
                    
                    if stream:
                        return self._simulate_streaming(content)
                    else:
                        usage = UsageStats()
                        return CompletionResponse(content, usage)
            
            else:
                # Handle error responses
                try:
                    error_data = response.json()
                    raise Exception(f"API Error Response: {json.dumps(error_data, indent=2)}")
                except json.JSONDecodeError:
                    raise Exception(f"API Error Response: {response.text}")
        
        def _simulate_streaming(self, content: str):
            """
            Simulate streaming by yielding chunks of the response
            
            Args:
                content: Full response content to stream
                
            Yields:
                StreamingChunk objects
            """
            # Split content into words for more natural streaming
            words = content.split(' ')
            for i, word in enumerate(words):
                # Add space before word except for first word
                chunk_text = word if i == 0 else ' ' + word
                yield StreamingChunk(chunk_text)


def create_sfassist_client(api_key: str, base_url: str, model: str = "claude-4-sonnet"):
    """
    Factory function to create SF Assist client
    
    Args:
        api_key: API key for authentication
        base_url: Base URL for the SF Assist endpoint
        model: Model name to use
        
    Returns:
        SFAssistClient instance
    """
    return SFAssistClient(api_key, base_url, model)
