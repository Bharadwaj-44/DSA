import requests

import json

from typing import List, Dict, Iterator, Optional

import time
 
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
 
class SnowflakeCortexClient:

    """

    Client for Snowflake Cortex API following company-specific format

    """

    def __init__(self, config_or_api_key, base_url: str = None, model: str = None):

        """

        Initialize Snowflake Cortex client

        Args:

            config_or_api_key: Either a config dict/object OR api_key string

            base_url: Base URL for the Snowflake Cortex endpoint (optional if config provided)

            model: Model name to use (optional, defaults from config or snowflake-llama-3.3-70b)

        """

        # Handle both config object and individual parameters

        if isinstance(config_or_api_key, dict):

            # Config dict provided

            config = config_or_api_key

            # Check if snowflake section exists

            if 'snowflake' in config:

                snowflake_config = config['snowflake']

                self.api_key = snowflake_config.get('api_key')

                self.base_url = snowflake_config.get('base_url', '').rstrip('/')

                self.model = snowflake_config.get('model', 'snowflake-llama-3.3-70b')

                self.app_id = snowflake_config.get('app_id', 'edadip')

                self.aplctn_cd = snowflake_config.get('aplctn_cd', 'edagnai')

            else:

                # Fallback to root level

                self.api_key = config.get('api_key')

                self.base_url = config.get('base_url', '').rstrip('/')

                self.model = config.get('model', 'llama3.1-70b')

                self.app_id = config.get('app_id', 'edadip')

                self.aplctn_cd = config.get('aplctn_cd', 'edagnai')

        elif hasattr(config_or_api_key, 'api_key') or hasattr(config_or_api_key, 'snowflake'):

            # Config object provided

            config = config_or_api_key

            # Check if snowflake attribute exists

            if hasattr(config, 'snowflake'):

                snowflake_config = config.snowflake

                self.api_key = getattr(snowflake_config, 'api_key', None)

                self.base_url = getattr(snowflake_config, 'base_url', '').rstrip('/')

                self.model = getattr(snowflake_config, 'model', 'llama3.1-70b')

                self.app_id = getattr(snowflake_config, 'app_id', 'edadip')

                self.aplctn_cd = getattr(snowflake_config, 'aplctn_cd', 'edagnai')

            else:

                # Fallback to root level attributes

                self.api_key = getattr(config, 'api_key', None)

                self.base_url = getattr(config, 'base_url', '').rstrip('/')

                self.model = getattr(config, 'model', 'llama3.1-70b')

                self.app_id = getattr(config, 'app_id', 'edadip')

                self.aplctn_cd = getattr(config, 'aplctn_cd', 'edagnai')

        else:

            # Individual parameters provided

            self.api_key = config_or_api_key

            self.base_url = base_url.rstrip('/') if base_url else ''

            self.model = model if model else 'llama3.1-70b'

            self.app_id = 'edadip'

            self.aplctn_cd = 'edagnai'

        # Debug output

        print(f"DEBUG INIT: api_key={'[SET]' if self.api_key else '[EMPTY]'}")

        print(f"DEBUG INIT: base_url={self.base_url if self.base_url else '[EMPTY]'}")

        print(f"DEBUG INIT: model={self.model}")

        self.chat = self.ChatCompletion(self)
 

    def _build_payload(self, messages: List[Dict[str, str]], system_message: str = None) -> Dict:
        """Build request payload for Snowflake Cortex in company-specific format"""
        # Extract system message
        sys_msg = system_message
        filtered_messages = []
        # ✅ DEBUG: Check what we received
        print(f"DEBUG _build_payload: Received system_message = {system_message is not None}")
        print(f"DEBUG _build_payload: Number of messages = {len(messages)}")
        for msg in messages:
            if msg.get('role') == 'system':
                if not sys_msg:
                    sys_msg = msg['content']
                    print(f"DEBUG: Extracted sys_msg from messages, length = {len(sys_msg)}")
            else:
                # Keep user and assistant messages
                filtered_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        # Default system message
        if not sys_msg:
            sys_msg = "You are a helpful AI assistant for data analysis and Python programming."
            print("DEBUG: Using default system message")
        # ✅ DEBUG: Check sys_msg before payload
        print(f"DEBUG: sys_msg is None? {sys_msg is None}")
        print(f"DEBUG: sys_msg length: {len(sys_msg) if sys_msg else 0}")
        print(f"DEBUG: sys_msg preview: {sys_msg[:100] if sys_msg else 'NONE!'}")
        # Keep last 10 messages for better context
        if len(filtered_messages) > 10:
            print(f"DEBUG: Trimming conversation from {len(filtered_messages)} to 10 messages")
            filtered_messages = filtered_messages[-10:]
        # ✅ CRITICAL: Include system message in BOTH places!
        all_messages = [
            {
                "role": "system",
                "content": sys_msg
            }
        ] + filtered_messages
        # Build payload
        payload = {
            "query": {
                "aplctn_cd": self.aplctn_cd,
                "app_id": self.app_id,
                "api_key": self.api_key,
                "method": "cortex",
                "model":{
                    "name":self.model,
                    "provider":"anthropic"
                },
                "sys_msg": sys_msg,  # ✅ Keep this field for API backend!
                "limit_convs": 0,
                "prompt": {
                    "messages": all_messages  # ✅ AND include in messages array for LLM!
                },
                "app_lvl_prefix": "edadip",
                "user_id": "",
                "session_id": "dsa_session"
            }
        }
        # ✅ DEBUG: Verify payload
        print(f"DEBUG: Payload sys_msg is None? {payload['query']['sys_msg'] is None}")
        return payload
 
    def _make_request(self, payload: Dict) -> requests.Response:
            
        """Make HTTP request to Snowflake Cortex API"""
        headers = {
        
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "api-key": self.api_key,  # ✅ ADD THIS - API key in header!
            "Authorization": f'Snowflake Token="{self.api_key}"'
        }
        print(f"DEBUG: Making request to {self.base_url}")
        print(f"DEBUG: Headers: {list(headers.keys())}")
        print(f"DEBUG: Payload: {json.dumps(payload, indent=2)}")
        response = requests.post(
        
            self.base_url,
            headers=headers,
            json=payload,
            verify=False,
            timeout=120
        )
        return response
 
 
    class ChatCompletion:

        """Chat completion interface for Snowflake Cortex"""

        def __init__(self, client):

            self.client = client

            self.completions = self  # Support OpenAI-style API: client.chat.completions.create()

        def create(self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs):

            """

            Create chat completion using Snowflake Cortex

            Args:

                model: Model name (will use client's configured model)

                messages: List of messages

                stream: Whether to stream the response

                **kwargs: Additional parameters

            Returns:

                CompletionResponse object or Iterator[StreamingChunk] for streaming

            """

            # Extract system message if present

           

            # Build payload

            payload = self.client._build_payload(messages, None)

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

                    raise Exception(f" Error Response: {json.dumps(error_data, indent=2)}")

                except json.JSONDecodeError:

                    raise Exception(f" Error Response: {response.text}")

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

                # Small delay to simulate streaming (optional, can remove for faster response)

                # time.sleep(0.01)
 
def create_snowflake_client(api_key: str, base_url: str, model: str = "snowflake-llama-3.3-70b"):

    """

    Factory function to create Snowflake Cortex client

    Args:

        api_key: API key for authentication

        base_url: Base URL for the Snowflake Cortex endpoint

        model: Model name to use

    Returns:

        SnowflakeCortexClient instance

    """

    return SnowflakeCortexClient(api_key, base_url, model)
 
