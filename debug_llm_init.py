#!/usr/bin/env python3
"""
Debug LLM client initialization
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_factory():
    """Test the LLM factory first"""
    print("🔍 Testing LLM Factory...")
    
    try:
        from app.llm.factory import LLMClientFactory
        
        # Test available providers
        providers = LLMClientFactory.get_available_providers()
        print(f"✅ Available providers: {providers}")
        
        # Test OpenAI client creation
        if 'openai' in providers:
            print("\n🧪 Testing OpenAI client creation...")
            openai_client = LLMClientFactory.create_client('openai')
            print(f"✅ OpenAI client created: {type(openai_client)}")
            
            # Test simple method call
            print("🧪 Testing OpenAI client method...")
            print(f"   Model: {openai_client.model}")
        
        # Test Ollama client creation
        if 'ollama' in providers:
            print("\n🧪 Testing Ollama client creation...")
            ollama_client = LLMClientFactory.create_client('ollama')
            print(f"✅ Ollama client created: {type(ollama_client)}")
            print(f"   Model: {ollama_client.model}")
            print(f"   Base URL: {ollama_client.base_url}")
            
    except Exception as e:
        print(f"❌ Error testing factory: {e}")
        import traceback
        traceback.print_exc()

def test_multimode_init():
    """Test MultiMode client initialization only"""
    print("\n🔍 Testing MultiMode Client Initialization...")
    
    try:
        from app.llm.multimode import MultiModeLLMClient
        
        print("🧪 Creating MultiModeLLMClient...")
        client = MultiModeLLMClient()
        print(f"✅ MultiModeLLMClient created")
        
        print(f"   Available clients: {list(client.clients.keys())}")
        print(f"   Task routing config: {client.task_routing_config}")
        
        # Test get_client_for_task
        print("\n🧪 Testing get_client_for_task...")
        test_client, test_model = client.get_client_for_task("general")
        print(f"✅ Got client for general task: {type(test_client)} with model {test_model}")
        
    except Exception as e:
        print(f"❌ Error testing MultiMode init: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_factory()
    test_multimode_init()
