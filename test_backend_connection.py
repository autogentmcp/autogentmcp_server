#!/usr/bin/env python3
"""
Quick test to verify backend connectivity and endpoint availability
"""

import requests
import json
import time

def test_backend_connection():
    """Test basic backend connectivity"""
    base_url = "http://localhost:8001"
    
    print("🔍 Testing Backend Connection...")
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Health Status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"   Health Response: {health_response.json()}")
        
        # Test root endpoint
        print("\n2. Testing root endpoint...")
        root_response = requests.get(f"{base_url}/", timeout=5)
        print(f"   Root Status: {root_response.status_code}")
        if root_response.status_code == 200:
            root_data = root_response.json()
            print(f"   Available endpoints: {root_data.get('endpoints', {})}")
        
        # Test enhanced query endpoint (non-streaming)
        print("\n3. Testing enhanced/query endpoint...")
        test_payload = {
            "query": "What data agents are available?",
            "session_id": "test_session_123",
            "conversation_id": "test_conv_123",
            "conversation_history": [],
            "include_analysis": True,
            "max_steps": 2
        }
        
        query_response = requests.post(
            f"{base_url}/enhanced/query",
            json=test_payload,
            timeout=30
        )
        print(f"   Query Status: {query_response.status_code}")
        if query_response.status_code == 200:
            response_data = query_response.json()
            print(f"   Query Response Keys: {list(response_data.keys())}")
            print(f"   Status: {response_data.get('status', 'unknown')}")
            if 'error' in response_data:
                print(f"   Error: {response_data['error']}")
        else:
            print(f"   Error Response: {query_response.text}")
        
        # Test streaming endpoint
        print("\n4. Testing orchestration/enhanced/stream endpoint...")
        try:
            stream_response = requests.post(
                f"{base_url}/orchestration/enhanced/stream",
                json=test_payload,
                timeout=10,
                stream=True,
                headers={'Accept': 'text/event-stream'}
            )
            print(f"   Stream Status: {stream_response.status_code}")
            
            if stream_response.status_code == 200:
                print("   Stream started successfully!")
                # Read first few events
                events_count = 0
                for line in stream_response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        print(f"   Stream Event: {decoded_line[:100]}...")
                        events_count += 1
                        if events_count >= 3:  # Just read first 3 events
                            break
            else:
                print(f"   Stream Error: {stream_response.text}")
                
        except Exception as e:
            print(f"   Stream Error: {e}")
        
        print("\n✅ Backend connectivity test completed!")
        
    except requests.ConnectionError:
        print("❌ Cannot connect to backend - is it running on localhost:8001?")
    except Exception as e:
        print(f"❌ Error testing backend: {e}")

if __name__ == "__main__":
    test_backend_connection()
