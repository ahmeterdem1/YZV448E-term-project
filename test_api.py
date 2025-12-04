#!/usr/bin/env python3
"""
API Test Script for PII Cleaner
Tests all endpoints and functionality
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health endpoint"""
    print("🏥 Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print("❌ Health check failed")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_process_text(text_content: str) -> Dict[str, Any]:
    """Test text processing endpoint"""
    print(f"📝 Testing text processing...")
    print(f"Input: {text_content}")
    
    try:
        payload = {"text_content": text_content}
        response = requests.post(
            f"{BASE_URL}/api/v1/process-text",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        
        if response.status_code == 201:
            print("✅ Text processing request successful")
            return result
        else:
            print("❌ Text processing failed")
            return {}
            
    except Exception as e:
        print(f"❌ Text processing error: {e}")
        return {}

def test_get_result(task_id: str) -> Dict[str, Any]:
    """Test result retrieval endpoint"""
    print(f"🔍 Testing result retrieval for task: {task_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/text/{task_id}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
            print("✅ Result retrieval successful")
            return result
        elif response.status_code == 404:
            print("⏳ Task not found (might still be processing)")
            return {}
        else:
            print("❌ Result retrieval failed")
            return {}
            
    except Exception as e:
        print(f"❌ Result retrieval error: {e}")
        return {}

def test_queue_status():
    """Test queue status endpoint"""
    print("📋 Testing queue status...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/queue")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Queue items: {len(result)}")
            for item in result:
                print(f"  - {item['id']}: {item['status']}")
            print("✅ Queue status check successful")
            return result
        else:
            print("❌ Queue status check failed")
            return []
            
    except Exception as e:
        print(f"❌ Queue status error: {e}")
        return []

def run_comprehensive_test():
    """Run comprehensive API test"""
    print("🚀 Starting comprehensive API test")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ Health check failed, stopping tests")
        return
    
    print("\n" + "=" * 50)
    
    # Test 2: Process multiple texts
    test_texts = [
        "Hello, my name is John Doe and my email is john.doe@example.com",
        "Contact me at +1-555-123-4567 or visit my website at https://johndoe.com",
        "My address is 123 Main Street, New York, NY 10001",
        "Username: johndoe123, ID: 987654321",
        "Call Sarah Johnson at sarah.j@company.org for more information"
    ]
    
    task_ids = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}/5:")
        result = test_process_text(text)
        if result and 'id' in result:
            task_ids.append(result['id'])
        time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 50)
    
    # Test 3: Check queue status
    test_queue_status()
    
    print("\n" + "=" * 50)
    
    
    print("\n🔍 Checking results:")
    for task_id in task_ids:
        print(f"\n--- Task: {task_id} ---")
        result = test_get_result(task_id)
        if result and 'text_content' in result:
            print(f"Processed text: {result['text_content']}")
    
    print("\n" + "=" * 50)
    
    # Test 5: Final queue status
    print("📋 Final queue status:")
    test_queue_status()
    
    print("\n✅ Comprehensive test completed!")

if __name__ == "__main__":
    run_comprehensive_test()