"""
Test script to check if the final response generation is working
"""
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_response_generation():
    """Test the response generation fix"""
    try:
        from app.orchestrator.simple_orchestrator import SimpleOrchestrator
        
        print("Creating SimpleOrchestrator...")
        orchestrator = SimpleOrchestrator()
        
        print("Testing with a simple data query...")
        # Test with a simple query that should use available tables
        result = await orchestrator.execute_workflow(
            "Show me data from purchase orders", 
            session_id="test_session"
        )
        
        print(f"Result: {result}")
        
        if result and "greeting" in result:
            print("✅ SUCCESS: Final response was generated!")
            print(f"Response: {result['greeting'][:200]}...")
        else:
            print("❌ FAILED: No final response generated")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_response_generation())
