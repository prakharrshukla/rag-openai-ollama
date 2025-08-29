#!/usr/bin/env python3
"""
Simple test script for the Medical RAG Q&A System
"""
import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from pypdf import PdfReader
        import requests
        print("✅ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_pdf():
    """Test if PDF file exists and can be read"""
    print("\nTesting PDF file...")
    pdf_path = Path("Good-Medical-Practice-2024---English-102607294.pdf")
    if not pdf_path.exists():
        print("❌ PDF file not found")
        return False
    
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        if len(reader.pages) > 0:
            print(f"✅ PDF loaded successfully ({len(reader.pages)} pages)")
            return True
        else:
            print("❌ PDF has no pages")
            return False
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False

def test_ollama():
    """Test Ollama connection"""
    print("\nTesting Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print(f"✅ Ollama running with {len(models)} models")
                for model in models[:3]:  # Show first 3 models
                    print(f"   - {model['name']}")
                return True
            else:
                print("⚠️  Ollama running but no models found")
                print("   Run: ollama pull llama3.2:1b")
                return False
        else:
            print("❌ Ollama not responding properly")
            return False
    except Exception as e:
        print("⚠️  Ollama not running or not accessible")
        print("   Install Ollama and run: ollama serve")
        return False

def main():
    """Run all tests"""
    print("🔍 RAG OpenAI Ollama - Medical Q&A System - Test Script")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_pdf():
        tests_passed += 1
    
    if test_ollama():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready to run the application.")
        print("\nTo start:")
        print("   Streamlit: streamlit run streamlit_app.py")
        print("   FastAPI:   uvicorn app:app --reload")
    elif tests_passed >= 2:
        print("⚠️  Most tests passed. Check warnings above.")
    else:
        print("❌ Some tests failed. Please fix issues before running.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
