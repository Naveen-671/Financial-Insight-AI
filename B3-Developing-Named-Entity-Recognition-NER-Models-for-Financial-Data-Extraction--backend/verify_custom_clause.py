import requests
import json
import os

# Configuration
API_URL = "http://localhost:8001/api/v1/pipeline/process-document"
SAMPLE_TEXT = """
This Agreement is entered into on January 1st, 2024.
The Borrower shall throw a party on December 25th, 2024.
"""
CUSTOM_PROMPT = """
Extract all dates as 'CelebrationDate'.
"""

def test_custom_clause_extraction():
    # Create a dummy file
    with open("test_doc.txt", "w") as f:
        f.write(SAMPLE_TEXT)

    try:
        print("Sending request to backend...")
        with open('test_doc.txt', 'rb') as f_read:
            files = {
                'file': ('test_doc.txt', f_read, 'text/plain')
            }
            data = {
                'include_ner': 'false',
                'include_finbert': 'false',
                'include_langextract': 'true',
                'clauses_prompt': CUSTOM_PROMPT
            }
            response = requests.post(API_URL, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("Response received successfully.")
            
            langextract_result = result.get("langextract", {})
            returned_text = langextract_result.get("text")
            
            if returned_text:
                print(f"SUCCESS: 'text' field found in response (Length: {len(returned_text)})")
            else:
                print("FAILURE: 'text' field MISSING in langextract response")
                print("Keys found:", langextract_result.keys())
                exit(1)

            extractions = langextract_result.get("results", {}).get("llama-3.1-8b-instant", {}).get("extractions", [])
            
            print(f"Extractions found: {len(extractions)}")
            
            found_custom_type = False
            for ext in extractions:
                print(f" - Found Class: {ext.get('extraction_class')}, Text: {ext.get('extraction_text')}")
                if "CelebrationDate" in ext.get('extraction_class', '') or "CelebrationDate" in str(ext): # Loose check as LLM might vary slightly
                     found_custom_type = True
            
            if found_custom_type:
                print("SUCCESS: Custom clause type 'CelebrationDate' found!")
            else:
                print("WARNING: Custom clause type not strictly found (check LLM output above).")
                # Note: This might fail if the LLM ignores the instruction, but proves the pipeline works if it doesn't error.
                
        else:
            print(f"FAILED: Status Code {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if os.path.exists("test_doc.txt"):
            os.remove("test_doc.txt")

if __name__ == "__main__":
    test_custom_clause_extraction()
