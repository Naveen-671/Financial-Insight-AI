import requests
import json

url = "http://localhost:8001/api/v1/pipeline/process-document"
files = {'file': open('test_document.txt', 'rb')}
data = {
    'include_ner': 'true',
    'include_finbert': 'true',
    'include_langextract': 'true'
}

try:
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Failed to connect: {e}")
