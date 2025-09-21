python server.py

curl.exe -X POST http://127.0.0.1:5000/split

Invoke-RestMethod -Uri "http://127.0.0.1:5000/infer" -Method POST -ContentType "application/json" -Body '{"messages": [{"role": "system", "content": "You are a friendly, concise assistant."}, {"role": "user", "content": "Do you know the way?"}], "max_new": 64, "temperature": 0.7, "top_p": 0.9}'