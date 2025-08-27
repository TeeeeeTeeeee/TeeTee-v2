# TeeTee-v2

Perfect — let’s put all the pieces together so you can actually run the whole split-model system end-to-end 🚀.

You’ve got **two Flask servers** (Node1 and Node2). Think of them like two little apps that need to run at the same time.

---

## Step 1. Start your virtual environment

In PowerShell:

```powershell
& c:/Users/hankaixin/Desktop/TeeTee-v2/.venv/Scripts/Activate.ps1
```

---

## Step 2. Run **Node2** (the back half) first

Node2 must be ready to accept halfway activations.

```powershell
$env:DTYPE="fp32"   # use fp16 only if you have a CUDA GPU
python apt.py
```

* Node2 runs on **port 5001**.
* You should see logs like:

  ```
  Node2 initialized with layers 11 to 21
  Starting Node2 on port 5001 ...
  ```

Check it’s alive:

```powershell
curl http://localhost:5001/health
```

---

## Step 3. Run **Node1** (the front half)

Open a **second PowerShell terminal**, activate the venv again, then:

```powershell
$env:NODE2_URL="http://127.0.0.1:5001"
python app.py
```

* Node1 runs on **port 5002**.
* You should see logs like:

  ```
  Model has 22 total layers
  Node1 will use layers 0 to 10
  Starting node1 server on port 5002 ...
  ```

Check it’s alive:

```powershell
curl http://localhost:5002/health
```

---

## Step 4. Send a prompt to Node1

Now you can actually query the split model.

**Option A — Single-step (default `/generate`)**

```powershell
$body = @{ prompt = "Hello, how are you today?" } | ConvertTo-Json
Invoke-WebRequest -Uri "http://localhost:5002/generate" -Method POST -Body $body -ContentType "application/json"
```

You’ll get JSON with only **one predicted token**.

---

**Option B — Multi-step (if you added the `/chat` loop)**

```powershell
$body = @{ prompt = "Hello, how are you today?"; max_new_tokens = 40 } | ConvertTo-Json
Invoke-WebRequest -Uri "http://localhost:5002/chat" -Method POST -Body $body -ContentType "application/json"
```

This will loop between Node1 and Node2, and you’ll see a **full sentence** in `"generated_text"`.

---

## Step 5. Explore

* `http://localhost:5002/verify` → Node1’s hash + info
* `http://localhost:5001/verify` → Node2’s hash + info
* `http://localhost:5002/node1_ra_report` → mock attestation proof

---

✅ That’s the workflow:

1. Start Node2.
2. Start Node1 (pointing it at Node2).
3. Query Node1’s routes (`/generate` for one token, `/chat` for sentences).

---

Do you want me to also give you a **diagram showing the flow** (Client → Node1 → Node2 → back to Client) so you can visualize what’s happening when you run those steps?
