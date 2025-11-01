# TeeTee LLM Server

Node.js server for running LLM inference using Phala Cloud's Confidential AI API with OpenAI-compatible interface.

## Features

- 🔐 Secure inference using Phala Cloud GPU TEE
- 🚀 OpenAI-compatible API interface
- 📡 Multiple model support (DeepSeek V3, Llama 3.3, GPT-OSS, Qwen)
- 🌊 Streaming and non-streaming responses
- ⚡ RESTful API endpoints
- 🔄 Easy integration with existing applications

## Setup

### 1. Install Dependencies

```bash
cd llm
npm install
```

### 2. Configure API Key

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your Phala Cloud API key:

```
PHALA_API_KEY=your_actual_api_key_here
PORT=3001
```

**Get your API key:**
1. Go to [Phala Dashboard](https://dashboard.phala.network)
2. Ensure you have at least $5 in your account
3. Navigate to Dashboard → Confidential AI API
4. Click Enable, then create your API key

### 3. Start the Server

```bash
npm start
```

For development with auto-reload:

```bash
npm run dev
```

## API Endpoints

### Health Check
```bash
GET /health
```

### List Available Models
```bash
GET /models
```

### Chat Completion (Non-streaming)
```bash
POST /chat
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is your model name?"}
  ],
  "model": "deepseek-v3",
  "temperature": 0.7,
  "max_tokens": 2000
}
```

### Chat Completion (Streaming)
```bash
POST /chat/stream
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Tell me a story"}
  ],
  "model": "deepseek-v3"
}
```

### Simple Inference
```bash
POST /inference
Content-Type: application/json

{
  "prompt": "What is artificial intelligence?",
  "system": "You are a helpful assistant",
  "model": "deepseek-v3"
}
```

## Available Models

| Model | Key | Context |
|-------|-----|---------|
| DeepSeek V3 0324 | `deepseek-v3` | 163K |
| Llama 3.3 70B | `llama-3.3-70b` | 131K |
| GPT OSS 120B | `gpt-oss-120b` | 131K |
| Qwen3 Coder | `qwen3-coder` | 262K |
| Qwen2.5 7B | `qwen-2.5-7b` | 32K |

## Example Usage

### cURL Example

```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "model": "deepseek-v3"
  }'
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:3001/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    messages: [
      { role: 'user', content: 'What is the meaning of life?' }
    ],
    model: 'deepseek-v3'
  })
});

const data = await response.json();
console.log(data.response);
```

### Python Example

```python
import requests

response = requests.post('http://localhost:3001/inference', json={
    'prompt': 'Explain quantum computing in simple terms',
    'model': 'deepseek-v3'
})

print(response.json()['response'])
```

## Security & Privacy

All inference requests run in GPU TEE (Trusted Execution Environment), ensuring:
- Hardware-level privacy protection
- Cryptographic proof of secure execution
- User data remains confidential during inference

Learn more about [verifying your AI responses](https://docs.phala.network/ai-agent-contract/getting-started/verify).

## Troubleshooting

### API Key Error
Make sure your `.env` file contains a valid `PHALA_API_KEY`.

### Port Already in Use
Change the `PORT` in your `.env` file to a different port.

### Module Not Found
Run `npm install` to ensure all dependencies are installed.

## License

MIT

