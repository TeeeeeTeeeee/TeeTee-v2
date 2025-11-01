require('dotenv').config();
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const OpenAI = require('openai');
const crypto = require('crypto');

// Phala TEE SDK
let TappdClient;
try {
  const pkg = require('@phala/dstack-sdk');
  TappdClient = pkg.TappdClient;
} catch (err) {
  console.warn('⚠️  Phala dstack-sdk not available. Attestation endpoints will return mock data.');
  console.warn('   This is normal for local development. In production TEE, attestation will work.');
}

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Initialize OpenAI client with Phala Cloud API
const client = new OpenAI({
  apiKey: process.env.PHALA_API_KEY,
  baseURL: "https://api.redpill.ai/v1"
});

// Available models (from Phala Cloud dashboard)
const MODELS = {
  'deepseek-v3': 'deepseek/deepseek-chat-v3-0324',
  'gpt-oss-120b': 'openai/gpt-oss-120b',
  'gpt-oss-20b': 'openai/gpt-oss-20b',
  'gemma-3-27b': 'google/gemma-3-27b-it',
  'qwen-2.5-7b': 'qwen/qwen-2.5-7b-instruct',
  'qwen-2.5-vl-72b': 'qwen/qwen2.5-vl-72b-instruct',
  'qwen3-vl-235b': 'qwen/qwen3-vl-235b-a22b-instruct'
};

// Health check endpoint
app.get('/health', (req, res) => {
  const apiKeyConfigured = !!process.env.PHALA_API_KEY;
  const isTEE = !!TappdClient;
  
  res.json({ 
    status: 'ok', 
    message: 'LLM Server is running',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    config: {
      port: process.env.PORT || 3001,
      apiKeyConfigured: apiKeyConfigured,
      baseUrl: 'https://api.redpill.ai/v1',
      modelsAvailable: Object.keys(MODELS).length,
      teeEnabled: isTEE,
      attestationAvailable: isTEE
    },
    environment: process.env.NODE_ENV || 'production',
    endpoints: {
      llm: ['/chat', '/chat/stream', '/inference'],
      attestation: ['/attest', '/attest/quick', '/tee/info'],
      utility: ['/health', '/models']
    }
  });
});

// List available models
app.get('/models', (req, res) => {
  res.json({ 
    models: MODELS,
    count: Object.keys(MODELS).length
  });
});

// Chat completion endpoint (non-streaming)
app.post('/chat', async (req, res) => {
  try {
    const { 
      messages, 
      model = 'deepseek-v3',
      temperature = 0.7,
      max_tokens = 2000
    } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ 
        error: 'Messages array is required' 
      });
    }

    const modelName = MODELS[model] || MODELS['deepseek-v3'];

    const response = await client.chat.completions.create({
      model: modelName,
      messages: messages,
      temperature: temperature,
      max_tokens: max_tokens,
      stream: false
    });

    res.json({
      success: true,
      model: modelName,
      response: response.choices[0].message.content,
      usage: response.usage,
      id: response.id,
      created: response.created
    });

  } catch (error) {
    console.error('Error in chat endpoint:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'Internal server error'
    });
  }
});

// Chat completion endpoint (streaming)
app.post('/chat/stream', async (req, res) => {
  try {
    const { 
      messages, 
      model = 'deepseek-v3',
      temperature = 0.7,
      max_tokens = 2000
    } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ 
        error: 'Messages array is required' 
      });
    }

    const modelName = MODELS[model] || MODELS['deepseek-v3'];

    // Set headers for SSE (Server-Sent Events)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const stream = await client.chat.completions.create({
      model: modelName,
      messages: messages,
      temperature: temperature,
      max_tokens: max_tokens,
      stream: true
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      if (content) {
        res.write(`data: ${JSON.stringify({ content })}\n\n`);
      }
    }

    res.write('data: [DONE]\n\n');
    res.end();

  } catch (error) {
    console.error('Error in streaming chat endpoint:', error);
    res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
    res.end();
  }
});

// Simple inference endpoint
app.post('/inference', async (req, res) => {
  try {
    const { 
      prompt, 
      system = "You are a helpful assistant",
      model = 'deepseek-v3',
      temperature = 0.7,
      max_tokens = 2000
    } = req.body;

    if (!prompt) {
      return res.status(400).json({ 
        error: 'Prompt is required' 
      });
    }

    const modelName = MODELS[model] || MODELS['deepseek-v3'];

    const messages = [
      { role: "system", content: system },
      { role: "user", content: prompt }
    ];

    const response = await client.chat.completions.create({
      model: modelName,
      messages: messages,
      temperature: temperature,
      max_tokens: max_tokens,
      stream: false
    });

    res.json({
      success: true,
      model: modelName,
      prompt: prompt,
      response: response.choices[0].message.content,
      usage: response.usage
    });

  } catch (error) {
    console.error('Error in inference endpoint:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'Internal server error'
    });
  }
});

// ============================================================================
// PHALA TEE ATTESTATION ENDPOINTS
// ============================================================================

// Get TEE base image information
app.get('/tee/info', async (req, res) => {
  if (!TappdClient) {
    return res.json({
      success: false,
      error: 'TEE not available',
      note: 'Running in local development mode. Deploy to Phala TEE for attestation.',
      isMockMode: true
    });
  }

  try {
    const client = new TappdClient();
    const info = await client.info();
    res.json({
      success: true,
      info: info,
      note: 'This server is running inside Phala TEE'
    });
  } catch (error) {
    console.error('Error getting TEE info:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Generate attestation report with custom data (POST)
app.post('/attest', async (req, res) => {
  if (!TappdClient) {
    // Mock response for local development
    return res.json({
      success: true,
      isMockMode: true,
      note: 'Running in local development mode. This is mock attestation data.',
      userData: req.body.userData || 'no-data-provided',
      mockQuote: 'MOCK_QUOTE_DATA_' + Date.now(),
      mockRtmrs: {
        rtmr0: 'mock_rtmr0_hash',
        rtmr1: 'mock_rtmr1_hash',
        rtmr2: 'mock_rtmr2_hash',
        rtmr3: 'mock_rtmr3_hash'
      }
    });
  }

  try {
    const { userData } = req.body;
    
    if (!userData) {
      return res.status(400).json({
        success: false,
        error: 'userData is required in request body'
      });
    }

    const client = new TappdClient();
    
    // Hash the user data
    const hash = crypto.createHash('sha256').update(userData).digest();
    
    // Get TDX quote
    const quoteResult = await client.tdxQuote(hash.slice(0, 32));
    
    // Replay RTMRs
    const rtmrs = quoteResult.replayRtmrs();
    
    res.json({
      success: true,
      userData: userData,
      quote: quoteResult.quote,
      eventLog: quoteResult.event_log,
      rtmrs: rtmrs,
      note: 'Attestation generated inside Phala TEE'
    });
  } catch (error) {
    console.error('Error generating attestation:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Quick attestation endpoint (GET) with default data
app.get('/attest/quick', async (req, res) => {
  if (!TappdClient) {
    // Mock response for local development
    return res.json({
      success: true,
      isMockMode: true,
      note: 'Running in local development mode. This is mock attestation data.',
      userData: `attestation-${Date.now()}`,
      mockQuote: 'MOCK_QUOTE_DATA_' + Date.now(),
      mockRtmrs: {
        rtmr0: 'mock_rtmr0_hash',
        rtmr1: 'mock_rtmr1_hash',
        rtmr2: 'mock_rtmr2_hash',
        rtmr3: 'mock_rtmr3_hash'
      },
      timestamp: new Date().toISOString()
    });
  }

  try {
    const client = new TappdClient();
    
    // Use current timestamp + LLM server identifier
    const userData = `llm-attestation-${Date.now()}`;
    const hash = crypto.createHash('sha256').update(userData).digest();
    
    // Get TDX quote
    const quoteResult = await client.tdxQuote(hash.slice(0, 32));
    
    // Replay RTMRs
    const rtmrs = quoteResult.replayRtmrs();
    
    res.json({
      success: true,
      userData: userData,
      quote: quoteResult.quote,
      eventLog: quoteResult.event_log,
      rtmrs: rtmrs,
      timestamp: new Date().toISOString(),
      note: 'Attestation generated inside Phala TEE'
    });
  } catch (error) {
    console.error('Error generating attestation:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// LLM inference with attestation (combines chat + attestation proof)
app.post('/inference/attested', async (req, res) => {
  try {
    const { 
      prompt, 
      system = "You are a helpful assistant",
      model = 'deepseek-v3',
      temperature = 0.7,
      max_tokens = 2000,
      includeAttestation = true
    } = req.body;

    if (!prompt) {
      return res.status(400).json({ 
        error: 'Prompt is required' 
      });
    }

    const modelName = MODELS[model] || MODELS['deepseek-v3'];

    const messages = [
      { role: "system", content: system },
      { role: "user", content: prompt }
    ];

    // Get LLM response
    const response = await client.chat.completions.create({
      model: modelName,
      messages: messages,
      temperature: temperature,
      max_tokens: max_tokens,
      stream: false
    });

    const llmResponse = response.choices[0].message.content;

    // Generate attestation for the response
    let attestation = null;
    if (includeAttestation) {
      if (TappdClient) {
        try {
          const teeClient = new TappdClient();
          const attestationData = `${prompt}::${llmResponse}::${Date.now()}`;
          const hash = crypto.createHash('sha256').update(attestationData).digest();
          const quoteResult = await teeClient.tdxQuote(hash.slice(0, 32));
          
          attestation = {
            quote: quoteResult.quote,
            rtmrs: quoteResult.replayRtmrs(),
            attestationData: attestationData,
            timestamp: new Date().toISOString()
          };
        } catch (attErr) {
          console.error('Attestation generation failed:', attErr);
          attestation = { error: 'Attestation failed', isMockMode: true };
        }
      } else {
        attestation = {
          isMockMode: true,
          note: 'Running in development mode',
          mockQuote: 'MOCK_QUOTE_' + Date.now()
        };
      }
    }

    res.json({
      success: true,
      model: modelName,
      prompt: prompt,
      response: llmResponse,
      usage: response.usage,
      attestation: attestation,
      verified: !!attestation && !attestation.isMockMode
    });

  } catch (error) {
    console.error('Error in attested inference endpoint:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data || 'Internal server error'
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    message: err.message 
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 LLM Server running on port ${PORT}`);
  console.log(`📡 Using Phala Cloud Confidential AI API`);
  console.log(`🔗 Base URL: https://api.redpill.ai/v1`);
  console.log(`\n🔒 TEE Attestation: ${TappdClient ? '✅ ENABLED' : '⚠️  MOCK MODE (local dev)'}`);
  console.log(`\nAvailable endpoints:`);
  console.log(`\n  LLM Endpoints:`);
  console.log(`    GET  /health - Health check`);
  console.log(`    GET  /models - List available models`);
  console.log(`    POST /chat - Chat completion (non-streaming)`);
  console.log(`    POST /chat/stream - Chat completion (streaming)`);
  console.log(`    POST /inference - Simple inference`);
  console.log(`    POST /inference/attested - Inference with TEE attestation proof`);
  console.log(`\n  TEE Attestation Endpoints:`);
  console.log(`    GET  /tee/info - Get TEE base image information`);
  console.log(`    GET  /attest/quick - Quick attestation with default data`);
  console.log(`    POST /attest - Generate attestation with custom data`);
  console.log(`\n⚠️  Make sure to set PHALA_API_KEY in your .env file\n`);
});

module.exports = app;

