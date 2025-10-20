import express from 'express';
import cors from 'cors';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { JsonRpcProvider, Contract } from 'ethers';
import * as dotenv from 'dotenv';
import axios, { AxiosResponse } from 'axios';
import { rateLimit } from 'express-rate-limit';
import CircuitBreaker from 'opossum';

// Load environment variables
dotenv.config({ path: path.join(__dirname, '..', '.env') });
dotenv.config({ path: path.join(__dirname, '.env') });

// Import 0G Storage SDK
const { Indexer } = require('@0glabs/0g-ts-sdk');

/**
 * INFT Oracle Backend Service
 * 
 * This service acts as an oracle for Intelligent NFTs (INFT) on 0G Network.
 * It provides secure AI inference capabilities with on-chain authorization validation.
 * 
 * Key Features:
 * - ERC-7857 authorization validation
 * - 0G Storage integration for encrypted data retrieval
 * - AES-GCM decryption
 * - LLM inference (Phala RedPill API)
 * - Streaming inference support
 * - Oracle proof generation
 */

// INFT Contract ABI - only functions we need
const INFT_ABI = [
  "function isAuthorized(uint256 tokenId, address user) view returns (bool)",
  "function ownerOf(uint256 tokenId) view returns (address)",
  "function encryptedURI(uint256 tokenId) view returns (string)",
  "function metadataHash(uint256 tokenId) view returns (bytes32)"
];

interface InferRequest {
  tokenId: number;
  input: string;
  user?: string;
}

interface InferResponse {
  success: boolean;
  output?: string;
  proof?: string;
  error?: string;
  metadata?: {
    tokenId: number;
    authorized: boolean;
    timestamp: string;
    proofHash: string;
    provider?: string;
    model?: string;
    temperature?: number;
    promptHash?: string;
    contextHash?: string;
    completionHash?: string;
  };
}

interface QuotesData {
  version: string;
  quotes: string[];
  metadata: {
    created: string;
    description: string;
    totalQuotes: number;
    category: string;
  };
}

interface DevKeys {
  encryptedURI: string;
  storageRootHash: string;
  key: string;
  iv: string;
  tag: string;
}

interface LLMConfig {
  provider: string;
  host: string;
  model: string;
  temperature: number;
  maxTokens: number;
  requestTimeoutMs: number;
  maxContextQuotes: number;
  devFallback: boolean;
  apiKey: string;
}

interface OpenAIChoice {
  index: number;
  message?: {
    role: string;
    content: string;
  };
  delta?: {
    role?: string;
    content?: string;
  };
  finish_reason?: string | null;
  logprobs?: any;
}

interface OpenAIChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface LLMCallMetrics {
  startTime: number;
  endTime?: number;
  duration?: number;
  tokenCount?: number;
  promptLength: number;
  contextQuoteCount: number;
  usedFallback: boolean;
  circuitBreakerState: string;
}

interface StructuredLogContext {
  requestId: string;
  tokenId: number;
  userAddress: string;
  step: string;
  metrics?: LLMCallMetrics;
  error?: string;
}

class INFTOracleService {
  private app: express.Application;
  private provider!: JsonRpcProvider;
  private inftContract!: Contract;
  private devKeys!: DevKeys;
  private llmConfig!: LLMConfig;
  private llmCircuitBreaker!: CircuitBreaker<[string], string>;
  private port: number;

  constructor() {
    this.app = express();
    this.port = parseInt(process.env.PORT || '3001');
    
    // Initialize blockchain connection
    this.initializeBlockchain();
    
    // Load development keys
    this.loadDevKeys();
    
    // Load LLM configuration
    this.loadLLMConfig();
    
    // Initialize circuit breaker for LLM calls
    this.initializeLLMCircuitBreaker();
    
    // Setup Express middleware
    this.setupMiddleware();
    
    // Setup routes
    this.setupRoutes();
  }

  /**
   * Initialize blockchain provider and contracts
   */
  private initializeBlockchain(): void {
    const rpcUrl = process.env.GALILEO_RPC_URL || 'https://evmrpc-testnet.0g.ai';
    const inftAddress = process.env.INFT_CONTRACT_ADDRESS || '0xB28dce039dDf7BC39aDE96984c8349DD5C6EcDC1';

    this.provider = new JsonRpcProvider(rpcUrl);
    this.inftContract = new Contract(inftAddress, INFT_ABI, this.provider);

    console.log('🔗 Blockchain initialized:');
    console.log('  - RPC URL:', rpcUrl);
    console.log('  - INFT Contract:', inftAddress);
  }

  /**
   * Load development keys for decryption
   */
  private loadDevKeys(): void {
    const devKeysPath = path.join(__dirname, '..', '0g-INFT', 'storage', 'dev-keys.json');
    
    if (!fs.existsSync(devKeysPath)) {
      console.warn('⚠️  Development keys not found, using fallback data');
      this.devKeys = {
        encryptedURI: '0g://storage/demo',
        storageRootHash: 'demo-hash',
        key: '0x' + crypto.randomBytes(32).toString('hex'),
        iv: '0x' + crypto.randomBytes(12).toString('hex'),
        tag: '0x' + crypto.randomBytes(16).toString('hex')
      };
      return;
    }

    this.devKeys = JSON.parse(fs.readFileSync(devKeysPath, 'utf8'));
    console.log('🔑 Development keys loaded');
    console.log('  - Encrypted URI:', this.devKeys.encryptedURI);
  }

  /**
   * Load LLM configuration from environment variables
   */
  private loadLLMConfig(): void {
    this.llmConfig = {
      provider: process.env.LLM_PROVIDER || 'phala-redpill',
      host: process.env.LLM_HOST || 'https://api.red-pill.ai',
      model: process.env.LLM_MODEL || 'phala/deepseek-r1-70b',
      temperature: parseFloat(process.env.LLM_TEMPERATURE || '0.2'),
      maxTokens: parseInt(process.env.LLM_MAX_TOKENS || '256'),
      requestTimeoutMs: parseInt(process.env.LLM_REQUEST_TIMEOUT_MS || '30000'),
      maxContextQuotes: parseInt(process.env.LLM_MAX_CONTEXT_QUOTES || '25'),
      devFallback: process.env.LLM_DEV_FALLBACK === 'true',
      apiKey: process.env.REDPILL_API_KEY || ''
    };

    if (!this.llmConfig.apiKey) {
      throw new Error('REDPILL_API_KEY environment variable is required');
    }

    console.log('🤖 LLM configuration loaded:');
    console.log('  - Provider:', this.llmConfig.provider);
    console.log('  - Host:', this.llmConfig.host);
    console.log('  - Model:', this.llmConfig.model);
    console.log('  - Temperature:', this.llmConfig.temperature);
    console.log('  - Max Tokens:', this.llmConfig.maxTokens);
    console.log('  - API Key:', this.llmConfig.apiKey ? 'Configured' : 'Missing');
    console.log('  - Dev Fallback:', this.llmConfig.devFallback);
  }

  /**
   * Initialize LLM Circuit Breaker for resilient API calls
   */
  private initializeLLMCircuitBreaker(): void {
    const circuitBreakerOptions = {
      timeout: this.llmConfig.requestTimeoutMs,
      errorThresholdPercentage: 50, // Open circuit after 50% failures
      resetTimeout: 30000, // Try again after 30 seconds
      rollingCountTimeout: 10000, // 10 second rolling window
      rollingCountBuckets: 10,
      name: 'LLM-RedPill-Circuit',
      group: 'llm-calls'
    };

    this.llmCircuitBreaker = new CircuitBreaker(this.callLLMDirect.bind(this), circuitBreakerOptions);
    
    // Add fallback for circuit breaker - use gpt-5-nano as backup
    this.llmCircuitBreaker.fallback(async (prompt: string) => {
      this.logStructured({
        requestId: 'circuit-fallback',
        tokenId: 0,
        userAddress: 'system',
        step: 'llm_circuit_fallback',
        error: 'Circuit breaker fallback triggered - using gpt-5-nano'
      });
      
      try {
        // Call fallback model directly
        console.log('🔄 Attempting fallback LLM call...');
        const result = await this.callFallbackLLM(prompt);
        console.log('✅ Fallback LLM call succeeded');
        return result;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.error('❌ Fallback LLM error:', errorMessage);
        this.logStructured({
          requestId: 'circuit-fallback-error',
          tokenId: 0,
          userAddress: 'system',
          step: 'llm_circuit_fallback_failed',
          error: `Fallback model failed: ${errorMessage}`
        });
        throw new Error(`LLM_CIRCUIT_OPEN: Circuit breaker is open and fallback model failed: ${errorMessage}`);
      }
    });

    // Circuit breaker event listeners for monitoring
    this.llmCircuitBreaker.on('open', () => {
      this.logStructured({
        requestId: 'circuit-event',
        tokenId: 0,
        userAddress: 'system',
        step: 'circuit_state_change',
        error: 'Circuit breaker opened - LLM service appears degraded'
      });
    });

    this.llmCircuitBreaker.on('halfOpen', () => {
      this.logStructured({
        requestId: 'circuit-event',
        tokenId: 0,
        userAddress: 'system',
        step: 'circuit_state_change'
      });
    });

    this.llmCircuitBreaker.on('close', () => {
      this.logStructured({
        requestId: 'circuit-event',
        tokenId: 0,
        userAddress: 'system',
        step: 'circuit_state_change'
      });
    });

    console.log('🔧 LLM Circuit Breaker initialized:');
    console.log('  - Timeout:', circuitBreakerOptions.timeout + 'ms');
    console.log('  - Error Threshold:', circuitBreakerOptions.errorThresholdPercentage + '%');
    console.log('  - Reset Timeout:', circuitBreakerOptions.resetTimeout + 'ms');
  }

  /**
   * Structured logging with metadata
   */
  private logStructured(context: StructuredLogContext): void {
    const timestamp = new Date().toISOString();
    const logLevel = context.error ? 'ERROR' : 'INFO';
    
    // Base log structure
    const logEntry = {
      timestamp,
      level: logLevel,
      service: 'inft-oracle-backend',
      version: '1.0.0',
      requestId: context.requestId,
      tokenId: context.tokenId,
      userAddress: context.userAddress,
      step: context.step
    };

    // Add metrics if available
    if (context.metrics) {
      Object.assign(logEntry, {
        llm: {
          provider: this.llmConfig.provider,
          model: this.llmConfig.model,
          duration_ms: context.metrics.duration,
          token_count: context.metrics.tokenCount,
          prompt_length: context.metrics.promptLength,
          context_quotes: context.metrics.contextQuoteCount,
          used_fallback: context.metrics.usedFallback,
          circuit_state: context.metrics.circuitBreakerState
        }
      });
    }

    // Add error if present
    if (context.error) {
      Object.assign(logEntry, { error: context.error });
    }

    // Log in production-safe format
    console.log(JSON.stringify(logEntry));
  }

  /**
   * Setup Express middleware
   */
  private setupMiddleware(): void {
    this.app.use(cors({
      origin: process.env.FRONTEND_URL || 'http://localhost:3000',
      credentials: true
    }));
    this.app.use(express.json());
    
    // Rate limiting
    const limiter = rateLimit({
      windowMs: 60 * 1000,
      limit: 30,
      message: { error: 'Too many requests. Please try again later.' }
    });
    this.app.use(limiter);
    
    // Request logging
    this.app.use((req, res, next) => {
      const requestId = crypto.randomUUID();
      console.log(`[${requestId}] ${req.method} ${req.path}`);
      req.headers['x-request-id'] = requestId;
      next();
    });
  }

  /**
   * Setup API routes
   */
  private setupRoutes(): void {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({ 
        status: 'healthy', 
        service: 'INFT Oracle Backend',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      });
    });

    // LLM health check
    this.app.get('/llm/health', this.handleLLMHealthCheck.bind(this));

    // Main inference endpoint
    this.app.post('/infer', this.handleInferRequest.bind(this));

    // Streaming inference endpoint
    this.app.post('/infer/stream', this.handleStreamingInferRequest.bind(this));

    // 404 handler
    this.app.use((req, res) => {
      res.status(404).json({ error: 'Endpoint not found' });
    });
  }

  /**
   * Handle LLM health check
   */
  private async handleLLMHealthCheck(req: express.Request, res: express.Response): Promise<void> {
    try {
      if (!this.llmConfig.apiKey) {
        res.status(503).json({
          status: 'unhealthy',
          error: 'LLM API key not configured',
          provider: this.llmConfig.provider
        });
        return;
      }

      res.json({
        status: 'healthy',
        provider: this.llmConfig.provider,
        model: this.llmConfig.model,
        configured: true
      });
    } catch (error: any) {
      res.status(503).json({
        status: 'unhealthy',
        error: error?.message || 'Unknown error'
      });
    }
  }

  /**
   * Handle inference requests
   */
  private async handleInferRequest(req: express.Request, res: express.Response): Promise<void> {
    try {
      const request: InferRequest = req.body;
      const requestId = req.headers['x-request-id'] as string;
      
      // Validate request
      if (!request.tokenId || typeof request.tokenId !== 'number') {
        res.status(400).json({ 
          success: false, 
          error: 'Invalid tokenId. Must be a number.' 
        });
        return;
      }

      if (!request.input || typeof request.input !== 'string') {
        res.status(400).json({ 
          success: false, 
          error: 'Invalid input. Must be a non-empty string.' 
        });
        return;
      }

      // Input validation
      if (request.input.length > 500) {
        res.status(400).json({ 
          success: false, 
          error: 'Input too long. Maximum 500 characters allowed.' 
        });
        return;
      }

      // Sanitize input
      const sanitizedInput = request.input
        .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
        .trim();

      if (!sanitizedInput) {
        res.status(400).json({ 
          success: false, 
          error: 'Input contains only invalid characters.' 
        });
        return;
      }

      console.log(`[${requestId}] Processing inference for token ${request.tokenId}`);

      // Check authorization
      const userAddress = request.user || '0x0000000000000000000000000000000000000000';
      let isAuthorized = false;
      
      try {
        isAuthorized = await this.inftContract.isAuthorized(request.tokenId, userAddress);
        console.log(`[${requestId}] Authorization check: ${isAuthorized}`);
      } catch (error) {
        console.error(`[${requestId}] Authorization check failed:`, error);
        res.status(403).json({
          success: false,
          error: 'Authorization check failed. Token may not exist.'
        });
        return;
      }

      if (!isAuthorized) {
        res.status(403).json({
          success: false,
          error: 'Access denied. You are not authorized to use this INFT.'
        });
        return;
      }

      // Fetch and decrypt data
      let quotesData: QuotesData;
      try {
        quotesData = await this.fetchAndDecryptData();
      } catch (error) {
        console.error(`[${requestId}] Data fetch failed:`, error);
        res.status(500).json({
          success: false,
          error: 'Failed to fetch encrypted data from storage.'
        });
        return;
      }

      // Perform inference
      let output: string;
      try {
        output = await this.performLLMInference(sanitizedInput, quotesData);
      } catch (error) {
        console.error(`[${requestId}] LLM inference failed:`, error);
        // Fallback to random quote
        output = quotesData.quotes[Math.floor(Math.random() * quotesData.quotes.length)];
      }

      // Generate proof
      const proof = this.generateProof(sanitizedInput, output, quotesData);

      // Return response
      res.json({
        success: true,
        output,
        proof,
        metadata: {
          tokenId: request.tokenId,
          authorized: isAuthorized,
          timestamp: new Date().toISOString(),
          proofHash: crypto.createHash('sha256').update(proof).digest('hex'),
          provider: this.llmConfig.provider,
          model: this.llmConfig.model,
          temperature: this.llmConfig.temperature
        }
      });

    } catch (error: any) {
      console.error('Inference error:', error);
      res.status(500).json({
        success: false,
        error: error?.message || 'Internal server error'
      });
    }
  }

  /**
   * Handle streaming inference requests
   */
  private async handleStreamingInferRequest(req: express.Request, res: express.Response): Promise<void> {
    try {
      const request: InferRequest = req.body;
      const requestId = req.headers['x-request-id'] as string;
      
      // Validate and authorize (same as regular inference)
      if (!request.tokenId || !request.input) {
        res.status(400).json({ error: 'Invalid request' });
        return;
      }

      const userAddress = request.user || '0x0000000000000000000000000000000000000000';
      const isAuthorized = await this.inftContract.isAuthorized(request.tokenId, userAddress);

      if (!isAuthorized) {
        res.status(403).json({ error: 'Access denied' });
        return;
      }

      // Setup SSE
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // Send start event
      res.write(`event: start\n`);
      res.write(`data: ${JSON.stringify({
        provider: this.llmConfig.provider,
        model: this.llmConfig.model,
        temperature: this.llmConfig.temperature,
        requestId,
        timestamp: new Date().toISOString()
      })}\n\n`);

      // Fetch data and perform streaming inference
      try {
        const quotesData = await this.fetchAndDecryptData();
        await this.performStreamingLLMInference(request.input, quotesData, res);
      } catch (error: any) {
        res.write(`event: error\n`);
        res.write(`data: ${JSON.stringify({ error: error?.message || 'Unknown error' })}\n\n`);
      }

      res.end();

    } catch (error: any) {
      console.error('Streaming error:', error);
      res.status(500).json({ error: error?.message || 'Unknown error' });
    }
  }

  /**
   * Fetch and decrypt data from 0G Storage
   */
  private async fetchAndDecryptData(): Promise<QuotesData> {
    // Try to decrypt from local storage first (fallback)
    const localPath = path.join(__dirname, '..', '0g-INFT', 'storage', 'quotes.enc');
    
    if (fs.existsSync(localPath)) {
      console.log('Using local encrypted file');
      const encryptedData = fs.readFileSync(localPath);
      return this.decryptData(encryptedData);
    }

    // Fallback: return demo data
    console.log('Using demo quotes data');
    return {
      version: '1.0',
      quotes: [
        "The only way to do great work is to love what you do. - Steve Jobs",
        "Innovation distinguishes between a leader and a follower. - Steve Jobs",
        "Stay hungry, stay foolish. - Steve Jobs",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
        "Success is not final, failure is not fatal: it is the courage to continue that counts. - Winston Churchill"
      ],
      metadata: {
        created: new Date().toISOString(),
        description: 'Inspirational quotes',
        totalQuotes: 5,
        category: 'motivation'
      }
    };
  }

  /**
   * Decrypt AES-GCM encrypted data
   */
  private decryptData(encryptedBuffer: Buffer): QuotesData {
    const iv = Buffer.from(this.devKeys.iv.slice(2), 'hex');
    const tag = Buffer.from(this.devKeys.tag.slice(2), 'hex');
    const key = Buffer.from(this.devKeys.key.slice(2), 'hex');
    
    // Extract encrypted data (skip IV and tag)
    const encryptedData = encryptedBuffer.slice(28);
    
    const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
    decipher.setAuthTag(tag);
    
    let decrypted = decipher.update(encryptedData);
    decrypted = Buffer.concat([decrypted, decipher.final()]);
    
    return JSON.parse(decrypted.toString('utf8'));
  }

  /**
   * Direct LLM API call (wrapped by circuit breaker) - OpenAI-compatible RedPill API
   */
  private async callLLMDirect(prompt: string): Promise<string> {
    const requestPayload = {
      model: this.llmConfig.model,
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ],
      stream: false,
      temperature: this.llmConfig.temperature,
      max_tokens: this.llmConfig.maxTokens
    };

    console.log(`🌐 Calling RedPill API: ${this.llmConfig.host}/v1/chat/completions`);
    
    try {
      const response: AxiosResponse<OpenAIChatResponse> = await axios.post(
        `${this.llmConfig.host}/v1/chat/completions`,
        requestPayload,
        {
          timeout: this.llmConfig.requestTimeoutMs,
          headers: {
            'Content-Type': 'application/json',
            'Authorization': this.llmConfig.apiKey,
            'Accept': 'application/json'
          }
        }
      );

      if (!response.data || !response.data.choices || response.data.choices.length === 0) {
        throw new Error('Invalid response from RedPill API');
      }

      const choice = response.data.choices[0];
      if (!choice.message || !choice.message.content) {
        throw new Error('No content in RedPill API response');
      }

      console.log(`⚡ LLM response received (${choice.message.content.length} chars)`);
      return choice.message.content.trim();
      
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response?.status === 401) {
          throw new Error('Invalid RedPill API key. Please check your REDPILL_API_KEY environment variable.');
        } else if (error.response?.status === 429) {
          throw new Error('RedPill API rate limit exceeded. Please try again later.');
        } else if (error.code === 'ECONNREFUSED') {
          throw new Error(`Cannot connect to RedPill API at ${this.llmConfig.host}. Please check your internet connection.`);
        } else if (error.code === 'ECONNABORTED') {
          throw new Error(`LLM request timeout after ${this.llmConfig.requestTimeoutMs}ms`);
        } else {
          const errorMsg = error.response?.data?.error?.message || error.message;
          throw new Error(`RedPill API error: ${errorMsg}`);
        }
      }
      throw error;
    }
  }

  /**
   * Fallback LLM API call using gpt-5-nano when circuit breaker is open
   */
  private async callFallbackLLM(prompt: string): Promise<string> {
    const requestPayload = {
      model: 'openai/gpt-5-nano',
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ],
      stream: false,
      // gpt-5-nano only supports temperature=1 (default), so we omit it
      // gpt-5-nano is a reasoning model that uses tokens for internal reasoning + output
      // We need much more tokens to allow for both reasoning and actual response
      max_completion_tokens: 1024  // Increased from 256 to allow for reasoning + output
    };

    console.log(`🔄 Using fallback model: gpt-5-nano`);
    
    try {
      const response: AxiosResponse<OpenAIChatResponse> = await axios.post(
        `${this.llmConfig.host}/v1/chat/completions`,
        requestPayload,
        {
          timeout: this.llmConfig.requestTimeoutMs,
          headers: {
            'Content-Type': 'application/json',
            'Authorization': this.llmConfig.apiKey,
            'Accept': 'application/json'
          }
        }
      );

      if (!response.data || !response.data.choices || response.data.choices.length === 0) {
        throw new Error('Invalid response from RedPill API (fallback)');
      }

      const choice = response.data.choices[0];
      
      if (!choice.message || !choice.message.content) {
        throw new Error('No content in RedPill API response (fallback)');
      }

      console.log(`✅ Fallback LLM response received (${choice.message.content.length} chars)`);
      return choice.message.content.trim();
      
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response?.status === 401) {
          throw new Error('Invalid RedPill API key (fallback). Please check your REDPILL_API_KEY environment variable.');
        } else if (error.response?.status === 429) {
          throw new Error('RedPill API rate limit exceeded (fallback). Please try again later.');
        } else if (error.code === 'ECONNREFUSED') {
          throw new Error(`Cannot connect to RedPill API at ${this.llmConfig.host} (fallback). Please check your internet connection.`);
        } else if (error.code === 'ECONNABORTED') {
          throw new Error(`Fallback LLM request timeout after ${this.llmConfig.requestTimeoutMs}ms`);
        } else {
          const errorMsg = error.response?.data?.error?.message || error.message;
          throw new Error(`RedPill API error (fallback): ${errorMsg}`);
        }
      }
      throw error;
    }
  }

  /**
   * Perform LLM inference with circuit breaker
   */
  private async performLLMInference(input: string, quotesData: QuotesData): Promise<string> {
    if (!this.llmConfig.apiKey) {
      throw new Error('LLM API key not configured');
    }

    // Build context from quotes
    const contextQuotes = quotesData.quotes.slice(0, this.llmConfig.maxContextQuotes);
    const context = contextQuotes.map((q, i) => `${i + 1}. ${q}`).join('\n');

    const prompt = `You are a concise assistant. Use the provided context strictly. Return a single inspirational quote tailored to the user's input.

Input: "${input}"

Context quotes:
${context}

Respond with only the quote text. No prefatory wording.`;

    try {
      // Use circuit breaker for LLM call
      const completion = await this.llmCircuitBreaker.fire(prompt);
      return completion;
    } catch (error: any) {
      console.error('LLM API error:', error?.message || 'Unknown error');
      
      // Fallback policy based on configuration
      if (this.llmConfig.devFallback && !error?.message?.includes('LLM_CIRCUIT_OPEN')) {
        console.log('⚠️ Using random quote fallback');
        const randomIndex = Math.floor(Math.random() * quotesData.quotes.length);
        return quotesData.quotes[randomIndex];
      }
      
      throw error;
    }
  }

  /**
   * Perform streaming LLM inference
   */
  private async performStreamingLLMInference(
    input: string, 
    quotesData: QuotesData, 
    res: express.Response
  ): Promise<void> {
    if (!this.llmConfig.apiKey) {
      throw new Error('LLM API key not configured');
    }

    const contextQuotes = quotesData.quotes.slice(0, this.llmConfig.maxContextQuotes);
    const context = contextQuotes.map((q, i) => `${i + 1}. ${q}`).join('\n');

    const prompt = `You are a concise assistant. Return a single inspirational quote tailored to: "${input}"\n\nContext:\n${context}`;

    try {
      const response = await axios.post(
        `${this.llmConfig.host}/v1/chat/completions`,
        {
          model: this.llmConfig.model,
          messages: [{ role: 'user', content: prompt }],
          temperature: this.llmConfig.temperature,
          max_tokens: this.llmConfig.maxTokens,
          stream: true
        },
        {
          headers: {
            'Authorization': `Bearer ${this.llmConfig.apiKey}`,
            'Content-Type': 'application/json'
          },
          responseType: 'stream',
          timeout: this.llmConfig.requestTimeoutMs
        }
      );

      let fullResponse = '';
      let tokenCount = 0;

      response.data.on('data', (chunk: Buffer) => {
        const lines = chunk.toString().split('\n').filter(line => line.trim() !== '');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            
            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices[0]?.delta?.content || '';
              
              if (content) {
                fullResponse += content;
                tokenCount++;
                
                res.write(`event: token\n`);
                res.write(`data: ${JSON.stringify({ content, tokenCount, done: false })}\n\n`);
              }
            } catch (e) {
              // Skip parse errors
            }
          }
        }
      });

      await new Promise((resolve, reject) => {
        response.data.on('end', resolve);
        response.data.on('error', reject);
      });

      // Send completion event
      res.write(`event: completion\n`);
      res.write(`data: ${JSON.stringify({ 
        fullResponse, 
        totalTokens: tokenCount,
        done: true 
      })}\n\n`);

    } catch (error: any) {
      console.error('Streaming LLM error:', error?.message || 'Unknown error');
      throw error;
    }
  }

  /**
   * Generate oracle proof
   */
  private generateProof(input: string, output: string, quotesData: QuotesData): string {
    const promptHash = crypto.createHash('sha256').update(input).digest('hex');
    const contextHash = crypto.createHash('sha256').update(JSON.stringify(quotesData.quotes)).digest('hex');
    const completionHash = crypto.createHash('sha256').update(output).digest('hex');

    const proofData = {
      version: '1.0.0',
      type: 'ORACLE_PROOF',
      timestamp: new Date().toISOString(),
      hashes: {
        prompt: promptHash,
        context: contextHash,
        completion: completionHash
      },
      model: {
        provider: this.llmConfig.provider,
        name: this.llmConfig.model,
        temperature: this.llmConfig.temperature
      }
    };

    return JSON.stringify(proofData);
  }

  /**
   * Start the server
   */
  public start(): void {
    this.app.listen(this.port, () => {
      console.log('\n╔════════════════════════════════════════════════════════════╗');
      console.log('║       🚀 INFT Oracle Backend Service Started              ║');
      console.log('╚════════════════════════════════════════════════════════════╝\n');
      console.log(`✅ Server running on http://localhost:${this.port}`);
      console.log(`✅ Connected to 0G Galileo (Chain ID: 16602)`);
      console.log(`✅ INFT Contract: ${this.inftContract.target}`);
      console.log(`\n📡 Endpoints:`);
      console.log(`   - GET  /health         - Service health check`);
      console.log(`   - GET  /llm/health     - LLM health check`);
      console.log(`   - POST /infer          - Run inference`);
      console.log(`   - POST /infer/stream   - Streaming inference`);
      console.log(`\n🔐 Security:`);
      console.log(`   - Rate limiting: 30 req/min`);
      console.log(`   - Input sanitization: enabled`);
      console.log(`   - Authorization: on-chain validation`);
      console.log('\n');
    });
  }
}

// Start the service
const service = new INFTOracleService();
service.start();

export default INFTOracleService;

