/**
 * Test client for LLM Server
 * Run with: node test-client.js
 */

const SERVER_URL = 'http://localhost:3001';

// Test health endpoint
async function testHealth() {
  console.log('\n🔍 Testing /health endpoint...');
  try {
    const response = await fetch(`${SERVER_URL}/health`);
    const data = await response.json();
    console.log('✅ Health check:', data);
  } catch (error) {
    console.error('❌ Health check failed:', error.message);
  }
}

// Test models endpoint
async function testModels() {
  console.log('\n🔍 Testing /models endpoint...');
  try {
    const response = await fetch(`${SERVER_URL}/models`);
    const data = await response.json();
    console.log('✅ Available models:', data);
  } catch (error) {
    console.error('❌ Models check failed:', error.message);
  }
}

// Test inference endpoint
async function testInference() {
  console.log('\n🔍 Testing /inference endpoint...');
  try {
    const response = await fetch(`${SERVER_URL}/inference`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: 'What is 2+2? Answer in one sentence.',
        model: 'deepseek-v3'
      })
    });
    
    const data = await response.json();
    console.log('✅ Inference result:');
    console.log('   Model:', data.model);
    console.log('   Response:', data.response);
    console.log('   Usage:', data.usage);
  } catch (error) {
    console.error('❌ Inference failed:', error.message);
  }
}

// Test chat endpoint
async function testChat() {
  console.log('\n🔍 Testing /chat endpoint...');
  try {
    const response = await fetch(`${SERVER_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'Tell me a very short joke about programming.' }
        ],
        model: 'deepseek-v3',
        temperature: 0.7,
        max_tokens: 100
      })
    });
    
    const data = await response.json();
    console.log('✅ Chat result:');
    console.log('   Model:', data.model);
    console.log('   Response:', data.response);
    console.log('   Usage:', data.usage);
  } catch (error) {
    console.error('❌ Chat failed:', error.message);
  }
}

// Test streaming chat endpoint
async function testChatStream() {
  console.log('\n🔍 Testing /chat/stream endpoint...');
  try {
    const response = await fetch(`${SERVER_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [
          { role: 'user', content: 'Count from 1 to 5, one number per line.' }
        ],
        model: 'deepseek-v3'
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    console.log('✅ Streaming response:');
    process.stdout.write('   ');
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            console.log('\n   Stream completed.');
            break;
          }
          try {
            const parsed = JSON.parse(data);
            if (parsed.content) {
              process.stdout.write(parsed.content);
            }
          } catch (e) {
            // Ignore parse errors
          }
        }
      }
    }
    console.log();
  } catch (error) {
    console.error('❌ Streaming chat failed:', error.message);
  }
}

// Run all tests
async function runTests() {
  console.log('🚀 Starting LLM Server Tests');
  console.log('=' .repeat(50));
  
  await testHealth();
  await testModels();
  await testInference();
  await testChat();
  await testChatStream();
  
  console.log('\n' + '='.repeat(50));
  console.log('✅ All tests completed!');
}

// Run tests
runTests().catch(error => {
  console.error('❌ Test suite failed:', error);
  process.exit(1);
});

