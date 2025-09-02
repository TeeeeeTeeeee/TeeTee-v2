import type { NextApiRequest, NextApiResponse } from 'next';
import OpenAI from 'openai';

const client = new OpenAI();

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  if (!process.env.OPENAI_API_KEY) {
    return res.status(500).json({ error: 'Missing OPENAI_API_KEY in server environment' });
  }

  try {
    const { messages, model } = req.body as {
      messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }> | string;
      model?: string;
    };

    if (!messages || (Array.isArray(messages) && messages.length === 0)) {
      return res.status(400).json({ error: 'messages array required' });
    }

    const usedModel = model || 'gpt-5-nano-2025-08-07';

    // Build a single text prompt for the Responses API
    const prompt = Array.isArray(messages)
      ? messages.map((m) => `${m.role}: ${m.content}`).join('\n')
      : messages;

    const response = await client.responses.create({
      model: usedModel,
      input: prompt,
    });

    const text = (response as any).output_text ?? '';
    return res.status(200).json({ text });
  } catch (err: any) {
    console.error('OpenAI API error:', err);
    return res.status(500).json({ error: err?.message || 'Unknown error' });
  }
}
