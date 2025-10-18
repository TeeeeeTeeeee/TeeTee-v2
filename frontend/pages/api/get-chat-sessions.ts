import type { NextApiRequest, NextApiResponse } from 'next';
import { getSessionsByWallet, type ChatSession } from '../../utils/json-storage';

type SuccessResponse = {
  sessions: ChatSession[];
  total: number;
};

type ErrorResponse = { error: string };

type ResponseBody = SuccessResponse | ErrorResponse;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseBody>,
) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { walletAddress } = req.query;

    if (!walletAddress || typeof walletAddress !== 'string') {
      return res.status(400).json({ error: 'Wallet address is required' });
    }

    // Get sessions from local JSON storage
    const sessions = getSessionsByWallet(walletAddress);

    return res.status(200).json({
      sessions,
      total: sessions.length,
    });
  } catch (err: any) {
    console.error('get-chat-sessions error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}


