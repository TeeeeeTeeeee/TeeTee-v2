import type { NextApiRequest, NextApiResponse } from 'next';
import { connectToDatabase } from '../../utils/mongodb';

type ChatSession = {
  _id: string;
  walletAddress: string;
  filename: string;
  rootHash: string | null;
  txHash: string;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
};

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

    const { db } = await connectToDatabase();
    const collection = db.collection('chatSessions');

    // Find all sessions for this wallet address (normalized to lowercase)
    const sessions = await collection
      .find({ walletAddress: walletAddress.toLowerCase() })
      .sort({ createdAt: -1 }) // Most recent first
      .toArray();

    // Transform MongoDB documents to JSON-serializable format
    const formattedSessions: ChatSession[] = sessions.map((session) => ({
      _id: session._id.toString(),
      walletAddress: session.walletAddress,
      filename: session.filename,
      rootHash: session.rootHash,
      txHash: session.txHash,
      messageCount: session.messageCount,
      createdAt: session.createdAt.toISOString(),
      updatedAt: session.updatedAt.toISOString(),
    }));

    return res.status(200).json({
      sessions: formattedSessions,
      total: formattedSessions.length,
    });
  } catch (err: any) {
    console.error('get-chat-sessions error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}


