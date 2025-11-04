import type { NextApiRequest, NextApiResponse } from 'next';
import { getSessionsByWallet, type ChatSession } from '../../utils/json-storage';
import { connectToDatabase } from '../../utils/mongodb';

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

    let sessions: ChatSession[] = [];

    // Try to get sessions from MongoDB first
    try {
      const { db } = await connectToDatabase();
      const chatSessionsCollection = db.collection('chatSessions');
      
      const mongoSessions = await chatSessionsCollection
        .find({ walletAddress: walletAddress.toLowerCase() })
        .sort({ createdAt: -1 })
        .toArray();
      
      // Convert MongoDB documents to ChatSession type
      sessions = mongoSessions.map(doc => ({
        _id: doc._id as string,
        walletAddress: doc.walletAddress,
        filename: doc.filename,
        preview: doc.preview,
        rootHash: doc.rootHash,
        txHash: doc.txHash,
        messageCount: doc.messageCount,
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
      }));
      
      console.log(`Loaded ${sessions.length} sessions from MongoDB for wallet ${walletAddress}`);
    } catch (mongoError: any) {
      console.error('MongoDB read error, falling back to local JSON:', mongoError);
      // Fall back to local JSON storage if MongoDB fails
      sessions = getSessionsByWallet(walletAddress);
      console.log(`Loaded ${sessions.length} sessions from local JSON for wallet ${walletAddress}`);
    }

    return res.status(200).json({
      sessions,
      total: sessions.length,
    });
  } catch (err: any) {
    console.error('get-chat-sessions error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}


