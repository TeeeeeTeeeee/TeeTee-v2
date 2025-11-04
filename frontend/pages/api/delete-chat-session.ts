import type { NextApiRequest, NextApiResponse } from 'next';
import { connectToDatabase } from '../../utils/mongodb';

type SuccessResponse = {
  success: boolean;
  message: string;
};

type ErrorResponse = { error: string };

type ResponseBody = SuccessResponse | ErrorResponse;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseBody>,
) {
  if (req.method !== 'DELETE') {
    res.setHeader('Allow', 'DELETE');
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { sessionId } = req.body;

    if (!sessionId || typeof sessionId !== 'string') {
      return res.status(400).json({ error: 'Session ID is required' });
    }

    // Delete the session from MongoDB only
    const { db } = await connectToDatabase();
    const chatSessionsCollection = db.collection('chatSessions');
    
    const result = await chatSessionsCollection.deleteOne({ sessionId: sessionId });
    
    if (result.deletedCount === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }
    
    console.log(`Deleted session ${sessionId} from MongoDB`);

    return res.status(200).json({
      success: true,
      message: 'Chat session deleted successfully',
    });
  } catch (err: any) {
    console.error('delete-chat-session error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}



