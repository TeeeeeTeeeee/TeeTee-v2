import type { NextApiRequest, NextApiResponse } from 'next';
import { deleteSession } from '../../utils/json-storage';

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

    // Delete the session from local JSON storage
    const deleted = deleteSession(sessionId);

    if (!deleted) {
      return res.status(404).json({ error: 'Session not found' });
    }

    return res.status(200).json({
      success: true,
      message: 'Chat session deleted successfully',
    });
  } catch (err: any) {
    console.error('delete-chat-session error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}

