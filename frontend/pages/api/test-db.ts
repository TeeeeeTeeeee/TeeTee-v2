import type { NextApiRequest, NextApiResponse } from 'next';
import { getAllSessions } from '../../utils/json-storage';
import fs from 'node:fs';
import path from 'node:path';

type ResponseBody = {
  success: boolean;
  message: string;
  details?: any;
  error?: string;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseBody>,
) {
  try {
    console.log('Testing JSON storage...');
    
    const dataDir = path.join(process.cwd(), 'data');
    const sessionsFile = path.join(dataDir, 'chat-sessions.json');
    
    // Check if data directory exists
    const dataDirExists = fs.existsSync(dataDir);
    
    // Check if sessions file exists
    const sessionsFileExists = fs.existsSync(sessionsFile);
    
    // Get all sessions
    const sessions = getAllSessions();
    
    return res.status(200).json({
      success: true,
      message: 'JSON storage is working!',
      details: {
        dataDirectory: dataDir,
        dataDirExists,
        sessionsFileExists,
        totalSessions: sessions.length,
        sessions: sessions.slice(0, 5), // Return first 5 sessions as sample
      },
    });
  } catch (error: any) {
    console.error('JSON storage test failed:', error);
    
    return res.status(500).json({
      success: false,
      message: 'Failed to access JSON storage',
      error: error?.message || String(error),
      details: {
        errorType: error?.constructor?.name,
      },
    });
  }
}


