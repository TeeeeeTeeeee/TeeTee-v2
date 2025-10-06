import type { NextApiRequest, NextApiResponse } from 'next';
import { connectToDatabase } from '../../utils/mongodb';

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
    console.log('Testing MongoDB connection...');
    console.log('MONGODB_URI exists:', !!process.env.MONGODB_URI);
    
    const { client, db } = await connectToDatabase();
    
    // Try to ping the database
    await db.command({ ping: 1 });
    
    // Get database stats
    const stats = await db.stats();
    
    return res.status(200).json({
      success: true,
      message: 'Successfully connected to MongoDB!',
      details: {
        database: db.databaseName,
        collections: stats.collections || 0,
        dataSize: stats.dataSize || 0,
        indexes: stats.indexes || 0,
      },
    });
  } catch (error: any) {
    console.error('MongoDB connection test failed:', error);
    
    return res.status(500).json({
      success: false,
      message: 'Failed to connect to MongoDB',
      error: error?.message || String(error),
      details: {
        errorType: error?.constructor?.name,
        code: error?.code,
      },
    });
  }
}


