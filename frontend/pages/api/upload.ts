import type { NextApiRequest, NextApiResponse, PageConfig } from 'next';
import formidable, { type Fields, type Files, type File } from 'formidable';
import fs from 'node:fs';

export const config: PageConfig = {
  api: {
    bodyParser: false,
  },
};

type SuccessResponse = {
  filename: string;
  size: number;
  rootHash: string;
  txHash: string;
};

type ErrorResponse = { error: string };

type ResponseBody = SuccessResponse | ErrorResponse;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseBody>,
) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const form = formidable({
    multiples: false,
    maxFileSize: 1024 * 1024 * 1024, // 1GB limit
  });

  try {
    const { fields, files } = await new Promise<{ fields: Fields; files: Files }>((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) return reject(err);
        resolve({ fields, files });
      });
    });

    res.status(200).json({
      filename: 'stub',
      size: 0,
      rootHash: 'stub',
      txHash: 'stub',
    });
  } catch (err: any) {
    console.error('upload-multipart error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}
