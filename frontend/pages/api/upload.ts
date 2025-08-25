import type { NextApiRequest, NextApiResponse, PageConfig } from 'next';

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