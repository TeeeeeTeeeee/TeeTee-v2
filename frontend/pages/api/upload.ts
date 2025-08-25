import type { NextApiRequest, NextApiResponse, PageConfig } from 'next';

export const config: PageConfig = {
  api: {
    bodyParser: false,
  },
};
