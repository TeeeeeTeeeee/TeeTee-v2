import type { NextApiRequest, NextApiResponse } from "next";

const LLM_SERVER_URL = "https://f39ca1bc5d8d918a378cd8e1d305d5ac3e75dc81-3001.dstack-pha-prod7.phala.network";

type ErrorResponse = {
  error: string;
  details?: any;
};

type SuccessResponse = {
  success: boolean;
  model?: string;
  response?: string;
  usage?: any;
  [key: string]: any;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<SuccessResponse | ErrorResponse>
) {
  // Only allow POST requests
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const {
      endpoint = "chat", // 'chat', 'inference', or 'chat/stream'
      messages,
      prompt,
      model = "deepseek-v3",
      temperature = 0.7,
      max_tokens = 2000,
      system = "You are a helpful assistant",
    } = req.body;

    // Validate required fields based on endpoint
    if (endpoint === "chat" && (!messages || !Array.isArray(messages))) {
      return res.status(400).json({
        error: "Messages array is required for chat endpoint",
      });
    }

    if (endpoint === "inference" && !prompt) {
      return res.status(400).json({
        error: "Prompt is required for inference endpoint",
      });
    }

    // Build the request body based on endpoint
    let requestBody: any;
    if (endpoint === "inference") {
      requestBody = {
        prompt,
        system,
        model,
        temperature,
        max_tokens,
      };
    } else {
      // chat or chat/stream
      requestBody = {
        messages,
        model,
        temperature,
        max_tokens,
      };
    }

    // Make request to LLM server
    const llmResponse = await fetch(`${LLM_SERVER_URL}/${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const data = await llmResponse.json();

    if (!llmResponse.ok) {
      return res.status(llmResponse.status).json({
        error: data.error || "LLM server error",
        details: data.details || data,
      });
    }

    return res.status(200).json(data);
  } catch (error: any) {
    console.error("LLM API error:", error);
    return res.status(500).json({
      error: "Failed to communicate with LLM server",
      details: error.message,
    });
  }
}

