import "@nomicfoundation/hardhat-toolbox";
import path from "path";
import dotenv from "dotenv";

dotenv.config({ path: path.resolve(process.cwd(), ".env") });

const config = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
      viaIR: true, // Fix for "Stack too deep" error
    },
  },
  networks: {
    // 0G Galileo Testnet
    "0g-testnet": {
      url: process.env.RPC_URL_0G || "https://evmrpc-testnet.0g.ai/",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 16602,
    },
    // 0G Mainnet
    "0g-mainnet": {
      url: process.env.MAINNET_RPC_URL || "https://evmrpc.0g.ai",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 16661,
      gasPrice: "auto",
    },
  },
};

export default config;

