import "@nomicfoundation/hardhat-toolbox";
import path from "path";
import dotenv from "dotenv";

dotenv.config({ path: path.resolve(process.cwd(), ".env") });

const config = {
  solidity: "0.8.20",
  networks: {
    "0g-testnet": {
      url: process.env.RPC_URL_0G,
      accounts: [process.env.PRIVATE_KEY], 
    },
  },
};

export default config;

