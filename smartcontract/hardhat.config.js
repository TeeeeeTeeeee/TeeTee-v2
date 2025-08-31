require("@nomicfoundation/hardhat-toolbox");
const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, ".env") });

/** @type import('hardhat/config').HardhatUserConfig */
// hardhat.config.js
module.exports = {
  solidity: "0.8.20",
  networks: {
    "0g-testnet": {
      url: process.env.RPC_URL_0G,
      accounts: [process.env.PRIVATE_KEY], 
    },
  },
};

