const { ethers } = require("hardhat");

async function main() {
  const contractAddress = "0xEF439FBFc73232d9368Af87D91ecaFB0FA221373";
  const userAddress = "0x0020cE4969A6Ec50885E083784A495483Db7A62c";
  
  const Subscription = await ethers.getContractFactory("Subscription");
  const contract = Subscription.attach(contractAddress);
  
  const credits = await contract.getPromptCredits(userAddress);
  const isSubscribed = await contract.isSubscribed(userAddress);
  const balance = await ethers.provider.getBalance(contractAddress);
  
  console.log("User prompt credits:", credits.toString());
  console.log("User subscribed:", isSubscribed);
  console.log("Contract balance:", ethers.formatEther(balance), "0G");
}

main().catch(console.error);
