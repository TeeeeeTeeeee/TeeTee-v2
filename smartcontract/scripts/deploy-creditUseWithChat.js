import pkg from 'hardhat';
const { ethers } = pkg;

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("🚀 Deploying with:", await deployer.getAddress());

  // Compile and get contract factory
  const CreditUseWithChat = await ethers.getContractFactory("CreditUse", deployer);

  // Deploy contract
  console.log("📦 Deploying CreditUseWithChat...");
  const creditUseWithChat = await CreditUseWithChat.deploy();
  await creditUseWithChat.waitForDeployment();
  const address = await creditUseWithChat.getAddress();
  console.log("✅ Deployed at:", address);

  // Log basic info
  const amount = await creditUseWithChat.BUNDLE_AMOUNT();
  const price = await creditUseWithChat.BUNDLE_PRICE();
  const owner = await creditUseWithChat.owner();
  console.log("   - Bundle amount:", amount.toString());
  console.log("   - Bundle price:", ethers.formatEther(price), "0G");
  console.log("   - Contract owner:", owner);

  // // Buy credits
  // console.log("💳 Buying 1 bundle of credits...");
  // const buyTx = await creditUse.buyCredits({ value: price });
  // await buyTx.wait();
  // const deployerCredits = await creditUse.checkUserCredits(deployer.address);
  // console.log("   - Deployer credits:", deployerCredits.toString());

  // //  Create new LLM with first host (pass array length as llmId to create new)
  // console.log("📝 Creating new LLM with first host...");
  // const totalLLMs = await creditUse.getTotalLLMs();
  // const registerTx = await creditUse.registerLLM(
  //   totalLLMs,                               // llmId - array length for new entry
  //   deployer.address,                        // host1
  //   "0x0000000000000000000000000000000000000000", // host2 - empty (address zero)
  //   "https://shard1.com",                   // shardUrl1
  //   "",                                      // shardUrl2 - empty
  //   "TestModel",                             // modelName
  //   100,                                     // totalTimeHost1 in minutes
  //   0                                        // totalTimeHost2 - 0 (empty)
  // );
  // await registerTx.wait();
  // const llmId = Number(totalLLMs);
  // console.log("   - Created LLM ID:", llmId);
  // console.log("   - Status: Waiting for second host");
  
  // // Join as second host (update existing entry, leave host1 fields as 0/empty)
  // console.log("🤝 Joining as second host...");
  // const joinTx = await creditUse.registerLLM(
  //   llmId,                                   // llmId - existing entry
  //   "0x0000000000000000000000000000000000000000", // host1 - empty (keep existing)
  //   deployer.address,                        // host2
  //   "",                                      // shardUrl1 - empty (keep existing)
  //   "https://shard2.com",                   // shardUrl2
  //   "",                                      // modelName - empty (keep existing)
  //   0,                                       // totalTimeHost1 - 0 (keep existing)
  //   50                                       // totalTimeHost2 in minutes
  // );
  // await joinTx.wait();
  // console.log("   - Second host joined successfully");
  // console.log("   - Status: Complete");

  // // Use credits on the LLM
  // const tokensUsed = 20;
  // console.log(`💻 Using ${tokensUsed} credits on LLM ID ${llmId}...`);
  // const useTx = await creditUse.usePrompt(llmId, tokensUsed);
  // await useTx.wait();
  // const poolBalance = (await creditUse.getHostedLLM(llmId)).poolBalance;
  // console.log("   - Pool balance after usePrompt:", poolBalance.toString());

  // console.log("⏱️ Reporting downtime for hosts...");
  // const downtimeTx = await creditUse.reportDowntime(llmId, 10, 5); // host1 10min, host2 5min
  // await downtimeTx.wait();
  // const llmData = await creditUse.getHostedLLM(llmId);
  // console.log("   - Downtime host1:", llmData.downtimeHost1.toString());
  // console.log("   - Downtime host2:", llmData.downtimeHost2.toString());

  // // Withdraw rewards to hosts
  // console.log("💸 Withdrawing rewards to hosts...");
  // const withdrawTx = await creditUse.withdrawToHosts(llmId);
  // await withdrawTx.wait();
  // console.log("✅ Withdraw executed. Check host wallets for received funds.");
}

main().catch((e) => {
  console.error("❌ Deployment or test failed:", e);
  process.exitCode = 1;
});
