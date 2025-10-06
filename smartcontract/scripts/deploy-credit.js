import pkg from 'hardhat';
const { ethers } = pkg;

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("🚀 Deploying with:", await deployer.getAddress());

  // Compile and get contract factory
  const CreditUse = await ethers.getContractFactory("CreditUse", deployer);

  // Deploy contract
  console.log("📦 Deploying CreditUse...");
  const creditUse = await CreditUse.deploy();
  await creditUse.waitForDeployment();
  const address = await creditUse.getAddress();
  console.log("✅ Deployed at:", address);

  // Log basic info
  const amount = await creditUse.BUNDLE_AMOUNT();
  const price = await creditUse.BUNDLE_PRICE();
  const owner = await creditUse.owner();
  console.log("   - Bundle amount:", amount.toString());
  console.log("   - Bundle price:", ethers.formatEther(price), "0G");
  console.log("   - Contract owner:", owner);

  // Buy credits
  console.log("💳 Buying 1 bundle of credits...");
  const buyTx = await creditUse.buyCredits({ value: price });
  await buyTx.wait();
  const deployerCredits = await creditUse.checkUserCredits(deployer.address);
  console.log("   - Deployer credits:", deployerCredits.toString());

  //  Register a hosted LLM
  console.log("📝 Registering a Hosted LLM...");
  const registerTx = await creditUse.registerLLM(
    deployer.address,       // host1
    deployer.address,       // host2
    "https://shard1.com",  // shardUrl1
    "https://shard2.com",  // shardUrl2
    "TestModel",            // modelName
    100,                    // totalTimeHost1 in minutes
    50                      // totalTimeHost2 in minutes
  );
  await registerTx.wait();
  const llmId = 0;
  console.log("   - Registered LLM ID:", llmId);

  // Use credits on the LLM
  const tokensUsed = 20;
  console.log(`💻 Using ${tokensUsed} credits on LLM ID ${llmId}...`);
  const useTx = await creditUse.usePrompt(llmId, tokensUsed);
  await useTx.wait();
  const poolBalance = (await creditUse.getHostedLLM(llmId)).poolBalance;
  console.log("   - Pool balance after usePrompt:", poolBalance.toString());

  console.log("⏱️ Reporting downtime for hosts...");
  const downtimeTx = await creditUse.reportDowntime(llmId, 10, 5); // host1 10min, host2 5min
  await downtimeTx.wait();
  const llmData = await creditUse.getHostedLLM(llmId);
  console.log("   - Downtime host1:", llmData.downtimeHost1.toString());
  console.log("   - Downtime host2:", llmData.downtimeHost2.toString());

  // Withdraw rewards to hosts
  console.log("💸 Withdrawing rewards to hosts...");
  const withdrawTx = await creditUse.withdrawToHosts(llmId);
  await withdrawTx.wait();
  console.log("✅ Withdraw executed. Check host wallets for received funds.");
}

main().catch((e) => {
  console.error("❌ Deployment or test failed:", e);
  process.exitCode = 1;
});
