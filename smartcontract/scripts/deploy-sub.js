const { ethers } = require("hardhat");

async function main() {
    const [deployer] = await ethers.getSigners();
    console.log("🚀 Deploying with:", await deployer.getAddress());

    const Subscription = await ethers.getContractFactory("Subscription", deployer);
    console.log("📦 Deploying Subscription...");
    const subscription = await Subscription.deploy();
    await subscription.waitForDeployment();

    const address = await subscription.getAddress();
    console.log("✅ Deployed at:", address);

    const price = await subscription.SUBSCRIPTION_PRICE();
    const duration = await subscription.SUBSCRIPTION_DURATION();
    console.log("   - Price:", ethers.formatEther(price), "0G");
    console.log("   - Duration (days):", (duration / 86400n).toString());
}

main().catch((e) => {
  console.error("❌ Deployment failed:", e);
  process.exitCode = 1;
});