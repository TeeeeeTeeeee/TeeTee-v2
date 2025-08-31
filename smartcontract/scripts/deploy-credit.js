const { ethers } = require("hardhat");

async function main() {
    const [deployer] = await ethers.getSigners();
    console.log("🚀 Deploying with:", await deployer.getAddress());

    const Credit = await ethers.getContractFactory("Credit", deployer);
    console.log("📦 Deploying Credit...");
    const credit = await Credit.deploy();
    await credit.waitForDeployment();

    const address = await credit.getAddress();
    console.log("✅ Deployed at:", address);

    const amount = await credit.BUNDLE_AMOUNT();
    const price = await credit.BUNDLE_PRICE();
    console.log("   - Bundle amount:", amount.toString());
    console.log("   - Bundle price:", ethers.formatEther(price), "0G");
}

main().catch((e) => {
  console.error("❌ Deployment failed:", e);
  process.exitCode = 1;
});