import pkg from 'hardhat';
const { ethers } = pkg;

async function main() {
    const [deployer] = await ethers.getSigners();
    console.log("🚀 Deploying with:", await deployer.getAddress());

    const CreditUse = await ethers.getContractFactory("CreditUse", deployer);
    console.log("📦 Deploying CreditUse...");
    const creditUse = await CreditUse.deploy();
    await creditUse.waitForDeployment();

    const address = await creditUse.getAddress();
    console.log("✅ Deployed at:", address);

    const amount = await creditUse.BUNDLE_AMOUNT();
    const price = await creditUse.BUNDLE_PRICE();
    console.log("   - Bundle amount:", amount.toString());
    console.log("   - Bundle price:", ethers.formatEther(price), "0G");
}

main().catch((e) => {
  console.error("❌ Deployment failed:", e);
  process.exitCode = 1;
});