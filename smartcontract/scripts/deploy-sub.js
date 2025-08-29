const { ethers } = require("hardhat");

async function main() {
    console.log("🚀 Starting deployment of Subscription contract from sub.sol...");
    
    // Get the contract factory
    const Subscription = await ethers.getContractFactory("Subscription");
    
    // Deploy the contract
    console.log("📦 Deploying Subscription contract...");
    const subscription = await Subscription.deploy();
    
    // Wait for deployment to finish
    await subscription.waitForDeployment();
    
    // Get the contract address
    const address = await subscription.getAddress();
    
    console.log("✅ Subscription contract deployed successfully!");
    console.log("📍 Contract address:", address);
    console.log("�� Subscription price: 0.001 0G");
    console.log("⏱️  Subscription duration: 30 days");
    console.log("�� Prompt cost: 0.00001 0G per prompt");
    console.log("🎁 Prompts per subscription: 100");
    
    // Verify the deployment by checking contract constants
    const subscriptionPrice = await subscription.SUBSCRIPTION_PRICE();
    const subscriptionDuration = await subscription.SUBSCRIPTION_DURATION();
    const promptCost = await subscription.PROMPT_COST();
    
    console.log("🔍 Contract verification:");
    console.log("   - Subscription price:", ethers.formatEther(subscriptionPrice), "0G");
    console.log("   - Subscription duration:", subscriptionDuration.toString(), "seconds");
    console.log("   - Prompt cost:", ethers.formatEther(promptCost), "0G");
    
    console.log("\n�� Next steps:");
    console.log("1. Copy the contract address above");
    console.log("2. Update your .env file with:");
    console.log(`   NEXT_PUBLIC_SUBSCRIPTION_ADDRESS=${address}`);
    console.log("3. Use this address in your frontend");
}

main().catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exitCode = 1;
});