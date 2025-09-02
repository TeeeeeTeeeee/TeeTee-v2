const hre = require("hardhat");

async function main() {
  const INFTFixed = await hre.ethers.getContractFactory("INFTFixed");
  const abi = INFTFixed.interface.fragments.map(fragment => fragment.format('json'));
  console.log('Fixed INFT ABI:');
  console.log(JSON.stringify(abi.map(f => JSON.parse(f)), null, 2));
}

main().catch(console.error);
