// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract CreditUse {
    uint256 public constant BUNDLE_AMOUNT = 200;
    uint256 public constant BUNDLE_PRICE = 0.001 ether; // 0.001 0G for 200 credits
    uint256 public constant CREDIT_PRICE_WEI = BUNDLE_PRICE / BUNDLE_AMOUNT;

    address public owner;

    mapping(address => uint256) public userCredits;

    struct HostedLLMEntry {
        address host1;
        address host2;
        string shardUrl1;
        string shardUrl2;
        string modelName;
        uint256 poolBalance;
    }

    HostedLLMEntry[] public hostedLLMs;

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function buyCredits() external payable {
        // Allow paying for 1 or more bundles at once
        require(msg.value % BUNDLE_PRICE == 0, "Send a multiple of 0.001 0G");
        uint256 bundles = msg.value / BUNDLE_PRICE;
        require(bundles > 0, "No value sent");

        userCredits[msg.sender] += bundles * BUNDLE_AMOUNT;
    }

    function checkUserCredits(address user) public view returns (uint256) {
        return userCredits[user];
    }

    function usePrompt(uint256 llmId) external {
        require(userCredits[msg.sender] > 0, "No prompt credits");
        require(llmId < hostedLLMs.length, "Invalid LLM");

        userCredits[msg.sender] -= 1;
        hostedLLMs[llmId].poolBalance += 1;
    }

    function getHostedLLM(uint256 id) external view returns (HostedLLMEntry memory) {
        require(id < hostedLLMs.length, "Invalid LLM");
        return hostedLLMs[id];
    }

    function withdrawToHosts(uint256 llmId) external onlyOwner {
        require(llmId < hostedLLMs.length, "Invalid LLM");
        HostedLLMEntry storage entry = hostedLLMs[llmId];
        uint256 credits = entry.poolBalance;
        require(credits > 0, "Nothing to withdraw");

        uint256 rem = credits % 2;
        uint256 evenCredits = credits - rem;

        if (rem == 1) {
            entry.poolBalance = 1;
        } else {
            entry.poolBalance = 0;
        }

        uint256 perHostAmount = (evenCredits / 2) * CREDIT_PRICE_WEI;
        uint256 totalAmount = perHostAmount * 2;
        require(address(this).balance >= totalAmount, "Insufficient contract balance");
        (bool s1, ) = payable(entry.host1).call{value: perHostAmount}("");
        require(s1, "Transfer to host1 failed");
        (bool s2, ) = payable(entry.host2).call{value: perHostAmount}("");
        require(s2, "Transfer to host2 failed");
    }
}
