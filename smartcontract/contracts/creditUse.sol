// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract CreditUse {
    uint256 public constant BUNDLE_AMOUNT = 200;
    uint256 public constant BUNDLE_PRICE = 0.001 ether; // 0.001 0G for 200 credits

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

}