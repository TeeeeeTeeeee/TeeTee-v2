// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Credit {
    uint256 public constant BUNDLE_AMOUNT = 200;
    uint256 public constant BUNDLE_PRICE = 0.001 ether; // 0.001 0G for 200 credits

    mapping(address => uint256) public promptCredits;

    function buy200() external payable {
        require(msg.value == BUNDLE_PRICE, "Price: 0.001 0G");
        promptCredits[msg.sender] += BUNDLE_AMOUNT;
    }

    function getPromptCredits(address user) public view returns (uint256) {
        return promptCredits[user];
    }

    function usePrompt() external {
        require(promptCredits[msg.sender] > 0, "No prompt credits");
        promptCredits[msg.sender] -= 1;
    }
}