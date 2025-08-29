// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Subscription {
    uint256 public constant SUBSCRIPTION_DURATION = 30 days;
    uint256 public constant SUBSCRIPTION_PRICE = 0.001 ether; // 0.001 0G
    uint256 public constant PROMPT_COST = 0.00001 ether; // 0.00001 0G per prompt

    mapping(address => uint256) public subscriptions;
    mapping(address => uint256) public promptCredits; // Track user's prompt credits

    // User subscribes
    function subscribe() external payable {
        require(msg.value == SUBSCRIPTION_PRICE, "Subscription requires exactly 0.001 0G");

        if (block.timestamp < subscriptions[msg.sender]) {
            // extend subscription if already active
            subscriptions[msg.sender] += SUBSCRIPTION_DURATION;
        } else {
            // start new subscription
            subscriptions[msg.sender] = block.timestamp + SUBSCRIPTION_DURATION;
        }
        // Add prompt credits when subscribing (100 prompts for 0.001 0G)
        promptCredits[msg.sender] += 100;
    }

    // Check subscription status
    function isSubscribed(address user) public view returns (bool) {
        return subscriptions[user] >= block.timestamp;
    }

    // Use a prompt - deduct credits
    function usePrompt() external {
        require(isSubscribed(msg.sender), "No active subscription");
        require(promptCredits[msg.sender] > 0, "No prompt credits available");
        
        promptCredits[msg.sender] -= 1;
    }

    // Check remaining prompt credits
    function getPromptCredits(address user) public view returns (uint256) {
        return promptCredits[user];
    }

    // Refund function
    function refund() external {
        require(isSubscribed(msg.sender), "No active subscription to refund");
        require(address(this).balance >= SUBSCRIPTION_PRICE, "Contract has insufficient funds");

        // Reset subscription and credits
        subscriptions[msg.sender] = 0;
        promptCredits[msg.sender] = 0;

        // Send refund
        payable(msg.sender).transfer(SUBSCRIPTION_PRICE);
    }
}