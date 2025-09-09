// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Subscription {
    uint256 public constant SUBSCRIPTION_DURATION = 30 days;
    uint256 public constant SUBSCRIPTION_PRICE = 0.005 ether; // 0.001 0G

    mapping(address => uint256) public subscriptions;

    // User subscribes
    function subscribe() external payable {
        require(msg.value == SUBSCRIPTION_PRICE, "Subscription requires exactly 0.005 0G");

        if (block.timestamp < subscriptions[msg.sender]) {
            // extend subscription if already active
            subscriptions[msg.sender] += SUBSCRIPTION_DURATION;
        } else {
            // start new subscription
            subscriptions[msg.sender] = block.timestamp + SUBSCRIPTION_DURATION;
        }

    }
    
    // Check subscription status
    function isSubscribed(address user) public view returns (bool) {
        return subscriptions[user] >= block.timestamp;
    }

}