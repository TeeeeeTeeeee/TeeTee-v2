// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract CreditUse {
    uint256 public constant BUNDLE_AMOUNT = 200;
    uint256 public constant BUNDLE_PRICE = 0.001 ether; // 0.001 0G for 200 credits
    uint256 public constant CREDIT_PRICE_WEI = BUNDLE_PRICE / BUNDLE_AMOUNT;

    address public owner;

    mapping(address => uint256) public userCredits;
    uint256 public generalPool; // Pool of funds before being allocated to specific LLMs

    struct HostedLLMEntry {
        address host1;
        address host2;
        string shardUrl1;
        string shardUrl2;
        string modelName;
        uint256 poolBalance;      // Pool balance in wei (0G) allocated to this LLM
        uint256 totalTimeHost1;   // total tracking time in minutes for host1
        uint256 totalTimeHost2;   // total tracking time in minutes for host2
        uint256 downtimeHost1;
        uint256 downtimeHost2;
        uint256 lastWithdrawHost1; // Last pool balance when host1 withdrew
        uint256 lastWithdrawHost2; // Last pool balance when host2 withdrew
        bool isComplete;          // true if both hosts are registered
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
        generalPool += msg.value; // Add money to general pool
    }

    function checkUserCredits(address user) public view returns (uint256) {
        return userCredits[user];
    }

    function getGeneralPool() public view returns (uint256) {
        return generalPool;
    }

    // Internal helper function to calculate withdrawable amount for a host
    function _calculateHostReward(uint256 llmId, uint8 hostNumber) internal view returns (uint256) {
        require(llmId < hostedLLMs.length, "Invalid LLM");
        require(hostNumber == 1 || hostNumber == 2, "Invalid host number");
        
        HostedLLMEntry storage entry = hostedLLMs[llmId];
        uint256 reward = 0;
        
        if (hostNumber == 1) {
            uint256 newEarnings = entry.poolBalance - entry.lastWithdrawHost1;
            if (newEarnings > 0 && entry.totalTimeHost1 > 0) {
                uint256 uptimePercent = ((entry.totalTimeHost1 - entry.downtimeHost1) * 1e18) / entry.totalTimeHost1;
                reward = (newEarnings * uptimePercent) / (2 * 1e18);
            }
        } else {
            uint256 newEarnings = entry.poolBalance - entry.lastWithdrawHost2;
            if (newEarnings > 0 && entry.totalTimeHost2 > 0) {
                uint256 uptimePercent = ((entry.totalTimeHost2 - entry.downtimeHost2) * 1e18) / entry.totalTimeHost2;
                reward = (newEarnings * uptimePercent) / (2 * 1e18);
            }
        }
        
        return reward;
    }

    // Check how much a host can withdraw
    function checkWithdrawable(uint256 llmId, address host) public view returns (uint256) {
        require(llmId < hostedLLMs.length, "Invalid LLM");
        HostedLLMEntry storage entry = hostedLLMs[llmId];
        
        if (host == entry.host1) {
            return _calculateHostReward(llmId, 1);
        } else if (host == entry.host2) {
            return _calculateHostReward(llmId, 2);
        } else {
            revert("Not a host of this LLM");
        }
    }

    function usePrompt(uint256 llmId, uint256 tokensUsed) external {
        require(tokensUsed > 0, "Zero tokens");
        require(llmId < hostedLLMs.length, "Invalid LLM");
        require(userCredits[msg.sender] >= tokensUsed, "Insufficient credits");
        
        // Calculate the actual money value for these credits
        uint256 moneyAmount = tokensUsed * CREDIT_PRICE_WEI;
        require(generalPool >= moneyAmount, "Insufficient general pool");
        
        userCredits[msg.sender] -= tokensUsed;
        generalPool -= moneyAmount; // Deduct from general pool
        hostedLLMs[llmId].poolBalance += moneyAmount; // Add money (not credits) to LLM pool
    }

    // Register or update LLM - if field is 0/empty, it keeps existing value
    function registerLLM(
        uint256 llmId,              // Pass hostedLLMs.length to create new, or existing ID to update
        address host1,
        address host2,
        string calldata shardUrl1,
        string calldata shardUrl2,
        string calldata modelName,
        uint256 totalTimeHost1,
        uint256 totalTimeHost2
    ) external returns (uint256) {
        // Create new entry if llmId >= array length
        if (llmId >= hostedLLMs.length) {
            require(bytes(modelName).length > 0, "Model name required for new entry");
            
            hostedLLMs.push(
                HostedLLMEntry({
                    host1: host1,
                    host2: host2,
                    shardUrl1: shardUrl1,
                    shardUrl2: shardUrl2,
                    modelName: modelName,
                    poolBalance: 0,
                    totalTimeHost1: totalTimeHost1,
                    totalTimeHost2: totalTimeHost2,
                    downtimeHost1: 0,
                    downtimeHost2: 0,
                    lastWithdrawHost1: 0,
                    lastWithdrawHost2: 0,
                    isComplete: (host1 != address(0) && host2 != address(0))
                })
            );
            return hostedLLMs.length - 1;
        }
        
        // Update existing entry - only update non-zero/non-empty fields
        HostedLLMEntry storage entry = hostedLLMs[llmId];
        
        if (host1 != address(0)) {
            entry.host1 = host1;
        }
        if (host2 != address(0)) {
            entry.host2 = host2;
        }
        if (bytes(shardUrl1).length > 0) {
            entry.shardUrl1 = shardUrl1;
        }
        if (bytes(shardUrl2).length > 0) {
            entry.shardUrl2 = shardUrl2;
        }
        if (bytes(modelName).length > 0) {
            entry.modelName = modelName;
        }
        if (totalTimeHost1 > 0) {
            entry.totalTimeHost1 = totalTimeHost1;
        }
        if (totalTimeHost2 > 0) {
            entry.totalTimeHost2 = totalTimeHost2;
        }
        
        // Update completion status
        entry.isComplete = (entry.host1 != address(0) && entry.host2 != address(0));
        
        return llmId;
    }
    
    function reportDowntime(uint256 llmId, uint256 minutesDownHost1, uint256 minutesDownHost2) external onlyOwner {
        require(llmId < hostedLLMs.length, "Invalid LLM");
        HostedLLMEntry storage entry = hostedLLMs[llmId];

        if (minutesDownHost1 > 0) {
            entry.downtimeHost1 += minutesDownHost1;
            if (entry.downtimeHost1 > entry.totalTimeHost1) {
                entry.downtimeHost1 = entry.totalTimeHost1; 
            }
        }

        if (minutesDownHost2 > 0) {
            entry.downtimeHost2 += minutesDownHost2;
            if (entry.downtimeHost2 > entry.totalTimeHost2) {
                entry.downtimeHost2 = entry.totalTimeHost2;
            }
        }
    }

    // Stop/unregister a host from an LLM - pays out remaining rewards before stopping
    function stopLLM(uint256 llmId, uint8 hostNumber) external {
        require(llmId < hostedLLMs.length, "Invalid LLM");
        require(hostNumber == 1 || hostNumber == 2, "Invalid host number (use 1 or 2)");
        
        HostedLLMEntry storage entry = hostedLLMs[llmId];
        
        // Check authorization
        address hostAddress = (hostNumber == 1) ? entry.host1 : entry.host2;
        require(msg.sender == hostAddress || msg.sender == owner, "Not authorized");
        
        // Pay out remaining rewards before stopping
        uint256 reward = _calculateHostReward(llmId, hostNumber);
        if (reward > 0 && address(this).balance >= reward) {
            (bool success, ) = payable(hostAddress).call{value: reward}("");
            require(success, "Final payment failed");
        }
        
        // Clear host data
        if (hostNumber == 1) {
            entry.host1 = address(0);
            entry.shardUrl1 = "";
            entry.totalTimeHost1 = 0;
            entry.downtimeHost1 = 0;
            entry.lastWithdrawHost1 = 0;
        } else {
            entry.host2 = address(0);
            entry.shardUrl2 = "";
            entry.totalTimeHost2 = 0;
            entry.downtimeHost2 = 0;
            entry.lastWithdrawHost2 = 0;
        }
        
        // Update completion status - will be false if either host is now address(0)
        // This will make it appear in the getIncompleteLLMs() list
        entry.isComplete = (entry.host1 != address(0) && entry.host2 != address(0));
    }

    function getHostedLLM(
        uint256 id
    ) external view returns (HostedLLMEntry memory) {
        require(id < hostedLLMs.length, "Invalid LLM");
        return hostedLLMs[id];
    }

    // Host can withdraw their own portion
    function withdraw(uint256 llmId) external {
        require(llmId < hostedLLMs.length, "Invalid LLM");
        HostedLLMEntry storage entry = hostedLLMs[llmId];
        
        uint8 hostNumber;
        if (msg.sender == entry.host1) {
            hostNumber = 1;
        } else if (msg.sender == entry.host2) {
            hostNumber = 2;
        } else {
            revert("Not a host of this LLM");
        }
        
        uint256 reward = _calculateHostReward(llmId, hostNumber);
        require(reward > 0, "Nothing to withdraw");
        require(address(this).balance >= reward, "Insufficient contract balance");
        
        // Update last withdraw marker
        if (hostNumber == 1) {
            entry.lastWithdrawHost1 = entry.poolBalance;
        } else {
            entry.lastWithdrawHost2 = entry.poolBalance;
        }
        
        // Pay the host
        (bool success, ) = payable(msg.sender).call{value: reward}("");
        require(success, "Transfer failed");
    }
    
    // Owner can still withdraw for both hosts at once (legacy function)
    function withdrawToHosts(uint256 llmId) external onlyOwner {
        require(llmId < hostedLLMs.length, "Invalid LLM");
        HostedLLMEntry storage entry = hostedLLMs[llmId];
        require(entry.poolBalance > 0, "Nothing to withdraw");

        uint256 host1Reward = _calculateHostReward(llmId, 1);
        uint256 host2Reward = _calculateHostReward(llmId, 2);

        uint256 totalAmount = host1Reward + host2Reward;
        require(totalAmount > 0, "Nothing to withdraw");
        require(address(this).balance >= totalAmount, "Insufficient contract balance");

        entry.lastWithdrawHost1 = entry.poolBalance;
        entry.lastWithdrawHost2 = entry.poolBalance;

        // Pay hosts
        if (host1Reward > 0) {
            (bool s1, ) = payable(entry.host1).call{value: host1Reward}("");
            require(s1, "Transfer to host1 failed");
        }

        if (host2Reward > 0) {
            (bool s2, ) = payable(entry.host2).call{value: host2Reward}("");
            require(s2, "Transfer to host2 failed");
        }
    }

    function editRegistedLLM(
        uint256 id,
        address host1,
        address host2,
        string calldata shardUrl1,
        string calldata shardUrl2,
        string calldata modelName
    ) external onlyOwner {
        require(id < hostedLLMs.length, "Invalid LLM");

        HostedLLMEntry storage e = hostedLLMs[id];

        if (host1 != address(0)) {
            e.host1 = host1;
        }
        if (host2 != address(0)) {
            e.host2 = host2;
            // If we're adding host2 and it wasn't complete, mark as complete
            if (!e.isComplete && e.host1 != address(0)) {
                e.isComplete = true;
            }
        }
        if (bytes(shardUrl1).length != 0) {
            e.shardUrl1 = shardUrl1;
        }
        if (bytes(shardUrl2).length != 0) {
            e.shardUrl2 = shardUrl2;
        }
        if (bytes(modelName).length != 0) {
            e.modelName = modelName;
        }
    }

    // Get all incomplete LLMs (slots waiting for second host)
    function getIncompleteLLMs() external view returns (uint256[] memory) {
        uint256 count = 0;
        
        // Count incomplete LLMs
        for (uint256 i = 0; i < hostedLLMs.length; i++) {
            if (!hostedLLMs[i].isComplete) {
                count++;
            }
        }
        
        // Create array of incomplete LLM IDs
        uint256[] memory incompleteLLMs = new uint256[](count);
        uint256 index = 0;
        
        for (uint256 i = 0; i < hostedLLMs.length; i++) {
            if (!hostedLLMs[i].isComplete) {
                incompleteLLMs[index] = i;
                index++;
            }
        }
        
        return incompleteLLMs;
    }

    // Get total count of hosted LLMs
    function getTotalLLMs() external view returns (uint256) {
        return hostedLLMs.length;
    }
}
