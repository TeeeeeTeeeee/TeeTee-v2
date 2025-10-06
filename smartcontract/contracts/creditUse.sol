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
        uint256 totalTimeHost1;   // total tracking time in minutes for host1
        uint256 totalTimeHost2;   // total tracking time in minutes for host2
        uint256 downtimeHost1;
        uint256 downtimeHost2;
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
    }

    function checkUserCredits(address user) public view returns (uint256) {
        return userCredits[user];
    }

    function usePrompt(uint256 llmId, uint256 tokensUsed) external {
        require(tokensUsed > 0, "Zero tokens");
        require(llmId < hostedLLMs.length, "Invalid LLM");
        require(userCredits[msg.sender] >= tokensUsed, "Insufficient credits");
        
        userCredits[msg.sender] -= tokensUsed;
        hostedLLMs[llmId].poolBalance += tokensUsed;
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

    function getHostedLLM(
        uint256 id
    ) external view returns (HostedLLMEntry memory) {
        require(id < hostedLLMs.length, "Invalid LLM");
        return hostedLLMs[id];
    }

    function withdrawToHosts(uint256 llmId) external onlyOwner {
        require(llmId < hostedLLMs.length, "Invalid LLM");
        HostedLLMEntry storage entry = hostedLLMs[llmId];
        uint256 credits = entry.poolBalance;
        require(credits > 0, "Nothing to withdraw");

        entry.poolBalance = 0;

        uint256 host1Reward = 0;
        uint256 host2Reward = 0;

        if (entry.totalTimeHost1 > 0) {
            uint256 host1UptimePercent =
                ((entry.totalTimeHost1 - entry.downtimeHost1) * 1e18) / entry.totalTimeHost1;
            host1Reward = (credits * CREDIT_PRICE_WEI * host1UptimePercent) / (2 * 1e18);
        }

        if (entry.totalTimeHost2 > 0) {
            uint256 host2UptimePercent =
                ((entry.totalTimeHost2 - entry.downtimeHost2) * 1e18) / entry.totalTimeHost2;
            host2Reward = (credits * CREDIT_PRICE_WEI * host2UptimePercent) / (2 * 1e18);
        }

        uint256 totalAmount = host1Reward + host2Reward;
        require(address(this).balance >= totalAmount, "Insufficient contract balance");

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

    // function editRegistedLLM(
    //     uint256 id,
    //     address host1,
    //     address host2,
    //     string calldata shardUrl1,
    //     string calldata shardUrl2,
    //     string calldata modelName
    // ) external onlyOwner {
    //     require(id < hostedLLMs.length, "Invalid LLM");
    //     require(host1 != address(0) && host2 != address(0), "Invalid host");
    //     require(
    //         bytes(shardUrl1).length > 0 && bytes(shardUrl2).length > 0,
    //         "Invalid URL"
    //     );
    //     require(bytes(modelName).length > 0, "Invalid model");

    //     HostedLLMEntry storage e = hostedLLMs[id];
    //     e.host1 = host1;
    //     e.host2 = host2;
    //     e.shardUrl1 = shardUrl1;
    //     e.shardUrl2 = shardUrl2;
    //     e.modelName = modelName;
    // }

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
