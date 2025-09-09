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

    function usePrompt(uint256 llmId, uint256 tokensUsed) external {
        require(tokensUsed > 0, "Zero tokens");
        require(llmId < hostedLLMs.length, "Invalid LLM");
        require(userCredits[msg.sender] >= tokensUsed, "Insufficient credits");

        userCredits[msg.sender] -= tokensUsed;
        hostedLLMs[llmId].poolBalance += tokensUsed;
    }

    function registerLLM(
        address host1,
        address host2,
        string calldata shardUrl1,
        string calldata shardUrl2,
        string calldata modelName
    ) external onlyOwner returns (uint256 llmId) {
        require(host1 != address(0) && host2 != address(0), "Invalid host");
        require(
            bytes(shardUrl1).length > 0 && bytes(shardUrl2).length > 0,
            "Invalid URL"
        );
        require(bytes(modelName).length > 0, "Invalid model");
        hostedLLMs.push(
            HostedLLMEntry({
                host1: host1,
                host2: host2,
                shardUrl1: shardUrl1,
                shardUrl2: shardUrl2,
                modelName: modelName,
                poolBalance: 0
            })
        );
        llmId = hostedLLMs.length - 1;
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

        uint256 rem = credits % 2;
        uint256 evenCredits = credits - rem;

        if (rem == 1) {
            entry.poolBalance = 1;
        } else {
            entry.poolBalance = 0;
        }

        uint256 perHostAmount = (evenCredits / 2) * CREDIT_PRICE_WEI;
        uint256 totalAmount = perHostAmount * 2;
        require(
            address(this).balance >= totalAmount,
            "Insufficient contract balance"
        );
        (bool s1, ) = payable(entry.host1).call{value: perHostAmount}("");
        require(s1, "Transfer to host1 failed");
        (bool s2, ) = payable(entry.host2).call{value: perHostAmount}("");
        require(s2, "Transfer to host2 failed");
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
}
