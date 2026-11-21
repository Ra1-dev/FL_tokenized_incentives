// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

import {RewardPool} from "../src/RewardPool.sol";

interface BroadcastVm {
    function startBroadcast() external;
    function stopBroadcast() external;
}

/// @notice Deploy the measured pool policy on Sepolia or a local development chain.
contract Deploy {
    BroadcastVm private constant vm = BroadcastVm(address(uint160(uint256(keccak256("hevm cheat code")))));

    function run() external returns (RewardPool pool) {
        require(block.chainid == 11155111 || block.chainid == 31337, "Sepolia or local chain required");
        // Signing is supplied by the encrypted account selected on the command line.
        vm.startBroadcast();
        pool = new RewardPool(3, 3600, 3600);
        vm.stopBroadcast();
    }
}
