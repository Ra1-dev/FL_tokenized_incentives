// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/// @notice Experimental reward token; only its immutable pool can change supply.
contract RewardToken is ERC20 {
    address public immutable pool;
    error Unauthorized();

    constructor(address pool_) ERC20("Federated Learning Reward", "FLR") {
        if (pool_ == address(0)) revert Unauthorized();
        pool = pool_;
    }

    function mint(address recipient, uint256 amount) external {
        if (msg.sender != pool) revert Unauthorized();
        _mint(recipient, amount);
    }

    function burn(uint256 amount) external {
        if (msg.sender != pool) revert Unauthorized();
        _burn(msg.sender, amount);
    }
}
