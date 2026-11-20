// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

import {RewardPool} from "../src/RewardPool.sol";
import {RewardToken} from "../src/RewardToken.sol";

interface Vm {
    function prank(address) external;
    function warp(uint256) external;
    function expectRevert(bytes4) external;
}

contract RewardPoolTest {
    Vm private constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
    RewardPool private pool;
    RewardToken private token;
    address private constant ALICE = address(0xA11CE);
    address private constant BOB = address(0xB0B);
    bytes32 private constant DIGEST = keccak256("reported update");
    bytes32 private constant SALT = keccak256("test salt");

    function setUp() public {
        pool = new RewardPool(3, 10, 10);
        token = pool.token();
        _register(ALICE);
        _register(BOB);
    }

    function _register(address client) private {
        pool.grantInitial(client);
        vm.prank(client);
        token.approve(address(pool), 100e18);
        vm.prank(client);
        pool.register();
    }

    function _ids() private pure returns (address[] memory ids) {
        ids = new address[](2);
        ids[0] = ALICE;
        ids[1] = BOB;
    }

    function _openAndReveal(uint256 round, address[] memory ids) private {
        pool.openRound(round, ids);
        for (uint256 i; i < ids.length; ++i) {
            vm.prank(ids[i]);
            pool.commit(round, keccak256(abi.encode(DIGEST, SALT, round, ids[i])));
        }
        (uint256 commitDeadline, uint256 revealDeadline,,,) = pool.roundInfo(round);
        vm.warp(commitDeadline);
        for (uint256 i; i < ids.length; ++i) {
            vm.prank(ids[i]);
            pool.reveal(round, DIGEST, SALT);
        }
        vm.warp(revealDeadline);
    }

    function _round(uint256 round, uint256 a, uint256 b, bool slashAlice) private {
        address[] memory ids = _ids();
        _openAndReveal(round, ids);
        uint256[] memory shares = new uint256[](2);
        shares[0] = a;
        shares[1] = b;
        bool[] memory flags = new bool[](2);
        flags[0] = slashAlice;
        pool.postScores(round, ids, shares, flags);
        pool.settle(round);
    }

    function testFuzz_SettlementConservesBudget(uint64 rawA, uint64 rawB) public {
        uint256 a = uint256(rawA) % (1e18 + 1);
        uint256 b = uint256(rawB) % (1e18 - a + 1);
        _round(1, a, b, false);
        uint256 minted = 1000 * (a + b);
        require(pool.unvested(ALICE) == 1000 * a, "alice amount");
        require(pool.unvested(BOB) == 1000 * b, "bob amount");
        require(pool.totalUnminted() == 1000e18 - minted, "budget conservation");
        require(token.totalSupply() == 200e18 + minted, "supply conservation");
        require(token.balanceOf(address(pool)) == token.totalSupply(), "pool collateral");
    }

    struct AccountOracle {
        uint256 stakeAmount;
        uint256 ready;
        uint256 paid;
        uint256 matured;
        uint256[] byIssuance;
    }

    function _advanceOracle(AccountOracle memory account, uint256 round, uint256 delay, uint256 reward, bool slash)
        private pure returns (uint256 burned)
    {
        // Keep a full issuance history, independently of the contract's ring slots.
        if (round > delay) {
            uint256 amount = account.byIssuance[round - delay];
            account.ready += amount;
            account.matured += amount;
            account.byIssuance[round - delay] = 0;
        }
        if (slash) {
            burned = account.stakeAmount / 10;
            account.stakeAmount -= burned;
            for (uint256 issue = 1; issue < round; ++issue) {
                uint256 loss = account.byIssuance[issue] / 10;
                account.byIssuance[issue] -= loss;
                burned += loss;
            }
        }
        if (delay == 0) {
            account.ready += reward;
            account.matured += reward;
        } else {
            account.byIssuance[round] = reward;
        }
    }

    function _assertAccount(address client, AccountOracle memory account) private view returns (uint256 liability) {
        uint256 locked;
        for (uint256 issue = 1; issue < account.byIssuance.length; ++issue) locked += account.byIssuance[issue];
        require(pool.stake(client) == account.stakeAmount, "stake history");
        require(pool.unvested(client) == locked && pool.claimable(client) == account.ready, "vesting history");
        require(token.balanceOf(client) == account.paid, "claimed balance");
        require(account.ready + account.paid == account.matured, "matured rewards never lost");
        return account.stakeAmount + account.ready + locked;
    }

    function testFuzz_StatefulAccountingInvariants(uint256 entropy, uint8 rawDelay, uint8 extraRounds) public {
        uint256 delay = uint256(rawDelay) % 51;
        uint256 count = delay + 4 + uint256(extraRounds) % 4;
        pool = new RewardPool(delay, 10, 10);
        token = pool.token();
        _register(ALICE);
        _register(BOB);
        address[] memory ids = _ids();
        AccountOracle[] memory accounts = new AccountOracle[](2);
        for (uint256 i; i < 2; ++i) {
            accounts[i].stakeAmount = 100e18;
            accounts[i].byIssuance = new uint256[](count + 1);
        }
        uint256 minted;
        uint256 burned;
        for (uint256 round = 1; round <= count; ++round) {
            entropy = uint256(keccak256(abi.encode(entropy, round)));
            uint256[] memory shares = new uint256[](2);
            shares[0] = entropy % (1e18 + 1);
            shares[1] = (entropy >> 64) % (1e18 - shares[0] + 1);
            bool[] memory flags = new bool[](2);
            flags[0] = entropy & 1 != 0;
            flags[1] = entropy & 2 != 0;
            _openAndReveal(round, ids);
            pool.postScores(round, ids, shares, flags);
            pool.settle(round);
            for (uint256 i; i < 2; ++i) {
                uint256 beforeMatured = accounts[i].matured;
                uint256 reward = 1000 * shares[i];
                minted += reward;
                burned += _advanceOracle(accounts[i], round, delay, reward, flags[i]);
                require(accounts[i].matured >= beforeMatured, "maturity monotonicity");
                if (accounts[i].ready != 0 && (entropy >> (128 + i)) & 1 != 0) {
                    vm.prank(ids[i]);
                    pool.claim();
                    accounts[i].paid += accounts[i].ready;
                    accounts[i].ready = 0;
                }
            }
            uint256 liabilities = _assertAccount(ALICE, accounts[0]) + _assertAccount(BOB, accounts[1]);
            require(token.balanceOf(address(pool)) == liabilities, "all liabilities collateralized");
            require(token.totalSupply() == 200e18 + minted - burned, "mint burn conservation");
            require(pool.totalUnminted() + minted == round * 1000e18, "cumulative budget");
        }
    }

    function testFuzz_CommitmentRejectsOtherRoundAndSender(bytes32 digest, bytes32 salt) public {
        _round(1, 0, 0, false);
        pool.openRound(2, _ids());
        vm.prank(ALICE);
        pool.commit(2, keccak256(abi.encode(digest, salt, uint256(1), ALICE)));
        vm.prank(BOB);
        pool.commit(2, keccak256(abi.encode(digest, salt, uint256(2), ALICE)));
        (uint256 deadline,,,,) = pool.roundInfo(2);
        vm.warp(deadline);
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.InvalidCommitment.selector);
        pool.reveal(2, digest, salt);
        vm.prank(BOB);
        vm.expectRevert(RewardPool.InvalidCommitment.selector);
        pool.reveal(2, digest, salt);
        require(!pool.revealed(2, ALICE) && !pool.revealed(2, BOB), "invalid commitments never reveal");
    }

    function testMaturityBeforeSlashAndClaim() public {
        _round(1, 1e18, 0, false);
        _round(2, 1e18, 0, false);
        _round(3, 1e18, 0, false);
        _round(4, 1e18, 0, true);
        require(pool.claimable(ALICE) == 1000e18, "mature amount protected");
        require(pool.unvested(ALICE) == 2800e18, "old tranches slashed before new reward");
        require(pool.stake(ALICE) == 90e18, "stake slash");
        require(token.balanceOf(address(pool)) == 3990e18, "burn conservation");
        vm.prank(ALICE);
        pool.claim();
        require(token.balanceOf(ALICE) == 1000e18 && pool.claimable(ALICE) == 0, "claim transfer");
        require(token.balanceOf(address(pool)) == 2990e18, "remaining liabilities");
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.NothingToClaim.selector);
        pool.claim();
    }

    function testAbsentClientMatures() public {
        _round(1, 1e18, 0, false);
        _round(2, 0, 0, false);
        _round(3, 0, 0, false);
        address[] memory ids = new address[](1);
        ids[0] = BOB;
        _openAndReveal(4, ids);
        pool.postScores(4, ids, new uint256[](1), new bool[](1));
        pool.settle(4);
        require(pool.claimable(ALICE) == 1000e18 && pool.unvested(ALICE) == 0, "absent maturity");
        require(pool.totalUnminted() == 3000e18, "empty rounds unminted");
    }

    function testDisabledVestingIsImmediatelyClaimable() public {
        pool = new RewardPool(0, 10, 10);
        token = pool.token();
        _register(ALICE);
        _register(BOB);
        _round(1, 1e18, 0, false);
        require(pool.claimable(ALICE) == 1000e18 && pool.unvested(ALICE) == 0, "immediate maturity");
        _round(2, 0, 0, true);
        require(pool.claimable(ALICE) == 1000e18, "claimable rewards protected");
        require(pool.stake(ALICE) == 90e18, "stake still slashable");
        vm.prank(ALICE);
        pool.claim();
        require(token.balanceOf(ALICE) == 1000e18, "immediate claim");
    }

    function testOnlyPoolChangesTokenSupply() public {
        vm.expectRevert(RewardToken.Unauthorized.selector);
        token.mint(ALICE, 1);
        vm.expectRevert(RewardToken.Unauthorized.selector);
        token.burn(1);
        require(token.decimals() == 18, "base units");
    }

    function testSetupAuthorizationAndNoDuplicateEndowment() public {
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.Unauthorized.selector);
        pool.grantInitial(address(3));
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.grantInitial(ALICE);
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.register();
        pool.openRound(1, _ids());
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.grantInitial(address(3));
    }

    function testCommitRevealBoundariesAndSenderBinding() public {
        pool.openRound(1, _ids());
        vm.prank(ALICE);
        pool.commit(1, keccak256(abi.encode(DIGEST, SALT, uint256(1), ALICE)));
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.commit(1, keccak256("replacement"));
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.Deadline.selector);
        pool.reveal(1, DIGEST, SALT);
        (uint256 commitDeadline, uint256 revealDeadline,,,) = pool.roundInfo(1);
        vm.warp(commitDeadline);
        vm.prank(BOB);
        vm.expectRevert(RewardPool.Deadline.selector);
        pool.commit(1, keccak256("late"));
        vm.prank(BOB);
        vm.expectRevert(RewardPool.InvalidCommitment.selector);
        pool.reveal(1, DIGEST, SALT);
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.InvalidCommitment.selector);
        pool.reveal(1, DIGEST, bytes32(0));
        vm.prank(ALICE);
        pool.reveal(1, DIGEST, SALT);
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.reveal(1, DIGEST, SALT);
        vm.warp(revealDeadline);
        vm.prank(BOB);
        vm.expectRevert(RewardPool.Deadline.selector);
        pool.reveal(1, DIGEST, SALT);
    }

    function testScoresRejectExcessSharesAndDuplicateIds() public {
        address[] memory ids = _ids();
        _openAndReveal(1, ids);
        uint256[] memory shares = new uint256[](2);
        shares[0] = 1e18;
        shares[1] = 1;
        bool[] memory flags = new bool[](2);
        vm.expectRevert(RewardPool.ExcessShares.selector);
        pool.postScores(1, ids, shares, flags);
        shares[1] = 0;
        ids[1] = ALICE;
        vm.expectRevert(RewardPool.InvalidInput.selector);
        pool.postScores(1, ids, shares, flags);
        ids[1] = BOB;
        vm.prank(ALICE);
        vm.expectRevert(RewardPool.Unauthorized.selector);
        pool.postScores(1, ids, shares, flags);
        pool.postScores(1, ids, shares, flags);
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.postScores(1, ids, shares, flags);
        pool.settle(1);
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.settle(1);
    }

    function testMissingRevealCannotEarn() public {
        address[] memory ids = _ids();
        pool.openRound(1, ids);
        uint256[] memory shares = new uint256[](2);
        shares[0] = 1;
        bool[] memory flags = new bool[](2);
        vm.expectRevert(RewardPool.Deadline.selector);
        pool.postScores(1, ids, shares, flags);
        (, uint256 deadline,,,) = pool.roundInfo(1);
        vm.warp(deadline);
        vm.expectRevert(RewardPool.InvalidInput.selector);
        pool.postScores(1, ids, shares, flags);
        shares[0] = 0;
        pool.postScores(1, ids, shares, flags);
        pool.settle(1);
        require(pool.totalUnminted() == 1000e18, "unrevealed round unpaid");
    }

    function testInvalidSamplingAndRoundOrder() public {
        address[] memory ids = _ids();
        ids[1] = ALICE;
        vm.expectRevert(RewardPool.InvalidInput.selector);
        pool.openRound(1, ids);
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.openRound(2, _ids());
        pool.openRound(1, _ids());
        vm.expectRevert(RewardPool.InvalidState.selector);
        pool.openRound(2, _ids());
    }
}
