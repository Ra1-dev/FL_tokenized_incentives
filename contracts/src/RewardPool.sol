// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

import {RewardToken} from "./RewardToken.sol";

/// @notice Coordinator-posted scores with commit/reveal and exact integer settlement.
/// @dev Scores are trusted inputs. Commitments bind a digest, not update quality.
contract RewardPool {
    uint256 public constant SCALE = 1e18;
    uint256 public constant ROUND_POOL = 1000e18;
    uint256 public constant INITIAL_STAKE = 100e18;
    uint256 public constant SLASH_DENOMINATOR = 10;
    address public immutable coordinator;
    RewardToken public immutable token;
    uint256 public immutable vestingDelay;
    uint256 public immutable commitWindow;
    uint256 public immutable revealWindow;
    uint256 public currentRound;
    uint256 public lastSettledRound;
    uint256 public totalUnminted;

    struct Round {
        uint256 commitDeadline;
        uint256 revealDeadline;
        uint256 remainder;
        bool posted;
        bool settled;
        address[] sampled;
    }
    struct Tranche {
        uint256 amount;
        uint256 unlockRound;
    }
    struct Score {
        uint256 share;
        bool slash;
    }

    address[] private participants;
    mapping(address => bool) public endowed;
    mapping(address => bool) public registered;
    mapping(address => uint256) public stake;
    mapping(address => uint256) public claimable;
    mapping(address => mapping(uint256 => Tranche)) public vestingTranche;
    mapping(uint256 => Round) private rounds;
    mapping(uint256 => mapping(address => bool)) public sampled;
    mapping(uint256 => mapping(address => bytes32)) public commitments;
    mapping(uint256 => mapping(address => bool)) public revealed;
    mapping(uint256 => mapping(address => bytes32)) public updateDigests;
    mapping(uint256 => mapping(address => Score)) public scores;

    error Unauthorized();
    error InvalidState();
    error InvalidInput();
    error Deadline();
    error InvalidCommitment();
    error ExcessShares();
    error NothingToClaim();

    event Endowed(address indexed client, uint256 amount);
    event Registered(address indexed client, uint256 stakeAmount);
    event RoundOpened(uint256 indexed round, address[] clients, uint256 commitDeadline, uint256 revealDeadline);
    event Committed(uint256 indexed round, address indexed client, bytes32 commitment);
    event Revealed(uint256 indexed round, address indexed client, bytes32 updateDigest);
    event ScoresPosted(uint256 indexed round, address[] clients, uint256[] shares, bool[] slashFlags);
    event Matured(address indexed client, uint256 indexed round, uint256 amount);
    event Slashed(address indexed client, uint256 indexed round, uint256 stakeBurned, uint256 rewardsBurned);
    event RewardVested(address indexed client, uint256 indexed round, uint256 amount, uint256 unlockRound);
    event Settled(uint256 indexed round, uint256 minted, uint256 unminted);
    event Claimed(address indexed client, uint256 amount);

    modifier onlyCoordinator() {
        if (msg.sender != coordinator) revert Unauthorized();
        _;
    }

    constructor(uint256 vestingDelay_, uint256 commitWindow_, uint256 revealWindow_) {
        if (vestingDelay_ > 50 || commitWindow_ == 0 || revealWindow_ == 0) {
            revert InvalidInput();
        }
        coordinator = msg.sender;
        vestingDelay = vestingDelay_;
        commitWindow = commitWindow_;
        revealWindow = revealWindow_;
        token = new RewardToken(address(this));
    }

    function grantInitial(address client) external onlyCoordinator {
        if (currentRound != 0 || endowed[client] || client == address(0)) revert InvalidState();
        endowed[client] = true;
        emit Endowed(client, INITIAL_STAKE);
        token.mint(client, INITIAL_STAKE);
    }

    function register() external {
        if (currentRound != 0 || registered[msg.sender] || !endowed[msg.sender]) revert InvalidState();
        registered[msg.sender] = true;
        participants.push(msg.sender);
        stake[msg.sender] = INITIAL_STAKE;
        emit Registered(msg.sender, INITIAL_STAKE);
        if (!token.transferFrom(msg.sender, address(this), INITIAL_STAKE)) revert InvalidState();
    }

    function openRound(uint256 round, address[] calldata ids) external onlyCoordinator {
        if (round != lastSettledRound + 1 || currentRound != lastSettledRound) revert InvalidState();
        if (ids.length == 0) revert InvalidInput();
        Round storage r = rounds[round];
        for (uint256 i; i < ids.length; ++i) {
            address client = ids[i];
            if (!registered[client] || sampled[round][client]) revert InvalidInput();
            sampled[round][client] = true;
            r.sampled.push(client);
        }
        currentRound = round;
        r.commitDeadline = block.timestamp + commitWindow;
        r.revealDeadline = r.commitDeadline + revealWindow;
        emit RoundOpened(round, ids, r.commitDeadline, r.revealDeadline);
    }

    function commit(uint256 round, bytes32 commitment) external {
        Round storage r = rounds[round];
        if (round != currentRound || !sampled[round][msg.sender] || commitments[round][msg.sender] != bytes32(0)) {
            revert InvalidState();
        }
        if (block.timestamp >= r.commitDeadline) revert Deadline();
        if (commitment == bytes32(0)) revert InvalidCommitment();
        commitments[round][msg.sender] = commitment;
        emit Committed(round, msg.sender, commitment);
    }

    function reveal(uint256 round, bytes32 updateDigest, bytes32 salt) external {
        Round storage r = rounds[round];
        if (round != currentRound || !sampled[round][msg.sender] || revealed[round][msg.sender]) revert InvalidState();
        if (block.timestamp < r.commitDeadline || block.timestamp >= r.revealDeadline) revert Deadline();
        if (commitments[round][msg.sender] != keccak256(abi.encode(updateDigest, salt, round, msg.sender))) {
            revert InvalidCommitment();
        }
        revealed[round][msg.sender] = true;
        updateDigests[round][msg.sender] = updateDigest;
        emit Revealed(round, msg.sender, updateDigest);
    }

    function postScores(uint256 round, address[] calldata ids, uint256[] calldata shares1e18, bool[] calldata slashFlags)
        external onlyCoordinator
    {
        Round storage r = rounds[round];
        if (round != currentRound || r.posted || r.settled || r.sampled.length == 0) revert InvalidState();
        if (block.timestamp < r.revealDeadline) revert Deadline();
        if (ids.length != r.sampled.length || ids.length != shares1e18.length || ids.length != slashFlags.length) {
            revert InvalidInput();
        }
        uint256 total = 0;
        // Use the recorded sample order so duplicate ids cannot receive multiple entries.
        for (uint256 i; i < ids.length; ++i) {
            if (ids[i] != r.sampled[i] || (!revealed[round][ids[i]] && shares1e18[i] != 0)) revert InvalidInput();
            if (shares1e18[i] > SCALE) revert ExcessShares();
            total += shares1e18[i];
            if (total > SCALE) revert ExcessShares();
            scores[round][ids[i]] = Score(shares1e18[i], slashFlags[i]);
        }
        r.posted = true;
        emit ScoresPosted(round, ids, shares1e18, slashFlags);
    }

    function settle(uint256 round) external {
        Round storage r = rounds[round];
        if (round != currentRound || !r.posted || r.settled) revert InvalidState();
        r.settled = true;
        uint256 slots = vestingDelay + 1;
        // Maturity precedes slashing, including for clients absent from this sample.
        uint256 participantCount = participants.length;
        for (uint256 i; i < participantCount; ++i) {
            address client = participants[i];
            for (uint256 j; j < slots; ++j) {
                Tranche storage tranche = vestingTranche[client][j];
                if (tranche.amount != 0 && tranche.unlockRound <= round) {
                    uint256 amount = tranche.amount;
                    delete vestingTranche[client][j];
                    claimable[client] += amount;
                    emit Matured(client, round, amount);
                }
            }
        }
        uint256 minted = 0;
        uint256 totalBurned = 0;
        for (uint256 i; i < r.sampled.length; ++i) {
            address client = r.sampled[i];
            Score memory score = scores[round][client];
            if (score.slash) {
                uint256 stakeBurned = stake[client] / SLASH_DENOMINATOR;
                stake[client] -= stakeBurned;
                uint256 rewardsBurned = 0;
                for (uint256 j; j < slots; ++j) {
                    Tranche storage tranche = vestingTranche[client][j];
                    uint256 burned = tranche.amount / SLASH_DENOMINATOR;
                    tranche.amount -= burned;
                    rewardsBurned += burned;
                }
                totalBurned += stakeBurned + rewardsBurned;
                emit Slashed(client, round, stakeBurned, rewardsBurned);
            }
            uint256 amount = ROUND_POOL * score.share / SCALE;
            if (amount != 0) {
                if (vestingDelay == 0) {
                    claimable[client] += amount;
                    emit Matured(client, round, amount);
                } else {
                    uint256 slot = round % slots;
                    if (vestingTranche[client][slot].amount != 0) revert InvalidState();
                    vestingTranche[client][slot] = Tranche(amount, round + vestingDelay);
                }
                minted += amount;
                emit RewardVested(client, round, amount, round + vestingDelay);
            }
        }
        r.remainder = ROUND_POOL - minted;
        totalUnminted += r.remainder;
        lastSettledRound = round;
        emit Settled(round, minted, r.remainder);
        // Complete every ledger effect before interacting with the immutable token.
        if (totalBurned != 0) token.burn(totalBurned);
        if (minted != 0) token.mint(address(this), minted);
    }

    function claim() external {
        uint256 amount = claimable[msg.sender];
        if (amount == 0) revert NothingToClaim();
        claimable[msg.sender] = 0;
        emit Claimed(msg.sender, amount);
        if (!token.transfer(msg.sender, amount)) revert InvalidState();
    }

    function roundInfo(uint256 round) external view returns (uint256, uint256, uint256, bool, bool) {
        Round storage r = rounds[round];
        return (r.commitDeadline, r.revealDeadline, r.remainder, r.posted, r.settled);
    }

    function registeredClients() external view returns (address[] memory) {
        return participants;
    }

    function sampledClients(uint256 round) external view returns (address[] memory) {
        return rounds[round].sampled;
    }

    function unvested(address client) external view returns (uint256 amount) {
        for (uint256 i; i <= vestingDelay; ++i) amount += vestingTranche[client][i].amount;
    }
}
