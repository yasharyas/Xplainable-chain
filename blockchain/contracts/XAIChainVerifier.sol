// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title XAIChainVerifier
 * @dev Stores AI explanations for blockchain transaction analysis on-chain
 * @notice This contract provides immutable storage for explainable AI predictions
 */
contract XAIChainVerifier {
    struct Explanation {
        bytes32 txHash;           // Transaction hash being analyzed
        string ipfsHash;          // IPFS CID of full explanation
        uint8 riskScore;          // 0-100 risk score
        uint256 timestamp;        // When stored
        address auditor;          // Who submitted
        bool verified;            // Verification status
    }
    
    mapping(bytes32 => Explanation) public explanations;
    mapping(bytes32 => bool) public exists;
    
    event ExplanationStored(
        bytes32 indexed txHash,
        string ipfsHash,
        uint8 riskScore,
        address auditor
    );
    
    event ExplanationVerified(
        bytes32 indexed txHash,
        address verifier
    );
    
    /**
     * @dev Store explanation on-chain
     * @param _txHash Hash of the transaction being analyzed
     * @param _ipfsHash IPFS CID containing full explanation data
     * @param _riskScore Risk score from 0-100
     */
    function storeExplanation(
        bytes32 _txHash,
        string memory _ipfsHash,
        uint8 _riskScore
    ) public returns (bool) {
        require(_riskScore <= 100, "Invalid risk score");
        require(!exists[_txHash], "Explanation already exists");
        
        explanations[_txHash] = Explanation({
            txHash: _txHash,
            ipfsHash: _ipfsHash,
            riskScore: _riskScore,
            timestamp: block.timestamp,
            auditor: msg.sender,
            verified: false
        });
        
        exists[_txHash] = true;
        
        emit ExplanationStored(_txHash, _ipfsHash, _riskScore, msg.sender);
        return true;
    }
    
    /**
     * @dev Verify an existing explanation
     * @param _txHash Hash of the transaction to verify
     */
    function verifyExplanation(bytes32 _txHash) public returns (bool) {
        require(exists[_txHash], "Explanation does not exist");
        explanations[_txHash].verified = true;
        emit ExplanationVerified(_txHash, msg.sender);
        return true;
    }
    
    /**
     * @dev Get explanation details
     * @param _txHash Hash of the transaction
     */
    function getExplanation(bytes32 _txHash) 
        public 
        view 
        returns (Explanation memory) 
    {
        require(exists[_txHash], "Explanation does not exist");
        return explanations[_txHash];
    }
    
    /**
     * @dev Check if explanation exists
     * @param _txHash Hash of the transaction
     */
    function hasExplanation(bytes32 _txHash) public view returns (bool) {
        return exists[_txHash];
    }
}
