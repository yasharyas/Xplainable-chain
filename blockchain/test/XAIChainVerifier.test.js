const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("XAIChainVerifier", function () {
    let xaiChainVerifier;
    let owner;
    let addr1;
    
    beforeEach(async function () {
        [owner, addr1] = await ethers.getSigners();
        
        const XAIChainVerifier = await ethers.getContractFactory("XAIChainVerifier");
        xaiChainVerifier = await XAIChainVerifier.deploy();
        await xaiChainVerifier.waitForDeployment();
    });
    
    describe("Deployment", function () {
        it("Should deploy successfully", async function () {
            expect(await xaiChainVerifier.getAddress()).to.be.properAddress;
        });
    });
    
    describe("Store Explanation", function () {
        it("Should store explanation correctly", async function () {
            const txHash = ethers.keccak256(ethers.toUtf8Bytes("test_transaction"));
            const ipfsHash = "QmTest123";
            const riskScore = 75;
            
            await expect(xaiChainVerifier.storeExplanation(txHash, ipfsHash, riskScore))
                .to.emit(xaiChainVerifier, "ExplanationStored")
                .withArgs(txHash, ipfsHash, riskScore, owner.address);
            
            const explanation = await xaiChainVerifier.getExplanation(txHash);
            expect(explanation.ipfsHash).to.equal(ipfsHash);
            expect(explanation.riskScore).to.equal(riskScore);
            expect(explanation.auditor).to.equal(owner.address);
        });
        
        it("Should reject invalid risk score", async function () {
            const txHash = ethers.keccak256(ethers.toUtf8Bytes("test_transaction"));
            const ipfsHash = "QmTest123";
            const invalidRiskScore = 101;
            
            await expect(
                xaiChainVerifier.storeExplanation(txHash, ipfsHash, invalidRiskScore)
            ).to.be.revertedWith("Invalid risk score");
        });
        
        it("Should reject duplicate explanations", async function () {
            const txHash = ethers.keccak256(ethers.toUtf8Bytes("test_transaction"));
            const ipfsHash = "QmTest123";
            const riskScore = 75;
            
            await xaiChainVerifier.storeExplanation(txHash, ipfsHash, riskScore);
            
            await expect(
                xaiChainVerifier.storeExplanation(txHash, ipfsHash, riskScore)
            ).to.be.revertedWith("Explanation already exists");
        });
    });
    
    describe("Verify Explanation", function () {
        it("Should verify explanation", async function () {
            const txHash = ethers.keccak256(ethers.toUtf8Bytes("test_transaction"));
            const ipfsHash = "QmTest123";
            const riskScore = 75;
            
            await xaiChainVerifier.storeExplanation(txHash, ipfsHash, riskScore);
            
            await expect(xaiChainVerifier.verifyExplanation(txHash))
                .to.emit(xaiChainVerifier, "ExplanationVerified")
                .withArgs(txHash, owner.address);
            
            const explanation = await xaiChainVerifier.getExplanation(txHash);
            expect(explanation.verified).to.be.true;
        });
    });
    
    describe("Query Functions", function () {
        it("Should check if explanation exists", async function () {
            const txHash = ethers.keccak256(ethers.toUtf8Bytes("test_transaction"));
            
            expect(await xaiChainVerifier.hasExplanation(txHash)).to.be.false;
            
            await xaiChainVerifier.storeExplanation(txHash, "QmTest", 50);
            
            expect(await xaiChainVerifier.hasExplanation(txHash)).to.be.true;
        });
    });
});
