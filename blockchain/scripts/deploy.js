const hre = require("hardhat");
const fs = require('fs');
const path = require('path');

async function main() {
    console.log("Deploying XAIChainVerifier to Polygon Amoy (Mumbai replacement)...");
    
    const XAIChainVerifier = await hre.ethers.getContractFactory("XAIChainVerifier");
    const contract = await XAIChainVerifier.deploy();
    
    // Wait for deployment to complete (ethers v6 syntax)
    await contract.waitForDeployment();
    
    const contractAddress = await contract.getAddress();
    
    console.log(`[OK] Contract deployed to: ${contractAddress}`);
    console.log(`* View on Polygonscan: https://amoy.polygonscan.com/address/${contractAddress}`);
    
    // Save address for frontend and backend
    const contractData = {
        address: contractAddress,
        network: "amoy",
        chainId: 80002,
        deployedAt: new Date().toISOString()
    };
    
    // Save to frontend
    const frontendPath = path.join(__dirname, '../../frontend/public/abi/contract-address.json');
    fs.mkdirSync(path.dirname(frontendPath), { recursive: true });
    fs.writeFileSync(frontendPath, JSON.stringify(contractData, null, 2));
    
    console.log("* Contract address saved to frontend/public/abi/");
    
    // Save contract ABI
    const artifact = await hre.artifacts.readArtifact("XAIChainVerifier");
    fs.writeFileSync(
        path.join(__dirname, '../../frontend/public/abi/XAIChainVerifier.json'),
        JSON.stringify(artifact, null, 2)
    );
    
    console.log("* Contract ABI saved to frontend/public/abi/");
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
