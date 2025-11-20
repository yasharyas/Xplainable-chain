// Test contract deployment by calling a read method
const hre = require("hardhat");

async function main() {
    const contractAddress = "0x69d090E6ED8F7C3879A48aD15C0D73680699C1f0";
    
    console.log("Testing contract functionality on Polygon Amoy...");
    console.log("Contract:", contractAddress);
    
    const XAIChainVerifier = await hre.ethers.getContractFactory("XAIChainVerifier");
    const contract = XAIChainVerifier.attach(contractAddress);
    
    // Test 1: Call a read method (hasExplanation)
    const testTxHash = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";
    try {
        const hasExpl = await contract.hasExplanation(testTxHash);
        console.log("[OK] Contract responding to read calls");
        console.log("Test tx has explanation:", hasExpl);
    } catch (error) {
        console.log("[ERROR] Contract read failed:", error.message);
        process.exit(1);
    }
    
    // Test 2: Check if we can get explanation for non-existent tx (should not revert)
    try {
        const explanation = await contract.getExplanation(testTxHash);
        console.log("[OK] getExplanation callable (returns empty for non-existent tx)");
    } catch (error) {
        // This is expected for non-existent explanation
        if (error.message.includes("Explanation does not exist")) {
            console.log("[OK] Contract correctly reverts for non-existent explanation");
        } else {
            console.log("[ERROR] Unexpected error:", error.message);
            process.exit(1);
        }
    }
    
    console.log("\n[RESULT] Contract deployment is FUNCTIONAL");
    console.log("Contract can receive calls and responds correctly");
}

main().catch((error) => {
    console.error("[ERROR] Test failed:", error);
    process.exitCode = 1;
});
