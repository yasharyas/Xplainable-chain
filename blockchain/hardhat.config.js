require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
    solidity: {
        version: "0.8.20",
        settings: {
            optimizer: {
                enabled: true,
                runs: 200
            }
        }
    },
    networks: {
        mumbai: {
            url: process.env.POLYGON_MUMBAI_RPC || "https://rpc-amoy.polygon.technology",
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
            chainId: 80002,  // Polygon Amoy testnet (Mumbai is deprecated)
            gas: 5000000,
            gasPrice: 30000000000  // 30 Gwei (increased for Amoy)
        },
        amoy: {
            url: process.env.POLYGON_MUMBAI_RPC || "https://rpc-amoy.polygon.technology",
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
            chainId: 80002,
            gas: 5000000,
            gasPrice: 30000000000
        },
        localhost: {
            url: "http://127.0.0.1:8545"
        }
    },
    etherscan: {
        apiKey: {
            polygonMumbai: process.env.POLYGONSCAN_API_KEY || process.env.ETHERSCAN_API_KEY,
            polygonAmoy: process.env.POLYGONSCAN_API_KEY || process.env.ETHERSCAN_API_KEY
        },
        customChains: [
            {
                network: "polygonMumbai",
                chainId: 80001,
                urls: {
                    apiURL: "https://api-testnet.polygonscan.com/api",
                    browserURL: "https://mumbai.polygonscan.com"
                }
            },
            {
                network: "polygonAmoy",
                chainId: 80002,
                urls: {
                    apiURL: "https://api-amoy.polygonscan.com/api",
                    browserURL: "https://amoy.polygonscan.com"
                }
            }
        ]
    },
    paths: {
        sources: "./contracts",
        tests: "./test",
        cache: "./cache",
        artifacts: "./artifacts"
    }
};
