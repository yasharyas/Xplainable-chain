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
            url: "https://polygon-testnet.public.blastapi.io",
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
            chainId: 80001,
            gas: 5000000,
            gasPrice: 8000000000
        },
        localhost: {
            url: "http://127.0.0.1:8545"
        }
    },
    etherscan: {
        apiKey: {
            polygonMumbai: process.env.ETHERSCAN_API_KEY
        },
        customChains: [
            {
                network: "polygonMumbai",
                chainId: 80001,
                urls: {
                    apiURL: "https://api-testnet.polygonscan.com/api",
                    browserURL: "https://mumbai.polygonscan.com"
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
