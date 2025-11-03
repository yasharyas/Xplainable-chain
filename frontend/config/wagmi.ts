import { getDefaultConfig } from '@rainbow-me/rainbowkit';
import { mainnet, polygon, polygonMumbai, sepolia, hardhat } from 'wagmi/chains';

export const config = getDefaultConfig({
  appName: 'XAI-Chain',
  projectId: process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID || 'YOUR_PROJECT_ID',
  chains: [mainnet, polygon, sepolia, polygonMumbai, hardhat],
  ssr: true,
});
