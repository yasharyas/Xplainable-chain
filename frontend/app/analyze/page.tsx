'use client';

import { useState } from 'react';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useAccount } from 'wagmi';
import Link from 'next/link';
import { Shield, ArrowLeft, Loader2, AlertTriangle, CheckCircle } from 'lucide-react';

export default function AnalyzePage() {
  const { isConnected } = useAccount();
  const [txHash, setTxHash] = useState('');
  const [network, setNetwork] = useState<'ethereum' | 'polygon' | 'polygon-amoy'>('ethereum');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const analyzeTx = async () => {
    if (!txHash) {
      setError('Please enter a transaction hash');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/analyze/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tx_hash: txHash,
          network: network,  // Use selected network from dropdown
          // DO NOT send transaction_data - let backend fetch from blockchain
        }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed');
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze transaction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="container mx-auto px-4 py-6 flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <Link href="/" className="text-gray-300 hover:text-white transition">
            <ArrowLeft className="w-6 h-6" />
          </Link>
          <Shield className="w-8 h-8 text-purple-400" />
          <h1 className="text-2xl font-bold text-white">Analyze Transaction</h1>
        </div>
        <ConnectButton />
      </header>

      {/* Main Content */}
      <section className="container mx-auto px-4 py-12 max-w-4xl">
        {/* Causal XAI Banner */}
        <div className="mb-6 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl p-6 shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xl font-bold text-white mb-2">🔬 New: Causal Explainable AI</h3>
              <p className="text-purple-100 text-sm">
                Go beyond correlation with causal inference. Identify true cause-and-effect relationships in fraud detection.
              </p>
            </div>
            <Link
              href="/analyze/causal"
              className="px-6 py-3 bg-white text-purple-600 font-semibold rounded-lg hover:bg-purple-50 transition-colors whitespace-nowrap"
            >
              Try Causal XAI →
            </Link>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-8">
          <h2 className="text-3xl font-bold text-white mb-6">Transaction Analysis</h2>
          
          <div className="space-y-6">
            {/* Network Selector */}
            <div>
              <label className="block text-gray-300 mb-2 font-medium">
                Blockchain Network
              </label>
              <select
                value={network}
                onChange={(e) => setNetwork(e.target.value as 'ethereum' | 'polygon' | 'polygon-amoy')}
                className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
              >
                <option value="ethereum">Ethereum Mainnet</option>
                <option value="polygon">Polygon Mainnet</option>
                <option value="polygon-amoy">Polygon Amoy Testnet</option>
              </select>
            </div>

            {/* Input */}
            <div>
              <label className="block text-gray-300 mb-2 font-medium">
                Transaction Hash
              </label>
              <input
                type="text"
                value={txHash}
                onChange={(e) => setTxHash(e.target.value)}
                placeholder={
                  network === 'ethereum' 
                    ? "0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060" 
                    : network === 'polygon'
                    ? "0x... (Polygon transaction hash)"
                    : "0x... (Polygon Amoy transaction hash)"
                }
                className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>

            {/* Analyze Button */}
            <button
              onClick={analyzeTx}
              disabled={loading || !isConnected}
              className="w-full px-6 py-4 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg font-semibold transition flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <span>Analyze Transaction</span>
              )}
            </button>

            {!isConnected && (
              <p className="text-yellow-400 text-sm text-center">
                Please connect your wallet to analyze transactions
              </p>
            )}

            {/* Error */}
            {error && (
              <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="text-red-300">{error}</div>
              </div>
            )}

            {/* Results */}
            {result && (
              <div className="space-y-4 mt-8">
                {/* Risk Badge */}
                <div className={`p-6 rounded-lg border-2 ${
                  result.is_malicious 
                    ? 'bg-red-900/30 border-red-500' 
                    : 'bg-green-900/30 border-green-500'
                }`}>
                  <div className="flex items-center space-x-3 mb-4">
                    {result.is_malicious ? (
                      <AlertTriangle className="w-8 h-8 text-red-400" />
                    ) : (
                      <CheckCircle className="w-8 h-8 text-green-400" />
                    )}
                    <div>
                      <h3 className="text-2xl font-bold text-white">
                        {result.is_malicious ? 'Malicious Transaction' : 'Legitimate Transaction'}
                      </h3>
                      <p className={result.is_malicious ? 'text-red-300' : 'text-green-300'}>
                        Risk Score: {result.risk_score}/100
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Confidence:</span>
                      <span className="text-white ml-2">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Transaction Hash:</span>
                      <span className="text-white ml-2 truncate block">{result.tx_hash}</span>
                    </div>
                  </div>
                </div>

                {/* Features */}
                <div className="bg-gray-900/50 rounded-lg p-6">
                  <h4 className="text-xl font-bold text-white mb-4">Analyzed Features</h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {Object.entries(result.features).map(([key, value]: any) => (
                      <div key={key} className="bg-gray-800/50 rounded p-3">
                        <div className="text-gray-400 text-xs uppercase mb-1">
                          {key.replace(/_/g, ' ')}
                        </div>
                        <div className="text-white font-semibold">
                          {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : value}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Storage Info */}
                <div className="bg-gray-900/50 rounded-lg p-6">
                  <h4 className="text-xl font-bold text-white mb-4">Verification</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">IPFS Hash:</span>
                      <span className="text-purple-400 font-mono">{result.ipfs_hash}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Blockchain Hash:</span>
                      <span className="text-purple-400 font-mono">{result.blockchain_hash}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Timestamp:</span>
                      <span className="text-white">{new Date(result.timestamp).toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
