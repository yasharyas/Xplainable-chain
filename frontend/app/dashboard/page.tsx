'use client';

import { useState, useEffect } from 'react';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import Link from 'next/link';
import { Shield, ArrowLeft, TrendingUp, AlertTriangle, Activity, Clock } from 'lucide-react';

export default function DashboardPage() {
  const [stats, setStats] = useState<any>(null);
  const [transactions, setTransactions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch audit trail
      const response = await fetch('http://localhost:8000/api/audit/');
      const data = await response.json();
      
      setTransactions(data.audit_trail || []);
      
      // Calculate stats
      const totalTx = data.audit_trail?.length || 0;
      const malicious = data.audit_trail?.filter((tx: any) => tx.is_malicious).length || 0;
      const avgRisk = totalTx > 0 
        ? data.audit_trail.reduce((sum: number, tx: any) => sum + tx.risk_score, 0) / totalTx 
        : 0;
      
      setStats({
        total: totalTx,
        malicious,
        legitimate: totalTx - malicious,
        avgRisk: avgRisk.toFixed(1),
      });
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
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
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        </div>
        <ConnectButton />
      </header>

      {/* Main Content */}
      <section className="container mx-auto px-4 py-12">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-white text-xl">Loading dashboard...</div>
          </div>
        ) : (
          <>
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
              <div className="bg-gray-800/50 backdrop-blur-lg rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-2">
                  <Activity className="w-8 h-8 text-blue-400" />
                  <h3 className="text-gray-400 font-medium">Total Analyzed</h3>
                </div>
                <p className="text-4xl font-bold text-white">{stats?.total || 0}</p>
              </div>

              <div className="bg-gray-800/50 backdrop-blur-lg rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-2">
                  <AlertTriangle className="w-8 h-8 text-red-400" />
                  <h3 className="text-gray-400 font-medium">Malicious</h3>
                </div>
                <p className="text-4xl font-bold text-red-400">{stats?.malicious || 0}</p>
              </div>

              <div className="bg-gray-800/50 backdrop-blur-lg rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-2">
                  <Shield className="w-8 h-8 text-green-400" />
                  <h3 className="text-gray-400 font-medium">Legitimate</h3>
                </div>
                <p className="text-4xl font-bold text-green-400">{stats?.legitimate || 0}</p>
              </div>

              <div className="bg-gray-800/50 backdrop-blur-lg rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-2">
                  <TrendingUp className="w-8 h-8 text-purple-400" />
                  <h3 className="text-gray-400 font-medium">Avg Risk</h3>
                </div>
                <p className="text-4xl font-bold text-white">{stats?.avgRisk || 0}</p>
              </div>
            </div>

            {/* Transaction List */}
            <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-8">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center space-x-3">
                <Clock className="w-8 h-8 text-purple-400" />
                <span>Recent Transactions</span>
              </h2>

              {transactions.length === 0 ? (
                <div className="text-center py-12">
                  <p className="text-gray-400 text-lg">No transactions analyzed yet</p>
                  <Link 
                    href="/analyze"
                    className="inline-block mt-4 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition"
                  >
                    Analyze Your First Transaction
                  </Link>
                </div>
              ) : (
                <div className="space-y-4">
                  {transactions.map((tx, idx) => (
                    <div 
                      key={idx}
                      className={`p-6 rounded-lg border-l-4 ${
                        tx.is_malicious 
                          ? 'bg-red-900/20 border-red-500' 
                          : 'bg-green-900/20 border-green-500'
                      }`}
                    >
                      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                              tx.is_malicious 
                                ? 'bg-red-500 text-white' 
                                : 'bg-green-500 text-white'
                            }`}>
                              {tx.is_malicious ? 'Malicious' : 'Legitimate'}
                            </span>
                            <span className="text-gray-400 text-sm">
                              Risk: {tx.risk_score}/100
                            </span>
                          </div>
                          <p className="text-white font-mono text-sm mb-1">
                            {tx.tx_hash}
                          </p>
                          <p className="text-gray-400 text-xs">
                            {new Date(tx.timestamp).toLocaleString()}
                          </p>
                        </div>

                        <div className="flex items-center space-x-6 text-sm">
                          <div>
                            <span className="text-gray-400">Confidence:</span>
                            <span className="text-white ml-2 font-semibold">
                              {(tx.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-400">IPFS:</span>
                            <span className="text-purple-400 ml-2 font-mono">
                              {tx.ipfs_hash?.substring(0, 12)}...
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </section>
    </main>
  );
}
