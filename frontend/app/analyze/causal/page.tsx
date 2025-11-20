'use client';

import { useState } from 'react';
import CausalGraphViz from '@/components/CausalGraphViz';

export default function CausalAnalysisPage() {
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    transaction_hash: '0x1234567890abcdef',
    gas_price: '50',
    value: '1.5',
    gas_used: '21000',
    sender_tx_count: '10',
    contract_age: '365',
    is_contract_creation: 'false',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const features = {
        gas_price: parseFloat(formData.gas_price),
        value: parseFloat(formData.value),
        gas_used: parseFloat(formData.gas_used),
        sender_tx_count: parseFloat(formData.sender_tx_count),
        contract_age: parseFloat(formData.contract_age),
        is_contract_creation: formData.is_contract_creation === 'true' ? 1 : 0,
      };

      const response = await fetch('http://localhost:8000/api/analyze/causal/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transaction_hash: formData.transaction_hash,
          features,
          treatment_features: ['gas_price', 'value', 'sender_tx_count'],
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to perform causal analysis');
      }

      const data = await response.json();
      setAnalysisResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Causal Explainable AI
          </h1>
          <p className="text-gray-600 text-lg">
            Beyond correlation: Identify true causal relationships in blockchain fraud detection
          </p>
        </div>

        {/* Research Contribution Banner */}
        <div className="bg-gradient-to-r from-purple-500 to-blue-500 text-white p-6 rounded-lg mb-8 shadow-lg">
          <h2 className="text-2xl font-bold mb-2">🔬 Novel Research Feature</h2>
          <p className="text-purple-100">
            This is the first implementation of <strong>Causal Explainable AI for Blockchain Fraud Detection</strong>.
            Unlike traditional correlation-based methods (SHAP, LIME), causal XAI identifies genuine cause-and-effect
            relationships while revealing spurious correlations created by confounders.
          </p>
        </div>

        {/* Causal Graph Visualization */}
        <div className="mb-8">
          <CausalGraphViz />
        </div>

        {/* Analysis Form */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Run Causal Analysis</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Transaction Hash
                </label>
                <input
                  type="text"
                  value={formData.transaction_hash}
                  onChange={(e) => setFormData({ ...formData, transaction_hash: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Gas Price (Gwei)
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={formData.gas_price}
                  onChange={(e) => setFormData({ ...formData, gas_price: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Value (ETH)
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={formData.value}
                  onChange={(e) => setFormData({ ...formData, value: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Gas Used
                </label>
                <input
                  type="number"
                  value={formData.gas_used}
                  onChange={(e) => setFormData({ ...formData, gas_used: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Sender Transaction Count
                </label>
                <input
                  type="number"
                  value={formData.sender_tx_count}
                  onChange={(e) => setFormData({ ...formData, sender_tx_count: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Contract Age (days)
                </label>
                <input
                  type="number"
                  value={formData.contract_age}
                  onChange={(e) => setFormData({ ...formData, contract_age: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900"
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 transition-all"
            >
              {loading ? 'Analyzing...' : 'Perform Causal Analysis'}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600 font-semibold">Error: {error}</p>
            </div>
          )}
        </div>

        {/* Analysis Results */}
        {analysisResult && (
          <div className="space-y-6">
            {/* Causal Effects */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Causal Effects (ACE)</h3>
              <div className="space-y-3">
                {Object.entries(analysisResult.causal_effects).map(([feature, effectData]: [string, any]) => {
                  const effect = effectData.average_causal_effect;
                  const strength = effectData.strength;
                  const mechanism = effectData.mechanism;
                  return (
                    <div key={feature} className="p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-gray-900 capitalize">{feature.replace(/_/g, ' ')}</span>
                        <div className="text-right">
                          <div className="font-bold text-lg" style={{ color: effect > 0 ? '#ef4444' : '#10b981' }}>
                            {effect > 0 ? '+' : ''}{effect.toFixed(4)}
                          </div>
                          <div className="text-xs text-gray-500">Average Causal Effect</div>
                        </div>
                      </div>
                      <div className="text-sm text-gray-600 mt-2">
                        <div><strong>Strength:</strong> {strength}</div>
                        <div className="mt-1"><strong>Mechanism:</strong> {mechanism}</div>
                      </div>
                      {effectData.controlled_for && effectData.controlled_for.length > 0 && (
                        <div className="text-xs text-gray-500 mt-2">
                          Controlled for: {effectData.controlled_for.join(', ')}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Correlation vs Causation Comparison */}
            {analysisResult.comparison && analysisResult.comparison.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Correlation vs Causation</h3>
                <div className="space-y-3">
                  {analysisResult.comparison.map((comparison: any) => (
                    <div key={comparison.feature} className="p-4 bg-gray-50 rounded-lg">
                      <div className="font-medium text-gray-900 capitalize mb-2">
                        {comparison.feature.replace(/_/g, ' ')}
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm mb-2">
                        <div>
                          <span className="text-gray-600">Correlation:</span>
                          <span className="ml-2 font-semibold">{comparison.correlation?.toFixed(4) || 'N/A'}</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Causal Effect:</span>
                          <span className="ml-2 font-semibold">{comparison.causal_effect?.toFixed(4) || 'N/A'}</span>
                        </div>
                      </div>
                      <div className="text-xs text-gray-600 mb-2">
                        Difference: {comparison.difference?.toFixed(4)} | Type: {comparison.relationship_type}
                      </div>
                      {comparison.interpretation && (
                        <div className="text-sm text-gray-700 italic mt-2">
                          {comparison.interpretation}
                        </div>
                      )}
                      {comparison.is_spurious && (
                        <div className="mt-2 px-3 py-1 bg-yellow-100 text-yellow-800 text-xs rounded-full inline-block">
                          ⚠️ Spurious Correlation Detected
                        </div>
                      )}
                      {comparison.is_suppressed && (
                        <div className="mt-2 px-3 py-1 bg-blue-100 text-blue-800 text-xs rounded-full inline-block">
                           Suppressed Effect Detected
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Confounders */}
            {analysisResult.confounders && Object.keys(analysisResult.confounders).length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Confounding Variables</h3>
                <p className="text-gray-600 text-sm mb-4">
                  These variables create spurious correlations and must be controlled for accurate causal inference.
                </p>
                <div className="space-y-3">
                  {Object.entries(analysisResult.confounders).map(([treatment, confounderData]: [string, any]) => (
                    <div key={treatment} className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                      <div className="font-medium text-gray-900 capitalize mb-2">
                        {treatment.replace(/_/g, ' ')}
                      </div>
                      {confounderData.confounders && confounderData.confounders.length > 0 && (
                        <div className="text-sm text-gray-700 mb-1">
                          <strong>Confounders:</strong> {confounderData.confounders.join(', ')}
                        </div>
                      )}
                      {confounderData.mediators && confounderData.mediators.length > 0 && (
                        <div className="text-sm text-gray-700 mb-1">
                          <strong>Mediators:</strong> {confounderData.mediators.join(', ')}
                        </div>
                      )}
                      <div className="text-xs text-gray-600 mt-2">
                        Backdoor paths: {confounderData.backdoor_paths || 0} | 
                        Adjustment needed: {confounderData.adjustment_needed ? 'Yes' : 'No'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Interpretation */}
            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 border border-green-200">
              <h3 className="text-xl font-bold text-gray-900 mb-3"> Interpretation</h3>
              <p className="text-gray-700 whitespace-pre-line">{analysisResult.interpretation}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
