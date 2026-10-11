import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { SystemMetrics, ModelStatus } from '../types';
import { FileText, CheckCircle, AlertTriangle, Cpu, TrendingUp } from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [cnnStatus, setCnnStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const m = await api.getMetrics();
        const c = await api.getCNNStatus();
        setMetrics(m);
        setCnnStatus(c);
      } catch (err) {
        console.error('Error fetching dashboard stats', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="p-8 text-center text-slate-500">Loading system stats...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">System Dashboard</h2>
        <p className="text-sm text-slate-500">
          Handwritten Feedback Form Extraction & Verification Overview
        </p>
      </div>

      {/* CNN Model Banner */}
      <div className={`p-4 rounded-lg border ${cnnStatus?.status === 'READY' ? 'bg-emerald-50 border-emerald-200 text-emerald-900' : 'bg-amber-50 border-amber-200 text-amber-900'}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Cpu className="w-6 h-6" />
            <div>
              <h4 className="font-semibold text-sm">
                CNN Verification Engine Status: <span className="font-mono">{cnnStatus?.status}</span>
              </h4>
              <p className="text-xs text-slate-600">
                {cnnStatus?.notice || 'ResNet-18 digit verification active.'}
              </p>
            </div>
          </div>
          <span className="px-3 py-1 bg-white rounded border text-xs font-mono font-bold shadow-sm">
            {cnnStatus?.architecture}
          </span>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
          <div className="flex justify-between items-center text-slate-500 text-xs font-semibold uppercase mb-1">
            Total Documents
            <FileText className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900">{metrics?.total_documents || 0}</div>
          <p className="text-xs text-slate-400 mt-1">{metrics?.total_pages || 0} pages processed</p>
        </div>

        <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
          <div className="flex justify-between items-center text-slate-500 text-xs font-semibold uppercase mb-1">
            Agreement Rate
            <CheckCircle className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900">
            {((metrics?.agreement_rate || 0) * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400 mt-1">{metrics?.agreements || 0} perfect matches</p>
        </div>

        <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
          <div className="flex justify-between items-center text-slate-500 text-xs font-semibold uppercase mb-1">
            Conflict Rate
            <AlertTriangle className="w-4 h-4 text-rose-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900">
            {((metrics?.conflict_rate || 0) * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400 mt-1">{metrics?.conflicts || 0} flagged for review</p>
        </div>

        <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
          <div className="flex justify-between items-center text-slate-500 text-xs font-semibold uppercase mb-1">
            Total Decisions
            <TrendingUp className="w-4 h-4 text-purple-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900">{metrics?.total_verification_decisions || 0}</div>
          <p className="text-xs text-slate-400 mt-1">Dual-engine evaluated</p>
        </div>
      </div>
    </div>
  );
};
