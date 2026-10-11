import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { ModelStatus } from '../types';
import { Cpu, AlertCircle, CheckCircle2 } from 'lucide-react';

export const ModelStatusPage: React.FC = () => {
  const [status, setStatus] = useState<ModelStatus | null>(null);

  useEffect(() => {
    api.getCNNStatus().then(setStatus).catch(console.error);
  }, []);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Custom CNN Model & Verification Engine Status</h2>
        <p className="text-sm text-slate-500">
          Pluggable Custom CNN Digit Classifier Interface
        </p>
      </div>

      <div className="bg-white p-6 rounded-lg shadow border border-slate-200 space-y-6">
        <div className="flex items-center space-x-4 pb-4 border-b">
          <div className="p-3 bg-blue-50 text-blue-600 rounded-lg">
            <Cpu className="w-8 h-8" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900">
              Architecture: <span className="font-mono text-blue-600">{status?.architecture || 'Custom CNN'}</span>
            </h3>
            <div className="flex items-center space-x-2 mt-1">
              <span className="text-xs text-slate-500">Weights File:</span>
              <span className="text-xs font-mono bg-slate-100 px-2 py-0.5 rounded">
                {status?.weights_exist ? 'pretrained_model.pt (Custom Ready)' : 'Plug Custom CNN Weights'}
              </span>
            </div>
          </div>
        </div>

        {/* Status Callout */}
        {status?.status === 'READY' ? (
          <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-900 flex items-start space-x-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-bold text-sm">Custom CNN Engine Active & Ready</h4>
              <p className="text-xs text-emerald-800 mt-0.5">
                Custom CNN classification interface is loaded and ready for custom model weights.
              </p>
            </div>
          </div>
        ) : (
          <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg text-amber-900 flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-bold text-sm">Custom CNN Ready for Training / Weights</h4>
              <p className="text-xs text-amber-800 mt-0.5">
                The application starts cleanly with pluggable Custom CNN interface.
              </p>
            </div>
          </div>
        )}

        <div className="space-y-3 pt-2">
          <h4 className="text-sm font-bold text-slate-800">Verification Engine Principles</h4>
          <ul className="text-xs text-slate-600 space-y-2 list-disc list-inside">
            <li>Targeted verification for <strong>Student Mobile Number</strong> (10 boxes) and <strong>Cut Off Marks</strong>.</li>
            <li>Writer-independent validation prevents memorization of student handwriting styles.</li>
            <li>Character-level disagreement analysis highlights conflicting digit positions without re-running entire pages.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
