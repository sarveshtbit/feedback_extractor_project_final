import React, { useState } from 'react';
import { X, Check } from 'lucide-react';

interface ManualReviewModalProps {
  isOpen: boolean;
  fieldName: string;
  currentValue: string;
  onClose: () => void;
  onSubmit: (fieldName: string, correctedValue: string, operator: string) => void;
}

export const ManualReviewModal: React.FC<ManualReviewModalProps> = ({
  isOpen,
  fieldName,
  currentValue,
  onClose,
  onSubmit,
}) => {
  const [value, setValue] = useState(currentValue);
  const [operator, setOperator] = useState('human_verifier_01');

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(fieldName, value, operator);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-md w-full p-6 shadow-xl relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-600"
        >
          <X className="w-5 h-5" />
        </button>
        <h3 className="text-lg font-bold text-slate-900 mb-1 capitalize">
          Verify Field: {fieldName.replace(/_/g, ' ')}
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          Override or confirm the final value stored in the database.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Corrected / Verified Value
            </label>
            <input
              type="text"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:outline-none"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Reviewer / Operator ID
            </label>
            <input
              type="text"
              value={operator}
              onChange={(e) => setOperator(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
          </div>

          <div className="flex justify-end space-x-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-md"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm bg-blue-600 text-white hover:bg-blue-700 rounded-md font-medium flex items-center"
            >
              <Check className="w-4 h-4 mr-1.5" />
              Save Verification
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
