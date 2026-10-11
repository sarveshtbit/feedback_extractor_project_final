import React, { useState } from 'react';
import { api } from '../services/api';
import { Upload, FileCheck, AlertCircle, Loader2 } from 'lucide-react';

export const UploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      const res = await api.uploadDocument(file);
      setResult(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Document upload and processing failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Upload Feedback Forms</h2>
        <p className="text-sm text-slate-500">
          Upload scanned image (PNG/JPEG) or multi-page PDF documents for automated extraction & verification.
        </p>
      </div>

      <form onSubmit={handleUpload} className="bg-white p-6 rounded-lg shadow border border-slate-200 space-y-4">
        <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
          <Upload className="w-10 h-10 mx-auto text-slate-400 mb-3" />
          <p className="text-sm font-medium text-slate-700">
            Drag & Drop scanned forms here, or click to browse
          </p>
          <p className="text-xs text-slate-400 mt-1">Supports PNG, JPG, JPEG, and multi-page PDF files</p>
          <input
            type="file"
            accept=".png,.jpg,.jpeg,.pdf"
            onChange={handleFileChange}
            className="mt-4 text-xs mx-auto text-slate-500"
          />
        </div>

        {file && (
          <div className="flex items-center justify-between p-3 bg-slate-50 rounded border text-sm">
            <span className="font-medium text-slate-800">{file.name}</span>
            <span className="text-xs text-slate-500 font-mono">{(file.size / 1024).toFixed(1)} KB</span>
          </div>
        )}

        {error && (
          <div className="p-3 bg-rose-50 border border-rose-200 rounded text-rose-800 text-sm flex items-center">
            <AlertCircle className="w-4 h-4 mr-2" />
            {error}
          </div>
        )}

        {result && (
          <div className="p-4 bg-emerald-50 border border-emerald-200 rounded text-emerald-900 text-sm space-y-1">
            <div className="flex items-center font-bold">
              <FileCheck className="w-4 h-4 mr-2" />
              Document Uploaded & Processed Successfully!
            </div>
            <p className="text-xs text-emerald-800">
              Document ID: <span className="font-mono">{result.document_id}</span> | Status: <span className="font-bold">{result.status}</span> | Total Pages: {result.total_pages}
            </p>
          </div>
        )}

        <button
          type="submit"
          disabled={!file || uploading}
          className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded-md text-sm transition-colors flex justify-center items-center"
        >
          {uploading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Processing Page & Running Dual-Engine Extraction...
            </>
          ) : (
            'Start Form Processing'
          )}
        </button>
      </form>
    </div>
  );
};
