import React, { useState } from 'react';
import { api } from '../services/api';
import { Activity, Upload, Play, CheckCircle, Loader2 } from 'lucide-react';

export const TrainingPage: React.FC = () => {
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [uploadingDs, setUploadingDs] = useState(false);
  const [dsInfo, setDsInfo] = useState<any>(null);

  // Training parameters
  const [epochs, setEpochs] = useState(30);
  const [lr, setLr] = useState(0.0001);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<any>(null);

  const handleDatasetUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!datasetFile) return;

    setUploadingDs(true);
    try {
      const formData = new FormData();
      formData.append('file', datasetFile);
      const res = await api.startTraining('default_v1', epochs, lr);
      setTrainResult(res);
    } catch (err: any) {
      console.error('Dataset upload/train error', err);
    } finally {
      setUploadingDs(false);
    }
  };

  const handleStartTraining = async () => {
    setTraining(true);
    setTrainResult(null);
    try {
      const res = await api.startTraining('default_v1', epochs, lr);
      setTrainResult(res);
    } catch (err: any) {
      console.error('Training error', err);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">CNN Model Training & Evaluation</h2>
        <p className="text-sm text-slate-500">
          Fine-tune ResNet-18 digit classifier on writer-independent handwritten digit callsets.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload Dataset */}
        <div className="bg-white p-6 rounded-lg shadow border border-slate-200 space-y-4">
          <h3 className="text-sm font-bold text-slate-800 flex items-center">
            <Upload className="w-4 h-4 mr-2 text-blue-600" />
            1. Upload Digit Dataset (.zip)
          </h3>
          <p className="text-xs text-slate-500">
            Zip file containing subfolders 0/ to 9/ with optional metadata.csv (writer_id, image_path, label).
          </p>
          <input
            type="file"
            accept=".zip"
            onChange={(e) => e.target.files && setDatasetFile(e.target.files[0])}
            className="text-xs text-slate-500"
          />
          {datasetFile && (
            <button
              onClick={handleDatasetUpload}
              disabled={uploadingDs}
              className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-semibold flex justify-center items-center"
            >
              {uploadingDs ? <Loader2 className="w-3.5 h-3.5 animate-spin mr-1" /> : 'Validate & Save Dataset'}
            </button>
          )}
        </div>

        {/* Hyperparameters & Start */}
        <div className="bg-white p-6 rounded-lg shadow border border-slate-200 space-y-4">
          <h3 className="text-sm font-bold text-slate-800 flex items-center">
            <Play className="w-4 h-4 mr-2 text-emerald-600" />
            2. Configure & Start Training
          </h3>
          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-slate-700 font-semibold mb-1">Epochs</label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full px-3 py-1.5 border rounded text-xs"
              />
            </div>
            <div>
              <label className="block text-slate-700 font-semibold mb-1">Learning Rate</label>
              <input
                type="number"
                step="0.00001"
                value={lr}
                onChange={(e) => setLr(Number(e.target.value))}
                className="w-full px-3 py-1.5 border rounded text-xs"
              />
            </div>
          </div>
          <button
            onClick={handleStartTraining}
            disabled={training}
            className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white rounded text-xs font-bold flex justify-center items-center"
          >
            {training ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
                Training ResNet-18 Model...
              </>
            ) : (
              'Start Training Run'
            )}
          </button>
        </div>
      </div>

      {trainResult && (
        <div className="bg-white p-6 rounded-lg shadow border border-slate-200 space-y-3">
          <div className="flex items-center text-emerald-700 font-bold text-sm">
            <CheckCircle className="w-5 h-5 mr-2" />
            Training Run Completed Successfully!
          </div>
          <div className="text-xs text-slate-600 font-mono space-y-1">
            <p>Training ID: {trainResult.training_id}</p>
            <p>Model Version Registered: {trainResult.version_name}</p>
            <p>Best Validation Loss: {trainResult.val_loss}</p>
          </div>
        </div>
      )}
    </div>
  );
};
