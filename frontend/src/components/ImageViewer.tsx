import React, { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

interface ImageViewerProps {
  imageSrc: string;
  selectedField?: string;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({ imageSrc, selectedField }) => {
  const [zoom, setZoom] = useState(1.0);

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 3.0));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.5));
  const handleReset = () => setZoom(1.0);

  return (
    <div className="bg-slate-900 rounded-lg p-4 text-white flex flex-col h-full">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-semibold">Original Form ROI Visualizer</h3>
        <div className="flex space-x-2">
          <button
            onClick={handleZoomOut}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs self-center font-mono">{Math.round(zoom * 100)}%</span>
          <button
            onClick={handleZoomIn}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200"
            title="Reset"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-auto bg-black rounded flex items-center justify-center relative min-h-[400px]">
        {imageSrc ? (
          <div style={{ transform: `scale(${zoom})`, transformOrigin: 'top center' }} className="transition-transform">
            <img src={imageSrc} alt="Form Original" className="max-w-full h-auto object-contain rounded" />
          </div>
        ) : (
          <p className="text-slate-500 text-sm">Select a page to view original form image</p>
        )}
      </div>
    </div>
  );
};
