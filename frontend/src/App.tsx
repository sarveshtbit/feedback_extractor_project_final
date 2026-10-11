import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { DashboardPage } from './pages/DashboardPage';
import { UploadPage } from './pages/UploadPage';
import { ResultsPage } from './pages/ResultsPage';
import { ModelStatusPage } from './pages/ModelStatusPage';
import { TrainingPage } from './pages/TrainingPage';

export function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardPage />;
      case 'upload':
        return <UploadPage />;
      case 'results':
        return <ResultsPage />;
      case 'model_status':
        return <ModelStatusPage />;
      case 'training':
        return <TrainingPage />;
      default:
        return <DashboardPage />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderContent()}
      </main>
      <footer className="bg-white border-t border-slate-200 py-4 text-center text-xs text-slate-500">
        Bannari Amman Institute of Technology — Career Guidance Program Feedback Digitization Platform &copy; 2026
      </footer>
    </div>
  );
}

export default App;
