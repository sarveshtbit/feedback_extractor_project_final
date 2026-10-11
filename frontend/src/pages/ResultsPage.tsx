import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { DocumentItem, PageResult } from '../types';
import { ImageViewer } from '../components/ImageViewer';
import { ComparisonTable } from '../components/ComparisonTable';
import { ManualReviewModal } from '../components/ManualReviewModal';
import { Download, FileText, ChevronRight } from 'lucide-react';

export const ResultsPage: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null);
  const [docDetails, setDocDetails] = useState<any>(null);
  const [selectedPageId, setSelectedPageId] = useState<number | null>(null);
  const [pageResult, setPageResult] = useState<PageResult | null>(null);
  
  // Verification Modal State
  const [modalOpen, setModalOpen] = useState(false);
  const [activeFieldName, setActiveFieldName] = useState('');
  const [activeFieldValue, setActiveFieldValue] = useState('');

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const docs = await api.getDocuments();
      setDocuments(docs);
      if (docs.length > 0) {
        handleSelectDoc(docs[0].document_id);
      }
    } catch (err) {
      console.error('Error fetching documents', err);
    }
  };

  const handleSelectDoc = async (docId: number) => {
    setSelectedDocId(docId);
    try {
      const details = await api.getDocumentDetails(docId);
      setDocDetails(details);
      if (details.pages && details.pages.length > 0) {
        handleSelectPage(details.pages[0].page_id);
      }
    } catch (err) {
      console.error('Error fetching document details', err);
    }
  };

  const handleSelectPage = async (pageId: number) => {
    setSelectedPageId(pageId);
    try {
      const res = await api.getPageResults(pageId);
      setPageResult(res);
    } catch (err) {
      console.error('Error fetching page results', err);
    }
  };

  const handleOpenVerifyModal = (fieldName: string, currentValue: string) => {
    setActiveFieldName(fieldName);
    setActiveFieldValue(currentValue);
    setModalOpen(true);
  };

  const handleSaveVerification = async (fieldName: string, correctedValue: string, operator: string) => {
    if (!selectedPageId) return;
    try {
      await api.verifyField(selectedPageId, fieldName, correctedValue, operator);
      handleSelectPage(selectedPageId);
    } catch (err) {
      console.error('Error saving field verification', err);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Form Extraction Results</h2>
          <p className="text-sm text-slate-500">
            Compare Gemini Vision extractions with independent ResNet-18 CNN predictions
          </p>
        </div>

        {selectedDocId && (
          <a
            href={`/api/v1/documents/${selectedDocId}/export/csv`}
            download
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-md text-sm font-medium flex items-center shadow-sm"
          >
            <Download className="w-4 h-4 mr-2" />
            Export Document CSV
          </a>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Document & Page Selector Sidebar */}
        <div className="bg-white p-4 rounded-lg shadow border border-slate-200 space-y-4">
          <h3 className="text-xs font-bold uppercase text-slate-500">Documents</h3>
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {documents.map((doc) => (
              <button
                key={doc.document_id}
                onClick={() => handleSelectDoc(doc.document_id)}
                className={`w-full text-left px-3 py-2 rounded text-xs flex justify-between items-center ${
                  selectedDocId === doc.document_id ? 'bg-blue-50 text-blue-700 font-bold' : 'hover:bg-slate-50 text-slate-700'
                }`}
              >
                <div className="truncate">
                  <div className="font-medium truncate">{doc.filename}</div>
                  <div className="text-[10px] text-slate-400 font-mono">ID: {doc.document_id} | {doc.status}</div>
                </div>
                <ChevronRight className="w-3.5 h-3.5 ml-1 flex-shrink-0" />
              </button>
            ))}
          </div>

          {docDetails && docDetails.pages && (
            <div>
              <h4 className="text-xs font-bold uppercase text-slate-500 mb-2">Pages ({docDetails.pages.length})</h4>
              <div className="grid grid-cols-3 gap-1.5">
                {docDetails.pages.map((p: any) => (
                  <button
                    key={p.page_id}
                    onClick={() => handleSelectPage(p.page_id)}
                    className={`py-1.5 text-xs rounded border text-center font-mono ${
                      selectedPageId === p.page_id
                        ? 'bg-blue-600 text-white border-blue-600 font-bold'
                        : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    Page {p.page_number}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-3 space-y-6">
          {pageResult ? (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ImageViewer imageSrc={`/${pageResult.aligned_image_path || pageResult.original_image_path}`} />
                <div className="space-y-4">
                  <div className="bg-white p-4 rounded-lg shadow border border-slate-200 space-y-2">
                    <h4 className="text-xs font-bold uppercase text-slate-500">Page Status Overview</h4>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-600">Processing Status:</span>
                      <span className="font-mono font-semibold text-slate-900">{pageResult.status}</span>
                    </div>
                  </div>

                  <ComparisonTable
                    pageResult={pageResult}
                    onVerifyField={handleOpenVerifyModal}
                  />
                </div>
              </div>
            </>
          ) : (
            <div className="p-12 text-center bg-white rounded-lg border text-slate-400">
              Select a document and page to inspect extraction results.
            </div>
          )}
        </div>
      </div>

      <ManualReviewModal
        isOpen={modalOpen}
        fieldName={activeFieldName}
        currentValue={activeFieldValue}
        onClose={() => setModalOpen(false)}
        onSubmit={handleSaveVerification}
      />
    </div>
  );
};
