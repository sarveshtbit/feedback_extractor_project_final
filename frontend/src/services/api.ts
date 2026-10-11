import axios from 'axios';
import { DocumentItem, PageResult, ModelStatus, SystemMetrics } from '../types';

const API_BASE = '/api/v1';

export const api = {
  getHealth: async () => {
    const res = await axios.get(`${API_BASE}/health`);
    return res.data.data;
  },

  getMetrics: async (): Promise<SystemMetrics> => {
    const res = await axios.get(`${API_BASE}/metrics`);
    return res.data.data;
  },

  getDocuments: async (): Promise<DocumentItem[]> => {
    const res = await axios.get(`${API_BASE}/documents`);
    return res.data.data;
  },

  getDocumentDetails: async (docId: number) => {
    const res = await axios.get(`${API_BASE}/documents/${docId}`);
    return res.data.data;
  },

  uploadDocument: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await axios.post(`${API_BASE}/documents/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return res.data.data;
  },

  getPageResults: async (pageId: number): Promise<PageResult> => {
    const res = await axios.get(`${API_BASE}/pages/${pageId}/results`);
    return res.data.data;
  },

  verifyField: async (pageId: number, fieldName: string, correctedValue: string, operator = 'human_verifier') => {
    const res = await axios.post(`${API_BASE}/pages/${pageId}/verify`, {
      field_name: fieldName,
      corrected_value: correctedValue,
      operator
    });
    return res.data.data;
  },

  getCNNStatus: async (): Promise<ModelStatus> => {
    const res = await axios.get(`${API_BASE}/models/cnn/status`);
    return res.data.data;
  },

  startTraining: async (datasetVersion: string, epochs = 30, lr = 0.0001) => {
    const res = await axios.post(`${API_BASE}/training/start`, {
      dataset_version: datasetVersion,
      epochs,
      learning_rate: lr
    });
    return res.data.data;
  }
};
