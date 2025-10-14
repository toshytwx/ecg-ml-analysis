import React, { useState } from 'react';
import styled from 'styled-components';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import FileUploader from './components/FileUploader';
import PredictionResult from './components/PredictionResult';
import Header from './components/Header';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
`;

const MainContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  background: white;
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  overflow: hidden;
`;

interface Prediction {
  predicted_age_group: number;
  confidence: number;
  all_probabilities: number[];
}

function App() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async (heaFile: File, datFile: File) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append('heaFile', heaFile);
    formData.append('datFile', datFile);

    try {
      const response = await fetch('http://localhost:3001/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const data: Prediction = await response.json();
      setPrediction(data);
      toast.success('Prediction completed successfully!');
    } catch (err: any) {
      const errorMessage = err.message || 'An unknown error occurred.';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppContainer>
      <MainContent>
        <Header />
        <FileUploader onFileUpload={handleFileUpload} loading={loading} />
        {error && (
          <div style={{ padding: '20px', color: 'red', textAlign: 'center' }}>
            Error: {error}
          </div>
        )}
        {prediction && <PredictionResult prediction={prediction} />}
      </MainContent>
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />
    </AppContainer>
  );
}

export default App;