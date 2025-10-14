import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';

const UploadContainer = styled.div`
  padding: 40px 20px;
  text-align: center;
`;

const DropzoneContainer = styled.div<{ $isDragActive: boolean; $hasFiles: boolean }>`
  border: 3px dashed ${props => 
    props.$isDragActive ? '#667eea' : 
    props.$hasFiles ? '#4CAF50' : '#ddd'
  };
  border-radius: 15px;
  padding: 40px 20px;
  margin: 20px 0;
  background: ${props => 
    props.$isDragActive ? '#f0f4ff' : 
    props.$hasFiles ? '#f0fff0' : '#fafafa'
  };
  transition: all 0.3s ease;
  cursor: pointer;

  &:hover {
    border-color: #667eea;
    background: #f0f4ff;
  }
`;

const DropzoneText = styled.div`
  font-size: 1.1rem;
  color: #666;
  margin-bottom: 20px;
`;

// FileInput is handled by react-dropzone

const FileInfo = styled.div`
  margin: 20px 0;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 10px;
  border-left: 4px solid #667eea;
`;

const FileName = styled.div`
  font-weight: 600;
  color: #333;
  margin-bottom: 5px;
`;

const FileSize = styled.div`
  font-size: 0.9rem;
  color: #666;
`;

const Button = styled.button<{ disabled: boolean }>`
  background: ${props => props.disabled ? '#ccc' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'};
  color: white;
  border: none;
  padding: 15px 30px;
  border-radius: 25px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  transition: all 0.3s ease;
  margin: 20px 10px;

  &:hover {
    transform: ${props => props.disabled ? 'none' : 'translateY(-2px)'};
    box-shadow: ${props => props.disabled ? 'none' : '0 10px 20px rgba(102, 126, 234, 0.3)'};
  }
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 10px;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  color: #e74c3c;
  background: #fdf2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
  padding: 15px;
  margin: 20px 0;
`;

interface FileUploaderProps {
  onFileUpload: (heaFile: File, datFile: File) => void;
  loading: boolean;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload, loading }) => {
  const [heaFile, setHeaFile] = useState<File | null>(null);
  const [datFile, setDatFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError(null);
    const file = acceptedFiles[0];
    
    if (file) {
      if (file.name.toLowerCase().endsWith('.hea')) {
        setHeaFile(file);
      } else if (file.name.toLowerCase().endsWith('.dat')) {
        setDatFile(file);
      } else {
        setError('Please upload only .hea or .dat files');
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.hea', '.dat']
    },
    multiple: false
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!heaFile || !datFile) {
      setError('Please upload both .hea and .dat files');
      return;
    }
    onFileUpload(heaFile, datFile);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <UploadContainer>
      <h2 style={{ marginBottom: '30px', color: '#333' }}>
        Upload ECG Files for Age Prediction
      </h2>
      
      <form onSubmit={handleSubmit}>
        <DropzoneContainer 
          {...getRootProps()} 
          $isDragActive={isDragActive}
          $hasFiles={!!(heaFile && datFile)}
        >
          <input {...getInputProps()} />
          <DropzoneText>
            {isDragActive
              ? 'Drop the files here...'
              : 'Drag & drop your .hea and .dat files here, or click to select files'
            }
          </DropzoneText>
        </DropzoneContainer>

        {heaFile && (
          <FileInfo>
            <FileName>📄 {heaFile.name}</FileName>
            <FileSize>Size: {formatFileSize(heaFile.size)}</FileSize>
          </FileInfo>
        )}

        {datFile && (
          <FileInfo>
            <FileName>📄 {datFile.name}</FileName>
            <FileSize>Size: {formatFileSize(datFile.size)}</FileSize>
          </FileInfo>
        )}

        {error && <ErrorMessage>{error}</ErrorMessage>}

        <Button 
          type="submit" 
          disabled={loading || !heaFile || !datFile}
        >
          {loading && <LoadingSpinner />}
          {loading ? 'Processing...' : 'Predict Age Group'}
        </Button>
      </form>
    </UploadContainer>
  );
};

export default FileUploader;
