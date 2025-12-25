import React from 'react';
import styled from 'styled-components';
import ECGVisualizer from './ECGVisualizer';

const ResultContainer = styled.div`
  padding: 40px 20px;
  background: #f8f9fa;
  border-top: 1px solid #e9ecef;
`;

const ResultHeader = styled.div`
  text-align: center;
  margin-bottom: 30px;
`;

const Title = styled.h2`
  color: #333;
  margin-bottom: 20px;
`;

const PredictionCard = styled.div`
  background: white;
  border-radius: 15px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  margin-bottom: 30px;
`;

const PredictionInfo = styled.div`
  display: flex;
  justify-content: space-around;
  margin-bottom: 30px;
  flex-wrap: wrap;
  gap: 20px;
`;

const InfoItem = styled.div`
  text-align: center;
  flex: 1;
  min-width: 150px;
`;

const InfoLabel = styled.div`
  font-size: 0.9rem;
  color: #666;
  margin-bottom: 5px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const InfoValue = styled.div<{ highlight?: boolean }>`
  font-size: ${props => props.highlight ? '2rem' : '1.5rem'};
  font-weight: 700;
  color: ${props => props.highlight ? '#667eea' : '#333'};
`;

const ConfidenceBar = styled.div`
  width: 100%;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
  margin: 10px 0;
`;

const ConfidenceFill = styled.div<{ width: number }>`
  height: 100%;
  width: ${props => props.width}%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 4px;
  transition: width 0.3s ease;
`;

const ProbabilitiesSection = styled.div`
  margin-top: 30px;
`;

const ProbabilitiesTitle = styled.h3`
  color: #333;
  margin-bottom: 20px;
  text-align: center;
`;

const ProbabilityItem = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 15px;
  padding: 10px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
`;

const AgeLabel = styled.div`
  min-width: 120px;
  font-weight: 600;
  color: #333;
`;

const ProbabilityBarContainer = styled.div`
  flex: 1;
  height: 20px;
  background: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
  margin: 0 15px;
  position: relative;
`;

const ProbabilityBar = styled.div<{ width: number; isPredicted: boolean }>`
  height: 100%;
  width: ${props => props.width}%;
  background: ${props => 
    props.isPredicted 
      ? 'linear-gradient(90deg, #4CAF50 0%, #45a049 100%)'
      : 'linear-gradient(90deg, #2196F3 0%, #1976D2 100%)'
  };
  border-radius: 10px;
  transition: width 0.3s ease;
`;

const ProbabilityValue = styled.div`
  min-width: 60px;
  text-align: right;
  font-weight: 600;
  color: #333;
`;

// 2. Оновлюємо інтерфейс, щоб він приймав signal_data
interface PredictionResultProps {
  prediction: {
    predicted_age_group: number;
    confidence: number;
    all_probabilities: number[];
    signal_data?: number[]; // <--- Додано це поле (опціональне, щоб не ламалось якщо даних немає)
  };
}

const PredictionResult: React.FC<PredictionResultProps> = ({ prediction }) => {
  console.log('PredictionResult received:', prediction);
  // 3. Деструктуризація signal_data
  const { predicted_age_group, confidence, all_probabilities, signal_data } = prediction;
  
  const ageLabels = [
    '18-25', '26-30', '31-35', '36-40', '41-45',
    '46-50', '51-55', '56-60', '61-65', '66-70',
    '71-75', '76-80', '81-85', '86-90', '91+'
  ];

  return (
    <ResultContainer>
      <ResultHeader>
        <Title>🎯 Prediction Results</Title>
      </ResultHeader>
      
      <PredictionCard>
        <PredictionInfo>
          <InfoItem>
            <InfoLabel>Predicted Age Group</InfoLabel>
            <InfoValue highlight>{predicted_age_group}</InfoValue>
          </InfoItem>
          <InfoItem>
            <InfoLabel>Confidence</InfoLabel>
            <InfoValue>{(confidence * 100).toFixed(1)}%</InfoValue>
            <ConfidenceBar>
              <ConfidenceFill width={confidence * 100} />
            </ConfidenceBar>
          </InfoItem>
        </PredictionInfo>
        {signal_data && signal_data.length > 0 && (
          <ECGVisualizer data={signal_data} />
        )}
        <ProbabilitiesSection>
          <ProbabilitiesTitle>All Age Group Probabilities</ProbabilitiesTitle>
          {(all_probabilities || []).map((prob, index) => (
            <ProbabilityItem key={index}>
              <AgeLabel>Age {ageLabels[index]}</AgeLabel>
              <ProbabilityBarContainer>
                <ProbabilityBar 
                  width={prob * 100} 
                  isPredicted={index + 1 === predicted_age_group}
                />
              </ProbabilityBarContainer>
              <ProbabilityValue>{(prob * 100).toFixed(1)}%</ProbabilityValue>
            </ProbabilityItem>
          ))}
        </ProbabilitiesSection>
      </PredictionCard>
    </ResultContainer>
  );
};

export default PredictionResult;