import React from 'react';
import {
  LineChart,
  Line,
  YAxis,
  XAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import styled from 'styled-components';

const ChartContainer = styled.div`
  margin: 30px 0;
  padding: 20px;
  background: white;
  border-radius: 15px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
`;

const ChartTitle = styled.h3`
  color: #333;
  margin-bottom: 20px;
  text-align: center;
  font-size: 1.2rem;
`;

interface ECGVisualizerProps {
  data: number[];
}

const ECGVisualizer: React.FC<ECGVisualizerProps> = ({ data }) => {
  // Recharts потребує масиву об'єктів
  const chartData = data.map((val, idx) => ({
    index: idx,
    value: val,
  }));

  if (!data || data.length === 0) return null;

  return (
    <ChartContainer>
      <ChartTitle>ECG Signal Preview</ChartTitle>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="index" hide />
            <YAxis domain={['auto', 'auto']} hide />
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
              itemStyle={{ color: '#667eea' }}
              labelStyle={{ display: 'none' }}
              formatter={(value: number) => [value.toFixed(3), 'Amplitude']}
            />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#667eea" 
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false} // Вимикаємо анімацію для продуктивності
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ChartContainer>
  );
};

export default ECGVisualizer;