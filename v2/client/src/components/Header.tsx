import React from 'react';
import styled from 'styled-components';

const HeaderContainer = styled.header`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 40px 20px;
  text-align: center;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  margin: 0 0 10px 0;
  font-weight: 700;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  margin: 0;
  opacity: 0.9;
`;

const Header: React.FC = () => {
  return (
    <HeaderContainer>
      <Title>🫀 ECG Age Prediction</Title>
      <Subtitle>Upload your ECG files to predict age group using AI</Subtitle>
    </HeaderContainer>
  );
};

export default Header;

