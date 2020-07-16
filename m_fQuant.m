% Implement uniform quantization
function [m_fOutput]= m_fQuant(m_fInput, s_nLevels, s_fDynRange)
% Output:
%   m_fOutput -  quantized output
% input arguments:
%   m_fInput - input signal 
%   s_fDynRange - dynamic range
%   s_nLevels - quantizer dictionary size
 

s_fDelta = 2*s_fDynRange / s_nLevels;

s_fMaxVal = s_fDynRange - (s_fDelta/2);

% Apply uniform quantization
m_fQuantSig = s_fDelta*floor((m_fInput + s_fDynRange) / s_fDelta) - s_fMaxVal;

% Truncate words outside [-s_fMaxVal,s_fMaxVal]
m_fQuantSig(find(m_fQuantSig > s_fMaxVal)) = s_fMaxVal;
m_fQuantSig(find(m_fQuantSig < -s_fMaxVal)) = -s_fMaxVal;

m_fOutput = m_fQuantSig;