function [m_fHhat, s_fRate] = m_fEncDec(m_fH, s_fDesRate, stSettings)

% Encode and decode a matrix using a lossy source code
%
% Syntax
% -------------------------------------------------------
% [m_fHhat, s_fRate] = m_fEncDec(m_fH, s_fDesRate, stSettings)
%
% INPUT:
% -------------------------------------------------------
% m_fH - data to encode (matrix)
% s_fDesRate - desired code rate (positive scalar)
% stSettings - code parameters (struct)
%
% OUTPUT:
% -------------------------------------------------------
% m_fHhat  - decoded data
% s_fRate  - code rate used (may be different from desired one)


switch stSettings.type
    % Dithered lattice quantization
    case 1
         s_nDim = 2;
          v_nLatticeScale = [10e-7, 10e-6, 10e-5, 10e-4, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20];     
    % Uniform quantization with random unitary rotation    
    case 2
        s_nDim = 16;
        
    % Subsampling with 3 bits quantizers
    case 3
        s_nDim = 1;
        % Dithered scalar quantization
    case 4
        s_nDim = 1;
        
end

 % Reshape
v_fH = m_fH(:);
% Zero pad if needed
s_nExtraZero = mod(length(v_fH),s_nDim);
if (s_nExtraZero) s_nExtraZero = s_nDim - s_nExtraZero; end     
v_fH = [v_fH; zeros(s_nExtraZero,1)];
% Scale to fit lattice - did not add in order to avoid error scaling
m_fEncInput = reshape(v_fH, s_nDim, []);

switch stSettings.type
    % Dithered lattice quantization
    case 1 
        % Hexagonal lattice
        s_nMaxIdx1 = 150;
        s_nMaxIdx2 = 150;
        m_fGenMat = [2, 0; 1, sqrt(3)]';
        
        % Find candidate matrices for which there is no overloading       
        s_fScale = max(vecnorm(m_fEncInput))/(min(s_nMaxIdx1,s_nMaxIdx2));
        v_nCandidateLattices = v_nLatticeScale(find(v_nLatticeScale >= s_fScale));
        if isempty(v_nCandidateLattices)
            v_nCandidateLattices = max(v_nLatticeScale);
        end
        
        m_fLattice = zeros(2,(2*s_nMaxIdx1 + 1)*(2*s_nMaxIdx2 + 1));
        idx = 1;
        for kk=-s_nMaxIdx1:s_nMaxIdx1
            for ll=-s_nMaxIdx2:s_nMaxIdx2
                m_fLattice(:,idx) =[kk;ll];
                idx = idx+1;
            end
        end
        % Update generator matrix by desired resolution
        s_fDelta = 1/(s_fDesRate ); %max(s_nMaxIdx1,s_nMaxIdx2)
        
        % Find lattice which achieves rate closest to desired one
        %s_fRateGap = inf;
%        for kkk=1:length(v_nCandidateLattices)
        kkk=1;
            % Scale generating matrix according to candidate lattices which
            % do not overload the input
            m_fGenMat2 = m_fGenMat * s_fDelta * v_nCandidateLattices(kkk);             
            m_fLattice = m_fGenMat2*m_fLattice;

            % Generate dither
            m_fDither = 0.5*diag(m_fGenMat2*[1;1])*(rand(size(m_fEncInput)) - 0.5);
            % Quantize
           % v_fLatIdx = dsearchn(m_fLattice',(m_fEncInput + m_fDither)');
            v_fLatIdx =  nearestneighbour((m_fEncInput + m_fDither),m_fLattice);
            m_fQ =  m_fLattice(:,v_fLatIdx);
            % Entopy coding_
            v_nCount = ones((2*s_nMaxIdx1 + 1)*(2*s_nMaxIdx2 + 1),1);
            v_nCount(unique(v_fLatIdx)) = histc(v_fLatIdx, unique(v_fLatIdx));
            s_nCode = arithenco(v_fLatIdx,v_nCount);
            % count number of bits in s_nCode into s_fRate
            s_fRate = length(s_nCode)/ length(m_fH(:));
%            if (abs(s_fRate - s_fDesRate)<s_fRateGap)
%                s_fRateGap = abs(s_fRate - s_fDesRate)
                % Decode by projecting back from lattice and substracting dither
                m_fDecOutput = m_fQ - m_fDither;
%               norm(m_fDecOutput-m_fEncInput)
%           end
%        end
        


      
    % Uniform quantization with random unitary rotation    
    case 2 
        
        % Generate random rotation using WH matrix
        s_nRnd = randsrc(s_nDim,1,[-1, 1]);
     	m_fRotMat = hadamard(s_nDim)*diag(s_nRnd) /sqrt(s_nDim);
        % Quantize
        m_fQ =  m_fQuant(m_fRotMat*m_fEncInput, floor(2^s_fDesRate), 1);
        % Recover
        m_fDecOutput = m_fRotMat'*m_fQ; 
        s_fRate = log2(floor(2^s_fDesRate));
        
       
        
    % Subsampling with 3 bits quantizers
    case 3
        % Get overall number of samples to keep
        s_nMaskSize = floor(length(v_fH)*s_fDesRate/3);
        % Generate random mask
        m_fDecOutput = zeros(size(v_fH));
        v_nMaskIdx = randi(length(v_fH),s_nMaskSize,1);
        m_fDecOutput(v_nMaskIdx) = m_fQuant(v_fH(v_nMaskIdx),8,2);
        s_fRate = length(unique(v_nMaskIdx))*3 / length(v_fH);
        
     % Dithered scalar quantization
    case 4
        s_fDynRange = 2*std(m_fEncInput);
        s_fDelta = 2*s_fDynRange / floor(2^s_fDesRate);
        m_fDither = (s_fDelta/2)*(rand(size(m_fEncInput)) - 0.5);
        m_fQ = m_fQuant(m_fEncInput + m_fDither, floor(2^s_fDesRate), s_fDynRange);
        
        
        % Entropy coding
        v_fLatIdx =  m_fQ(:)/(s_fDelta/2);
        v_fLatIdx = round(v_fLatIdx - min(v_fLatIdx) + 1); 
        v_nCount = ones(max(v_fLatIdx),1);
        v_nCount(unique(v_fLatIdx)) = histc(v_fLatIdx, unique(v_fLatIdx));
        s_nCode = arithenco(v_fLatIdx,v_nCount);
        % count number of bits in s_nCode into s_fRate
        s_fRate = length(s_nCode)/ length(m_fH(:));
        m_fDecOutput = m_fQ - m_fDither; 
end


% Remove zero padding
v_fDecOutput = m_fDecOutput(:);
v_fDecOutput = v_fDecOutput(1:end-s_nExtraZero);
m_fHhat = reshape(v_fDecOutput, size(m_fH));