function [m_fHhat, s_fRate] = m_fQuantizeData(m_fH, s_fDesRate, stSettings)

% Encode and decode a matrix using a lossy source code
%
% Syntax
% -------------------------------------------------------
% [m_fHhat, s_fRate] = m_fEncDec(m_fH, s_fDesRate, stSettings)
%
% INPUT:
% -------------------------------------------------------
% m_fH - data to encode (matrix or vector)
% s_fDesRate - desired code rate (positive scalar)
% stSettings - code parameters (struct)
%
% OUTPUT:
% -------------------------------------------------------
% m_fHhat  - decoded data  (matrix or vector)
% s_fRate  - code rate used (may be different from desired one)


% Global variables - used to prevent re-generating lattices repeatedly
global gm_fGenMat3D;
global gm_fLattice3D;
global gm_fGenMat2D;
global gm_fLattice2D;

switch stSettings.type
    % Dithered 3-D lattice quantization
    case 1
        s_nDim = 3;
        % Dithered 2-D lattice quantization
    case 2
        s_nDim = 2;
        % Dithered scalar quantization
    case 3
        s_nDim = 1;
        % QSGD (non-subtractice dithered scalar quantization)
    case 4
        s_nDim = 1;
        % Uniform quantization with random unitary rotation
    case 5
        s_nDim = 16;
        % Subsampling with 3 bits quantizers
    case 6
        s_nDim = 1;
        
end

% Reshape into vector
v_fH = m_fH(:);
% Zero pad if needed
s_nExtraZero = mod(length(v_fH),s_nDim);
if (s_nExtraZero) s_nExtraZero = s_nDim - s_nExtraZero; end
v_fH = [v_fH; zeros(s_nExtraZero,1)];
% Encoder input divided into blocks
m_fEncInput = reshape(v_fH, s_nDim, []);
% Scaling of the standard deviation - experimental value
if (stSettings.scale == 1)
    % Max absolute value scaling
    s_fScale= max(max(abs(m_fEncInput)))+ 1e-10;
else %if (stSettings.scale == 2)
    % Standard deviation scaling
    s_fRatio = 2 + s_fDesRate/5;
    s_fScale= s_fRatio*sqrt(sum(std(m_fEncInput').^2) + 1e-10);
end

% Dithered lattice quantization
if ((stSettings.type == 1) || (stSettings.type == 2))
    % Normalize by twice the standard deviation to guarantee non-overloading
    %   s_fScale= s_fRatio*sqrt(sum(std(m_fEncInput').^2) + 1e-10);
    %   s_fScale= max(max(abs(m_fEncInput)))+ 1e-10;
    m_fEncInput = m_fEncInput/s_fScale;
    % 3-D lattice
    if (stSettings.type == 1)
        % Generate lattices if not previously generated
        if isempty(gm_fLattice3D)
            % Hexagonal lattice basic generator matrix
            m_fGenMat = inv([1, 0, 0; 0, 1, 0; 1/2, 1/2, 1/2]);
            [gm_fLattice3D, gm_fGenMat3D] = m_fGetLattice(m_fGenMat,s_fDesRate);
        end
        m_fGenMat_t = gm_fGenMat3D;
        m_fLattice = gm_fLattice3D;
        % 2-D lattice
    else
        % Generate lattices if not previously generated
        if isempty(gm_fLattice2D)
            % Hexagonal lattice basic generator matrix
            m_fGenMat =  [2, 0; 1, sqrt(3)]';
            [gm_fLattice2D, gm_fGenMat2D] = m_fGetLattice(m_fGenMat,s_fDesRate);
        end
        m_fGenMat_t = gm_fGenMat2D;
        m_fLattice = gm_fLattice2D;
    end
    if (stSettings.OptSearch == 1)
       % Generate dither
       m_fDither = m_fGenDither(m_fGenMat_t,size(m_fEncInput,2)); 
        
       % Quantize
       % v_fLatIdx = dsearchn(m_fLattice',(m_fEncInput + m_fDither)');
       v_fLatIdx =  nearestneighbour((m_fEncInput + m_fDither),m_fLattice);
       m_fQ =  m_fLattice(:,v_fLatIdx);
       
       % Entopy coding
       v_nCount = ones(size(m_fLattice(:)));
       v_nCount(unique(v_fLatIdx)) = histc(v_fLatIdx, unique(v_fLatIdx));
       s_nCode = arithenco(v_fLatIdx,v_nCount);
       % count number of bits in s_nCode into s_fRate
       s_fRate = length(s_nCode)/ length(v_fH);
    else     
        % Generate dither
        m_fDither = 0.5*m_fGenMat_t*(rand(size(m_fEncInput)) - 0.5);
        % Quantize
        m_fQ = m_fGenMat_t * round(inv(m_fGenMat_t)*(m_fEncInput + m_fDither));
        % No entropy coding
        s_fRate = s_fDesRate;
    end
    

    % Decode by projecting back from lattice and substracting dither
    m_fDecOutput = s_fScale*(m_fQ - m_fDither);
    
% Dithered scalar quantization
elseif (stSettings.type == 3)
    % Dynamic range determines the number of decision regions
    % s_fDynRange= max(max(abs(m_fEncInput)))+ 1e-10;
    % s_fDynRange = s_fRatio*(std(m_fEncInput)+ 1e-10);
    s_fDynRange = s_fScale;
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
    s_fRate = length(s_nCode)/ length(v_fH);
    m_fDecOutput = m_fQ - m_fDither;
    
% QSGD (non-subtractice dithered scalar quantization)
elseif (stSettings.type == 4)
    % Normalize by the input max (while the analysis in the paper uses the
    % norm, they state that in practice they use the max value)
    %s_fScale= max(max(abs(m_fEncInput)))+ 1e-10;
    m_fEncInput = m_fEncInput/s_fScale;
     % Dynamic range determines the number of decision regions
    s_fDynRange = 1;
%    s_fScale = 1;
%    s_fDynRange = s_fRatio*(std(m_fEncInput)+ 1e-10);
    s_fDelta = 2*s_fDynRange / floor(2^s_fDesRate);
    m_fDither = (s_fDelta/2)*(rand(size(m_fEncInput)) - 0.5);
    m_fDecOutput = s_fScale*m_fQuant(m_fEncInput + m_fDither, floor(2^s_fDesRate), s_fDynRange);
    
    s_fRate = log2(floor(2^s_fDesRate));
    
% Uniform quantization with random unitary rotation
elseif (stSettings.type == 5)
    % Normalize by the input max  
    %s_fScale= max(max(abs(m_fEncInput)))+ 1e-10;
    % Generate random rotation using WH matrix
    s_nRnd = randsrc(s_nDim,1,[-1, 1]);
    m_fRotMat = hadamard(s_nDim)*diag(s_nRnd) /sqrt(s_nDim);
    % Quantize
    m_fQ =  m_fQuant(m_fRotMat*m_fEncInput, floor(2^s_fDesRate), 1);
    % Recover
    m_fDecOutput = m_fRotMat'*m_fQ;
    s_fRate = log2(floor(2^s_fDesRate));
    
    
    
% Subsampling with 3 bits quantizers
elseif (stSettings.type == 6)
    % Normalize by the input max  
    %s_fScale= max(max(abs(m_fEncInput)))+ 1e-10;
    % Get overall number of samples to keep
    s_nMaskSize = floor(length(v_fH)*s_fDesRate/3);
    % Generate random mask
    m_fDecOutput = zeros(size(v_fH));
    v_nMaskIdx = randi(length(v_fH),s_nMaskSize,1);
    m_fDecOutput(v_nMaskIdx) = s_fScale*m_fQuant(v_fH(v_nMaskIdx)/s_fScale,8,1);
    s_fRate = length(unique(v_nMaskIdx))*3 / length(v_fH);
    

    
end


% Remove zero padding
v_fDecOutput = m_fDecOutput(:);
v_fDecOutput = v_fDecOutput(1:end-s_nExtraZero);
% Re-scale
m_fHhat =  reshape(v_fDecOutput, size(m_fH));