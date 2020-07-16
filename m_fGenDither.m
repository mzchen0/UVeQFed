function m_fDither = m_fGenDither(m_fGenMat,s_nSamples)

% Generate dither uniformly distrubted over basis cell
%
% Syntax
% -------------------------------------------------------
% m_fDither = m_fGenDither(m_fGenMat,s_nSamples)
%
% INPUT:
% -------------------------------------------------------
% m_fGenMat - basic lattice generator matix (matrix)
% s_nSamples - number of samples to generate (positive scalar) 
%
% OUTPUT:
% -------------------------------------------------------
% m_fDither  - dither vectors points (matrix) 



s_nDim = size(m_fGenMat,1); 
s_nMaxDim =  1;
m_fBaseLattice = zeros(s_nDim,(2*s_nMaxDim + 1)^s_nDim);
idx = 1;
if (s_nDim == 2)
    for kk=-s_nMaxDim:s_nMaxDim
        for ll=-s_nMaxDim:s_nMaxDim
            m_fBaseLattice(:,idx) =[kk;ll];
            if (norm(m_fBaseLattice(:,idx))==0) s_nZeroIdx = idx; end
            idx = idx+1;
        end
    end
elseif (s_nDim == 3)
    for kk=-s_nMaxDim:s_nMaxDim
        for ll=-s_nMaxDim:s_nMaxDim
            for jj=-s_nMaxDim:s_nMaxDim
                m_fBaseLattice(:,idx) =[kk;ll;jj];
                if (norm(m_fBaseLattice(:,idx))==0) s_nZeroIdx = idx; end
                idx = idx+1;
            end
        end
    end
end
m_fLattice = m_fGenMat*m_fBaseLattice;

% Randomize points and select those that are closest to center
m_fPoints = diag(max(abs(m_fLattice')))*(rand(s_nDim, 3*s_nSamples) - 0.5);

v_fLatIdx =  nearestneighbour(m_fPoints,m_fLattice);
v_fIdx = find(v_fLatIdx==s_nZeroIdx);

if(length(v_fIdx) >=s_nSamples)
    m_fDither = m_fPoints(:,v_fIdx(1:s_nSamples));
else
    m_fDither = [m_fPoints(:,v_fIdx), m_fPoints(:,v_fIdx(1:(s_nSamples-length(v_fIdx))))];
end


