function [m_fLattice, m_fGenMat_t] = m_fGetLattice(m_fGenMat,s_fDesRate)

% Generate lattice points inside the unit ball for vector quantization
%
% Syntax
% -------------------------------------------------------
% [m_fLattice, m_fGenMat_t] = m_fGetLattice(m_fGenMat,s_fDesRate)
%
% INPUT:
% -------------------------------------------------------
% m_fGenMat - basic lattice generator matix (matrix)
% s_fDesRate - desired code rate (positive scalar) 
%
% OUTPUT:
% -------------------------------------------------------
% m_fLattice  - lattice points (matrix)
% m_fGenMat_t  - updated generator matrix (matrix)

s_nDim = size(m_fGenMat,1);
% Number of lattice points in unit ball
%s_fPoints = (pi^(s_nDim/2))/(gamma(1+(s_nDim/2))*det(m_fGenMat));
% Number of desired lattice points
%s_fDesPoints = floor(2^(s_fDesRate*s_nDim + 1));
% Generate basis lattice - supporting 2-D and 3-D lattices
s_nMaxDim =  floor(2^(s_fDesRate));
m_fBaseLattice = zeros(s_nDim,(2*s_nMaxDim + 1)^s_nDim);
idx = 1;
if (s_nDim == 2)
    for kk=-s_nMaxDim:s_nMaxDim
        for ll=-s_nMaxDim:s_nMaxDim
            m_fBaseLattice(:,idx) =[kk;ll];
            idx = idx+1;
        end
    end
elseif (s_nDim == 3)
    for kk=-s_nMaxDim:s_nMaxDim
        for ll=-s_nMaxDim:s_nMaxDim
            for jj=-s_nMaxDim:s_nMaxDim
                m_fBaseLattice(:,idx) =[kk;ll;jj];
                idx = idx+1;
            end
        end
    end
end
% Scale generator matrix
m_fGenMat_t = m_fGenMat/(sqrt(det(m_fGenMat))*s_nMaxDim);
% Save all resulting lattice points inside the unit cube
m_fLattice = m_fGenMat_t*m_fBaseLattice;
m_fLattice = m_fLattice - 2*(m_fLattice > 1) + 2* (m_fLattice < -1);
