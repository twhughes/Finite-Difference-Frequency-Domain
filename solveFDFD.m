function [Ex, Ey, Ez, Hx, Hy, Hz, omega] = solveFDFD(pol, L0, wvlen, xrange, yrange, eps_r, SRC, Npml)
%% Input Parameters
% L0: length unit (e.g., L0 = 1e-9 for nm)
% wvlen: wavelength in L0
% xrange: [xmin xmax], range of domain in x-direction including PML
% yrange: [ymin ymax], range of domain in y-direction including PML
% eps_r: Nx-by-Ny array of relative permittivity
% SRC: Nx-by-Ny array of current source density (electric or magnetic)
% Npml: [Nx_pml Ny_pml], number of cells in x- and y-normal PML

%% Output Parameters
% Ez, Hx, Hy: Nx-by-Ny arrays of H- and E-field components
% dL: [dx dy] in L0
% A: system matrix of A x = b
% omega: angular frequency for given wvlen

%% Set up the domain parameters.
eps0 = 8.854e-12 * L0;  % vacuum permittivity in farad/L0
mu0 = pi * 4e-7 * L0;  % vacuum permeability in henry/L0
c0 = 1/sqrt(eps0*mu0);  % speed of light in vacuum in L0/sec

N = size(eps_r);  % [Nx Ny]
L = [diff(xrange) diff(yrange)];  % [Lx Ly]
dL = L./(N);  % [dx dy]

M = prod(N); 

omega = 2*pi*c0/wvlen;  % angular frequency in rad/sec

%% Deal with the s_factor
[Sxf, Sxb, Syf, Syb] = S_create(L0, wvlen, xrange, yrange, N, Npml); 

%% Construct derivate matrices with PML
Dyb = Syb * createDws('y', 'b', dL, N); 
Dxb = Sxb * createDws('x', 'b', dL, N); 
Dxf = Sxf * createDws('x', 'f', dL, N); 
Dyf = Syf * createDws('y', 'f', dL, N); 

%% Average the epsilon space and create the epsilon tensors 
eps_x = bwdmean_w(eps0 * eps_r, 'x');
eps_y = bwdmean_w(eps0 * eps_r, 'y'); 
eps_z = eps0 * eps_r; 

vector_eps_x = reshape(eps_x, M, 1); 
vector_eps_y = reshape(eps_y, M, 1); 
vector_eps_z = reshape(eps_z, M, 1); 

T_eps_x = spdiags(vector_eps_x, 0, M, M); 
T_eps_y = spdiags(vector_eps_y, 0, M, M); 
T_eps_z = spdiags(vector_eps_z, 0, M, M); 


%% Construct solver for either TE or TM polarization
    %% TE: Hz, Ex, Ey
    %% TM: Ez, Hx, Hy
if (strcmp(pol, 'TE'))
    %% Construct A matrix and b vector
    A = Dxf* T_eps_x^-1 *Dxb + Dyf* T_eps_y^-1* Dyb + omega^2*mu0*speye(M); 
    mz = reshape(SRC, M, 1); 
    b = 1i*omega*mz; 
    
    %% Solve the system of equations
    hz = A\b; 
    
    %% Find electric fields
    ex = -1i/omega * T_eps_y^-1 * Dyb * hz; 
    ey = 1i/omega * T_eps_x^-1 * Dxb * hz; 
    
    
    %% Return field matrices
    Hz = reshape(hz, N);
    Ex = reshape(ex, N); 
    Ey = reshape(ey, N); 
    
    Ez = []; 
    Hx = []; 
    Hy = []; 
        
    
elseif (strcmp(pol, 'TM'))
    %% Construct A matrix and b vector
    A = Dxb * mu0^-1 * Dxf + Dyb * mu0^-1 * Dyf + omega^2*T_eps_z; 
    jz = reshape(SRC, M, 1); 
    b = 1i * omega * jz; 
    
    %% Solve the system of equations
    ez = A\b; 
    
    %% Find magnetic fields
    hx = -1/(1i*omega) * mu0^-1 * Dyb * ez; 
    hy = 1/(1i*omega) * mu0^-1 * Dxb * ez; 

    %% Return field matrices
    Ez = reshape(ez, N);
    Hx = reshape(hx, N); 
    Hy = reshape(hy, N); 
    
    Ex = []; 
    Ey = []; 
    Hz = []; 
    
else
    error('Invalid polarization. Please specify either TE or TM'); 
end
    

end
