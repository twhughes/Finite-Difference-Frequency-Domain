function Dws = createDws(w, s, dL, N)
% w: one of 'x', 'y', 'z'
% s: one of 'f' and 'b'
% dL: [dx dy dz] for 3D; [dx dy] for 2D
% N: [Nx Ny Nz] for 3D; [Nx Ny] for 2D

dw = dL('xyz' == w);  % one of dx, dy, dz
sign = diff('bf' == s);  % +1 for s=='f'; -1 for s=='b'
M = prod(N);  % total number of cells in domain


% Dws = zeros(M);  % Initialize Dws into the appropriate square matrix

% Compute Nx, Ny, and Nz
Nx = N(1); 
Ny = N(2); 
if (length(N) == 3)
    Nz = N(3); 
else
    Nz = 1; 
end

% Check the direction of the derivative matrix to be created
switch w 
    case 'x'
        % Construct a block of the derivative matrix in the x-direction
        block_x = -speye(N(1)) + circshift(speye(N(1)), -1); 

        % Create Dws = Dxs by stacking up the blocks
        for n = 0 : 1 : (Ny*Nz - 1)
            Dws(1 + n*Nx : (n+1)*Nx, 1 + n*Nx : (n+1)*Nx) = 1/dw * block_x; 
        end
        
    case 'y'
        % Construct the block for Dys
%         block_y = zeros(Nx*Ny); 
        for n = 0 : 1 : Ny-1
            block_y(1 + n*Nx : (n+1)*Nx , 1 + n*Nx : (n+1)*Nx) = -speye(Nx); 

            if (n < Ny - 1)
                block_y(1 + n*Nx : (n+1)*Nx , 1 + (n+1)*Nx : (n+2)*Nx) = speye(Nx); 
            else
                block_y(1 + n*Nx : (n+1)*Nx , 1 : Nx) = speye(Nx); 
            end
        end
        
        % Construct Dws = Dys out of blocks
        for n = 0 : 1 : Nz-1
            Dws(1 + n*(Nx*Ny) : (n+1)*(Nx*Ny) , 1 + n*(Nx*Ny) : (n+1)*(Nx*Ny)) = 1/dw * block_y; 
        end
        
    case 'z'
        % Construct Dzs by stacking identity matrices
        for n = 0 : 1 : Nz - 1
            Dws(1 + n*(Nx*Ny) : (n+1)*(Nx*Ny) , 1 + n*(Nx*Ny) : (n+1)*(Nx*Ny)) = -speye(Nx*Ny); 

            if n < Nz - 1
                Dws(1 + n*(Nx*Ny) : (n+1)*(Nx*Ny) , 1 + (n+1)*(Nx*Ny) : (n+2)*(Nx*Ny)) = speye(Nx*Ny); 
            else
                Dws(1 + n*(Nx*Ny) : (n+1)*(Nx*Ny) , 1 : Nx*Ny) = speye(Nx*Ny); 
            end
        end
        % Rescale Dws = Dzs
        Dws = 1/dw * Dws; 
        
end

% Check if the matrix needs to be transposed for backwards matrices
if sign == -1
    Dws = -transpose(Dws); 
end

end

