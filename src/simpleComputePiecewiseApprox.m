% Generates a piecewise linear approximation.
%
% Inputs:
%   b_blocks     - Vector of Y-intercepts (b) for each block
%   a_blocks     - Vector of Slopes (a) for each block
%   x_data       - Vector of X-axis data (e.g., SOC or time)
%   block_ranges - Vector of normalized breakpoints (e.g., [0, 0.5, 1])
%                  defining relative start/end of blocks.
%
% Output:
%   y_line       - The resulting approximated lines vector
function y_line = simpleComputePiecewiseApprox(b_blocks, a_blocks, x_data, block_ranges)

    % Determine data size and number of blocks
    N_total  = length(x_data);
    N_blocks = length(block_ranges)-1;

    % Pre-allocate the output vector
    y_line = zeros(N_total, 1);

    % Loop through each block to compute segments
    for block = 1:N_blocks
        % Define the range of indices for the current block based on the ranges
        idx_range = floor(block_ranges(block)*N_total)+1 : floor(block_ranges(block+1)*N_total);
        
        % Extract the subset of x-data corresponding to these indices
        x_block = x_data(idx_range);
        
        % Compute the linear values (y = ax + b) for this segment
        y_line(idx_range) = x_block * a_blocks(block) + b_blocks(block);
    end 
end
