
% Generates a piecewise linear approximation based on intersection points of sequential line segments.
%
% Inputs:
%   b_blocks      - Vector of Y-intercepts (b) for each block
%   a_blocks      - Vector of Slopes (a) for each block
%   x_data        - Vector of X-axis data (SOC)
%
% Output:
%   y_line - The resulting approximated vector
function y_line = computePiecewiseApprox(b_blocks, a_blocks, x_data)
    % Determine sizes
    N_blocks = length(a_blocks);
    N_total  = length(x_data);
    
    % Initialize intercepts vector
    % Boundaries: 0, cut_1, cut_2, ..., cut_(N-1), N_total
    block_intercepts = zeros(1, N_blocks + 1);
    block_intercepts(1) = 0;

    % Calculate transition indices
    for k = 1:N_blocks
        if k < N_blocks
            % Calculate where line K and line K+1 intersect: 
            % a1*x + b1 = a2*x + b2  =>  x = (b2 - b1) / (a1 - a2)
            x_intersect = (b_blocks(k+1) - b_blocks(k)) / ...
                          (a_blocks(k) - a_blocks(k+1));
            
            % Find the index in data closest to this x-value
            [~, index] = min(abs(x_data - x_intersect));
            
            block_intercepts(k+1) = index;
        end
    end
    
    % Set the final boundary to the end of the data
    block_intercepts(end) = N_total;

    % Construct the result vector 
    % (Pre-allocate output to match the size of input data)
    y_line = zeros(N_total, 1); 

    for k = 1:N_blocks
        % Define the range for this block
        idx_start = block_intercepts(k) + 1;
        idx_end   = block_intercepts(k+1);
        
        % Safety check: ensure start is before end
        if idx_start <= idx_end
            idx_range = idx_start : idx_end;
            
            % Extract x segment
            x_block = x_data(idx_range);
            
            % Ensure column vector for calculation consistency
            if isrow(x_block), x_block = x_block'; end
            
            % Calculate y = ax + b
            segment_vals = x_block * a_blocks(k) + b_blocks(k);
            
            % Store in the final vector
            y_line(idx_range) = segment_vals;
        end
    end
end
