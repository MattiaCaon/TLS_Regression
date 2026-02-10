% --- 1. SETUP DATA ---
% Ensure x and y are column vectors
x = x(:);
y = y(:);
N = length(x);

% --- 2. CONFIGURATION ---
% TLS requires assumptions about the noise standard deviation on x and y.
% If unknown, assume they are equal (sx=1, sy=1).
sx = 1.0; 
sy = 1.0; 

N_iter = 5; % Number of iterations for the solver

% --- 3. INITIALIZATION ---
% Weighting matrix construction
% W is a 2N x 2N matrix. 
% Top-left block scales x-residuals, Bottom-right scales y-residuals.
si = diag([1/sx 1/sy]);
W = kron(si, eye(N));

% Initial Guess using standard Least Squares (LS)
% This provides a starting point for the coefficient 'a'
as = x \ y;     % Standard LS slope
xs = x;         % Initial estimate of "true" x is just the measured x

% --- 4. ITERATIVE TLS SOLVER ---
% Based on the "Inversion" section of your template
fprintf('Iter |  Slope (a)  | Residual Norm\n');
fprintf('-----------------------------------\n');

for iter = 1:N_iter
    % 1. Calculate current residuals based on estimated parameters
    ys = as * xs;
    
    dx = x - xs;       % Residual on x
    dy = y - ys;       % Residual on y
    d = [dx; dy];      % Stacked residual vector
    
    % 2. Construct Jacobian Matrix G
    % Gx represents derivatives for the x-equation
    % Gy represents derivatives for the y-equation
    Gx = [eye(N), zeros(N,1)];  
    Gy = [as*eye(N), xs];       
    G = [Gx; Gy];
    
    % 3. Apply Weights
    WG = W * G;
    Wd = W * d;
    
    % 4. Solve for updates (Gauss-Newton step)
    % xx contains updates for [x_1 ... x_N, a]
    xx = WG \ Wd;
    
    % 5. Update estimates
    xs = xs + xx(1:N);      % Update true x estimates
    as = as + xx(N+1);      % Update slope a
    
    % (Optional) Calculate weighted residual norm for monitoring
    res_norm = norm(Wd);
    fprintf('%4d | % .6f | %.4e\n', iter, as, res_norm);
end

% --- 5. RESULTS ---
a_tls = as; % Final estimated coefficient

fprintf('\nFinal TLS Coefficient (a): %.6f\n', a_tls);

% Plotting the result
figure;
plot(x, y, 'ko', 'DisplayName', 'Data'); hold on;
plot(x, a_tls*x, 'r-', 'LineWidth', 2, 'DisplayName', 'TLS Fit');
legend show;
grid on;
title(sprintf('TLS Fitting (a = %.4f)', a_tls));
xlabel('x'); ylabel('y');