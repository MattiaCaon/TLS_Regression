clear, clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modello
N = 10;
xt = randn(N,1); % incognite
at = 2; % parametro
yt = at*xt; % misura
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rumori sui dati
sx = .3;
sy = .1;
Nr = 1e5;
xm = xt + sx*randn(N,Nr); % riferimento
ym = yt + sy*randn(N,Nr); % misura da addestrare
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for r = 1:Nr
    Z = [xm(:,r)/sx ym(:,r)/sy];
    [U,S,V] = svd(Z);
    v2 = V(:,2);
    as = -v2(1)/v2(2)*sy/sx; % riportiamo le misure al di fuori dello spazio normalizzato
    Z1 = U(:,1)*S(1,1)*V(:,1)';
    xs = Z1(:,1)*sx;
    Jx(r) = mean((xt - xs).^2); 
    a_stim_SVD(r) = as;
end
rms_x_svd = sqrt(mean(Jx))
T{4,1} = 'TLS SVD';
T{4,2} = mean(a_stim_SVD - at);
T{4,3} = std(a_stim_SVD - at);
T{4,4} = sqrt(mean((a_stim_SVD - at).^2));
T