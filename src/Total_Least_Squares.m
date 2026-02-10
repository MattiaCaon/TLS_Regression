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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lower bound: non può andar meglio del caso dove si conoscono con
% precisione i parametri
si = diag([1/sx 1/sy]);
W = kron(si,eye(N));
Gx = eye(N);
Gy = at*eye(N);
G = [Gx;Gy];
WG = W*G;
Gi = pinv(WG);
Cx = Gi*Gi';
rms_x_lb_I = sqrt(mean(diag(Cx)))
if 0 % controllo
    d = [xm;ym];
    Wd = W*d;
    xs = Gi*Wd;
    err_x = xt - xs;
    rms_x_lb_exp = sqrt(mean(err_x(:).^2))
    return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lower bound II: includo i parametri ma linearizzo rispetto alla soluzione
% corretta
Gx = [eye(N) zeros(N,1)];
Gy = [at*eye(N) xt];
G = [Gx;Gy];
WG = W*G;
Gi = pinv(WG);
Gi = pinv(WG);
Cx = Gi*Gi';
s = diag(Cx);
rms_x_lb_II = sqrt(mean(s(1:N)))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inversione rispetto a parametri e misure di riferimento (a,b,x)
% matrice di sbiancamento
si = diag([1/sx 1/sy]);
W = kron(si,eye(N));
N_iter = 5;
for r = 1:Nr
    % Inizializzazione: prendo il riferimento per buono e inverto rispetto
    % ai parametri
    as = xm(:,r)\ym(:,r);
    xs = xm(:,r);
    a_stim_LS(r) = as;
    % Iterazioni: linearizzo rispetto a (a,b,x) stimati e inverto
    for iter = 1:N_iter
        Jx(r,iter) = mean((xt - xs).^2); % errore su x (per validare il metodo)
        ys = as*xs;
        dx = xm(:,r) - xs;
        dy = ym(:,r) - ys;
        Gx = [eye(N) zeros(N,1)];
        Gy = [as*eye(N) xs];
        G = [Gx;Gy];
        d = [dx;dy];
        WG = W*G;
        Wd = W*d;
        Jd(r,iter) = mean(Wd.^2); % residuo (misurabile)
        
        xx = WG\Wd;
        xs = xs + xx(1:N);
        as = as + xx(N+1);
    end
    a_stim_TLS(r) = as;
end

rms_x = sqrt(mean(Jx,1))
rms_res = sqrt(mean(Jd,1));
figure
subplot(2,1,1), plot(1:N_iter,rms_res), grid, xlabel('iteration')
title('weighted residual on x and y'), ylim([0 max(ylim)])
subplot(2,1,2), 
plot([1 N_iter],rms_x_lb_I*[1 1],'k-'), hold on
plot([1 N_iter],rms_x_lb_II*[1 1],'r-'),
plot(1:N_iter,rms_x,'b'), grid, hold on, xlabel('iteration')
title('error on x'), ylim([0 max(ylim)])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Confronto con LS
T{1,1} = 'LS vs TLS';
T{1,2} = 'bias';
T{1,3} = 'std';
T{1,4} = 'RMSE';
T{2,1} = 'LS';
T{3,1} = 'TLS';
T{2,2} = mean(a_stim_LS - at);
T{2,3} = std(a_stim_LS - at);
T{2,4} = sqrt(mean((a_stim_LS - at).^2));
T{3,2} = mean(a_stim_TLS - at);
T{3,3} = std(a_stim_TLS - at);
T{3,4} = sqrt(mean((a_stim_TLS - at).^2));

% Istogrammi
[h,bin] = hist(a_stim_LS - at,51);
h_LS = h/Nr;
[h] = hist(a_stim_TLS - at,bin);
h_TLS = h/Nr;
figure
subplot(2,1,1), plot(bin,h_LS,bin,h_TLS), grid
legend('LS','TLS'), xlabel('a')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PER COMPRENDERE IL BIAS DEL LS
% il problema che viene risolto è: ym = xm*a
% => xm'*ym = xm'*xm*as => as = xm'*ym/(xm'*xm)
% Assumiamo per semplicità che:
% 1) ny = 0 => ym = y = a*x
% 2) nx sia piccolo rispetto a x, in modo da approssimare
% 1/(xm'*xm) = 1/(x'*x + 2*x'*nx + nx'*nx) = 
% = 1/(x'*x) - 1/(x'*x)^2*(2*x'*nx + nx'*nx)
% In questo modo si ha:
% as = xm'*ym/(xm'*xm) = a*(x+nx)'*x*{1/(x'*x) - 1/(x'*x)^2*(2*x'*nx + nx'*nx)}
% Approssimando ulteriormente al nominatore (x+nx) = x e prendendo il
% valore atteso si ha:
% Prendendo i valori attesi si ha:
% E[as] = a*(1 - N*sx^2/(x'*x))
% che rivela come ci sia una sottostima metodica
as_th = at*(1 - N*sx^2/(xt'*xt));
bias_th = at - as_th
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLUZIONE DIRETTA CON SVD - solo per problemi lineari
% l'svd è molto limitata in quanto alla gestione pesi. Di fatto funziona
% solo con pesi costanti
clear Jx Jy
for r = 1:Nr
    Z = [xm(:,r)/sx ym(:,r)/sy];
    [U,S,V] = svd(Z);
    v2 = V(:,2);
    as = -v2(1)/v2(2)*sy/sx;
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

