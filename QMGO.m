%--------------------------------------------------------------------------------
% QMGO source codes

% Haogao Song, Lei Liu, Yi Chen, Qian Zhang, Huiling Chen

% Last update: 09 19 2025

% After use of code, please users cite the main paper: 
% QMGO: A Reinforcement Learning Decision Making Enhanced Moss Growth Optimizer for Feature Selection
% Haogao Song, Lei Liu, Yi Chen, Qian Zhang, Huiling Chen
%--------------------------------------------------------------------------------
function [best_Fit, best_X, Convergence_curve] = QMGO(N, Maxit, lb, ub, D, fobj)

%% 1. Global Parameters and Initialization
it = 1;
Convergence_curve = [];

best_Fit = inf;
best_X = zeros(1, D);

X = initialization(N, D, ub, lb);
X_Fit = zeros(1, N);

for i = 1:N
    X_Fit(i) = fobj(X(i, :));
    if X_Fit(i) < best_Fit
        best_X = X(i, :);
        best_Fit = X_Fit(1, i);
    end
end

w = 2;
rec_num = 10;
divide_num = D / 4;
d1 = 0.2;

newX = zeros(N, D);
newX_Fit = zeros(1, N);

rec = 1;
rX = zeros(N, D, rec_num);
rX_Fit = zeros(1, N, rec_num);

%% Q-Learning Initialization
action_num_param = 2;
state_num = 2;
Reward_table = ones(N, state_num, action_num_param);
Reward_table(:, 1, 1) = -1;
Q_table = zeros(N, state_num, action_num_param);
cur_state = ones(N, 1);
action_num = ones(N, 1);

%% 2. Main Optimization Loop
while it < Maxit
    %% MGO: Determination of wind direction
    divX = X;
    div_num_perm = randperm(D);

    for j = 1:max(divide_num, 1)
        th = best_X(div_num_perm(j));
        index = divX(:, div_num_perm(j)) > th;
        if sum(index) < size(divX, 1) / 2
            index = ~index;
        end
        divX = divX(index, :);
    end

    DD = best_X - divX;
    D_wind = sum(DD, 1) / size(divX, 1); % Implements Eq. (1)

    %% MGO: Calculate Step Sizes and Parameters
    alpha_MGO = size(divX, 1) / N; % Implements Eq. (6)
    gama = 1 / sqrt(1 - power(alpha_MGO, 2));
    % The term (1 - it / Maxit) corresponds to E in Eq. (5)
    step1 = w * (rand(size(D_wind)) - 0.5) * (1 - it / Maxit); % Implements Eq. (3)
    step2 = 0.1 * w * (rand(size(D_wind)) - 0.5) * (1 - it / Maxit) * (1 + 1/2 * (1 + tanh(alpha_MGO / gama)) * (1 - it / Maxit)); % Implements Eq. (4)
    step3 = 0.1 * (rand() - 0.5) * (1 - it / Maxit); % Implements step3 from Eq. (9)
    act = actCal(1 ./ 1 + (0.5 - 10 * (rand(size(D_wind))))); % Corresponds to 'flag' calculation in Eq. (8)

    %% INFO Strategy Initialization (at the beginning of each cryptobiosis cycle)
    if rec == 1
        rX(:, :, rec) = X;
        rX_Fit(1, :, rec) = X_Fit;
        rec = rec + 1;
        [~, ind] = sort(X_Fit);
        X_Best_INFO = X(ind(1), :);
        Fit_Best_INFO = X_Fit(ind(1));
        Fit_Worst_INFO = X_Fit(ind(end));
        X_Worst_INFO = X(ind(end), :);
        I = randi([2 5]);
        X_Better_INFO = X(ind(I), :);
        Fit_Better_INFO = X_Fit(ind(I));
    end

    %% Update Each Search Agent
    for i = 1:N
        newX(i, :) = X(i, :);

        %% MGO: Spore dispersal search
        % Implements Eq. (2)
        if rand() > d1
            newX(i, :) = newX(i, :) + step1 .* D_wind;
        else
            newX(i, :) = newX(i, :) + step2 .* D_wind;
        end

        %% MGO: Dual propagation search
        % Implements Eq. (7)
        if rand() < 0.8
            if rand() > 0.5
                newX(i, div_num_perm(1)) = best_X(div_num_perm(1)) + step3 * D_wind(div_num_perm(1));
            else
                newX(i, :) = (1 - act) .* newX(i, :) + act .* best_X;
            end
        end

        Flag4ub = newX(i, :) > ub;
        Flag4lb = newX(i, :) < lb;
        newX(i, :) = (newX(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;

        newX_Fit(i) = fobj(newX(i, :));

        %% Q-Learning: Adaptive Strategy Selection
        if (Q_table(i, cur_state(i, 1), 1) >= Q_table(i, cur_state(i, 1), 2))
            action_num(i, 1) = 1;

            %% Action 1: RF Strategy
            a1 = 2 * exp(-(4 * it / Maxit)^2); % Implements Eq. (13)
            tempp = newX(i, :);
            num_d = D;
            pick = randperm(D);
            pick = pick(1:num_d);
            k = randperm(num_d);
            % Implements the main update logic of Eq. (11)
            for j = 1:k
                a2 = rand();
                if rand > 0.5
                    tempp(pick(j)) = tempp(pick(j)) + a1 * (lb + a2 * (ub - lb));
                else
                    tempp(pick(j)) = tempp(pick(j)) - a1 * (lb + a2 * (ub - lb));
                end
                if tempp(pick(j)) > ub || tempp(pick(j)) < lb
                    tempp(pick(j)) = lb + rand() * (ub - lb);
                end
            end
            for j = (k + 1):num_d
                % Implements the 'Temp' logic from Eq. (12)
                if i == 1
                    Temp = best_X(pick(j));
                else
                    Temp = newX(i - 1, pick(j));
                end
                tempp(pick(j)) = (Temp + tempp(pick(j))) / 2;
            end
        else
            action_num(i, 1) = 2;

            %% Action 2: INFO
            alpha_INFO = 2 * exp(-4 * (it / Maxit));
            del = 2 * rand * alpha_INFO - alpha_INFO; % Corresponds to delta in Eq. (25)
            sigm = 2 * rand * alpha_INFO - alpha_INFO;

            %% INFO: Updating Rule Stage
            A1 = randperm(N);
            A1(A1 == i) = [];
            a = A1(1); b = A1(2); c = A1(3);
            
            e = 1e-25;
            epsi = e * rand;

            omg = max([newX_Fit(a) newX_Fit(b) newX_Fit(c)]); % Corresponds to beta in Eq. (19)
            MM = [(newX_Fit(a) - newX_Fit(b)) (newX_Fit(a) - newX_Fit(c)) (newX_Fit(b) - newX_Fit(c))];
            
            % Corresponds to w1, w2, w3 in Eq. (16), (17), (18)
            v(1) = cos(MM(1) + pi) * exp(-(MM(1)) / omg);
            v(2) = cos(MM(2) + pi) * exp(-(MM(2)) / omg);
            v(3) = cos(MM(3) + pi) * exp(-(MM(3)) / omg);
            Wt = sum(v);
            
            WM1 = del .* (v(1) .* (newX(a, :) - newX(b, :)) + v(2) .* (newX(a, :) - newX(c, :)) + ...
                v(3) .* (newX(b, :) - newX(c, :))) / (Wt + 1) + epsi; % Implements Eq. (15)
            
            omg = max([Fit_Best_INFO Fit_Better_INFO Fit_Worst_INFO]); % Corresponds to y in Eq. (24)
            MM = [(Fit_Best_INFO - Fit_Better_INFO) (Fit_Best_INFO - Fit_Better_INFO) (Fit_Better_INFO - Fit_Worst_INFO)];
            
            % Corresponds to v1, v2, v3 in Eq. (21), (22), (23)
            v(1) = cos(MM(1) + pi) * exp(-MM(1) / omg);
            v(2) = cos(MM(2) + pi) * exp(-MM(2) / omg);
            v(3) = cos(MM(3) + pi) * exp(-MM(3) / omg);
            Wt = sum(v);

            WM2 = del .* (v(1) .* (X_Best_INFO - X_Better_INFO) + v(2) .* (X_Best_INFO - X_Worst_INFO) + ...
                v(3) .* (X_Better_INFO - X_Worst_INFO)) / (Wt + 1) + epsi; % Implements Eq. (20)
            
            r = unifrnd(0.1, 0.5);
            MeanRule = r .* WM1 + (1 - r) .* WM2; % Implements Eq. (14)

            %% INFO: Convergence Acceleration
            % Implements Eq. (26), (27), (28), (29)
            if rand < 0.5
                X1_INFO = newX(i, :) + sigm .* (rand .* MeanRule) + randn .* (X_Best_INFO - newX(a, :)) / (Fit_Best_INFO - newX_Fit(a) + 1);
                X2_INFO = X_Best_INFO + sigm .* (rand .* MeanRule) + randn .* (newX(a, :) - newX(b, :)) / (newX_Fit(a) - newX_Fit(b) + 1);
            else
                X1_INFO = newX(a, :) + sigm .* (rand .* MeanRule) + randn .* (newX(b, :) - newX(c, :)) / (newX_Fit(b) - newX_Fit(c) + 1);
                X2_INFO = X_Better_INFO + sigm .* (rand .* MeanRule) + randn .* (newX(a, :) - newX(b, :)) / (newX_Fit(a) - newX_Fit(b) + 1);
            end

            %% INFO: Vector Combining Stage
            u = zeros(1, D);
            for j = 1:D
                mu = 0.05 * randn; % Implements Eq. (31)
                % Implements Eq. (30)
                if rand < 0.5
                    if rand < 0.5
                        u(j) = X1_INFO(j) + mu * abs(X1_INFO(j) - X2_INFO(j));
                    else
                        u(j) = X2_INFO(j) + mu * abs(X1_INFO(j) - X2_INFO(j));
                    end
                else
                    u(j) = newX(i, j);
                end
            end

            %% INFO: Local Search Stage
            if rand < 0.5
                L = rand < 0.5;
                v1 = (1 - L) * 2 * (rand) + L; % Implements v1 from Eq. (36)
                v2 = rand .* L + (1 - L); % Implements v2 from Eq. (37)
                Xavg = (newX(a, :) + newX(b, :) + newX(c, :)) / 3; % Implements Eq. (35)
                phi = rand;
                Xrnd = phi .* (Xavg) + (1 - phi) * (phi .* X_Better_INFO + (1 - phi) .* X_Best_INFO); % Corresponds to Xk in Eq. (34)
                Randn_val = L .* randn(1, D) + (1 - L) .* randn;
                if rand < 0.5
                    u = X_Best_INFO + Randn_val .* (MeanRule + randn .* (X_Best_INFO - newX(a, :))); % Implements Eq. (32)
                else
                    u = Xrnd + Randn_val .* (MeanRule + randn .* (v1 * X_Best_INFO - v2 * Xrnd)); % Implements Eq. (33)
                end
            end
            FU = u > ub;FL = u < lb;
            tempp = (u .* (~(FU + FL))) + ub .* FU + lb .* FL;
        end

        FU = tempp > ub;
        FL = tempp < lb;
        tempp = (tempp .* (~(FU + FL))) + ub .* FU + lb .* FL;

        temf = fobj(tempp);

        %% Q-Learning: Update Reward and Q-Table
        % Implements Reward Function logic from Eq. (38)
        if (temf < newX_Fit(i))
            newX_Fit(i) = temf;
            newX(i, :) = tempp;
            Reward_table(i, cur_state(i, 1), action_num(i, 1)) = +1;
        else
            Reward_table(i, cur_state(i, 1), action_num(i, 1)) = -1;
        end

        maxQ = max(Q_table(i, action_num(i, 1), :));
        studyrate = 0.1;
        r_val = Reward_table(i, cur_state(i, 1), action_num(i, 1));
        % Implements Q-value update from Eq. (10)
        Q_table(i, cur_state(i, 1), action_num(i, 1)) = Q_table(i, cur_state(i, 1), action_num(i, 1)) + ...
            studyrate * (r_val + 0.9 * maxQ - Q_table(i, cur_state(i, 1), action_num(i, 1)));

        cur_state(i, 1) = action_num(i, 1);

        rX(i, :, rec) = newX(i, :);
        rX_Fit(1, i, rec) = newX_Fit(i);

        if newX_Fit(i) < best_Fit
            best_X = newX(i, :);
            best_Fit = newX_Fit(i);
        end
    end

    [~, ind] = sort(newX_Fit);
    X_Best_INFO = newX(ind(1), :);
    Fit_Best_INFO = newX_Fit(ind(1));
    Fit_Worst_INFO = newX_Fit(ind(end));
    X_Worst_INFO = newX(ind(end), :);
    I = randi([2 5]);
    X_Better_INFO = newX(ind(I), :);
    Fit_Better_INFO = newX_Fit(ind(I));
    
    rec = rec + 1;
    
    %% MGO: Cryptobiosis Mechanism
    if rec > rec_num || it >= Maxit
        [lcost, Iindex] = min(rX_Fit, [], 3);
        for i = 1:N
            X(i, :) = rX(i, :, Iindex(i));
        end
        X_Fit = lcost;
        rec = 1;
    end

    Convergence_curve(it) = best_Fit;
    it = it + 1;
end
end

%% Helper Functions
function [act] = actCal(X)
    act = X;
    act(act >= 0.5) = 1;
    act(act < 0.5) = 0;
end