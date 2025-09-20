
% =============================================================================================================================================================================
% Differentiated Creative Search (DCS)
% CITATION:
% Duankhan P., Sunat K., Chiewchanwattana S., and Nasa-ngium P. "The Differentiated Creative Search (DCS): Leveraging Differentiated Knowledge-acquisition and Creative Realism
% to Address Complex Optimization Problems". (Accepted for publication in Expert Systems with Applications)
% =============================================================================================================================================================================
function [best_pos,convergence_curve] = DCS2(N,MaxFEs,Lb,Ub,dim,fobj)
    Lb = Lb * ones(1, dim);
    Ub = Ub * ones(1, dim);
    % Parameters
    trail_X = zeros(N,dim);
    newX = zeros(N,dim);
    convergence_curve = [];
    eta_qKR = zeros(1,N); %%%%%
    new_fitness = zeros(N,1);
    % Golden ratio
    golden_ratio = 2/(1 + sqrt(5)); %%%%%
    % High-performing individuals
    ngS = max(6,round(N * (golden_ratio/3))); %%%%% (best performance)

    % Initialize the population
    X = zeros(N,dim);
    for i = 1:N
        X(i,:) = Lb + rand(1,dim) .* (Ub - Lb);
    end

    % Initialize fitness values
    AllFitness = zeros(N,1);
    for i = 1:N
        AllFitness(i,1) = fobj(X(i, :));
    end
    % Generation
    FEs = 0;
    itr = 1;
    pc = 0.5; %%%%%
    % Best solution
    best_fitness = min(AllFitness);
    % Ranking-based self-improvement
    phi_qKR = 0.25 + 0.55 * ((0 + ((1:N)/N)) .^ 0.5); %%%%% （the individual's phi coefficient value the higher the disparity 0.35-0.8)
    while FEs < MaxFEs
        % Sort population by fitness values
        [X, AllFitness, ~] = PopSort(X,AllFitness);
        % Reset
        bestInd = 1;
        % Compute social impact factor
        lamda_t = 0.1 + (0.518 * ((1-(FEs/MaxFEs)^0.5))); %%%%% (The later the lower focus more on themselevs over time 0.618-0.1)
        % (Comply with the non-high performance individual start from the
        % number 7)
        for i = 1:N
            % Compute differentiated knowledge-acquisition rate
            eta_qKR(i) = (round(rand * phi_qKR(i)) + (rand <= phi_qKR(i)))/2; %%%%%（quantized knowledge acquisition rate, the lower the rank)
            jrand = floor(dim * rand + 1); %%%%% (some dimension)
            trail_X(i,:) = X(i,:);
            if i == N && rand < pc
                % Low-performing
                trail_X(i,:) = Lb + rand * (Ub - Lb); %%% 1
            elseif i <= ngS %%%%%
                % High-performing
                while true
                    r1 = round(N * rand + 0.5);
                    if r1 ~= i && r1 ~= bestInd
                        break
                    end
                end

                for d = 1:dim
                    if rand <= eta_qKR(i) || d == jrand
                        trail_X(i,d) = X(r1,d) + LnF3(golden_ratio,0.05,1,1); %%% 2
                    end
                end
            else
                % Average-performing
                while true
                    r1 = round(N * rand + 0.5);
                    if r1 ~= i && r1 ~= bestInd
                        break
                    end
                end

                while true
                    r2 = ngS + round((N - ngS) * rand + 0.5);
                    if r2 ~= i && r2 ~= bestInd && r2 ~= r1
                        break
                    end
                end

                % Compute learning ability
                omega_it = rand; %%%%% （Comply with the whole poppulation)
                for d = 1:dim
                    if rand <= eta_qKR(i) || d == jrand
                        % trail_X(i,d) = X(bestInd,d) + ((X(r2,d) - X(i,d)) * lamda_t) + ((X(r1,d) - X(i,d)) * omega_it); %%% 3
                    end
                end
            end
            % Boundary
            trail_X(i,:) = boundConstraint(trail_X(i,:),X(i,:),[Lb; Ub]);
            newX(i, :) = trail_X(i, :);
            new_fitness(i,1) = fobj(trail_X(i, :));
            FEs = FEs + 1;
            if new_fitness(i,1) <= AllFitness(i,1)
                X(i,:) = newX(i,:);
                AllFitness(i,1) = new_fitness(i,1);
                if new_fitness(i,1) < best_fitness
                    best_fitness = new_fitness(i,1);
                    bestInd = i;
                end
            end
        end
        best_pos = X(bestInd,:);
        best_cost = best_fitness;
        convergence_curve(itr) = best_fitness;
        itr = itr + 1;
    end
end


function [sorted_population, sorted_fitness, sorted_index] = PopSort(input_pop,input_fitness)
    [sorted_fitness, sorted_index] = sort(input_fitness,1,'ascend');
    sorted_population = input_pop(sorted_index,:);
end


function Y = LnF3(alpha, sigma, m, n)
    Z = laplacernd(m, n);
    Z = sign(rand(m,n)-0.5) .* Z;
    U = rand(m, n);
    R = sin(0.5*pi*alpha) .* tan(0.5*pi*(1-alpha*U)) - cos(0.5*pi*alpha);
    Y = sigma * Z .* (R) .^ (1/alpha);
end


function x = laplacernd(m, n)
    u1 = rand(m, n);
    u2 = rand(m, n);
    x = log(u1./u2);
end


function vi = boundConstraint(vi, pop, lu)
    % if the boundary constraint is violated, set the value to be the middle
    % of the previous value and the bound
    %
    % Version: 1.1   Date: 11/20/2007
    % Written by Jingqiao Zhang, jingqiao@gmail.com
    [NP, ~] = size(pop);  % the population size and the problem's dimension
    % check the lower bound
    xl = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;
    % check the upper bound
    xu = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end