%--------------------------------------------------------------------------------
% QMGO source codes

% Haogao Song, Lei Liu, Yi Chen, Qian Zhang, Huiling Chen

% Last update: 09 19 2025

% After use of code, please users cite the main paper: 
% QMGO: A Reinforcement Learning Decision Making Enhanced Moss Growth Optimizer for Feature Selection
% Haogao Song, Lei Liu, Yi Chen, Qian Zhang, Huiling Chen
%--------------------------------------------------------------------------------

% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end