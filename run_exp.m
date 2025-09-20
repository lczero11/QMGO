%% CEC17 Benchmark Testing with Status-Based Optimization (SBO)
clc;clear;close all;
addpath(genpath(pwd));

% Parameters: 
POP_SIZE = 30; Maxit = 10000; DIM = 30; LB = -100; UB = 100;

cec17_function_names = {
    'Shifted and Rotated Bent Cigar Function', ...
    'Shifted and Rotated Sum of Different Power Function', ...
    'Shifted and Rotated Zakharov Function', ...
    'Shifted and Rotated Rosenbrock''s Function', ...
    'Shifted and Rotated Rastrigin''s Function', ...
    'Shifted and Rotated Expanded Scaffer''s F6 Function', ...
    'Shifted and Rotated Lunacek Bi-Rastrigin Function', ...
    'Shifted and Rotated Non-Continuous Rastrigin''s Function', ...
    'Shifted and Rotated Levy Function', ...
    'Shifted and Rotated Schwefel''s Function', ...
    'Hybrid Function 1 (N=3)', ... % CEC17 F11
    'Hybrid Function 2 (N=3)', ... % CEC17 F12
    'Hybrid Function 3 (N=3)', ... % CEC17 F13
    'Hybrid Function 4 (N=4)', ... % CEC17 F14
    'Hybrid Function 5 (N=4)', ... % CEC17 F15
    'Hybrid Function 6 (N=4)', ... % CEC17 F16
    'Hybrid Function 7 (N=5)', ... % CEC17 F17
    'Hybrid Function 8 (N=5)', ... % CEC17 F18
    'Hybrid Function 9 (N=5)', ... % CEC17 F19
    'Hybrid Function 10 (N=6)', ...% CEC17 F20
    'Composition Function 1 (N=3)', ... % CEC17 F21
    'Composition Function 2 (N=3)', ... % CEC17 F22
    'Composition Function 3 (N=3)', ... % CEC17 F23
    'Composition Function 4 (N=4)', ... % CEC17 F24
    'Composition Function 5 (N=4)', ... % CEC17 F25
    'Composition Function 6 (N=4)', ... % CEC17 F26
    'Composition Function 7 (N=5)', ... % CEC17 F27
    'Composition Function 8 (N=5)', ... % CEC17 F28
    'Composition Function 9 (N=5)', ... % CEC17 F29
    'Composition Function 10 (N=6)'  % CEC17 F30
};

fprintf('\n=== CEC17 Benchmark Suite Evaluation ===\n');
fprintf('Algorithm: Status-Based Optimization (SBO)\n');
fprintf('Configuration: PopSize=%d, Dim=%d, Maxit=%d\n\n', POP_SIZE, DIM, Maxit);

% Initialize results storage
results = struct('fnum',[],'best_pos',[],'best_val',[],'runtime',[]);
start_time = tic;

for func_num = 1:30
    % Function-specific header
    fprintf('┌──────────────────────────────────────────────┐\n');
    fprintf('│  FUNCTION #%02d - %-38s \n', func_num, cec17_function_names{func_num});
    fprintf('└──────────────────────────────────────────────┘\n');
    
    % Initialize function
    fobj = @(x) cec17_func(x', func_num);
    func_start = tic;
    
    % Run optimization
    [best_pos, convergence_curve] = QMGO(POP_SIZE, Maxit, LB, UB, DIM, fobj);
    
    % Store results
    runtime = toc(func_start);
    results(func_num).fnum = func_num;
    results(func_num).best_pos = best_pos;
    results(func_num).best_val = convergence_curve(end);
    results(func_num).runtime = runtime;
    
    % Display function results 
    fprintf('├─ Best Fitness: %.3e\n', convergence_curve(end));
    fprintf('├─ Best Solution: [%s]\n', strjoin(arrayfun(@(x) sprintf('%.3f',x), best_pos, 'UniformOutput', false), ', '));
    fprintf('├─ Convergence: %.2e → %.2e\n', convergence_curve(1), convergence_curve(end));
    fprintf('└─ Runtime: %.2f seconds\n\n', runtime);
    
    % Progress update every 5 functions
    if mod(func_num,5) == 0
        fprintf('✓ Completed %d/30 functions (%.1f%%)\n', ...
                func_num, 100*func_num/30);
        fprintf('  Current average runtime: %.2f sec/function\n\n', ...
                toc(start_time)/func_num);
    end
end

%% Final Report
fprintf('\n════════════════ TEST SUMMARY ════════════════\n');
fprintf('Total computation time: %.2f minutes\n', toc(start_time)/60);
fprintf('Average runtime per function: %.2f seconds\n', mean([results.runtime]));