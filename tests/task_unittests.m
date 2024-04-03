classdef task_unittests < matlab.unittest.TestCase
    % TASK_UNITTESTS is a collection of functions to run unit tests on
    % various functions used to generate task-based variables for the agent.

    methods(Test)
        function test_constructormethods(obj)
            
            % test_constructormethods tests the output for all the
            % constructor methods within object Task().
            
            % INITIALIZE TASK AND OTHER VARS
            task = Task();
            task.C = linspace(-0.1,0.1,20);
            task.condition = 1;
            task.mu = 0.7;

            % COMPUTE EXPECTED OUTPUTS
            expected_p_cs = NaN(length(task.C), 2); 
            expected_p_cs(:, 1) = (task.C < 0) ./ sum(task.C < 0);
            expected_p_cs(:, 2) = (task.C >= 0) ./ sum(task.C >= 0);  

            expected_p_ras(1, 2) = 1 - task.mu;  
            expected_p_ras(2, 1) = 1 - task.mu;  
            expected_p_ras(1, 1) = task.mu;
            expected_p_ras(2, 2) = task.mu;

            % RUN TESTS
            obj.verifyEqual(task.p_cs,expected_p_cs, 'Incorrect p_cs');
            obj.verifyEqual(task.p_ras,expected_p_ras, 'Incorrect p_ras');
        end

        function test_statesample(obj)
            
            % test_statesample function tests the state_sample function
            % from the Task() object.
            
            % INITIALIZE TASK AND OTHER VARS
            task = Task();
            task.Theta = 0.5;
            task.state_sample();
            
            % RUN TESTS
            obj.verifyTrue(task.s_t == 0 || task.s_t == 1, 'Invalid state value');
            obj.verifyEqual(task.s_index, task.s_t + 1, 's_index calculation is incorrect');

            % Test case 1: State sample with Theta = 0 (no chance of success)
            task.Theta = 0;
            task.state_sample();
            assert(task.s_t == 0, 'Test case 1 failed: State should be 0 when Theta is 0.');
            assert(task.s_index == 1, 'Test case 1 failed: s_index should be 1 when state is 0.');
        
            % Test case 2: State sample with Theta = 1 (always successful)
            task.Theta = 1;
            task.state_sample();
            assert(task.s_t == 1, 'Test case 2 failed: State should be 1 when Theta is 1.');
            assert(task.s_index == 2, 'Test case 2 failed: s_index should be 2 when state is 1.');
        
            % Test case 3: State sample with 0 < Theta < 1
            task.Theta = 0.5;
            task.state_sample();
            assert(task.s_t == 0 || task.s_t == 1, 'Test case 3 failed: State should be either 0 or 1.');
            assert(task.s_index == task.s_t + 1, 'Test case 3 failed: s_index should be s_t + 1.');

        end

        function test_contrastsample(obj)
            
            % test_contrastsample function tests the contrast_sample function
            % from the Task() object.
            
            % INITIALIZE TASK AND OTHER VARS
            task = Task();
            task.s_index = 1;
            task.C = linspace(-0.1,0.1,20); 
            task.contrast_sample();
            
            % COMPUTE EXPECTED OUTPUTS
            expected_condiff = 0.1; % expected upper limit of contrast difference
            expected_length = 1; % expected length of contrast difference

            % RUN TESTS
            obj.verifyLessThanOrEqual(task.c_t,expected_condiff, ...
                'Contrast difference out of range') % check if generated contrast difference is out of range
            obj.verifyEqual(length(task.c_t), expected_length, 'Length of task.c_t no as expected'); % 
            obj.verifyLessThan(task.c_t,0,'Contrast difference does not match the task generated state')
        end

        function test_rewardsample(obj)
            
            % test_rewardsample function tests the reward_sample function
            % from the Task() object.
            
            % INITIALIZE TASK AND OTHER VARS
            task = Task();
            task.state_sample();
            task.reward_sample(1);

            % EXPECTED OUTPUTS
            expected_length = 1;

            % RUN TESTS
            obj.verifyEqual(length(task.r_t), expected_length, ...
                'Length of task.r_t no as expected'); % check if length of task generated contrast difference
            obj.verifyLessThanOrEqual(task.r_t,1,'task.r_t exceeds 1') % check if generated contrast difference is according to the state
        end
    end
end