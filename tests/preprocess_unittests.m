classdef preprocess_unittests < matlab.unittest.TestCase

    methods(Test)

        function test_flipmu(obj)

            % INTIALIZE VARS
            preprocess_obj = preprocess_LR();
            num_trials = 10; % number of trials on which the test needs to be run on
            preprocess_obj.mu = linspace(0.5,1,num_trials).';
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);
            preprocess_obj.data.congruence = repelem(0,num_trials,1);
            preprocess_obj.flipped_mu = NaN(num_trials,1);
            preprocess_obj.flip_mu;

            % EXPECTED
            expected_flippedmu = 1-preprocess_obj.mu;

            % RUN TEST
            obj.verifyEqual(preprocess_obj.flipped_mu,expected_flippedmu, ...
                'Expected and actual congruence flipped mu do not match.')
        end

        function test_computeactiondeprew(obj)

            % INTIALIZE VARS
            preprocess_obj = preprocess_LR();
            num_trials = 10; % number of trials on which the test needs to be run on
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);
            rng(23)
            preprocess_obj.obtained_reward = randi([0, 1],num_trials,1);
            rng(123)
            preprocess_obj.action = randi([0, 1],num_trials,1);
            preprocess_obj.recoded_reward = NaN(num_trials,1);
            preprocess_obj.compute_action_dep_rew;

            % EXPECTED
            expected_recodedrew = NaN(num_trials,1);
            for n = 1:num_trials
                expected_recodedrew(n) = preprocess_obj.obtained_reward(n) + (preprocess_obj.action(n)*((-1) .^ ...
                        (2 + preprocess_obj.obtained_reward(n))));
            end

            % RUN TEST
            obj.verifyEqual(preprocess_obj.recoded_reward,expected_recodedrew, ...
                'Expected and actual recoded rewards array do not match.')
        end

        function test_computemu(obj)

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            num_trials = 10; % number of trials on which the test needs to be run on
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);
            preprocess_obj.data.contrast = repelem(1,num_trials,1); % set actual mu < or > 0.5
            preprocess_obj.mu_t = NaN(num_trials,1);
            preprocess_obj.mu_t_1 = NaN(num_trials,1);
            preprocess_obj.compute_mu;

            % EXPECTED
            expected_mu_t_1 = NaN(num_trials,1);
            expected_mu_t = NaN(num_trials,1);
           for i = 2:height(preprocess_obj.data)
                if preprocess_obj.data.contrast(i) == 1 % if actual mu < 0.5
                    expected_mu_t_1(i) = 1-preprocess_obj.flipped_mu(i-1);
                    expected_mu_t(i) = 1-preprocess_obj.flipped_mu(i);
                else
                    expected_mu_t_1(i) = preprocess_obj.flipped_mu(i-1);
                    expected_mu_t(i) = preprocess_obj.flipped_mu(i);
                end
           end

           % RUN TEST
            obj.verifyEqual(preprocess_obj.mu_t,expected_mu_t, ...
                'Expected and actual mu for current trial array do not match.')
            obj.verifyEqual(preprocess_obj.mu_t_1,expected_mu_t_1, ...
                'Expected and actual mu for previous trial array do not match.')
        end

        function test_computestatedeppe(obj)

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            num_trials = 10; % number of trials on which the test needs to be run on
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);
            rng(123)
            preprocess_obj.state = randi([0, 1],num_trials,1);
            preprocess_obj.absolute_lr = 0;

            preprocess_obj.flip_mu;
            preprocess_obj.compute_mu;
            preprocess_obj.compute_action_dep_rew;
            preprocess_obj.compute_state_dep_pe;

            % EXPECTED
            expected_pe = zeros(num_trials,1);
            expected_up = zeros(num_trials,1);
            for i = 2:height(preprocess_obj.data)
                if preprocess_obj.state(i) == 0
                    expected_pe(i) = preprocess_obj.recoded_reward(i) - preprocess_obj.mu_t_1(i);
                else
                    expected_pe(i) = (1-preprocess_obj.recoded_reward(i))-preprocess_obj.mu_t_1(i);
                end
                expected_up(i) = preprocess_obj.mu_t(i) - preprocess_obj.mu_t_1(i);
            end
            preprocess_obj.data.pe(preprocess_obj.data.trials == 1,1) = 0;
            if preprocess_obj.absolute_lr == 1 % for absolute LR analysis
                expected_pe = abs(expected_pe);
                expected_up = abs(expected_up);
            end

            % RUN TEST
            obj.verifyEqual(preprocess_obj.data.pe,expected_pe, ...
                'Expected and actual PE arrays do not match.')
            obj.verifyEqual(preprocess_obj.data.up,expected_up, ...
                'Expected and actual UP arrays do not match.')
        end

        function test_computeconfirm(obj)

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            num_trials = 10; % number of trials on which the test needs to be run on
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);
            preprocess_obj.obtained_reward = randi([0, 1],num_trials,1);
            preprocess_obj.state = randi([0, 1],num_trials,1);
            preprocess_obj.action = randi([0, 1],num_trials,1);
            preprocess_obj.data.contrast = repelem(1,num_trials,1); % set actual mu < or > 0.5

            preprocess_obj.compute_confirm;

            % EXPECTED
            expected_confirmrew = zeros(num_trials,1);
            for i = 1:num_trials
                if preprocess_obj.data.contrast(i) == 1 % actual mu < 0.5
                    if preprocess_obj.state(i) == preprocess_obj.action(i) % the less rewarding state and action combination
                        expected_confirmrew(i) = 1-preprocess_obj.obtained_reward(i);
                    else
                        expected_confirmrew(i) = preprocess_obj.obtained_reward(i);
                    end
                else
                    if preprocess_obj.state(i) == preprocess_obj.action(i) % the less rewarding state and action combination
                        expected_confirmrew(i) = 1-preprocess_obj.obtained_reward(i);
                    else
                        expected_confirmrew(i) = preprocess_obj.obtained_reward(i);
                    end
                end
            end

            % RUN TEST
            obj.verifyEqual(preprocess_obj.data.confirm_rew,expected_confirmrew, ...
                'Expected and actual confirmation bias arrays do not match.')
        end

        function test_removeconditions(obj)

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            num_trials = 10; % number of trials on which the test needs to be run on
            preprocess_obj.removed_cond = 1;
            preprocess_obj.agent = 0;
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);
            preprocess_obj.condition = repelem(1,num_trials,1);
            preprocess_obj.remove_conditions;

            % EXPECTED
            preprocess_obj.data = preprocess_obj.data(preprocess_obj.condition ~= preprocess_obj.removed_cond,:);
            if preprocess_obj.agent == 0
                expected_condition = preprocess_obj.data.choice_cond;
            else
                expected_condition = preprocess_obj.data.condition;
            end

            % RUN TEST
            obj.verifyEqual(preprocess_obj.condition,expected_condition, ...
                'Expected and actual condition arrays do not match.')
        end

        function test_computenormalise(obj)

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            num_trials = 10;
            var_normalise = linspace(0,1,10).';
            normalised = preprocess_obj.compute_normalise(var_normalise);

            % EXPECTED
            expected_normalise = NaN(num_trials,1);
            denom = max(var_normalise) - min(var_normalise);
            for i = 1:length(var_normalise) % normalise
                num = var_normalise(i) - min(var_normalise);
                expected_normalise(i) = num./denom;
            end

            % RUN TESTS
            obj.verifyEqual(normalised,expected_normalise, ...
                'Normalisation not working as expected.')
        end

        function test_computeru(obj)

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            num_trials = 10;
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);
            preprocess_obj.condition = repelem(1,num_trials,1);

            preprocess_obj.compute_ru;

            % EXPECTED
            expected_ru = NaN(num_trials,1);
            for i = 1:height(preprocess_obj.data)
                if preprocess_obj.condition(i) == 1
                    expected_ru(i) = 0;
                else
                    expected_ru(i) = 1;
                end
            end

            % RUN TESTS
            obj.verifyEqual(preprocess_obj.data.ru,expected_ru, ...
                'Categorical variable for RU not correct.')
        end

        function test_addvars(testCase)
            % Test adding a single column to an empty table

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();  % Replace YourClass with the actual class name
            num_trials = 10;
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);
            
            new_column = linspace(0,1,num_trials).';
            varName = 'new_column';
            preprocess_obj.add_vars(new_column, varName);

            % RUN TESTS
            testCase.assertClass(preprocess_obj.data, 'table');
            testCase.verifySize(preprocess_obj.data.(varName), [num_trials, 1]);
            testCase.verifyEqual(preprocess_obj.data.(varName), new_column);
        end

        function test_removezerope(testCase)
            % Test removing trials with PE = 0 from a table with zero PEs

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();  
            num_trials = 10;
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);

            preprocess_obj.data = table([1; 2; 0; 4; 5; 1; 5; 6; 2; 1], 'VariableNames', {'pe'});
            preprocess_obj.remove_zero_pe();

            % Check if rows with PE = 0 are removed
            expected_data = preprocess_obj.data(preprocess_obj.data.pe ~= 0,:);
            testCase.verifyEqual(preprocess_obj.data, expected_data);
        end

        function test_addsplithalf(testCase)
            % Test for adding split half variable

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();  
            num_trials = 10;
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);

            preprocess_obj.data = table([1:10].', 'VariableNames', {'trials'});
            preprocess_obj.add_splithalf();

            % RUN TESTS
            expected_data = [0; 1; 0; 1; 0; 1; 0; 1; 0; 1];
            testCase.verifyEqual(preprocess_obj.data.splithalf, expected_data);
        end

        function test_addsaliencechoice(testCase)
            % Test for adding variable whether the salient choice was made

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();  
            num_trials = 10;
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:);

            contrast_left = [0.8, 0.9, 0.1, 0.2, 0.4, 0.5, 0.6, 0.4, 0.9, 0.1];
            contrast_right = [0.1, 0.3, 0.4, 0.4, 0.6, 0.3, 0.5, 0.9, 0.8, 0.4];
            choice = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0];

            preprocess_obj.data = table(contrast_left.',contrast_right.',choice.', ...
                'VariableNames', {'contrast_left','contrast_right','choice'});
            preprocess_obj.add_saliencechoice();
            
            % RUN TESTS
            expected_data = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0].';
            testCase.verifyEqual(preprocess_obj.data.salience_choice, expected_data);
        end
    end
end