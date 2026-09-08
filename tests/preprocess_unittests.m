classdef preprocess_unittests < matlab.unittest.TestCase
    % PREPROCESS_UNITTESTS is a collection of functions to run unit tests on
    % various functions used for data preprocessing.
    %
    % These tests build minimal, hand-constructed inputs for each function
    % rather than loading the real dataset -- every property/table column a
    % function needs is set directly by the test, and expected outputs are
    % concrete numbers worked out by hand (not the same formula/loop copied
    % from the function under test). Where a test's setup would otherwise
    % need another preprocess_LR method's output (e.g. flipped_mu from
    % flip_mu), that intermediate value is also set by hand so each test
    % only exercises the one function it names.

    methods(Test)

        function test_flipmu(obj)
            % test_flipmu function tests the flip_mu function
            % from the preprocess_LR() object.

            % INITIALIZE VARS -- covers both congruence == 0 (incongruent,
            % gets flipped) and congruence == 1 (congruent, unchanged)
            preprocess_obj = preprocess_LR();
            preprocess_obj.mu = [0.7; 0.2; 0.3; 0.9];
            preprocess_obj.data = table([0;1;0;1],'VariableNames',{'congruence'});
            preprocess_obj.flipped_mu = NaN(4,1);
            preprocess_obj.flip_mu();

            % EXPECTED (hand-computed: congruence==0 -> 1-mu; congruence==1 -> mu)
            expected_flippedmu = [0.3; 0.2; 0.7; 0.9];

            % RUN TEST
            obj.verifyEqual(preprocess_obj.flipped_mu,expected_flippedmu, ...
                'Expected and actual congruence flipped mu do not match.')
        end

        function test_computeactiondeprew(obj)
            % test_computeactiondeprew function tests the compute_action_dep_rew
            % function from the preprocess_LR() object.

            % INITIALIZE VARS -- covers all 4 action x reward combinations
            preprocess_obj = preprocess_LR();
            preprocess_obj.obtained_reward = [0,0,1,1];
            preprocess_obj.action = [0,1,0,1];
            preprocess_obj.compute_action_dep_rew();

            % EXPECTED (hand-computed truth table: reward==0 keeps
            % recoded_reward==action; reward==1 flips it to 1-action)
            expected_recodedrew = [0,1,1,0];

            % RUN TEST
            obj.verifyEqual(preprocess_obj.recoded_reward,expected_recodedrew, ...
                'Expected and actual recoded rewards array do not match.')
        end

        function test_computemu(obj)
            % test_computemu function tests the compute_mu
            % function from the preprocess_LR() object.
            %
            % INITIALIZE VARS -- flipped_mu is set directly by hand (not via
            % flip_mu) so this test only exercises compute_mu. Covers both
            % contrast == 0 and contrast == 1.
            preprocess_obj = preprocess_LR();
            preprocess_obj.flipped_mu = [0.3; 0.6; 0.4; 0.8];
            preprocess_obj.data = table([0;0;1;1],'VariableNames',{'contrast'});
            preprocess_obj.mu_t = NaN(4,1);
            preprocess_obj.mu_t_1 = NaN(4,1);
            preprocess_obj.compute_mu();

            % EXPECTED (hand-computed: contrast==1 -> 1-flipped_mu, contrast==0
            % -> flipped_mu as-is. Trial 1 has no previous trial, stays NaN.)
            expected_mu_t_1 = [NaN; 0.3; 0.4; 0.6];
            expected_mu_t = [NaN; 0.6; 0.6; 0.2];

            % RUN TEST
            obj.verifyEqual(preprocess_obj.mu_t,expected_mu_t,'NaNEqualsNaN',true, ...
                'Expected and actual mu for current trial array do not match.')
            obj.verifyEqual(preprocess_obj.mu_t_1,expected_mu_t_1,'NaNEqualsNaN',true, ...
                'Expected and actual mu for previous trial array do not match.')
        end

        function test_computestatedeppe(obj)
            % test_computestatedeppe function tests the compute_state_dep_pe
            % function from the preprocess_LR() object.

            % INITIALIZE VARS -- recoded_reward/mu_t/mu_t_1 are set directly by
            % hand (not via compute_action_dep_rew/flip_mu/compute_mu) so this
            % test only exercises compute_state_dep_pe. Covers all 4
            % combinations of state (0/1) and trials==1 (PE forced to 0) vs.
            % trials~=1 (PE computed).
            preprocess_obj = preprocess_LR();
            preprocess_obj.state = [0;0;1;1];
            preprocess_obj.recoded_reward = [1;0;1;0];
            preprocess_obj.mu_t_1 = [0.3;0.4;0.5;0.6];
            preprocess_obj.mu_t = [0.35;0.42;0.55;0.58];
            preprocess_obj.data = table([1;2;1;2],'VariableNames',{'trials'});
            preprocess_obj.data.pe = NaN(4,1); % pe column must exist before indexed assignment
            preprocess_obj.compute_state_dep_pe();

            % EXPECTED (hand-computed: pe = recoded_reward - mu_t_1 for
            % state==0, (1-recoded_reward) - mu_t_1 for state==1, forced to 0
            % on trials==1; up = mu_t - mu_t_1, undefined/NaN on trial 1)
            expected_pe = [0; -0.4; 0; 0.4];
            expected_up = [NaN; 0.02; 0.05; -0.02];

            % RUN TEST
            obj.verifyEqual(preprocess_obj.data.pe,expected_pe, ...
                'Expected and actual PE arrays do not match.')
            obj.verifyEqual(preprocess_obj.data.up,expected_up,'NaNEqualsNaN',true, ...
                'Expected and actual UP arrays do not match.')
        end

        function test_computeconfirm(obj)
            % test_computeconfirm function tests the compute_confirm
            % function from the preprocess_LR() object.

            % INITIALIZE VARS -- covers all 2x2x2 combinations of contrast,
            % state/action match-vs-mismatch, and obtained_reward
            preprocess_obj = preprocess_LR();
            preprocess_obj.obtained_reward = [0,0,0,0,1,1,1,1].';
            preprocess_obj.state = [0,0,1,1,0,0,1,1].';
            preprocess_obj.action = [0,1,0,1,0,1,0,1].';
            preprocess_obj.data = table([1,0,1,0,1,0,1,0].','VariableNames',{'contrast'});
            preprocess_obj.compute_confirm();

            % EXPECTED (hand-derived: contrast==1 favors state~=action,
            % contrast==0 favors state==action; the favored combination keeps
            % the reward as-is, the other combination gets 1-reward)
            expected_confirmrew = [1;1;0;0;0;0;1;1];

            % RUN TEST
            obj.verifyEqual(preprocess_obj.data.confirm_rew,expected_confirmrew, ...
                'Expected and actual confirmation bias arrays do not match.')
        end

        function test_removeconditions(obj)
            % test_removeconditions function tests the remove_conditions
            % function from the preprocess_LR() object.

            % INITIALIZE VARS -- choice_cond/val use values distinct from
            % condition so a passing test actually confirms the right rows/
            % columns are kept, not just that the same numbers were echoed back
            preprocess_obj = preprocess_LR();
            preprocess_obj.removed_cond = 2;
            preprocess_obj.condition = [1;2;3];
            preprocess_obj.data = table([10;20;30],[100;200;300], ...
                'VariableNames',{'choice_cond','val'});
            preprocess_obj.remove_conditions();

            % EXPECTED (hand-picked: condition==2, i.e. row 2, is removed, so
            % only rows 1 and 3 -- and their choice_cond/val values -- remain)
            expected_condition = [10;30];
            expected_val = [100;300];

            % RUN TEST
            obj.verifyEqual(preprocess_obj.condition,expected_condition, ...
                'Expected and actual condition arrays do not match.')
            obj.verifyEqual(preprocess_obj.data.val,expected_val, ...
                'Data was not filtered to the expected rows.')
        end

        function test_computenormalise(obj)
            % test_computenormalise function tests the compute_normalise
            % function from the preprocess_LR() object.

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            var_normalise = [0;5;10];
            normalised = preprocess_obj.compute_normalise(var_normalise);

            % EXPECTED (hand-computed: (x-min)/(max-min) with min=0, max=10)
            expected_normalise = [0; 0.5; 1];

            % RUN TESTS
            obj.verifyEqual(normalised,expected_normalise, ...
                'Normalisation not working as expected.')
        end

        function test_computeru(obj)
            % test_computeru function tests the compute_ru
            % function from the preprocess_LR() object.

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            preprocess_obj.condition = [1;2;3];
            preprocess_obj.data = table((1:3).','VariableNames',{'placeholder'});
            preprocess_obj.compute_ru();

            % EXPECTED (hand-picked: ru is false only where condition==1)
            expected_ru = logical([0;1;1]);

            % RUN TESTS
            obj.verifyEqual(preprocess_obj.data.ru,expected_ru, ...
                'Categorical variable for RU not correct.')
        end

        function test_addvars(testCase)
            % test_addvars function tests the add_vars
            % function from the preprocess_LR() object.

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            preprocess_obj.data = table((1:5).','VariableNames',{'dummy'});
            new_column = linspace(0,1,5).';
            varName = 'new_column';
            preprocess_obj.add_vars(new_column, varName);

            % RUN TESTS
            testCase.assertClass(preprocess_obj.data, 'table');
            testCase.verifySize(preprocess_obj.data.(varName), [5, 1]);
            testCase.verifyEqual(preprocess_obj.data.(varName), new_column);
        end

        function test_removezerope(testCase)
            % test_removezerope function tests the remove_zero_pe
            % function from the preprocess_LR() object.

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            preprocess_obj.data = table([1; 2; 0; 4; 5; 1; 5; 6; 2; 1], 'VariableNames', {'pe'});
            preprocess_obj.remove_zero_pe();

            % EXPECTED (hand-picked: the single pe==0 row, index 3, is removed)
            expected_pe = [1; 2; 4; 5; 1; 5; 6; 2; 1];

            % RUN TESTS
            testCase.verifyEqual(preprocess_obj.data.pe, expected_pe);
        end

        function test_addsplithalf(testCase)
            % test_addsplithalf function tests the add_splithalf
            % function from the preprocess_LR() object.

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR();
            preprocess_obj.data = table([1;2;3;4;5], 'VariableNames', {'trials'});
            preprocess_obj.add_splithalf();

            % EXPECTED (hand-picked: splithalf is true only for even trial numbers)
            expected_splithalf = logical([0;1;0;1;0]);

            % RUN TESTS
            testCase.verifyEqual(preprocess_obj.data.splithalf, expected_splithalf);
        end

        function test_addsaliencechoice(testCase)
            % test_addsaliencechoice function tests the add_saliencechoice
            % function from the preprocess_LR() object.

            % INITIALIZE VARS -- covers contrast_left > contrast_right,
            % contrast_left < contrast_right, and the contrast_left ==
            % contrast_right edge case (which falls into the <= branch), each
            % crossed with choice == 0 and choice == 1
            preprocess_obj = preprocess_LR();
            contrast_left = [0.8, 0.9, 0.1, 0.2, 0.5, 0.5];
            contrast_right = [0.3, 0.2, 0.4, 0.5, 0.5, 0.5];
            choice = [0, 1, 0, 1, 0, 1];

            preprocess_obj.data = table(contrast_left.',contrast_right.',choice.', ...
                'VariableNames', {'contrast_left','contrast_right','choice'});
            preprocess_obj.add_saliencechoice();

            % EXPECTED (hand-derived truth table)
            expected_salience_choice = logical([1;0;0;1;0;1]);

            % RUN TESTS
            testCase.verifyEqual(preprocess_obj.data.salience_choice, expected_salience_choice);
        end
    end
end
