classdef agent_unittests < matlab.unittest.TestCase
    % AGENT_UNITTESTS is a collection of functions to run unit tests on
    % various functions used to make choices and learn for different agents.

    methods(Test)

        function test_constructormethods(obj)

            % test_constructormethods tests the output for all the
            % constructor methods within object Agent().
            
            % INITIALIZE VARS
            agent = Agent();

            % EXPECTED
            expected_p_o_giv_u = zeros(1,length(agent.set_o));
            expected_p_o_giv_u_norm = zeros(1,length(agent.set_o)); 
            expected_mu_for_ev = agent.mu;
            expected_mu_for_ev = expected_mu_for_ev/sum(expected_mu_for_ev);
            expected_c_t = agent.c0; 

            % RUN TESTS
            obj.verifyEqual(agent.p_o_giv_u,expected_p_o_giv_u, ...
                'Expected and actual probabilities of observations given contrast differences do not match.')
            obj.verifyEqual(agent.p_o_giv_u_norm,expected_p_o_giv_u_norm, ...
                'Expected and actual normalised probabilities of observations do not match.')
            obj.verifyEqual(agent.mu_for_ev,expected_mu_for_ev, ...
                'Expected and actual expected value do not match.')
            obj.verifyEqual(agent.c_t,expected_c_t, ...
                'Expected and actual prior distribution over mu do not match.')
        end

        function test_observation_sample(obj)
            % test_observationsample function tests the observation_sample function
            % from the Task() object.

            % INITIALIZE VARS
            agent = Agent();
            contrast_diff = 0.1; % set contrast difference
            rng(123) % set seed
            agent.observation_sample(contrast_diff);

            % EXPECTED
            rng(123) % same seed
            expected_o_t = normrnd(contrast_diff,agent.sigma);

            % RUN TESTS
            obj.verifyEqual(length(agent.o_t),length(expected_o_t), ...
                'Expected and actual observation do not have the same length.')
            assert(abs(agent.o_t - contrast_diff) <= 3 * agent.sigma, ...
                'Observation should be sampled from a normal distribution.');
            assert(agent.o_t == expected_o_t,'Observation sampled do not match.')

        end

        function test_p_s_giv_o(obj)
            % test_p_s_giv_o runs unit tests on p_s_giv_o().

            % INTIALIZE VARS
            agent = Agent();
        
            % RUN TESTS
            % Test case 1: No perceptual uncertainty and check for correct
            % state inference
            contrast_diff = -0.1; % s_t = 0
            agent.sigma = 0.0001; % high perceptual sensitivity
            agent.observation_sample(contrast_diff);
            agent.p_s_giv_o(agent.o_t);
            assert(agent.pi_0 > agent.pi_1,'State inference by agent incorrect.')

            contrast_diff = 0.1; % s_t = 1
            agent.sigma = 0.0001; % high perceptual sensitivity
            agent.observation_sample(contrast_diff);
            agent.p_s_giv_o(agent.o_t);
            assert(agent.pi_1 > agent.pi_0,'State inference by agent incorrect.')
        
            % Test case 2: Observing a negative contrast difference
            contrast_diff = -0.1;
            agent.observation_sample(contrast_diff);
            agent.p_s_giv_o(agent.o_t);
            assert(abs(agent.pi_0 + agent.pi_1 - 1) < eps, 'Test case 2 failed: pi_0 and pi_1 should sum up to 1.');
            assert(agent.pi_0 >= 0 && agent.pi_0 <= 1, 'Test case 2 failed: pi_0 should be between 0 and 1.');
            assert(agent.pi_1 >= 0 && agent.pi_1 <= 1, 'Test case 2 failed: pi_1 should be between 0 and 1.');

            % Test case 3: Observing a positive contrast difference
            contrast_diff = 0.1;
            agent.observation_sample(contrast_diff);
            agent.p_s_giv_o(agent.o_t);
            assert(abs(agent.pi_0 + agent.pi_1 - 1) < eps, 'Test case 3 failed: pi_0 and pi_1 should sum up to 1.');
            assert(agent.pi_0 >= 0 && agent.pi_0 <= 1, 'Test case 3 failed: pi_0 should be between 0 and 1.');
            assert(agent.pi_1 >= 0 && agent.pi_1 <= 1, 'Test case 3 failed: pi_1 should be between 0 and 1.');
           
        end

        function test_decide_p(obj)
            % test_decide_pe runs unit tests on decide_p().

            % INITIALIZE VARS
            agent = Agent();
            agent.sigma = 0.0001; % no perceptual uncertainty
            contrast_diff = -0.1; % contrast difference
            agent.observation_sample(contrast_diff)
            agent.decide_p();

            % RUN TESTS
            % Test case 1: check perceptual decision when pi_0 > pi_1
            assert(agent.p_d_0 == 0,'Incorrect perceptual decision.')
            assert(agent.d_t == 1 || agent.d_t == 0, 'Perceptual decision should be either 0 or 1.');
            % Test case 1: check perceptual decision when pi_0 < pi_1
            contrast_diff = 0.1; % contrast difference
            agent.observation_sample(contrast_diff)
            agent.decide_p();
            assert(agent.p_d_0 == 1,'Incorrect perceptual decision.')
            assert(agent.d_t == 1 || agent.d_t == 0, 'Perceptual decision should be either 0 or 1.');
        end

        function test_cat_bs(obj)
            % test_cat_bs runs unit tests on cat_bs().

            % INITIALIZE VARS
            agent = Agent();
            agent.sigma = 0.0001; % no perceptual uncertainty
            contrast_diff = -0.1; % contrast difference
            agent.observation_sample(contrast_diff)
            agent.cat_bs();

            % EXPECTED
            if agent.d_t == 0
                expected_pi_0 = 0;
                expected_pi_1 = 1;
            else
                expected_pi_0 = 1;
                expected_pi_1 = 0;
            end

            % RUN TESTS
            assert(agent.pi_0 == expected_pi_0 & agent.pi_1 == expected_pi_1,'Categorical belief states not correct.')
        end

        function test_eval_poly(obj)
            % test_eval_poly runs unit tests on eval_poly().

            % INITIALIZE VARS
            agent = Agent();
            agent.c_t = [2 3];
            poly_eval = agent.eval_poly();

            % EXPECTED
            expected_poly_int = polyint([agent.c_t, 0]); % expected anti-derivative of the polynomial
            expected_poly_eval = polyval(expected_poly_int, [0, 1]); % expected evaluate the polynomial
            expected_poly_eval = diff(expected_poly_eval); % expected difference of evaluated polynomial

            % RUN TESTS
            assert(isequal(poly_eval, expected_poly_eval), 'Evaluated polynomial is not as expected.');
        end

        function test_compute_valence(obj)
            % test_compute_valence runs unit tests for compute_valence().

            % INITIALIZE VARS
            agent = Agent();
            agent.pi_0 = 1; % set belief state
            agent.pi_1 = 0;
            agent.compute_valence;

            % EXPECTED
            expected_v_a_0 = (agent.pi_0 - agent.pi_1) * agent.E_mu_t + agent.pi_1; 
            expected_v_a_1 = (agent.pi_1 - agent.pi_0) * agent.E_mu_t + agent.pi_0;         
            expected_v_a_t = [expected_v_a_0, expected_v_a_1]; % concatenate action valences

            % RUN TESTS
            assert(isequal(agent.v_a_t, expected_v_a_t), 'Action valences is not as expected.');
        end

        function test_softmax(obj)
            % test_softmax runs unit tests for softmax().

            % INITIALIZE VARS
            agent = Agent();
            agent.v_a_t = [0.8,0.2]; % initialize EVs 
            agent.beta = 100; % beta parameter
            agent.softmax();

            % EXPECTED
            expected_p_a_t = exp(agent.v_a_t.*agent.beta) / sum(exp(agent.v_a_t.*agent.beta));

            % RUN TESTS
            assert(isequal(agent.p_a_t, expected_p_a_t), 'Choice probabilities is not as expected.');

        end

        function test_decide_e(obj)
            % test_decide_e runs unit tests on the decide_e().

            % INITIALIZE VARS
            agent = Agent();
            agent.sigma = 0.00001; % set high perceptual sensitivity

            % RUN TESTS

            % Test case: random agent
            agent.agent = 0; 
            rng(123) % set seed
            agent.decide_e(0.1);
            rng(123) % same seed
            expected_a_t = binornd(1, agent.p_a_t(2)); % action
            assert(isequal(agent.a_t, 0) || isequal(agent.a_t, 1), 'Action should be either 0 or 1 for a random decision agent.');
            assert(isequal(agent.a_t, expected_a_t), 'Action should be either 0 or 1 for a random decision agent.');

            % Test case: Bayesian agent
            agent.agent = 1;
            agent.c_t = [2 3]; % set c_t
            rng(123)
            agent.decide_e(0.1); % Call decide_e with an example o_t
            rng(123)
            expected_a_t = binornd(1, agent.p_a_t(2)); % action
            assert(isequal(agent.a_t, 0) || isequal(agent.a_t, 1), 'Test case 3 failed: Action should be either 0 or 1 for a normative Bayesian agent in simulation mode.');
            assert(isequal(agent.a_t, expected_a_t), 'Action should be either 0 or 1 for a random decision agent.');
            % Test case: Categorical agent
            agent.agent = 2;
            agent.c_t = [2 3];
            rng(123)
            agent.decide_e(0.1); % Call decide_e with an example o_t
            rng(123)
            expected_a_t = binornd(1, agent.p_a_t(2)); % action
            assert(isequal(agent.a_t, 0) || isequal(agent.a_t, 1), 'Test case 3 failed: Action should be either 0 or 1 for a normative Bayesian agent in simulation mode.');
            assert(isequal(agent.a_t, expected_a_t), 'Action should be either 0 or 1 for a random decision agent.');
        end

        function test_compute_action_dep_rew(obj)
            % test_compute_action_dep_rew runs unit tests on
            % compute_action_dep_rew().

            % INITIALIZE VARS
            agent = Agent();
            r = 1; % set task-generated reward
            agent.a_t = 1; % set action
            r_t = agent.compute_action_dep_rew(r);

            % EXPECTED
            expected_r_t = r + (agent.a_t *((-1) .^ (2 + r)));

            % RUN TEST
            assert(isequal(r_t, expected_r_t), 'Recoded reward does not match');

        end

        function test_compute_q(obj)
            % test_compute_q runs unit tests on compute_q().

            % INITIALIZE VARS
            agent = Agent();
            agent.pi_0 = 1; 
            agent.pi_1 = 0;
            agent.c_t = [2 4];
            r = 1;
            agent.a_t = 1;
            agent.compute_q(r);
            agent.G = agent.eval_poly;

            % EXPECTED
            expected_t = length(agent.c_t) + 1;
            expected_r_t = agent.compute_action_dep_rew(r); 
            expected_G = agent.eval_poly();
            expected_C = (agent.pi_0 - agent.pi_1) * ((1 - expected_G) .^ (1 - expected_r_t)) * (expected_G .^ expected_r_t) + agent.pi_1;
            expected_q_0 = (agent.pi_1 .^ expected_r_t) * (agent.pi_0 .^ (1-expected_r_t)) / expected_C;
            expected_q_1 = ((-1) .^ (expected_r_t + 1)) * (agent.pi_0 - agent.pi_1) / expected_C;

            % RUN TEST
            assert(isequal(agent.q_0, expected_q_0), 'Computed q_0 does not match');
            assert(isequal(agent.q_1, expected_q_1), 'Computed q_0 does not match');
        end

        function test_update_coefficients(obj)
            % test_update_coefficients runs unit tests on
            % update_coefficients().

            % INITIALIZE VARS
            agent = Agent();
            agent.c_t = [2 3];
            agent.t = length(agent.c_t) + 1;
            agent.q_0 = -0.5;
            agent.q_1 = 0.5;
            agent.update_coefficients()

            % EXPECTED
            c_t = [2 3];
            expected_d = zeros(1,agent.t);
            expected_d(end) = agent.q_0 * c_t(end);  % update last element
            for n = 0:(agent.t - 3)
                expected_d(end-(n+1)) = agent.q_1 * c_t(end-n) + agent.q_0 * c_t(end-(n+1)); % update all elements except last and first
            end
            expected_d(1) = agent.q_1 * c_t(1); % update first element
            expected_c_t = expected_d;

            % RUN TEST
            assert(isequal(agent.c_t, expected_c_t), 'Updated c_t does not match');
        end

    end
end