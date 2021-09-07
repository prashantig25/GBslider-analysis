function task_agent_int
    % this function simulates the interaction between task and agent.
    
    % initialization of required classes
    task = Task();
    agent = Agent();
    
    % number of trials
    T = task.T;
    
    % pre-allocation
    s = NaN(1,T);
    u = NaN(1,T); 
    o = NaN(1,T);
    r = NaN(1,T);
    e_mu_t = zeros(1,T);  
    a = zeros(1,T);  
    p_a0_t = zeros(1,T);  
    p_a1_t = zeros(1,T);  
    p_a_t = zeros(1,T);
    v_a_0 = zeros(1,T);
    v_a_1 = zeros(1,T);
    r = NaN(1,T);
    pi_1 = NaN(1,T);  
    pi_0 = NaN(1,T);  
    block = NaN(1,T);  
    corr = NaN(1,T);  
    trial = NaN(1,T);  
    for i = 1:T
        task.state_sample() % samples a trial's state
        task.contrast_sample() % generates contrast based on the state
        agent.observation_sample(task.c_t) % contrast based observation sampled
        agent.decide_e() % economic choice 
        task.reward_sample(agent.a_t) % economic choice based reward
%         agent.compute_action_dep_rew(task.r_t)
        agent.learn(task.r_t) % update based on reward
        if (task.s_t == 0 && agent.a_t == 0) || (task.s_t == 1 && agent.a_t == 1)
                corr(i) = 1;
        else
                corr(i) = 0;
        end
        s(i) = task.s_t
        u(i) = task.c_t
        o(i) = agent.o_t
        pi_1(i) = agent.pi_1 
        pi_0(i) = agent.pi_0
        e_mu_t(i) = agent.G
        v_a_0(i) = agent.v_a_t(1)
        v_a_1(i) = agent.v_a_t(2)
        a(i) = agent.a_t
        p_a0_t(i) = agent.p_a_t(1)
        p_a1_t(i) = agent.p_a_t(2)
        %p_a_t(T) = agent.p_a_t(T)
        r(i) = task.r_t
        trial(i) = i 
        %block(T) = sim_vars.block
    end
end