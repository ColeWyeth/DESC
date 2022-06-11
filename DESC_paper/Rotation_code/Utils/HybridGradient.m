classdef HybridGradient < handle
    properties
        lr {mustBeNumeric}
        beta_1 {mustBeNumeric}
        beta_2 {mustBeNumeric}
        decay_interval {mustBeNumeric}
        m_t
        v_t
        t {mustBeNumeric}
        strategy
    end
    methods
        function obj = HybridGradient(lr, beta_1, beta_2, decay_interval)
            if nargin == 4
                obj.lr = lr;
                obj.beta_1 = beta_1;
                obj.beta_2 = beta_2;
                obj.decay_interval = decay_interval;
                obj.t = 0;
                obj.strategy = 0; % adam 
            end
        end
        function step = GetStep(obj, grad)
            if obj.t == 0 
                obj.m_t = zeros(size(grad));
                obj.v_t = zeros(size(grad));
            end
            if obj.strategy == 0
                obj.t = obj.t + 1;
                obj.m_t = (obj.beta_1 * obj.m_t) + (1 - obj.beta_1)*grad;
                obj.v_t = (obj.beta_2 * obj.v_t) + (1 - obj.beta_2)*(grad.^2);
                corr_m_t = obj.m_t / (1 - obj.beta_1^obj.t); 
                corr_v_t = obj.v_t / (1 - obj.beta_2^obj.t);
                step = -obj.lr * corr_m_t ./ (sqrt(corr_v_t) + 10^(-8));
            end 
            if obj.strategy == 1
                obj.t = obj.t + 1;
                %step_size = (obj.lr/(2^fix(obj.t/obj.decay_interval)));
                step_size = 100*(obj.lr/(fix(obj.t/obj.decay_interval)+1));
                step = -step_size * grad;
            end
            % piecewise decay version
            %step_size = obj.lr/(2^fix(obj.t/100));
            %step = -step_size * corr_m_t ./ (sqrt(corr_v_t) + 10^(-8));
            
            % reciporical decay version
            %step_size = obj.lr/(obj.t);
            %step = -step_size * corr_m_t ./ (sqrt(corr_v_t) + 10^(-8));
        end
        function obj = stopAdam(obj)
            obj.strategy = 1;
        end
    end
end