classdef AdamGradient < handle
    properties
        lr {mustBeNumeric}
        beta_1 {mustBeNumeric}
        beta_2 {mustBeNumeric}
        m_t
        v_t
        t {mustBeNumeric}
    end
    methods
        function obj = AdamGradient(lr, beta_1, beta_2)
            if nargin == 3
                obj.lr = lr;
                obj.beta_1 = beta_1;
                obj.beta_2 = beta_2;
                obj.t = 0;
            end
        end
        function step = GetStep(obj, grad)
            if obj.t == 0 
                grad_length = size(grad, 2);
                obj.m_t = zeros(1, grad_length);
                obj.v_t = zeros(1, grad_length);
            end
            obj.t = obj.t + 1;
            obj.m_t = (obj.beta_1 * obj.m_t) + (1 - obj.beta_1)*grad;
            obj.v_t = (obj.beta_2 * obj.v_t) + (1 - obj.beta_2)*(grad.^2);
            corr_m_t = obj.m_t / (1 - obj.beta_1^obj.t); 
            corr_v_t = obj.v_t / (1 - obj.beta_2^obj.t);
            step = -obj.lr * corr_m_t ./ (sqrt(corr_v_t) + 10^(-8));
            
            % piecewise decay version
            %step_size = obj.lr/(2^fix(obj.t/100));
            %step = -step_size * corr_m_t ./ (sqrt(corr_v_t) + 10^(-8));
            
            % reciporical decay version
            %step_size = obj.lr/(obj.t);
            %step = -step_size * corr_m_t ./ (sqrt(corr_v_t) + 10^(-8));
        end
    end
end