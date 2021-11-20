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
        function obj = AdamGradient(lr, beta_1, beta_2, grad_length)
            if nargin == 4
                obj.lr = lr;
                obj.beta_1 = beta_1;
                obj.beta_2 = beta_2;
                obj.m_t = zeros(1, grad_length);
                obj.v_t = zeros(1, grad_length);
                obj.t = 0;
            end
        end
        function step = GetStep(obj, grad)
            obj.t = obj.t + 1;
            obj.m_t = (obj.beta_1 * obj.m_t) + (1 - obj.beta_1)*grad;
            obj.v_t = (obj.beta_2 * obj.v_t) + (1 - obj.beta_2)*(grad.^2);
            corr_m_t = obj.m_t / (1 - obj.beta_1^obj.t); 
            corr_v_t = obj.v_t / (1 - obj.beta_2^obj.t);
            step = -obj.lr * corr_m_t ./ (sqrt(corr_v_t) + 10^(-8));
        end
    end
end