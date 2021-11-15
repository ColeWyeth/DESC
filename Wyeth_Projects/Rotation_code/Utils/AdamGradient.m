classdef AdamGradient
    properties
        lr {mustBeNumeric}
        beta_1 {mustBeNumeric}
        beta_2 {mustBeNumeric}
        m_t
        v_t
    end
    methods
        function obj = AdamGradient(lr, beta_1, beta_2, grad_length)
            if nargin == 4
                obj.lr = lr;
                obj.beta_1 = beta_1;
                obj.beta_2 = beta_2;
                obj.m_t = zeros(1, grad_length);
                obj.v_t = zeros(1, grad_length);
            end
        end
    end
end