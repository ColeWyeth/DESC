classdef PiecewiseStepSize < handle
    properties
        learning_rate {mustBeNumeric}
        decay_interval {mustBeNumeric}
        t {mustBeNumeric}
    end
    methods
        function obj = PiecewiseStepSize(learning_rate, decay_interval)
            obj.learning_rate = learning_rate;
            obj.decay_interval = decay_interval;
            obj.t = 0;
        end
        function step = GetStep(obj, grad)
            obj.t = obj.t + 1;
            %step_size = (obj.learning_rate/(2^fix(obj.t/obj.decay_interval)));
            step_size = (obj.learning_rate/(fix(obj.t/obj.decay_interval)+1));
            step = -step_size * grad;
        end
    end
end