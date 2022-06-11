classdef ConstantStepSize < handle
    properties
        learning_rate {mustBeNumeric}
    end
    methods
        function obj = ConstantStepSize(learning_rate)
            obj.learning_rate = learning_rate;
        end
        function step = GetStep(obj, grad)
            step = -obj.learning_rate * grad;
        end
    end
end