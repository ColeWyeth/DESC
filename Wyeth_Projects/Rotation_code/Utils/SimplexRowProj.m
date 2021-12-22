function [projected] = SimplexRowProj(row)
% This is an iterative simplex projection algorithm for one row.
   w = sort(row); 
   Ti = 0; 
   for i = 1:size(w, 2)
       if sum(w(i:end)-w(i)) < 1
           Ti = i;  
           break 
       end
   end
   T = w(Ti) - (1 - sum(w(Ti:end)-w(Ti)))/length(w(Ti:end)); 
   projected = max(row - T, 0);
end

