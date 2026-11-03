function nll = getNLL(contrast,state,choices,p1,p2,nll)
% get p1 and p2 to compute nll based on participants' choices
% if contrast == 0
    if state == 0
        if choices == 1
            nll = nll - log(max(p1, eps));
        else
            nll = nll - log(max(p2, eps));
        end
    else
        if choices == 0
            nll = nll - log(max(p2, eps));
        else
            nll = nll - log(max(p1, eps));
        end
    end
% else
%     if state == 0
%         if choices == 1
%             nll = nll - log(max(p1, eps));
%         else
%             nll = nll - log(max(p2, eps));
%         end
%     else
%         if choices == 0
%             nll = nll - log(max(p2, eps));
%         else
%             nll = nll - log(max(p1, eps));
%         end
%     end
% end
end