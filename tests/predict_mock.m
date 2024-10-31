function predicted = predict_mock(~,tbl)
% function fitlm_mock is mock version of MATLAB's fitlm.
%
% INPUTS:
%   tbl: table to be fit to the linear model.
%   mdl: linear model
%
% OUTPUT:
%   lm: mock linear model
%
rng(123)
predicted = randn(height(tbl),1);
end