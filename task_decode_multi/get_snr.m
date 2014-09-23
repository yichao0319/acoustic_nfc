%% -----------------------------------
%% get_snr: function description
%% -----------------------------------
function [snr] = get_snr(preamble, rcv_preamble)
    % snr = log10( (abs(rcv_preamble).^2) / mean(mean(abs(rcv_preamble-preamble).^2)) * 10 ) * 10;

    %% H
    % fprintf('H\n');
    % h = rcv_preamble ./ preamble;
    % w = mean2(abs(rcv_preamble - preamble) .^ 2);
    % snr = pow2db(abs(h) .^ 2 / w)
    
    %% MMSE
    % fprintf('MMSE\n');
    s1 = abs( rcv_preamble .* conj(preamble) ) .^ 2;
    s0 = abs( mean2( rcv_preamble .* conj(preamble) ) ) .^ 2;
    w  = mean2( abs(rcv_preamble) .^ 2 ) - s0;
    snr = pow2db(s1 / w);

    %% EVM 
    % fprintf('EVM\n');
    % evm = abs((preamble - rcv_preamble)) .^ 2;
    % p0 = mean2( abs(preamble).^2 );
    % evm = evm / p0;
    % snr = ones(size(evm)) ./ evm;
    % snr = pow2db(snr)

end



