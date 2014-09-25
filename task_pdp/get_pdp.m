%% get_pdp: function description
function [pdp] = get_pdp(hfft)
    nsc = size(hfft, 1);

    pdp = zeros(nsc, 1);
    for t = 1:size(hfft, 2)
        pdp = pdp + abs(ifft(hfft(:, t)));
    end
    pdp = pdp / size(hfft, 2);

end
