%% get_H: Get channel coefficient
function [Hfft, sync_fft] = get_H(wav_packet_orig, preamble_offset, ...
                           f_min, Fs, Nfft, Nfft_useful, ...
                           Nofdm, slice_width, slice_count, ...
                           Ncp, Ncp_half, ...
                           modPreamble)

    swch_samples = (slice_width * slice_count + 1) * Nofdm;

    %--------------------------------------------------------------------------
    % 4.1) Get the sync symbols
    %--------------------------------------------------------------------------
    sync_ofdm = zeros(1,Nofdm*slice_count);
    % sync_ofdm_matrix = zeros(Nofdm,slice_count);
    % sync_ofdm_no_cp = zeros(Nfft,slice_count);

    for r = 1:slice_count
        offset1 = preamble_offset - 1 + (r-1)*Nofdm*slice_width;
        offset2 = offset1 + swch_samples;
        sync_ofdm((Nofdm*(r-1)+1):(Nofdm*r)) = ...
            wav_packet_orig((offset1+1):(offset1+Nofdm)) + ...
            i*wav_packet_orig((offset2+1):(offset2+Nofdm));
    end

    %--------------------------------------------------------------------------
    % 4.2) digital down convert (DDC)
    %--------------------------------------------------------------------------
    sync_ofdm_matrix = reshape(sync_ofdm,Nofdm,slice_count);
    for slice=1:slice_count
        base_idx = (slice-1)*Nofdm*slice_width;
        sync_ofdm_matrix(:,slice) = sync_ofdm_matrix(:,slice) .* ...
            exp(-i*2*pi*f_min*(base_idx:(base_idx+Nofdm-1)).'/Fs);
    end

    %--------------------------------------------------------------------------
    % 4.3) remove 1/2 CP 
    %--------------------------------------------------------------------------
    % Nfft rows x slice_count cols
    sync_ofdm_no_cp = sync_ofdm_matrix((Ncp-Ncp_half+1):(Nofdm-Ncp_half),:);

    %--------------------------------------------------------------------------
    % 4.4) apply FFT to obtain frequency-domain signals
    %--------------------------------------------------------------------------
    sync_fft = zeros(Nfft,slice_count);
    for slice=1:slice_count
        sync_fft(:,slice) = fft(sync_ofdm_no_cp(:,slice),Nfft).';
    end

    % extract only those subcarriers of interest
    sync_fft = sync_fft(1:Nfft_useful,:);
    sync_fft_norm = norm(sync_fft,'fro');

    %--------------------------------------------------------------------------
    % 4.5: compute channel coefficient
    %--------------------------------------------------------------------------
    Hfft = sync_fft ./ modPreamble;
    
end
