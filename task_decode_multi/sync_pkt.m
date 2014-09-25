%% sync_pkt: Perform synchronization
function [preamble_offset] = sync_pkt(slice_width, slice_count, Nofdm, Nfft, Nfft_useful, subCarrier_swch, modPreamble_fft_norm, start_offset, threshold, modPreamble_swch, wav_packet_orig)

    DEBUG1 = 1;
    DEBUG2 = 0;

    font_size = 18;

    
    % size of entire sandwich (note that there is one empty symbol at the end)
    swch_samples = (slice_width * slice_count + 1) * Nofdm;
    sample_count = length(wav_packet_orig);

    sync_ofdm = zeros(1,Nofdm*slice_count);
    sync_ofdm_matrix = zeros(Nofdm,slice_count);
    sync_ofdm_no_cp = zeros(Nfft,slice_count);

    p_lst = [];
    p_max = -1;
    % start_offset = 1;
    % start_offset = start_offsets{seed}(fi);
    % threshold = thresholds{seed}(fi);
    % fprintf('    start_offset = %d\n', start_offset);
    % fprintf('    threshold    = %f\n', threshold);

    if start_offset <= 0
        start_offset = 1; 
    end
    preamble_offset = start_offset - 1;

    step_size = swch_samples;
    for try_offset = start_offset:step_size:(sample_count - (swch_samples*2 - 1))
        max_offset = min(sample_count, try_offset + step_size + (swch_samples*2 - 1));
        try_p_cnt = max_offset - (try_offset + (swch_samples*2 - 1)) + 1;
      
        try_packets = ...
              wav_packet_orig((try_offset:(max_offset-swch_samples))) + ...
            i * wav_packet_orig((try_offset+swch_samples):max_offset);
        try_packets = reshape(try_packets,1,[]);

        modInnerProd = swch_prod(conj(modPreamble_swch), try_packets);
        modInnerProd = modInnerProd(1:try_p_cnt);

        scProd = zeros(1,max_offset-try_offset+1-swch_samples);
        for sc = 1:Nfft_useful
            scProd = scProd + swch_prod(conj(subCarrier_swch(sc,:)), try_packets).^2;
        end
      
        scNormSq = zeros(1,try_p_cnt);
        for slice = 1:slice_count
            base_idx = (slice-1)*Nofdm*slice_width;
            scNormSq = scNormSq + scProd(base_idx+1:base_idx+try_p_cnt);
        end
      
        try_p_lst = modInnerProd ./ (modPreamble_fft_norm * sqrt(scNormSq));

        [p_max,idx] = max(try_p_lst);
        p_lst = [p_lst try_p_lst];

        if DEBUG2, fprintf('    %d: p_max = %f, idx = %d\n', try_offset, p_max, idx); end

        % if (p_max >= 0.4) 
        if (threshold > 0 & p_max >= threshold)
            preamble_offset = idx + try_offset - 1;
            break;
        end
    end

    fh = figure;
    clf;
    plot(p_lst);
    % xlim([0 swch_samples*2]);
    set(gca, 'FontSize', font_size);
    set(gca, 'OuterPosition', [0 0 1 0.5]);
    xlim([0 400000]);
    print(fh, '-dpsc', './tmp/plist.ps');

    % fd=fopen('p2.txt','w');
    % fprintf(fd,'%f\n',p_lst);
    % fclose(fd);
    % fprintf('p_lst = %d x %d\n', size(p_lst));

    
    if threshold <= 0
        % fprintf('==============\n');
        % %% select the top N
        % num_cand = 10;
        % [tmp_lst, tmp_ind] = sort(p_lst, 'descend');
        % sel_lst = tmp_lst(1:num_cand);
        % sel_ind = tmp_ind(1:num_cand);

        % %% the correct one should locate in the second block
        % min_ind = min(sel_ind);
        % fprintf('min ind=%d\n', min_ind);
        % cand_ind_std = min_ind + step_size - 1;
        % cand_ind_end = cand_ind_std + step_size;
        % fprintf('  range = %d:%d\n', cand_ind_std, cand_ind_end);

        % %% find the max one in the second block
        % max_p = 0;
        % max_ind = 0;
        % for li = [1:num_cand]
        %     fprintf('  %d: %f\n', sel_ind(li), sel_lst(li));
        %     if sel_ind(li) >= cand_ind_std & sel_ind(li) <= cand_ind_end & sel_lst(li) > max_p
        %         max_p = sel_lst(li);
        %         max_ind = sel_ind(li);
        %     end
        % end

        % preamble_offset = max_ind + preamble_offset - 1;
        % fprintf('sel ind=%d: %f\n', max_ind, max_p);
        % fprintf('==============\n');

        %% =============================================
        %% if we have only 1 packet
        % [p_max, idx] = max(p_lst);
        % preamble_offset = idx + start_offset - 1;


        %% =============================================
        %% if we have multiple packets
        [tmp_lst, tmp_ind] = sort(p_lst, 'descend');
        avg_cc = mean(p_lst);
        top3_avg_cc = mean(tmp_lst(1:10));
        fprintf('avg cc = %f, top 3 avg cc = %f\n', avg_cc, top3_avg_cc);
        this_thresh = (avg_cc + top3_avg_cc) / 2;
        tmp_idx_list = find(p_lst > this_thresh);

        %% every swch_samples*2, find a packet
        %% XXX: either the first or the second is the true start of the first packet
        prev = find_nearby_largest(p_lst, tmp_idx_list(1));
        for li = tmp_idx_list(2:end)
            if li >= prev + swch_samples/2
                prev2 = li;
                break;
            end
        end
        prev2 = find_nearby_largest(p_lst, prev2);
        %% XXX: even the first one is slight larger, still use the second one
        if p_lst(prev2) > p_lst(prev) * 0.99  
            prev = prev2;
        end

        preamble_offset = [prev];
        for li = tmp_idx_list(2:end)
            if li >= prev + swch_samples*2
                prev = li;
                preamble_offset = [preamble_offset, prev];
            end
        end

        preamble_offset = preamble_offset + start_offset - 1;
    end

end


%% find_nearby_largest
function [ind] = find_nearby_largest(p_lst, ind)
    new_ind = ind; 
    new_p   = p_lst(ind);

    for ii = ind+1:ind+20
        if p_lst(ii) > new_p
            new_ind = ii;
            new_p = p_lst(ii);
        end
    end

    ind = new_ind;
end