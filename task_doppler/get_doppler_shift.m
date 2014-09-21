%% ====================================
%% Yi-Chao@UT Austin
%%
%% e.g.
%%   get_doppler_shift('../data/rcv_pkts/exp0918/', 'freq18k.v2.exp3')
%% ====================================

%% get_doppler_shift: function description
function [outputs] = get_doppler_shift(input_dir, filename)
    addpath('../util/matlab/change_point_detection/');

    DEBUG0 = 0;
    DEBUG1 = 1;
    DEBUG2 = 1;  %% progress
    DEBUG3 = 0;  %% basic info
    DEBUG4 = 1;  %% process info
    DEBUG5 = 0;  %% final output
    DEBUG6 = 1;  %% show frequency shift


    %% parameters for change point detection
    num_bootstrap = 100;   % number of iterations for Bootstrap Analysis
    conf_threshold = 0.99;  % the threshold for Signiﬁcance Testing
    rank_method = 0;
    filter_method = 0;


    %% ====================================
    %% Read the .wav file
    %% ====================================
    if DEBUG2, fprintf('Read file\n'); end

    file_path_name = [input_dir filename '.wav'];
    [wav_data, Fs, nbits] = wavread(file_path_name);
    wav_len = length(wav_data);
    wav_time = 0:1/Fs:(wav_len-1)/Fs;
    if DEBUG4, 
        fprintf('- wav_data %d x %d (Fs=%d, nbits=%d)\n', size(wav_data), Fs, nbits); 
        fprintf('  duration = %f\n', wav_time(end));
    end

    %% ====================================
    %% plot wav in time domain
    %% ====================================
    % fh = figure(1);
    % clf;
    % plot(wav_time, wav_data);
    % ylabel('Amplitude');
    % xlabel('Time (s)');
    % print(fh, '-dpsc', ['tmp/' filename '.time.ps']);


    %% ====================================
    %% Short-Time Fourier Transform
    %% ====================================
    if DEBUG2, fprintf('Short-time Fourier transform\n'); end

    % window = Fs/2; % Should be minimum twice the maximum frequency we want to analyze
    window = Fs/2;
    noverlap = floor(window/2); % 50% overlap
    Nfft = 44100;
    % Nfft = 2048;
    
    % Spectrogram takes the STFT of the signal
    % P matrix contains the power spectral density of each segment of the STFT
    [S,F,T,P] = spectrogram(wav_data, window, noverlap, Nfft, Fs);


    %% ====================================
    %% Plotting frequency-time Plot
    %% ====================================
    % if DEBUG4, fprintf('- plot S: %d x %d\n', size(S)); end;
    % fh = figure(2);
    % clf;
    % imagesc(T, F, real(S));
    % colorbar;
    % ylim([17500 18500]);
    % xlabel('Time (s)');
    % ylabel('Frequency (Hz)');
    % title('Time-Frequency plot of a Audio signal');
    % print(fh, '-dpsc', ['tmp/' filename '.freq-time.ps']);

    %% ====================================
    %% Plotting Power Spectral Density
    %% ====================================
    if DEBUG4, fprintf('- plot P: %d x %d\n', size(P)); end;
    fh = figure(3);
    clf;
    imagesc(T, F, 10*log10(P)); % frequency-time Plot of the signal
    colorbar;
    f_min = 16000;  %17500;
    f_max = 22000;  %18500;
    ylim([f_min f_max]);
    xlabel('Time (s)');
    ylabel('Power/Frequency (dB/Hz)');
    print(fh, '-dpsc', ['tmp/' filename '.psd.ps']);


    %% ====================================
    %% Calculate freq shift
    %% ====================================
    if DEBUG2, fprintf('Highest peak:\n'); end;
    % f_min = 17900;  f_min_ind = freq2ind(f_min, Fs/2, length(F));
    % f_max = 18100;  f_max_ind = freq2ind(f_max, Fs/2, length(F));
    f_min = 17000;  f_min_ind = freq2ind(f_min, Fs/2, length(F));
    f_max = 21000;  f_max_ind = freq2ind(f_max, Fs/2, length(F));
    target_band = 10*log10(P(f_min_ind:f_max_ind, :));
    % fh = figure(4); clf; imagesc(tmp); print(fh, '-dpsc', ['tmp/tmp.ps']);
    target_freq = F(f_min_ind:f_max_ind);
    target_f_min = 18000;
    target_f_max = 20000;
    target_f     = 19000;
    % plot_t = [6, 15, 18];
    % plot_t = [7, 9, 11];
    % plot_t = [5, 7, 9];
    % plot_t = [56];
    % plot_t = [11, 12, 14, 16, 17, 18];
    plot_t = [5, 9, 12];

    
    for t = 1:size(target_band, 2)
        if DEBUG6, fprintf('- t%d(%f):\n', t, T(t)); end;

        ts = target_band(:,t)';

        %% ====================================
        %% plot Power over frequency of this time period
        %% ====================================
        if ismember(t, plot_t), 
            % cps = detect_change_points(ts, num_bootstrap, conf_threshold, rank_method, filter_method, 'no_plot');
            fh = figure(4); clf; 
            plot(target_freq, ts); % hold on;
            % plot(target_freq(cps), ts(cps), 'or');
            xlim([target_freq(1), target_freq(end)]);
            xlabel('Frequency (Hz)'); ylabel('Power/Frequency (db/Hz)');
            print(fh, '-dpsc', ['tmp/'  filename '.t' int2str(t) '.ps']); 
        end;

        
        %% ====================================
        %% 1) First spike
        %% ====================================
        [spike_idx] = find_spikes(ts);
        peak_val = ts(spike_idx);
        [sort_val, sort_ind] = sort(peak_val, 'descend');
        
        first_val = sort_val(1);
        first_idx = spike_idx(sort_ind(1));
        [vr, vs] = doppler_velocity(target_freq(first_idx), target_f, -1, 0);
        if DEBUG6, fprintf('  1st peak: freq=%f, pwer=%f, v=%f\n', target_freq(first_idx), first_val, vr); end;


        %% ====================================
        %% 2) Second spike
        %% ====================================
        second_val = sort_val(2);
        second_idx = spike_idx(sort_ind(2));
        % if t==6, 
        %     fh = figure(4); clf; 
        %     plot(target_freq, ts); hold on;
        %     plot(target_freq(spike_idx), peak_val, 'or'); hold on;
        %     plot(target_freq(second_idx), second_val, 'xg');
        %     print(fh, '-dpsc', ['tmp/tmp.ps']); 
        % end
        [vr, vs] = doppler_velocity(target_freq(second_idx), target_f, -1, 0);
        if DEBUG6, fprintf('  2nd peak: freq=%f, pwer=%f, v=%f\n', target_freq(second_idx), second_val, vr); end;

        %% ====================================
        %% 3) Third spike
        %% ====================================
        if(length(sort_val) >= 3)
            third_val = sort_val(3);
            third_idx = spike_idx(sort_ind(3));
            [vr, vs] = doppler_velocity(target_freq(third_idx), target_f, -1, 0);
            if DEBUG6, fprintf('  3rd peak: freq=%f, pwer=%f, v=%f\n', target_freq(third_idx), third_val, vr); end;
        end


        %% ====================================
        %% 4) Centor frequency of the first spike
        %% ====================================
        [spk_min_ind, spk_max_ind] = find_spike_range(first_idx, ts);
        
        first_spk_freq = target_freq([spk_min_ind:spk_max_ind])';
        first_spk_pwer = ts([spk_min_ind:spk_max_ind]);
        % [spk_freq, spk_rss] = weighted_freq(first_spk_freq, first_spk_pwer);
        
        if ismember(t, plot_t), 
            fh = figure(5);
            clf;
            lh1 = plot(target_freq, ts); hold on;
            set(lh1, 'LineWidth', 2);
            plot(target_freq(first_idx), first_val, 'or'); hold on;
            lh2 = plot(first_spk_freq, first_spk_pwer, '.y-');
            set(lh2, 'LineWidth', 2);
            grid on;
            xlim([target_freq(1), target_freq(end)]);
            xlabel('Frequency (Hz)'); ylabel('Power/Frequency (db/Hz)');
            print(fh, '-dpsc', ['tmp/' filename '.spk' int2str(t) '.ps']); 
        end
        
        [vr, vs] = doppler_velocity(mean(first_spk_freq), target_f, -1, 0);
        if DEBUG6, fprintf('  1st spike center: freq=%f, pwer=%f, v=%f\n', mean(first_spk_freq), mean(first_spk_pwer), vr); end;


        %% ====================================
        %% 5) Centor frequency of the second spike
        %% ====================================
        [spk_min_ind, spk_max_ind] = find_spike_range(second_idx, ts);
        
        second_spk_freq = target_freq([spk_min_ind:spk_max_ind])';
        second_spk_pwer = ts([spk_min_ind:spk_max_ind]);
        
        [vr, vs] = doppler_velocity(mean(second_spk_freq), target_f, -1, 0);
        if DEBUG6, fprintf('  2nd spike center: freq=%f, pwer=%f, v=%f\n', mean(second_spk_freq), mean(second_spk_pwer), vr); end;


        %% ====================================
        %% 6) Centor frequency of the third spike
        %% ====================================
        if(length(sort_val) >= 3)
            [spk_min_ind, spk_max_ind] = find_spike_range(third_idx, ts);
            
            third_spk_freq = target_freq([spk_min_ind:spk_max_ind])';
            third_spk_pwer = ts([spk_min_ind:spk_max_ind]);
            
            [vr, vs] = doppler_velocity(mean(third_spk_freq), target_f, -1, 0);
            if DEBUG6, fprintf('  3rd spike center: freq=%f, pwer=%f, v=%f\n', mean(third_spk_freq), mean(third_spk_pwer), vr); end;
        end


        %% ====================================
        %% 7) Centor frequency
        %% ====================================
        [center_freq, avg_rss] = weighted_freq(target_freq', ts);
        [vr, vs] = doppler_velocity(center_freq, target_f, -1, 0);
        if DEBUG6, fprintf('  Center: freq=%f, pwer=%f, v=%f\n', center_freq, avg_rss, vr); end;


        %% ====================================
        %% 8) Change Point Detection
        %% ====================================
        cps = detect_change_points(ts, num_bootstrap, conf_threshold, rank_method, filter_method, 'no_plot');
        cps_freq = target_freq(cps);
        freq_shift_thresh = 0;
        idx = find(cps_freq <= (target_f_min-freq_shift_thresh));
        if length(idx) > 0
            cps_min_ind = cps(idx(end));
        else
            cps_min_ind = cps(1);
        end
            
        idx = find(cps_freq >= (target_f_max+freq_shift_thresh));
        if(length(idx) > 0)
            cps_max_ind = cps(idx(1));
        else
            cps_max_ind = cps(end);
        end
        
        cps_target_freq = target_freq([cps_min_ind:cps_max_ind])';
        cps_target_pwer = ts([cps_min_ind:cps_max_ind]);

        % [center_freq, avg_rss] = weighted_freq(cps_target_freq, cps_target_pwer);
        [vr, vs] = doppler_velocity(mean(cps_target_freq), target_f, -1, 0);
        if DEBUG6, fprintf('  CPs: freq=%f, pwer=%f, v=%f\n', mean(cps_target_freq), mean(cps_target_pwer), vr); end;
        
        if ismember(t, plot_t), 
            fh = figure(4); clf; 
            lh1 = plot(target_freq, ts); hold on;
            set(lh1, 'LineWidth', 2);
            lh2 = plot(target_freq(cps), ts(cps), 'or'); hold on;
            lh3 = plot(cps_target_freq, cps_target_pwer, '.y-.'); hold on;
            set(lh3, 'LineWidth', 2);
            plot(target_freq([cps_min_ind, cps_max_ind]), ts([cps_min_ind, cps_max_ind]), 'xg');
            xlim([target_freq(1), target_freq(end)]);
            xlabel('Frequency (Hz)'); ylabel('Power/Frequency (db/Hz)');
            grid on;
            legend([lh2, lh3], {'change points', 'sound'});
            print(fh, '-dpsc', ['tmp/' filename '.cps' int2str(t) '.ps']); 
        end

    end

    
end


%% weighted_freq: weighted center
function [center_freq, avg_rss] = weighted_freq(freq, rss)
  avg_rss = mean(rss);
  center_freq = sum(freq .* (rss/sum(rss)));
end

%% freq2ind
function [ind] = freq2ind(freq, max_freq, F_len)
    ind = floor(freq / max_freq * F_len);
end


%% find_spikes: function description
function [spike_idx] = find_spikes(ts)
    phase = 0;
    spike_idx = [-1];
    
    for ti = 1:length(ts)
        val = ts(ti);
        if phase == 0,
            phase = 1;
        elseif phase == 1,
            if prev_val > val,
                if spike_idx(1) == -1,
                    spike_idx = [ti-1];
                else
                    spike_idx = [spike_idx, ti-1];
                end
                
                phase = 2;
            end
        elseif phase == 2,
            if prev_val < val,
                phase = 1;
            end
        end

        prev_val = val;
    end
end

                
            
%% find_spike_range: function description
function [min_ind, max_ind] = find_spike_range(ind, rss)
    spike_thresh = 3; %% ignore small spike
    min_ind = ind;
    max_ind = ind;

    next_val = rss(ind);
    for t = [ind-1:-1:1]
        val = rss(t) - spike_thresh;
        if val <= next_val
            next_val = rss(t);
        else
            min_ind = t + 1;
            break;
        end
    end

    prev_val = rss(ind);
    for t = [ind+1:1:length(rss)]
        val = rss(t) - spike_thresh;
        if val <= prev_val
            prev_val = rss(t);
        else
            max_ind = t - 1;
            break;
        end
    end

end


%% doppler_velocity: function description
function [vr, vs] = doppler_velocity(f, f0, vr, vs)
    %% f = (c + vr) / (c + vs) * f0
    %%    c: sound speed
    %%    vr: velocity of the receiver (positive if moving toward, negative if otherwise)
    %%    vs: velocity of the sender (positive if moving away, negative if otherwise)
    c = 331 + 0.6 * 26;

    if(vr < 0)
        vr = f / f0 * (c + vs) - c;
    end

    if(vs < 0)
        vs = (c + vr) * f0 / f - c;
    end
end

    
