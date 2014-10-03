%% realtime_position: function description
function realtime_position()

    %% ====================================
    %% parameters
    %% ====================================
    Fs = 44100;
    window = floor(Fs/32);
    noverlap = floor(window/4); % 75% overlap
    Nfft = Fs;
    frame_size = Fs;

    f1 = [16999];
    f2 = [17999];
    half_band = 200;

    spk1_pos = [0, 0];
    spk2_pos = [0.60, 0];
    dist_of_spks = cal_distance(spk1_pos, spk2_pos);

    NUM_PARTICLE = 1000;
    num_valid_pat = NUM_PARTICLE;
    x_lim = [-0.1 1];
    y_lim = [-2 0];

    MOVE_SPEED_THRESH = 1;  %% m/s
    MOVE_DIST_THRESH = MOVE_SPEED_THRESH * (window/Fs);

    

    %% ====================================
    %% variables
    %% ====================================
    % duration = input('type the duration of the experiment:');
    duration = 100;
    traces1 = [-1];
    traces2 = [-1];
    traces = zeros(2, 1);
    % f_offsets = [-1];


    %% ======================
    %% particles
    %% ======================
    particle_pos = zeros(NUM_PARTICLE, 2, 10);
    for pidx = 1:num_valid_pat
        particle_pos(pidx, 1, 1) = rand * (x_lim(2)-x_lim(1)) + x_lim(1);
        particle_pos(pidx, 2, 1) = rand * (y_lim(2)-y_lim(1)) + y_lim(1);
    end


    har = dsp.AudioRecorder('SampleRate', 44100, ...
                            'NumChannels', 1, ...
                            'SamplesPerFrame', frame_size);
    disp('Speak into microphone now');

    base_time = 0;
    ti_all = 0;
    tic;
    while toc < duration,
        wav_data = step(har);
        % plot_wav(wav_data, Fs);
        fprintf('time: %f\n', toc);
        
        %% ====================================
        %% Short-Time Fourier Transform
        %% ====================================
        [S,F,T,P] = spectrogram(double(wav_data), window, noverlap, Nfft, Fs);
        T = T + base_time;
        base_time = base_time + frame_size / Fs;


        for ti = 1:length(T)
            ti_all = ti_all + 1;
            fprintf('  ti%d (%d):\n', ti, ti_all);

            %% ====================================
            %% Calculate freq and position shift
            %% ====================================
            
            %% for each tone from speaker 1
            avg_f_offset1 = 0;
            avg_v1 = 0;
            avg_d1 = 0;

            for fi = 1:length(f1)
                [peak_freq, freq_offset, vel, dist_offset] = get_shift(F, P, T, ti, Fs, f1(fi), half_band);

                avg_f_offset1 = avg_f_offset1 + freq_offset;
                avg_v1 = avg_v1 + vel;
                avg_d1 = avg_d1 + dist_offset;
            end

            avg_f_offset1 = avg_f_offset1 / length(f1);
            avg_v1 = avg_v1 / length(f1);
            avg_d1 = avg_d1 / length(f1);
            traces1(ti_all) = avg_d1;
            sum_movement1 = sum(traces1);
            
            %% for each tone from speaker 2
            avg_f_offset2 = 0;
            avg_v2 = 0;
            avg_d2 = 0;

            for fi = 1:length(f2)
                [peak_freq, freq_offset, vel, dist_offset] = get_shift(F, P, T, ti, Fs, f2(fi), half_band);

                avg_f_offset2 = avg_f_offset2 + freq_offset;
                avg_v2 = avg_v2 + vel;
                avg_d2 = avg_d2 + dist_offset;
            end

            avg_f_offset2 = avg_f_offset2 / length(f2);
            avg_v2 = avg_v2 / length(f2);
            avg_d2 = avg_d2 / length(f2);
            traces2(ti_all) = avg_d2;
            sum_movement2 = sum(traces2);

            
            %% ====================================
            %% Calculate absolute position
            %% ====================================
            valid_idx = [-1];
            for pidx = 1:num_valid_pat
                % fprintf('pidx=%d, t=%d\n', pidx, t);
                % size(particle_pos)
                
                [pos, valid] = update_position(particle_pos(pidx, :, ti_all), spk1_pos, spk2_pos, traces1(ti_all), traces2(ti_all), MOVE_DIST_THRESH);
                if valid == 1
                    if valid_idx(1) == -1
                        valid_idx = [pidx];
                    else
                        valid_idx = [valid_idx, pidx];
                    end
                    particle_pos(pidx, :, ti_all+1) = pos;
                end
            end
            if valid_idx(1) > 0
                tmp = particle_pos(valid_idx, :, :);
                particle_pos = tmp;
                num_valid_pat = size(particle_pos, 1);
            else
                particle_pos = zeros(NUM_PARTICLE, 2, 10);
                for pidx = 1:num_valid_pat
                    particle_pos(pidx, 1, ti_all+1) = rand * (x_lim(2)-x_lim(1)) + x_lim(1);
                    particle_pos(pidx, 2, ti_all+1) = rand * (y_lim(2)-y_lim(1)) + y_lim(1);
                end
                num_valid_pat = size(particle_pos, 1);
            end

            plot_particles(particle_pos(:, :, 1:ti_all+1), x_lim, y_lim);
        end

        plot_1d_trace(traces1, traces2);

    end

    release(har);
    disp('Recording complete');
end


%% doppler_velocity: function description
function [vr, vs] = doppler_velocity(f0, f_offset, vr, vs)
    %% f = (c + vr) / (c + vs) * f0
    %%    c: sound speed
    %%    vr: velocity of the receiver (positive if moving toward, negative if otherwise)
    %%    vs: velocity of the sender (positive if moving away, negative if otherwise)
    c = 331 + 0.6 * 26;

    %% ------------------
    %% XXX: filtering
    % idx = find(abs(f_offset) <= 1);
    % f_offset(idx) = 0;
    %% ------------------

    f = f0 + f_offset;

    if(vr < 0)
        vr = f / f0 * (c + vs) - c;
    end

    if(vs < 0)
        vs = (c + vr) * f0 / f - c;
    end
end


%% freq2ind
function [ind] = freq2ind(freq, max_freq, F_len)
    ind = floor(freq / max_freq * F_len);
end

%% get_shift: function description
function [peak_freq, freq_offset, vel, dist_offset] = get_shift(F, P, T, t, Fs, f1, half_band)
    f_min = f1 - half_band;
    f_max = f1 + half_band;
    f_min_ind = freq2ind(f_min, Fs/2, length(F));
    f_max_ind = freq2ind(f_max, Fs/2, length(F));    
    
    target_freq = F(f_min_ind:f_max_ind);
    ts = 10*log10(P(f_min_ind:f_max_ind, t))';
    
    if isinf(ts)
        % idx = find(ts, inf)
        peak_freq = 0;
        freq_offset = 0;
        vel = 0;
        dist_offset = 0;
        return;
    end

    %% ====================================
    %% peak of the first spike
    %% ====================================
    [tmp, peak_idx] = max(ts);
    peak_freq = target_freq(peak_idx);
    freq_offset = target_freq(peak_idx) - f1;
    [vel, vs] = doppler_velocity(f1, freq_offset, -1, 0);
    if t == 1
        dist_offset = vel * T(t);
    else
        dist_offset = vel * (T(t) - T(t-1));
    end
end


%% cal_distance: function description
function [dist] = cal_distance(pos1, pos2)
    dist = sqrt((pos1(1)-pos2(1))^2 + (pos1(2)-pos2(2))^2);
end


function [points] = get_intersection2(c1, r1, c2, r2)

    r = sqrt((c1(1)-c2(1))^2 + (c1(2)-c2(2))^2);
    if (r1 + r2 <= r || abs(r1-r2) >= r)
        error('This Two Circles Have No Cross!');
    end

    seta = acos((r1^2 + r^2 - r2^2)/2/r/r1);
    r_seta = atan2(c2(2)-c1(2), c2(1)-c1(1));

    alpha = [r_seta-seta, r_seta+seta];

    crossx = c1(1) + r1*cos(alpha);
    crossy = c1(2) + r1*sin(alpha);

    for si = 1:length(crossx)
        points(si, 1) = crossx(si);
        points(si, 2) = crossy(si);
    end
end

%% functionname: function description
function [pos, valid] = update_position(pre_pos, spk1_pos, spk2_pos, mv1, mv2, MOVE_DIST_THRESH)
    dist_of_spks = cal_distance(spk1_pos, spk2_pos);

    %% no movement
    if (mv1 == 0) & (mv2 == 0)
        pos = pre_pos;
        valid = 1;
        return;
    end

    %% move
    dist2spk1 = cal_distance(spk1_pos, pre_pos) - mv1;
    dist2spk2 = cal_distance(spk2_pos, pre_pos) - mv2;

    %% =================================================
    %% validate particles
    %% =================================================
    %% if no intersection, invalid the particle
    if dist_of_spks > (dist2spk1 + dist2spk2) | ... 
       (abs(dist2spk1 - dist2spk2) >= dist_of_spks)
        pos = [Inf, Inf];
        valid = 0;
        return;
    end

    %% get intersection
    points = get_intersection2(spk1_pos, dist2spk1, spk2_pos, dist2spk2);
    min_dist = 99;
    pos = pre_pos;
    
    for ps = 1:size(points, 1)
        this_dist = cal_distance(pre_pos, points(ps, :));

        if this_dist < min_dist | (ps == 1)
            min_dist = this_dist;
            pos = points(ps, :);
        end
    end

    %% ----------------------------------
    %% moving distance is too large
    if min_dist > MOVE_DIST_THRESH
        pos = [Inf, Inf];
        valid = 0;
        return;
    end
    %% ----------------------------------

    valid = 1;
end




%% ===============================
%% plots
%% ===============================

%% plot_power_spectral_density: function description
function plot_power_spectrum_density(T, F, P, f_min, f_max, fig_idx)
    fh = figure(fig_idx);
    clf;
    imagesc(T, F, 10*log10(P)); % frequency-time Plot of the signal
    colorbar;
    % f_min = 17500; %16000;  %
    % f_max = 18500; %22000;  %
    % f_min = 3200;
    % f_max = 3400;
    ylim([f_min f_max]);
    xlabel('Time (s)');
    ylabel('Power/Frequency (dB/Hz)');
    % print(fh, '-dpsc', ['tmp/' filename '.f' num2str(f_ctr) '.psd.ps']);
end


%% plot_wav: function description
function plot_wav(wav_data, Fs)
    fh = figure(11);
    clf;
    plot([1:length(wav_data)] / Fs, wav_data);
    xlabel('Time (s)');
    ylabel('Magnitude');
end


%% plot_1d_trace: function description
function plot_1d_trace(traces1, traces2)
    fh = figure(12);
    clf;
    font_size = 18;

    %% trace1
    subplot(2,1,1)
    lh1 = plot(traces1);
    set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh1, 'LineWidth', 2);
    set(lh1, 'marker', '.');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
    set(lh1, 'MarkerSize', 5);
    ylabel('dist. spk1 (m)', 'FontSize', font_size);
    
    %% trace1
    subplot(2,1,2)
    lh2 = plot(traces2);
    set(lh2, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh2, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh2, 'LineWidth', 2);
    set(lh2, 'marker', '.');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
    set(lh2, 'MarkerSize', 5);
    xlabel('Time (s)', 'FontSize', font_size);
    ylabel('dist. spk1 (m)', 'FontSize', font_size);

    % print(fh, '-dpsc', [filename '.ps']);
end

%% plot_particles: function description
function plot_particles(particle_pos, x_lim, y_lim)
    fh = figure(13);
    clf;
    font_size = 18;

    for pidx = 1:size(particle_pos, 1)
        posx = squeeze(particle_pos(pidx, 1, :));
        posy = squeeze(particle_pos(pidx, 2, :));

        plot(posx, posy, 'r-');
        plot(posx(1), posy(1), 'og');
        plot(posx(end), posy(end), 'ob');
        hold on;
    end

    %% center
    avgx = squeeze(mean(particle_pos(:, 1, :), 1));
    avgy = squeeze(mean(particle_pos(:, 2, :), 1));
    lh1 = plot(avgx, avgy);
    set(lh1, 'Color', 'y');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh1, 'LineWidth', 3);

    xlim(x_lim);
    ylim(y_lim);
    xlabel('X (m)', 'FontSize', font_size);
    ylabel('Y (m)', 'FontSize', font_size);

end
