%% ====================================
%% Yi-Chao@UT Austin
%%
%% e.g.
%%   get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp1', 17999, 19999, [-0.1, 0.5], [0, 1.1])
%% ====================================

%% get_doppler_shift: function description
function [mic_positions] = get_position_particle(input_dir, filename, f0, f1, x_lim, y_lim)
    addpath('../util/matlab/change_point_detection/');
    opengl software;

    DEBUG0 = 0;
    DEBUG1 = 1;
    DEBUG2 = 1;  %% progress
    DEBUG3 = 0;  %% basic info
    DEBUG4 = 1;  %% process info
    DEBUG5 = 0;  %% final output
    DEBUG6 = 1;  %% show frequency shift


    NUM_PARTICLE = 1000;
    MIN_NUM_PART = 200;
    MOVE_SPEED_THRESH = 1;  %% m/s

    spk1_pos = [0, 0];
    % spk2_pos = [0.50, 0.50];
    spk2_pos = [0.6, 0];
    dist_of_spks = cal_distance(spk1_pos, spk2_pos);



    %% ======================
    %% get traces
    %% ======================
    if DEBUG2, fprintf('Get traces\n'); end

    for fi = 1:length(f0)
        this_f0 = f0(fi);
        [tmp_traces, time_spk] = get_doppler_shift(input_dir, filename, this_f0);
        if fi == 1
            traces_spk1 = reshape(tmp_traces, 1, []);
        else
            traces_spk1(fi, :) = tmp_traces;
        end
    end

    for fi = 1:length(f1)
        this_f1 = f1(fi);
        [tmp_traces, time_spk] = get_doppler_shift(input_dir, filename, this_f1);
        if fi == 1
            traces_spk2 = reshape(tmp_traces, 1, []);
        else
            traces_spk2(fi, :) = tmp_traces;
        end
    end
    traces_spk1 = [traces_spk1(:,1) diff(traces_spk1, 1, 2)];
    traces_spk2 = [traces_spk2(:,1) diff(traces_spk2, 1, 2)];
    plot_1d_trace(traces_spk1, traces_spk2, time_spk, ['./tmp/' filename '.1d_traces']);
    fprintf('  final dist to spk1: %f\n', sum(traces_spk1));
    fprintf('  final dist to spk2: %f\n', sum(traces_spk2));

    traces_spk1 = mean(traces_spk1, 1);
    traces_spk2 = mean(traces_spk2, 1);

    MOVE_DIST_THRESH = MOVE_SPEED_THRESH * mean2(time_spk(2:end) - time_spk(1:end-1));
    fprintf('  move dist thresh = %f\n', MOVE_DIST_THRESH);


    %% ======================
    %% particles
    %% ======================
    particle_pos = zeros(NUM_PARTICLE, 2, length(time_spk)+1);
    for pidx = 1:NUM_PARTICLE
        particle_pos(pidx, 1, 1) = rand * (x_lim(2)-x_lim(1)) + x_lim(1);
        particle_pos(pidx, 2, 1) = rand * (y_lim(2)-y_lim(1)) + y_lim(1);
    end


    %% ======================
    %% moving the mic
    %% ======================
    if DEBUG2, fprintf('Moving the mic...\n'); end

    
    for ti = 1:length(time_spk)
        this_t = time_spk(ti);
        % if this_t > 1
        %     break;
        % end
        % fprintf('  t%d: %f\n', ti, this_t);
        % fprintf('    size of particles: %dx%d\n', size(particle_pos,1), size(particle_pos,3));
            
        num_valid = 0;
        valid_idx = [-1];
        for pidx = 1:size(particle_pos, 1)
            %% no movement
            if (traces_spk1(ti) == 0) & (traces_spk2(ti) == 0)
                particle_pos(pidx, :, ti+1) = particle_pos(pidx, :, ti);
                num_valid = num_valid + 1;
                valid_idx(num_valid) = pidx;
                continue;
            end

            %% move
            % fprintf('    pidx=%d, ti=%d\n', pidx, ti);
            dist2spk1 = cal_distance(spk1_pos, particle_pos(pidx, :, ti)) - traces_spk1(ti);
            dist2spk2 = cal_distance(spk2_pos, particle_pos(pidx, :, ti)) - traces_spk2(ti);
            % dist2spk1 = cal_distance(spk1_pos, mic_positions(:,1)) - sum_traces_spk1(ti);
            % dist2spk2 = cal_distance(spk2_pos, mic_positions(:,1)) - sum_traces_spk2(ti);

            %% =================================================
            %% validate particles
            %% =================================================
            %% if no intersection, invalid the particle
            if dist_of_spks > (dist2spk1 + dist2spk2) | ... 
               (abs(dist2spk1 - dist2spk2) >= dist_of_spks)
                fprintf('  t%d (%f): dist of spks (%f) > mic2spk1 (%f) + mic2spk2 (%f)\n', ti, this_t, dist_of_spks, dist2spk1, dist2spk2);
                % mic_positions(:, ti+1) = mic_positions(:,ti);
                particle_pos(pidx, 1, ti+1) = Inf;
                particle_pos(pidx, 2, ti+1) = Inf;
                continue;
            end

            %% get intersection
            % points = get_intersection(spk1_pos, dist2spk1, spk2_pos, dist2spk2);
            points = get_intersection2(spk1_pos, dist2spk1, spk2_pos, dist2spk2);
            min_dist = 99;
            new_pos = particle_pos(pidx, :, ti);
            
            for ps = 1:size(points, 1)
                this_dist = cal_distance(particle_pos(pidx, :, ti), points(ps, :));

                if this_dist < min_dist | (ps == 1)
                    min_dist = this_dist;
                    new_pos = points(ps, :);
                end
            end

            %% ----------------------------------
            %% moving distance is too large
            if min_dist > MOVE_DIST_THRESH
                fprintf('  t%d (%f): moving distance (%f) > threshold (%f)\n', ti, this_t, min_dist, MOVE_DIST_THRESH);

                particle_pos(pidx, 1, ti+1) = Inf;
                particle_pos(pidx, 2, ti+1) = Inf;
                continue;
            end
            %% ----------------------------------

            particle_pos(pidx, :, ti+1) = new_pos;
            num_valid = num_valid + 1;
            valid_idx(num_valid) = pidx;
        end


        %% -------------------------
        %% update the particles
        %% -------------------------
        %% remove invalid samples
        tmp = particle_pos(valid_idx, :, :);
        particle_pos = tmp;
        %% resampling
        if num_valid < MIN_NUM_PART
            fprintf(' >> resampling...\n');
            [centers, x_range, y_range] = find_center(particle_pos(:, :, 1:ti+1));
            fprintf('    center (%f, %f), x range: (%f, %f), y range (%f, %f)\n', centers(:, ti+1), x_range, y_range);
            % fprintf('    center (%f, %f), x std: %f, y std %f\n', centers(:, ti+1), max(abs(x_range-centers(1,ti+1))), max(abs(y_range-centers(2,ti+1))));
            
            tmp = ones(NUM_PARTICLE, 2, size(particle_pos, 3)) * Inf;
            tmp(1:num_valid, :, :) = particle_pos;
            particle_pos = tmp;

            num_new_part = NUM_PARTICLE - num_valid;
            particle_pos(num_valid+1:end, 1, ti+1) = normrnd(...
                                                centers(1,ti+1), ...
                                                max(abs(x_range-centers(1,ti+1))), ...
                                                num_new_part, 1 ...
                                                );
            particle_pos(num_valid+1:end, 2, ti+1) = normrnd(...
                                                centers(2,ti+1), ...
                                                max(abs(y_range-centers(2,ti+1))), ...
                                                num_new_part, 1 ...
                                                );
        end
    end

    fprintf('  number of valid particles: %d\n', length(find(particle_pos(:, 1, end) ~= Inf)));

    % mic_positions(:, end)
    plot_2d_trace(particle_pos, ['./tmp/' filename '.particle'], x_lim, y_lim);
    % plot_2d_trace_video(particle_pos, time_spk, ['./tmp/' filename '.particle'], x_lim, y_lim);
    mic_positions = find_center(particle_pos);
end


%% cal_distance: function description
function [dist] = cal_distance(pos1, pos2)
    dist = sqrt((pos1(1)-pos2(1))^2 + (pos1(2)-pos2(2))^2);
end

%% get_intersection: function description
function [points] = get_intersection(c1, r1, c2, r2)
    syms x y
    epr1 = (x - c1(1))^2 + (y - c1(2))^2 - r1^2;
    epr2 = (x - c2(1))^2 + (y - c2(2))^2 - r2^2;
    [x, y] = solve(epr1, epr2, x, y);
    x = double(x);
    y = double(y);
    
    if isempty(x)
        error('This Two Circles Have No Cross!')
    end

    for si = 1:length(x)
        points(si, 1) = x(si);
        points(si, 2) = y(si);
    end
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


%% find_center: function description
function [centers, x_range, y_range] = find_center(particle_traces)
    centers = zeros(2, size(particle_traces, 3));
    x_range = zeros(1, 2);
    y_range = zeros(1, 2);

    for ti = 1:size(particle_traces, 3)
        idx = find(particle_traces(:, 1, ti) ~= Inf);
        centers(1, ti) = mean2(particle_traces(idx, 1, ti));
        centers(2, ti) = mean2(particle_traces(idx, 2, ti));
    end

    x_range = [min(centers(1, :)), max(centers(1, :))];
    y_range = [min(centers(2, :)), max(centers(2, :))];
end



%% --------------------------------------------------------------------
%% ploting
%% --------------------------------------------------------------------

%% plot_1d_trace: function description
function plot_1d_trace(traces1, traces2, time, filename)
    fh = figure;
    clf;
    font_size = 18;

    %% trace1
    subplot(2,1,1)
    lh1 = plot(time, traces1);
    set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh1, 'LineWidth', 2);
    set(lh1, 'marker', '.');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
    set(lh1, 'MarkerSize', 5);
    ylabel('dist. spk1 (m)', 'FontSize', font_size);
    
    %% trace1
    subplot(2,1,2)
    lh2 = plot(time, traces2);
    set(lh2, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh2, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh2, 'LineWidth', 2);
    set(lh2, 'marker', '.');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
    set(lh2, 'MarkerSize', 5);
    xlabel('Time (s)', 'FontSize', font_size);
    ylabel('dist. spk1 (m)', 'FontSize', font_size);

    print(fh, '-dpsc', [filename '.ps']);
end


%% plot_2d_trace: function description
function plot_2d_trace(traces, filename, x_lim, y_lim)

    fh = figure;
    clf;
    font_size = 18;

    valid_particle_idx = [-1];
    valid_particle_x = 0;
    valid_particle_y = 0;

    xs = squeeze(traces(:, 1, :));
    ys = squeeze(traces(:, 2, :));
    % idx = find(xs(:, end) ~= Inf);
    % avg_xs = mean(xs(idx, :), 1);
    % avg_ys = mean(ys(idx, :), 1);
    avg_xs = zeros(1, size(traces, 3));
    avg_ys = zeros(1, size(traces, 3));
    for ti = 1:size(traces, 3)
        idx = find(xs(:, ti) ~= Inf);
        avg_xs(1, ti) = mean2(xs(idx, ti));
        avg_ys(1, ti) = mean2(ys(idx, ti));
    end

    for pidx = 1:size(traces, 1)
        if isinf(traces(pidx, 1, end)) & isinf(traces(pidx, 2, end))
            continue;
        end

        noninf_t = find(traces(pidx, 1, :) ~= Inf);
        lh1 = plot(squeeze(traces(pidx, 1, noninf_t)), squeeze(traces(pidx, 2, noninf_t)));
            
        set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
        set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
        set(lh1, 'LineWidth', 1);
        hold on;
        plot(traces(pidx, 1, noninf_t(1)), traces(pidx, 2, noninf_t(1)), 'og');
        plot(traces(pidx, 1, noninf_t(end)), traces(pidx, 2, noninf_t(end)), 'ob');
    end

    lh2 = plot(avg_xs, avg_ys);
    set(lh2, 'Color', 'y');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh2, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh2, 'LineWidth', 2);
    lh3 = plot(avg_xs(1), avg_ys(1), '^g');
    set(lh3, 'MarkerFaceColor', 'g');
    set(lh3, 'MarkerSize', 6);
    lh4 = plot(avg_xs(end), avg_ys(end), '^b');
    set(lh4, 'MarkerFaceColor', 'b');
    set(lh4, 'MarkerSize', 6);
    
    xlim(x_lim);
    ylim(y_lim);

    print(fh, '-dpsc', [filename '.ps']);
end


function plot_2d_trace_video(traces, time, filename, x_lim, y_lim)
    
    frame_rate = ceil(length(time) / time(end));
    fprintf('frame rate = %d\n', frame_rate);

    % mov = VideoWriter([filename '.avi']);
    % mov.FrameRate = frame_rate;
    % open(mov);
    mov = avifile([filename '.avi'], 'fps', frame_rate);
    
    for ti = 1:length(time)

        fh = figure;
        clf;
        font_size = 18;

        for pidx = 1:size(traces, 1)
            if isinf(traces(pidx, 1, ti)) && isinf(traces(pidx, 2, ti))
                continue;
            end

            noninf_t = find(traces(pidx, 1, 1:ti) ~= Inf);
            lh1 = plot(squeeze(traces(pidx, 1, noninf_t)), squeeze(traces(pidx, 2, noninf_t)));
            set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
            set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
            set(lh1, 'LineWidth', 2);
            hold on;
            lh2 = plot(traces(pidx, 1, noninf_t(:)), traces(pidx, 2, noninf_t(:)), 'og');
            lh3 = plot(traces(pidx, 1, noninf_t(1)), traces(pidx, 2, noninf_t(1)), 'ob');
            
        end

        xlim(x_lim);
        ylim(y_lim);

        % frame = getframe(fh);
        % writeVideo(mov, frame);
        mov = addframe(mov, fh);
    end

    mov = close(mov);
end


% function plot_2d_trace_video2(a_trace, time, filename, x_lim, y_lim)
    
%     frame_rate = ceil(length(time) / time(end));
%     fprintf('frame rate = %d\n', frame_rate);

%     mov = VideoWriter([filename '.avi']);
%     mov.FrameRate = frame_rate;
%     open(mov);
    
%     for ti = 1:length(time)

%         fh = figure;
%         clf;
%         % font_size = 18;

%         lh1 = plot(a_trace(1, 1:ti), a_trace(2, 1:ti));
%         set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
%         set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
%         set(lh1, 'LineWidth', 2);
%         set(lh1, 'marker', '.');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
%         set(lh1, 'MarkerSize', 5);
%         % xlabel('X (m)', 'FontSize', font_size);
%         % ylabel('Y (m)', 'FontSize', font_size);
%         % xlabel('X (m)');
%         % ylabel('Y (m)');
%         xlim(x_lim);
%         ylim(y_lim);
        
%         % frame = getframe(fh);
%         % writeVideo(mov, frame);
%         img = hardcopy(fh, '-dzbuffer', '-r0');
%         writeVideo(mov, im2frame(img));
%     end

%     close(mov);
% end

