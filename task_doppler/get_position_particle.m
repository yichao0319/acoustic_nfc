%% ====================================
%% Yi-Chao@UT Austin
%%
%% e.g.
%%   get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp1', 17999, 19999, [-0.1, 0.5], [0, 1.1])
%% ====================================

%% get_doppler_shift: function description
function move_dists = get_position_particle(input_dir, filename, f0, f1, x_lim, y_lim)
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

    spk1_pos = [0, 0];
    spk2_pos = [0.50, 0.50];
    % mic_pos = [0, 1];
    % mic_pos = [0, 0.5];
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
        fprintf('  t%d: %f\n', ti, this_t);

        for pidx = 1:NUM_PARTICLE
            %% if the particle is invalid
            if isinf(particle_pos(pidx, 1, ti)) & isinf(particle_pos(pidx, 2, ti))
                particle_pos(pidx, 1, ti+1) = Inf;
                particle_pos(pidx, 2, ti+1) = Inf;
                continue;
            end

            %% no movement
            if (traces_spk1(ti) == 0) & (traces_spk2(ti) == 0)
                particle_pos(pidx, :, ti+1) = particle_pos(pidx, :, ti);
                continue;
            end

            %% move
            dist2spk1 = cal_distance(spk1_pos, particle_pos(pidx, :, ti)) - traces_spk1(ti);
            dist2spk2 = cal_distance(spk2_pos, particle_pos(pidx, :, ti)) - traces_spk2(ti);
            % dist2spk1 = cal_distance(spk1_pos, mic_positions(:,1)) - sum_traces_spk1(ti);
            % dist2spk2 = cal_distance(spk2_pos, mic_positions(:,1)) - sum_traces_spk2(ti);
            
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

            particle_pos(pidx, :, ti+1) = new_pos;
        end
    end

    % mic_positions(:, end)
    % plot_2d_trace(mic_positions, ['./tmp/' filename '.2d_trace'], x_lim, y_lim);
    % plot_2d_trace_video(particle_pos, time_spk, ['./tmp/' filename '.particle'], x_lim, y_lim);
    % plot_2d_trace_video2(mic_positions, time_spk, ['./tmp/' filename '.2d_trace'], x_lim, y_lim);
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
function plot_2d_trace(a_trace, filename, x_lim, y_lim)
    fh = figure;
    clf;
    font_size = 18;

    lh1 = plot(a_trace(1, :), a_trace(2, :));
    set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh1, 'LineWidth', 2);
    set(lh1, 'marker', '.');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
    set(lh1, 'MarkerSize', 5);
    xlabel('X (m)', 'FontSize', font_size);
    ylabel('Y (m)', 'FontSize', font_size);
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
            if isinf(traces(pidx, 1, ti)) & isinf(traces(pidx, 2, ti))
                continue;
            end

            lh1 = plot(squeeze(traces(pidx, 1, 1:ti)), squeeze(traces(pidx, 2, 1:ti)));
            set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
            set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
            set(lh1, 'LineWidth', 2);
            hold on;
            lh2 = plot(traces(pidx, 1, ti), traces(pidx, 2, ti), 'ob');
            
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

