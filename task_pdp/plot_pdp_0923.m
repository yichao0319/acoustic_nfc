%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Yi-Chao Chen
%% 2014.02.22 @ UT Austin
%%
%% - Input:
%%
%%
%% - Output:
%%
%%
%% e.g.
%%
%%     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function get_pdp_0923()
    % addpath('../utils');
    
    %% --------------------
    %% DEBUG
    %% --------------------
    DEBUG0 = 0;
    DEBUG1 = 1;
    DEBUG2 = 1;

    PLOT_TRACE1 = 1;
    PLOT_TRACE2 = 0;


    %% --------------------
    %% Constant
    %% --------------------
    

    %% --------------------
    %% Variable
    %% --------------------
    input_rcv_dir   = '../processed_data/task_decode_multi/rcv_pkts/exp0923/';
    input_sent_dir  = '../processed_data/task_decode/sent_pkts/';
    output_dir      = '../processed_data/task_pdp/pdp/';
    output_fig_dir  = '../processed_data/task_pdp/figure/';

    slice_cnt = 10;
    slice_width = 4;
    nsc = 12;

    font_size = 18;


    %% --------------------
    %% Check input
    %% --------------------


    %% --------------------
    %% Main starts
    %% --------------------
    
    if(PLOT_TRACE1)
        fprintf('Plot Trace 1\n');

        seeds = [3];
        exps = [1:2];
        dists = [10:10:100];

        use_gamma = 0;
        use_p0    = 0;

        for this_exp = exps

            %% ===========================================
            %% power delay profile
            %% ===========================================
            pdps = zeros(length(dists), length(seeds), nsc);
        
            for di = [1:length(dists)]
                dist = dists(di);

                for si = [1:length(seeds)]
                    seed = seeds(si);

                    %% H
                    filename = [input_rcv_dir 'dist' int2str(dist) '.exp' int2str(this_exp) '.s' int2str(seed) '.h.txt'];
                    tmp = load(filename);
                    ncols = size(tmp, 2);
                    hfft = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) )'; 
                    num_pkts = size(hfft, 2) / slice_cnt;
                    fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));
                    fprintf('  num packets = %d\n', num_pkts);

                    %% Power Delay Profile
                    %% 1) all packets, all slices
                    this_pdp = get_pdp(hfft);
                    %% 2) all packets, first slices
                    % this_pdp = get_pdp(hfft(:, 1:slice_cnt:end));
                    %% 3) first packets, all slices
                    % this_pdp = get_pdp(hfft(:, 1:slice_cnt));
                    %% 4) first packets, first slices
                    % this_pdp = get_pdp(hfft(:, 1));

                    pdps(di, si, :) = reshape(this_pdp, 1, 1, []);
                end
            end

            % dlmwrite([output_dir 'exp' num2str(this_exp) '.dist' int2str(dists(di)) '.pdp.txt'], pdps, 'delimiter', '\t');
            % fprintf('  output PDP: %d x %d x %d\n', size(pdps));

            % plot PDP of each distance
            for di = [1:length(dists)]
                dist = dists(di);
                output_filename = [output_fig_dir 'dist' int2str(dists(di)) '.exp' num2str(this_exp)  '.pdp'];
                if(length(seeds) > 1)
                    avg_pdps = mean(squeeze(pdps(di, :, :)), 1);
                else
                    avg_pdps = squeeze(pdps(di, :, :));
                end
                plot_pdp(avg_pdps, output_filename);
            end


            %% ===========================================
            %% EDP
            %% ===========================================
            ind = 4;
            avg_pdps = squeeze(mean(pdps, 2));
            edps = avg_pdps(:, ind);
            
            %% find the best fit of gamma in path loss model
            best_gamma = find_pl_gamma(edps, dists);
            fprintf('  best gamma: %1.2g\n', best_gamma);

            %% plot EDP over distance
            output_filename = [output_fig_dir 'trace1.exp' num2str(this_exp) '.edp'];
            plot_edp_dist(edps, dists, output_filename);


            %% ===========================================
            %% estimate distance given edp
            %% ===========================================
            output_filename = [output_fig_dir 'trace1.exp' num2str(this_exp) '.err'];
            p0 = get_p0(edps, dists/100, best_gamma);
            if use_p0 == 0
                use_p0 = p0;
                use_gamma = best_gamma;
            end
            est_dists = est_dist_edp(edps, use_gamma, use_p0) * 100;

            for di = 1:length(dists)
                dist = dists(di);
                est_dist = est_dists(di);
                fprintf('|%f - %f| = %f\n', dist, est_dist, abs(dist - est_dist));
            end
            % mse = mean( (dists - est_dists).^2 );
            avg_err = mean(abs(dists - est_dists));
            normalized_avg_err = mean(abs(dists - est_dists) ./ dists);
            fprintf('  p0 = %f, gamma = %f\n', use_p0, use_gamma);
            fprintf('  avg err = %f\n', avg_err);
            fprintf('  normalized err = %f\n', normalized_avg_err);
            plot_dist_err(abs(dists - est_dists), dists, output_filename);

        end
    end  %% end PLOT_TRACE1


    %% ========================================================================

    if(PLOT_TRACE2)
        fprintf('Plot Trace 2\n');

        seeds = [3];
        exps = [1:2];
        dists = [10:10:120];
        angles = [90]; %[0, 90];

        %% use exp 1
        use_gamma = 0;
        use_p0    = 0;
        %% use trace 1, exp 1
        use_gamma = 0.05;
        use_p0    = 0.120760;
        %% use trace 2, dir 0, exp 1
        % use_gamma = 0.02;
        % use_p0    = 0.038776;
        

        for ai = 1:length(angles)
            this_angle = angles(ai);
            for this_exp = exps

                %% ===========================================
                %% power delay profile
                %% ===========================================
                pdps = zeros(length(dists), length(seeds), nsc);
            
                for di = [1:length(dists)]
                    dist = dists(di);

                    for si = [1:length(seeds)]
                        seed = seeds(si);

                        %% H
                        filename = [input_rcv_dir 'dist' int2str(dist) '.dir' int2str(this_angle) '.exp' int2str(this_exp) '.s' int2str(seed) '.h.txt'];
                        tmp = load(filename);
                        ncols = size(tmp, 2);
                        hfft = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) )'; 
                        num_pkts = size(hfft, 2) / slice_cnt;
                        fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));
                        fprintf('  num packets = %d\n', num_pkts);

                        %% Power Delay Profile
                        %% 1) all packets, all slices
                        this_pdp = get_pdp(hfft);
                        %% 2) all packets, first slices
                        % this_pdp = get_pdp(hfft(:, 1:slice_cnt:end));
                        %% 3) first packets, all slices
                        % this_pdp = get_pdp(hfft(:, 1:slice_cnt));
                        %% 4) first packets, first slices
                        % this_pdp = get_pdp(hfft(:, 1));

                        pdps(di, si, :) = reshape(this_pdp, 1, 1, []);
                    end
                end

                % dlmwrite([output_dir 'exp' num2str(this_exp) '.dist' int2str(dists(di)) '.pdp.txt'], pdps, 'delimiter', '\t');
                % fprintf('  output PDP: %d x %d x %d\n', size(pdps));

                % plot PDP of each distance
                for di = [1:length(dists)]
                    dist = dists(di);
                    output_filename = [output_fig_dir 'dist' int2str(dist) '.dir' int2str(this_angle) '.exp' int2str(this_exp) '.s' int2str(seed) '.pdp'];
                    if(length(seeds) > 1)
                        avg_pdps = mean(squeeze(pdps(di, :, :)), 1);
                    else
                        avg_pdps = squeeze(pdps(di, :, :));
                    end
                    plot_pdp(avg_pdps, output_filename);
                end


                %% ===========================================
                %% EDP
                %% ===========================================
                ind = 4;
                avg_pdps = squeeze(mean(pdps, 2));
                edps = avg_pdps(:, ind);
                
                %% find the best fit of gamma in path loss model
                best_gamma = find_pl_gamma(edps, dists);
                fprintf('  best gamma: %1.2g\n', best_gamma);

                %% plot PDP over distance
                output_filename = [output_fig_dir 'trace2.exp' num2str(this_exp) '.dir' int2str(this_angle) '.edp'];
                plot_edp_dist(edps, dists, output_filename);


                %% ===========================================
                %% estimate distance given edp
                %% ===========================================
                output_filename = [output_fig_dir 'trace2.exp' num2str(this_exp) '.dir' int2str(this_angle) '.err'];
                p0 = get_p0(edps, dists/100, best_gamma);
                if use_p0 == 0
                    use_p0 = p0;
                    use_gamma = best_gamma;
                end
                est_dists = est_dist_edp(edps, use_gamma, use_p0) * 100;

                for di = 1:length(dists)
                    dist = dists(di);
                    est_dist = est_dists(di);
                    fprintf('|%f - %f| = %f\n', dist, est_dist, abs(dist - est_dist));
                end
                % mse = mean( (dists - est_dists).^2 );
                avg_err = mean(abs(dists - est_dists));
                normalized_avg_err = mean(abs(dists - est_dists) ./ dists);
                fprintf('  p0 = %f, gamma = %f\n', use_p0, use_gamma);
                fprintf('  avg err = %f\n', avg_err);
                fprintf('  normalized err = %f\n', normalized_avg_err);
                plot_dist_err(abs(dists - est_dists), dists, output_filename);

            end
        end
    end  %% end PLOT_TRACE2

end

%% plot_pdp: function description
function plot_pdp(avg_pdps, output_filename)
    font_size = 18;

    fh = figure;
    clf;
    bar(avg_pdps);
    set(gca, 'FontSize', font_size);
    xlabel('Delay', 'FontSize', font_size);
    ylabel('Signal Strength', 'FontSize', font_size);
    % set(gca, 'XLim', [min(dists)-10 max(dists)+10]);  %% -Inf and Inf for automatic value
    % legend(legends, 'Location','EastOutside'); %% North | NorthEast | NorthOutside | Best | BestOutside
    print(fh, '-dpsc', [output_filename '.eps']);
end


%% plot_edp_dist: function description
function plot_edp_dist(edps, dists, output_filename)
    font_size = 18;
    gammas = [0.01, 0.02, 0.04, 0.05, 0.06];

    colors  = {'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k'};
    lines   = {'-', '-', '-', '-.', '-', '-', '-', '-.', '-', '-', '-', '-.'};
    markers = {'+', 'o', '*', '.', 'x', 's', 'd', '^', '>', '<', 'p', 'h', '+', 'o', '*', '.', 'x'};

    legends{1} = 'EDP';

    fh = figure;
    clf;
    lh1 = plot(dists, edps);
    set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh1, 'LineWidth', 4);
    set(lh1, 'marker', 'o');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
    hold on;

    for gi = [1:length(gammas)]
        this_gamma = gammas(gi);

        p0 = get_p0(edps, dists/100, this_gamma);
        
        legends{gi+1} = ['gamma=' num2str(this_gamma)];

        pr = path_loss(dists/100, p0, this_gamma);
        plot(dists, pr, ...
                    'Color', colors{gi}, ...
                    'LineStyle', lines{gi}, ...
                    'LineWidth', 2, ...
                    'marker', markers{gi});
        hold on;
    end

    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('Engry of Direct Path (EDP)', 'FontSize', font_size);
    legend(legends, 'Location', 'NorthEast'); %% North | NorthEast | NorthOutside | Best | BestOutside
    print(fh, '-dpsc', [output_filename '.eps']);

end


%% find_pl_gamma: function description
function [gamma] = find_pl_gamma(edps, dists)
    gammas = [0:0.01:0.1];

    min_mse = -1;
    gamma = -1;

    for gi = [1:length(gammas)]
        this_gamma = gammas(gi);

        p0 = get_p0(edps, dists/100, this_gamma);
        
        pr = path_loss(dists/100, p0, this_gamma)';

        ok_ind = intersect(find(~isnan(pr)), find(~isinf(pr)));
        mse = mean( (pr(ok_ind) - edps(ok_ind)).^2 );
        if(min_mse < 0 || mse < min_mse)
            min_mse = mse;
            gamma = this_gamma;
        end

        fprintf('    gamma=%1.2g: error=%1.2g\n', this_gamma, mse);
    end
end


%% path_loss: function description
function [pr] = path_loss(dist, p0, gamma)
    pr = p0 - 10*gamma*log10(dist);
end


%% est_dist_edp(edps, gamma)
function [dists] = est_dist_edp(edps, gamma, p0)
    dists = (10 .^ ((p0 - edps) / 10 / gamma))';
end

%% get_p0: function description
function [p0] = get_p0(edps, dists, gamma)
    ind_dist100 = find(dists == 1);
    if length(ind_dist100 > 0)
        p0 = edps(ind_dist100);
    else
        p0 = edps(end) + 10*gamma*log10(dists(end));
    end
end


%% plot_edp_dist: function description
function plot_dist_err(errs, dists, output_filename)
    font_size = 18;
    
    colors  = {'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k'};
    lines   = {'-', '-', '-', '-.', '-', '-', '-', '-.', '-', '-', '-', '-.'};
    markers = {'+', 'o', '*', '.', 'x', 's', 'd', '^', '>', '<', 'p', 'h', '+', 'o', '*', '.', 'x'};

    fh = figure;
    clf;
    lh1 = plot(dists, errs);
    set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh1, 'LineWidth', 4);
    set(lh1, 'marker', 'o');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
    
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('Estimation Error (cm)', 'FontSize', font_size);
    ylim([0 100]);
    print(fh, '-dpsc', [output_filename '.eps']);

end