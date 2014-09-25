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

function get_pdp_0718()
    % addpath('../utils');
    
    %% --------------------
    %% DEBUG
    %% --------------------
    DEBUG0 = 0;
    DEBUG1 = 1;
    DEBUG2 = 1;

    PLOT_EDP = 1;


    %% --------------------
    %% Constant
    %% --------------------
    seeds = [1:5];
    exps = [1:5];
    dists = [10, 50, 100, 150, 200, 250, 300];


    %% --------------------
    %% Variable
    %% --------------------
    input_rcv_dir   = '../processed_data/task_decode/rcv_pkts/exp0718/';
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
    
    %% --------------------
    %% EDP
    %% --------------------
    if(PLOT_EDP)
        fprintf('Plot EDP\n');

        for this_exp = [1:3]

            pdps = zeros(length(dists), length(seeds), nsc);
        
            for di = [1:length(dists)]
                dist = dists(di);

                for si = [1:length(seeds)]
                    seed = seeds(si);

                    %% H
                    filename = [input_rcv_dir 'rcv_packet.exp' num2str(this_exp) '.dist' int2str(dist) '.s' int2str(seed) '.h.txt'];
                    tmp = load(filename);
                    ncols = size(tmp, 2);
                    hfft = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                    fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));

                    %% Power Delay Profile
                    pdps(di, si, :) = reshape(abs(ifft(hfft(:,1))), 1, 1, []);
            
                end
            end

            % dlmwrite([output_dir 'exp' num2str(this_exp) '.dist' int2str(dists(di)) '.pdp.txt'], pdps, 'delimiter', '\t');
            % fprintf('  output PDP: %d x %d x %d\n', size(pdps));

            % plot PDP of each distance
            for di = [1:length(dists)]
                output_filename = [output_fig_dir 'exp' num2str(this_exp) '.dist' int2str(dists(di)) '.pdp'];
                avg_pdps = mean(squeeze(pdps(di, :, :)), 1);
                plot_pdp(avg_pdps, output_filename);
            end


            %% EDP
            ind = 4;
            ind_dist100 = find(dists == 100);
            avg_pdps = squeeze(mean(pdps, 2));
            p0 = avg_pdps(ind_dist100, ind);
            fprintf('  size of avg PDPs: %d x %d\n', size(avg_pdps));
            
            %% find the best fit of gamma in path loss model
            best_gamma = find_pl_gamma(avg_pdps(:, ind), dists, p0);
            fprintf('  best gamma: %1.2g\n', best_gamma);

            %% plot PDP over distance
            output_filename = [output_fig_dir 'exp' num2str(this_exp) '.edp'];
            plot_edp_dist(avg_pdps(:, ind), dists, p0, output_filename);
        end
    end  %% end PLOT_EDP

    
    %% --------------------
    %% EDP: move
    %% --------------------
    if(PLOT_EDP)
        % move_types = {'away', 'toward'};
        exps = [4:5];
        pdps = zeros(length(exps), length(seeds), nsc);

        for mi = [1:length(exps)]
            this_exp = exps(mi);

            for si = [1:length(seeds)]
                seed = seeds(si);

                %% H
                filename = [input_rcv_dir 'rcv_packet.exp' num2str(this_exp) '.dist100.s' int2str(seed) '.h.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                hfft = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));

                %% Power Delay Profile
                pdps(mi, si, :) = reshape(abs(ifft(hfft(:,1))), 1, 1, []);
            end
        end

        dlmwrite([output_dir 'exp0718.move.dist100.pdp'], pdps, 'delimiter', '\t');
        fprintf('  output PDP: %d x %d x %d\n', size(pdps));

        %% plot PDP of each distance
        for mi = [1:length(exps)]
            this_exp = exps(mi);

            output_filename = [output_fig_dir 'exp' num2str(this_exp) '.dist100.pdp'];
            avg_pdps = mean(squeeze(pdps(mi, :, :)), 1);
            plot_pdp(avg_pdps, output_filename);
        end
    end
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
function plot_edp_dist(edps, dists, p0, output_filename)
    font_size = 18;
    gammas = [0, 0.01, 0.02, 0.04, 0.08];

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
        gamma = gammas(gi);

        legends{gi+1} = ['gamma=' num2str(gamma)];

        pr = path_loss(dists/100, p0, gamma);
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


%% plot_edp_dist: function description
function plot_edp_dist2(edps, dists, output_filename)
    font_size = 18;
    gammas = [0, 0.01, 0.02, 0.04, 0.08];

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
        gamma = gammas(gi);

        legends{gi+1} = ['gamma=' num2str(gamma)];

        p0 = edps(end) + 10*gamma*log10(dists(end)/100);

        pr = path_loss(dists/100, p0, gamma);
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
function [gamma] = find_pl_gamma(edps, dists, p0)
    gammas = [0:0.01:0.1];

    min_mse = -1;
    gamma = -1;

    for gi = [1:length(gammas)]
        this_gamma = gammas(gi);
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

%% find_pl_gamma: function description
function [gamma] = find_pl_gamma2(edps, dists)
    gammas = [0:0.01:0.1];

    min_mse = -1;
    gamma = -1;

    for gi = [1:length(gammas)]
        this_gamma = gammas(gi);

        p0 = edps(end) + 10*this_gamma*log10(dists(end)/100);
        
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

