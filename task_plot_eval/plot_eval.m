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

function plot_eval()
    % addpath('../utils');
    
    %% --------------------
    %% DEBUG
    %% --------------------
    DEBUG0 = 0;
    DEBUG1 = 1;
    DEBUG2 = 1;


    %% --------------------
    %% Constant
    %% --------------------


    %% --------------------
    %% Variable
    %% --------------------
    input_rcv_dir   = '../processed_data/task_decode/rcv_pkts/exp0523/';
    input_sent_dir  = '../processed_data/task_decode/sent_pkts/';
    output_data_dir = '../processed_data/task_plot_eval/data/';
    output_fig_dir  = '../processed_data/task_plot_eval/figure/';

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
    %% BER
    %%   1. clean
    %% --------------------
    seeds = [1:3];
    dists = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    BERs = zeros(length(seeds), length(dists));
    for di = [1:length(dists)]
        dist = dists(di);
        
        for si = [1:length(seeds)]
            seed = seeds(si);

            filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_' int2str(seed) '.demod.txt'];
            filename_gt = [input_sent_dir 'sent_pkt' int2str(seed) '.demod'];

            rcv_demod = load(filename);
            sent_demod = load(filename_gt);
            fprintf('  rcv: %s (%d x %d)\n', filename, size(rcv_demod));
            fprintf('  sent: %s (%d x %d)\n', filename_gt, size(sent_demod));

            ber = get_ber(sent_demod, rcv_demod);
            fprintf('    BER = %1.2g\n', ber);

            BERs(si, di) = ber;
        end
    end

    avg_ber = mean(BERs, 1);
    std_ber = std(BERs, 1);
    
    fh = figure;
    clf;
    bar(dists, avg_ber);
    hold on;
    errorbar(dists, avg_ber, std_ber, 'r', 'linestyle', 'none');
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('Bit Error Rate', 'FontSize', font_size);
    print(fh, '-dpsc', [output_fig_dir 'ber.clean.eps']);


    %% --------------------
    %% BER
    %%   2. hand
    %% --------------------
    dists = [0, 10, 20, 30, 40];
    BERs = zeros(length(seeds), length(dists));
    for di = [1:length(dists)]
        dist = dists(di);
        
        for si = [1:length(seeds)]
            seed = seeds(si);

            filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_hand' int2str(seed) '.demod.txt'];
            filename_gt = [input_sent_dir 'sent_pkt' int2str(seed) '.demod'];

            rcv_demod = load(filename);
            sent_demod = load(filename_gt);
            fprintf('  rcv: %s (%d x %d)\n', filename, size(rcv_demod));
            fprintf('  sent: %s (%d x %d)\n', filename_gt, size(sent_demod));

            ber = get_ber(sent_demod, rcv_demod);
            fprintf('    BER = %1.2g\n', ber);

            BERs(si, di) = ber;
        end
    end

    avg_ber = mean(BERs, 1);
    std_ber = std(BERs, 1);
    
    fh = figure;
    clf;
    bar(dists, avg_ber);
    hold on;
    errorbar(dists, avg_ber, std_ber, 'r', 'linestyle', 'none');
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('Bit Error Rate', 'FontSize', font_size);
    print(fh, '-dpsc', [output_fig_dir 'ber.hand.eps']);


    %% --------------------
    %% BER
    %%   3. obstacle
    %% --------------------
    dists = [0, 10, 20, 30, 40];
    BERs = zeros(length(seeds), length(dists));
    for di = [1:length(dists)]
        dist = dists(di);
        
        for si = [1:length(seeds)]
            seed = seeds(si);

            filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_obs' int2str(seed) '.demod.txt'];
            filename_gt = [input_sent_dir 'sent_pkt' int2str(seed) '.demod'];

            rcv_demod = load(filename);
            sent_demod = load(filename_gt);
            fprintf('  rcv: %s (%d x %d)\n', filename, size(rcv_demod));
            fprintf('  sent: %s (%d x %d)\n', filename_gt, size(sent_demod));

            ber = get_ber(sent_demod, rcv_demod);
            fprintf('    BER = %1.2g\n', ber);

            BERs(si, di) = ber;
        end
    end

    avg_ber = mean(BERs, 1);
    std_ber = std(BERs, 1);
    
    fh = figure;
    clf;
    bar(dists, avg_ber);
    hold on;
    errorbar(dists, avg_ber, std_ber, 'r', 'linestyle', 'none');
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('Bit Error Rate', 'FontSize', font_size);
    print(fh, '-dpsc', [output_fig_dir 'ber.obs.eps']);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% --------------------
    %% channel coefficient -- stdev
    %%   1. clean
    %% --------------------
    seeds = [1:3];
    dists = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    Hs = zeros(length(seeds), length(dists), nsc, slice_cnt);
    
    for di = [1:length(dists)]
        dist = dists(di);

        for ci = [1:nsc]
            tmp_Hs{ci} = zeros(length(seeds), slice_cnt);
        end
        for si = [1:length(seeds)]
            seed = seeds(si);

            filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_' int2str(seed) '.h.txt'];
            
            hfft = load(filename);
            fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));

            tmp_Hs{ci} = hfft;
            Hs(si, di, :, :) = reshape(hfft, 1, 1, nsc, slice_cnt);
        end

        fh = figure;
        clf;
        lh1 = plot(mean(tmp_Hs{1}, 1), 'r-');
        hold on;
        lh2 = plot(mean(tmp_Hs{2}, 1), 'b-.');
        hold on;
        set(gca, 'FontSize', font_size);
        xlabel('slice', 'FontSize', font_size);
        ylabel('std of CSI', 'FontSize', font_size);
        print(fh, '-dpsc', [output_fig_dir 'csi.std.dist' int2str(dist) '.clean.eps']);

    end

    
    % fh = figure;
    % clf;
    % bar(dists, avg_ber);
    % hold on;
    % errorbar(dists, avg_ber, std_ber, 'r', 'linestyle', 'none');
    % set(gca, 'FontSize', font_size);
    % xlabel('Distance (cm)', 'FontSize', font_size);
    % ylabel('Bit Error Rate', 'FontSize', font_size);
    % print(fh, '-dpsc', [output_fig_dir 'ber.clean.eps']);
end


%% get_ber: function description
function [ber] = get_ber(sent_demod, rcv_demod)
    slice_cnt = 10;
    slice_width = 4;

    nerrs = 0;
    nbits = 0;

    for si = [1:slice_cnt]
        col_std = (si-1) * slice_width + 2;
        col_end = si * slice_width;

        nerrs = nerrs + nnz(sent_demod(:, col_std:col_end) - rcv_demod(:, col_std:col_end));
        nbits = nbits + prod(size(sent_demod(:, col_std:col_end)));
    end

    ber = nerrs / nbits;
end
