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

    BER_CLEAN = 0;
    BER_HAND  = 0;
    BER_OBS   = 0;
    BER_MOVE  = 0;

    H_CLEAN = 1;
    H_OBS = 0;
    H_HAND = 0;
    H_MOVE = 0;

    SNR_PHASE_ALL = 1;
    PHASE_TIME = 1;
    SNR_STD = 1;
    PHASE_STD = 1;
    PHASE_STD_ALL = 1;

    



    %% --------------------
    %% Constant
    %% --------------------
    seeds = [1:5];

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
    if(BER_CLEAN)
        fprintf('BER - clean:\n');

        dists = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        BERs = zeros(length(seeds), length(dists));
        for di = [1:length(dists)]
            dist = dists(di);
            
            for si = [1:length(seeds)]
                seed = seeds(si);

                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_' int2str(seed) '.demod.txt'];
                filename_gt = [input_sent_dir 'sent_pkt' int2str(seed) '.demod.txt'];

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
    end


    %% --------------------
    %% BER
    %%   2. hand
    %% --------------------
    if(BER_HAND)
        fprintf('BER - hand:\n');

        dists = [0, 10, 20, 30, 40];
        BERs = zeros(length(seeds), length(dists));
        for di = [1:length(dists)]
            dist = dists(di);
            
            for si = [1:length(seeds)]
                seed = seeds(si);

                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_hand' int2str(seed) '.demod.txt'];
                filename_gt = [input_sent_dir 'sent_pkt' int2str(seed) '.demod.txt'];

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
    end


    %% --------------------
    %% BER
    %%   3. obstacle
    %% --------------------
    if(BER_OBS)
        fprintf('BER - obstacle:\n');

        dists = [0, 10, 20, 30, 40];
        BERs = zeros(length(seeds), length(dists));
        for di = [1:length(dists)]
            dist = dists(di);
            
            for si = [1:length(seeds)]
                seed = seeds(si);

                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_obs' int2str(seed) '.demod.txt'];
                filename_gt = [input_sent_dir 'sent_pkt' int2str(seed) '.demod.txt'];

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
    end

    %% --------------------
    %% BER
    %%   4. move
    %% --------------------
    if(BER_MOVE)
        fprintf('BER - move:\n');

        BERs = zeros(length(seeds), 2);
        for si = [1:length(seeds)]
            seed = seeds(si);

            %% away
            filename = [input_rcv_dir 'rcv_packet_away_' int2str(seed) '.demod.txt'];
            filename_gt = [input_sent_dir 'sent_pkt' int2str(seed) '.demod.txt'];

            rcv_demod = load(filename);
            sent_demod = load(filename_gt);
            fprintf('  rcv: %s (%d x %d)\n', filename, size(rcv_demod));
            fprintf('  sent: %s (%d x %d)\n', filename_gt, size(sent_demod));

            ber = get_ber(sent_demod, rcv_demod);
            fprintf('    BER = %1.2g\n', ber);

            BERs(si, 1) = ber;

            %% toward
            filename = [input_rcv_dir 'rcv_packet_toward_' int2str(seed) '.demod.txt'];
            filename_gt = [input_sent_dir 'sent_pkt' int2str(seed) '.demod.txt'];

            rcv_demod = load(filename);
            sent_demod = load(filename_gt);
            fprintf('  rcv: %s (%d x %d)\n', filename, size(rcv_demod));
            fprintf('  sent: %s (%d x %d)\n', filename_gt, size(sent_demod));

            ber = get_ber(sent_demod, rcv_demod);
            fprintf('    BER = %1.2g\n', ber);

            BERs(si, 2) = ber;
        end
    
        avg_ber = mean(BERs, 1);
        std_ber = std(BERs, 1);
        
        fh = figure;
        clf;
        bar([10, 20], avg_ber);
        hold on;
        errorbar([10, 20], avg_ber, std_ber, 'r', 'linestyle', 'none');
        set(gca, 'FontSize', font_size);
        xlabel('Distance (cm)', 'FontSize', font_size);
        ylabel('Bit Error Rate', 'FontSize', font_size);
        print(fh, '-dpsc', [output_fig_dir 'ber.move.eps']);
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% --------------------
    %% channel coefficient -- stdev
    %%   1. clean
    %% --------------------
    if(H_CLEAN)
        fprintf('Channel Coefficient - clean:\n');

        dists = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        snr_stds = zeros(length(dists), length(seeds), slice_cnt);
        pdps = zeros(length(dists), length(seeds), nsc);
        phase_stds = zeros(length(dists), length(seeds), nsc);
        phase_diffs = zeros(length(dists), length(seeds), nsc);
        
        for di = [1:length(dists)]
            dist = dists(di);

            for si = [1:length(seeds)]
                seed = seeds(si);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% preamble
                filename = [input_sent_dir 'preamble' int2str(seed) '.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                preamble = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  preamble: %s (%d x %d)\n', filename, size(preamble));
                
                %% H
                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_' int2str(seed) '.h.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                hfft = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));

                %% Power Delay Profile
                pdps(di, si, :) = reshape(abs(ifft(hfft(:,1))), 1, 1, []);
                
                %% rcv preamble
                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_' int2str(seed) '.preamble.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                rcv_preamble = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  rcv_preamble: %s (%d x %d)\n', filename, size(rcv_preamble));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                

                snr = get_snr(preamble, rcv_preamble);
                phase = angle(hfft);
                % norm_phase = adjust_phase(phase);
                norm_phase = remove_phase_rotation(phase);


                %% plot SNR and phase of each file
                if(SNR_PHASE_ALL)
                    output_filename = [output_fig_dir './snr.phase.all_file/clean.dist' int2str(dist) '.' int2str(seed)];
                    plot_snr_phase_sc(snr, phase, output_filename);
                end

                %% plot phase over time of each file
                if(PHASE_TIME)
                    output_filename = [output_fig_dir ...
                                       './phase_over_time.all_file/clean.dist' int2str(dist) '.' int2str(seed)];
                    plot_phase_time(norm_phase, output_filename);
                end
                
                %% SNR std
                this_snr_std = std(snr, 0, 1);
                snr_stds(di, si, :) = reshape(this_snr_std, 1, 1, []);

                %% phase std
                this_phase_std = std(norm_phase, 0, 2);
                phase_stds(di, si, :) = reshape(this_phase_std, 1, 1, []);

                %% phase diff
                this_phase_diff = mean(abs(norm_phase(:, 2:end) - norm_phase(:, 1:end-1)), 2);
                phase_diffs(di, si, :) = reshape(this_phase_diff, 1, 1, []);

            end
        end

        %% plot std of SNR
        if SNR_STD
            output_filename = [output_fig_dir 'clean.snr_std'];
            plot_snr_std(dists, snr_stds, output_filename);
        end

        %% plot std of phase
        if PHASE_STD
            for si = [1:length(seeds)]
                seed = seeds(si);
                % output_filename = [output_fig_dir 'clean.phase_std'];
                % plot_phase_std2(dists, phase_stds, seed, output_filename);
                output_filename = [output_fig_dir 'clean.phase_diff'];
                plot_phase_diff(dists, phase_diffs, seed, output_filename);
            end
        end

        if PHASE_STD_ALL
            output_filename = [output_fig_dir 'clean.phase_diff'];
            plot_phase_diff_all(dists, phase_diffs, seeds, output_filename);
        end

        %% plot PDP of each distance
        for di = [1:length(dists)]
            output_filename = [output_fig_dir 'clean.dist' int2str(dists(di)) '.pdp'];
            avg_pdps = mean(squeeze(pdps(di, :, :)), 1);
            plot_pdp(avg_pdps, output_filename);
        end
    end

    
    %% --------------------
    %% channel coefficient -- stdev
    %%   2. hand
    %% --------------------
    if(H_HAND)
        fprintf('Channel Coefficient - hand:\n');

        dists = [0, 10, 20, 30, 40];
        snr_stds = zeros(length(dists), length(seeds), slice_cnt);
        phase_stds = zeros(length(dists), length(seeds), nsc);
        phase_diffs = zeros(length(dists), length(seeds), nsc);
        
        for di = [1:length(dists)]
            dist = dists(di);

            for si = [1:length(seeds)]
                seed = seeds(si);

                %% preamble
                filename = [input_sent_dir 'preamble' int2str(seed) '.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                preamble = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  preamble: %s (%d x %d)\n', filename, size(preamble));
                
                %% H
                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_hand' int2str(seed) '.h.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                hfft = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));

                %% Power Delay Profile
                pdps(di, si, :) = reshape(abs(ifft(hfft(:,1))), 1, 1, []);
                
                %% rcv preamble
                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_hand' int2str(seed) '.preamble.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                rcv_preamble = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  rcv_preamble: %s (%d x %d)\n', filename, size(rcv_preamble));
                
                snr = get_snr(preamble, rcv_preamble);
                phase = angle(hfft);
                norm_phase = adjust_phase(phase);

                %% plot SNR and phase of each file
                if(SNR_PHASE_ALL)
                    output_filename = [output_fig_dir './snr.phase.all_file/hand.dist' int2str(dist) '.' int2str(seed)];
                    plot_snr_phase_sc(snr, phase, output_filename);
                end

                %% plot phase over time of each file
                if(PHASE_TIME)
                    output_filename = [output_fig_dir ...
                                       './phase_over_time.all_file/hand.dist' int2str(dist) '.' int2str(seed)];
                    plot_phase_time(norm_phase, output_filename);
                end
                
                %% SNR std
                this_snr_std = std(snr, 0, 1);
                snr_stds(di, si, :) = reshape(this_snr_std, 1, 1, []);

                %% phase std
                this_phase_std = std(norm_phase, 0, 2);
                phase_stds(di, si, :) = reshape(this_phase_std, 1, 1, []);

                %% phase diff
                this_phase_diff = mean(abs(norm_phase(:, 2:end) - norm_phase(:, 1:end-1)), 2);
                phase_diffs(di, si, :) = reshape(this_phase_diff, 1, 1, []);

            end
        end

        %% plot std of SNR
        if SNR_STD
            output_filename = [output_fig_dir 'hand.snr_std'];
            plot_snr_std(dists, snr_stds, output_filename);
        end

        %% plot std of phase
        if PHASE_STD
            for si = [1:length(seeds)]
                seed = seeds(si);
                % output_filename = [output_fig_dir 'clean.phase_std'];
                % plot_phase_std2(dists, phase_stds, seed, output_filename);
                output_filename = [output_fig_dir 'hand.phase_diff'];
                plot_phase_diff(dists, phase_diffs, seed, output_filename);
            end
        end

        if PHASE_STD_ALL
            output_filename = [output_fig_dir 'hand.phase_diff'];
            plot_phase_diff_all(dists, phase_diffs, seeds, output_filename);
        end

        %% plot PDP of each distance
        for di = [1:length(dists)]
            output_filename = [output_fig_dir 'hand.dist' int2str(dists(di)) '.pdp'];
            avg_pdps = mean(squeeze(pdps(di, :, :)), 1);
            plot_pdp(avg_pdps, output_filename);
        end
    end


    %% --------------------
    %% channel coefficient -- stdev
    %%   3. obstacle
    %% --------------------
    if(H_OBS)
        fprintf('Channel Coefficient - obstacle:\n');

        dists = [0, 10, 20, 30, 40];
        snr_stds = zeros(length(dists), length(seeds), slice_cnt);
        phase_stds = zeros(length(dists), length(seeds), nsc);
        phase_diffs = zeros(length(dists), length(seeds), nsc);
        
        for di = [1:length(dists)]
            dist = dists(di);

            for si = [1:length(seeds)]
                seed = seeds(si);

                %% preamble
                filename = [input_sent_dir 'preamble' int2str(seed) '.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                preamble = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  preamble: %s (%d x %d)\n', filename, size(preamble));
                
                %% H
                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_obs' int2str(seed) '.h.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                hfft = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));

                %% Power Delay Profile
                pdps(di, si, :) = reshape(abs(ifft(hfft(:,1))), 1, 1, []);
                
                %% rcv preamble
                filename = [input_rcv_dir 'rcv_packet_dist' int2str(dist) '_obs' int2str(seed) '.preamble.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                rcv_preamble = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  rcv_preamble: %s (%d x %d)\n', filename, size(rcv_preamble));
                
                snr = get_snr(preamble, rcv_preamble);
                phase = angle(hfft);
                norm_phase = adjust_phase(phase);

                %% plot SNR and phase of each file
                if(SNR_PHASE_ALL)
                    output_filename = [output_fig_dir './snr.phase.all_file/obs.dist' int2str(dist) '.' int2str(seed)];
                    plot_snr_phase_sc(snr, phase, output_filename);
                end

                %% plot phase over time of each file
                if(PHASE_TIME)
                    output_filename = [output_fig_dir ...
                                       './phase_over_time.all_file/obs.dist' int2str(dist) '.' int2str(seed)];
                    plot_phase_time(norm_phase, output_filename);
                end
                
                %% SNR std
                this_snr_std = std(snr, 0, 1);
                snr_stds(di, si, :) = reshape(this_snr_std, 1, 1, []);

                %% phase std
                this_phase_std = std(norm_phase, 0, 2);
                phase_stds(di, si, :) = reshape(this_phase_std, 1, 1, []);

                %% phase diff
                this_phase_diff = mean(abs(norm_phase(:, 2:end) - norm_phase(:, 1:end-1)), 2);
                phase_diffs(di, si, :) = reshape(this_phase_diff, 1, 1, []);

            end
        end

        %% plot std of SNR
        if SNR_STD
            output_filename = [output_fig_dir 'obs.snr_std'];
            plot_snr_std(dists, snr_stds, output_filename);
        end

        %% plot std of phase
        if PHASE_STD
            for si = [1:length(seeds)]
                seed = seeds(si);
                % output_filename = [output_fig_dir 'clean.phase_std'];
                % plot_phase_std2(dists, phase_stds, seed, output_filename);
                output_filename = [output_fig_dir 'obs.phase_diff'];
                plot_phase_diff(dists, phase_diffs, seed, output_filename);
            end
        end

        if PHASE_STD_ALL
            output_filename = [output_fig_dir 'obs.phase_diff'];
            plot_phase_diff_all(dists, phase_diffs, seeds, output_filename);
        end

        %% plot PDP of each distance
        for di = [1:length(dists)]
            output_filename = [output_fig_dir 'obs.dist' int2str(dists(di)) '.pdp'];
            avg_pdps = mean(squeeze(pdps(di, :, :)), 1);
            plot_pdp(avg_pdps, output_filename);
        end
    end


    %% --------------------
    %% channel coefficient -- stdev
    %%   4. move
    %% --------------------
    if(H_MOVE)
        fprintf('Channel Coefficient - move:\n');

        move_types = {'away', 'toward'};
        snr_stds = zeros(length(move_types), length(seeds), slice_cnt);
        pdps = zeros(length(move_types), length(seeds), nsc);
        phase_stds = zeros(length(move_types), length(seeds), nsc);
        phase_diffs = zeros(length(move_types), length(seeds), nsc);

        for mi = [1:length(move_types)]
            move_type = char(move_types{mi});

            for si = [1:length(seeds)]
                seed = seeds(si);

                %% preamble
                filename = [input_sent_dir 'preamble' int2str(seed) '.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                preamble = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  preamble: %s (%d x %d)\n', filename, size(preamble));
                
                %% H
                filename = [input_rcv_dir 'rcv_packet_' move_type '_' int2str(seed) '.h.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                hfft = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  hfft: %s (%d x %d)\n', filename, size(hfft));

                %% Power Delay Profile
                pdps(mi, si, :) = reshape(abs(ifft(hfft(:,1))), 1, 1, []);
                
                %% rcv preamble
                filename = [input_rcv_dir 'rcv_packet_' move_type '_' int2str(seed) '.preamble.txt'];
                tmp = load(filename);
                ncols = size(tmp, 2);
                rcv_preamble = complex( tmp(:, 1:ncols/2), tmp(:, ncols/2+1:end) ); 
                fprintf('  rcv_preamble: %s (%d x %d)\n', filename, size(rcv_preamble));
                
                snr = get_snr(preamble, rcv_preamble);
                phase = angle(hfft);
                norm_phase = adjust_phase(phase);

                %% plot SNR and phase of each file
                if(SNR_PHASE_ALL)
                    output_filename = [output_fig_dir './snr.phase.all_file/' move_type '.' int2str(seed)];
                    plot_snr_phase_sc(snr, phase, output_filename);
                end

                %% plot phase over time of each file
                if(PHASE_TIME)
                    output_filename = [output_fig_dir ...
                                       './phase_over_time.all_file/' move_type '.' int2str(seed)];
                    plot_phase_time(norm_phase, output_filename);
                end
                
                %% SNR std
                this_snr_std = std(snr, 0, 1);
                snr_stds(mi, si, :) = reshape(this_snr_std, 1, 1, []);

                %% phase std
                this_phase_std = std(norm_phase, 0, 2);
                phase_stds(mi, si, :) = reshape(this_phase_std, 1, 1, []);

                %% phase diff
                this_phase_diff = mean(abs(norm_phase(:, 2:end) - norm_phase(:, 1:end-1)), 2);
                phase_diffs(mi, si, :) = reshape(this_phase_diff, 1, 1, []);
            end
        end

        %% plot std of SNR
        if SNR_STD
            output_filename = [output_fig_dir 'move.snr_std'];
            plot_snr_std([10, 20], snr_stds, output_filename);
        end

        %% plot std of phase
        if PHASE_STD
            for si = [1:length(seeds)]
                seed = seeds(si);
                % output_filename = [output_fig_dir 'clean.phase_std'];
                % plot_phase_std2(dists, phase_stds, seed, output_filename);
                output_filename = [output_fig_dir 'move.phase_diff'];
                plot_phase_diff([10, 20], phase_diffs, seed, output_filename);
            end
        end

        if PHASE_STD_ALL
            output_filename = [output_fig_dir 'move.phase_diff'];
            plot_phase_diff_all([10, 20], phase_diffs, seeds, output_filename);
        end

        %% plot PDP of each distance
        for mi = [1:length(move_types)]
            move_type = char(move_types{mi});

            output_filename = [output_fig_dir 'move.' move_type '.pdp'];
            avg_pdps = mean(squeeze(pdps(mi, :, :)), 1);
            plot_pdp(avg_pdps, output_filename);
        end
    end
end

%% -----------------------------------
%% get_ber: function description
%% -----------------------------------
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


%% -----------------------------------
%% get_ber: function description
%% -----------------------------------
function [snr] = get_snr(preamble, rcv_preamble)
    % snr = log10( (abs(rcv_preamble).^2) / mean(mean(abs(rcv_preamble-preamble).^2)) * 10 ) * 10;

    %% H
    % fprintf('H\n');
    % h = rcv_preamble ./ preamble;
    % w = mean2(abs(rcv_preamble - preamble) .^ 2);
    % snr = pow2db(abs(h) .^ 2 / w)
    
    %% MMSE
    % fprintf('MMSE\n');
    s1 = abs( rcv_preamble .* conj(preamble) ) .^ 2;
    s0 = abs( mean2( rcv_preamble .* conj(preamble) ) ) .^ 2;
    w  = mean2( abs(rcv_preamble) .^ 2 ) - s0;
    snr = pow2db(s1 / w);

    %% EVM 
    % fprintf('EVM\n');
    % evm = abs((preamble - rcv_preamble)) .^ 2;
    % p0 = mean2( abs(preamble).^2 );
    % evm = evm / p0;
    % snr = ones(size(evm)) ./ evm;
    % snr = pow2db(snr)

end


%% -----------------------------------
%% adjust_phase
%% -----------------------------------
function [phase] = adjust_phase(phase)
    % idx = find(phase < 0);
    % phase(idx) = phase(idx) + 2*pi;
    thresh = 4;
    for sci = [1:size(phase,1)]
        for si = [2:size(phase,2)]
            if (phase(sci, si) - phase(sci, si-1)) > thresh
                phase(sci, si) = phase(sci, si) - 2*pi;
            elseif (phase(sci, si-1) - phase(sci, si)) > thresh
                phase(sci, si) = phase(sci, si) + 2*pi;
            end
        end

        % if find(phase(sci, :) > 2*pi)
        %     phase(sci, :) = phase(sci, :) - 2*pi;
        % end
        % if find(phase(sci, :) < -2*pi)
        %     phase(sci, :) = phase(sci, :) + 2*pi;
        % end
    end
end

%% remove_phase_rotation: function description
function [phase] = remove_phase_rotation(phase)
    for si = [1:size(phase,2)]
        phase(:, si) = phase(:, si) - phase(1, si);
    end

    phase(find(phase < 0)) = phase(find(phase < 0)) + 2*pi;
    phase = adjust_phase(phase);
end



%% plot_snr_phase_sc: function description
function plot_snr_phase_sc(snr, phase, output_filename)
    num_slice = 3;
    font_size = 18;

    %% SNR
    fh = figure;
    clf;
    plot([1:size(snr,1)], snr(:,1), 'Color', 'b', ...
                                    'LineStyle', '-', ...
                                    'LineWidth', 4, ...
                                    'marker', 'o');
    hold on;
    plot([1:size(snr,1)], snr(:,2), 'Color', 'r', ...
                                    'LineStyle', '--', ...
                                    'LineWidth', 4, ...
                                    'marker', 'x');
    plot([1:size(snr,1)], snr(:,3), 'Color', 'g', ...
                                    'LineStyle', ':', ...
                                    'LineWidth', 4, ...
                                    'marker', '^');
    set(gca, 'FontSize', font_size);
    xlabel('Subcarrier', 'FontSize', font_size);
    ylabel('SNR (db)', 'FontSize', font_size);
    legend('slice 1', 'slice 2', 'slice 3');
    print(fh, '-dpsc', [output_filename '.snr.eps']);


    %% phase
    fh = figure;
    clf;
    plot([1:size(phase,1)], phase(:,1), 'Color', 'b', ...
                                    'LineStyle', '-', ...
                                    'LineWidth', 4, ...
                                    'marker', 'o');
    hold on;
    plot([1:size(phase,1)], phase(:,2), 'Color', 'r', ...
                                    'LineStyle', '--', ...
                                    'LineWidth', 4, ...
                                    'marker', 'x');
    plot([1:size(phase,1)], phase(:,3), 'Color', 'g', ...
                                    'LineStyle', ':', ...
                                    'LineWidth', 4, ...
                                    'marker', '^');
    set(gca, 'FontSize', font_size);
    xlabel('Subcarrier', 'FontSize', font_size);
    ylabel('Phase', 'FontSize', font_size);
    legend('slice 1', 'slice 2', 'slice 3');
    print(fh, '-dpsc', [output_filename '.phase.eps']);

end

%% plot_phase_time: function description
function plot_phase_time(phase, output_filename)
    font_size = 18;

    colors  = {'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k'};
    lines   = {'-', '-', '-', '-.', '-', '-', '-', '-.', '-', '-', '-', '-.'};
    markers = {'+', 'o', '*', '.', 'x', 's', 'd', '^', '>', '<', 'p', 'h', '+', 'o', '*', '.', 'x'};

    fh = figure;
    clf;
    for sci = [1:size(phase,1)]
        legends{sci} = ['sc' int2str(sci)];
        plot([1:size(phase,2)], phase(sci,:), 'Color', colors{sci}, ...
                                              'LineStyle', lines{sci}, ...
                                              'LineWidth', 2, ...
                                              'marker', markers{sci});
        hold on;
    end
    set(gca, 'FontSize', font_size);
    xlabel('Slice', 'FontSize', font_size);
    ylabel('Phase', 'FontSize', font_size);
    legend(legends, 'Location','EastOutside'); %% North | NorthEast | NorthOutside | Best | BestOutside
    print(fh, '-dpsc', [output_filename '.phase.eps']);
end

    

%% plot_snr_std: function description
function plot_snr_std(dists, snr_stds, output_filename)
    font_size = 18;

    snr_stds = reshape(snr_stds, size(snr_stds, 1), []);
    avg_snr_stds = mean(snr_stds, 2);
    std_snr_stds = std(snr_stds, 0, 2);

    fh = figure;
    bar(dists, avg_snr_stds);
    hold on;
    errorbar(dists, avg_snr_stds, std_snr_stds, 'r', 'linestyle', 'none');
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('Std of SNRs of subcarriers', 'FontSize', font_size);
    print(fh, '-dpsc', [output_filename '.eps']);
end


%% plot_phase_std: function description
function plot_phase_std(dists, phase_stds, output_filename)
    font_size = 18;

    colors  = {'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k'};
    lines   = {'-', '-', '-', '-.', '-', '-', '-', '-.', '-', '-', '-', '-.'};
    markers = {'+', 'o', '*', '.', 'x', 's', 'd', '^', '>', '<', 'p', 'h', '+', 'o', '*', '.', 'x'};

    fh = figure;
    clf;
    for sci = [1:size(phase_stds,3)]
        this_sci_phase_stds = squeeze(phase_stds(:, :, sci));
        avg_phase_stds = mean(this_sci_phase_stds, 2);
        std_phase_stds = std(this_sci_phase_stds, 0, 2);

        legends{sci} = ['sc' int2str(sci)];
        plot(dists, avg_phase_stds, ...
                    'Color', colors{sci}, ...
                    'LineStyle', lines{sci}, ...
                    'LineWidth', 2, ...
                    'marker', markers{sci});
        hold on;
    end
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('Std of phases of subcarriers', 'FontSize', font_size);
    legend(legends, 'Location','EastOutside'); %% North | NorthEast | NorthOutside | Best | BestOutside
    print(fh, '-dpsc', [output_filename '.eps']);
end


%% plot_phase_std: function description
function plot_phase_std2(dists, phase_stds, seed, output_filename)
    %% phase_stds: dist x seed x sc
    font_size = 18;

    % avg_phase_stds = squeeze(mean(phase_stds, 2));
    % std_phase_stds = squeeze(std(phase_stds, 0, 2));
    avg_phase_stds = squeeze(phase_stds(:,seed,:));
    % std_phase_stds = squeeze(phase_stds(:,1,:));

    for i = 1:12
        legends{i} = ['sc' int2str(i)];
    end

    fh = figure;
    clf;
    bar(dists, avg_phase_stds);
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('Std of phases over time', 'FontSize', font_size);
    set(gca, 'XLim', [min(dists)-10 max(dists)+10]);  %% -Inf and Inf for automatic value
    legend(legends, 'Location','EastOutside'); %% North | NorthEast | NorthOutside | Best | BestOutside
    print(fh, '-dpsc', [output_filename '.' int2str(seed) '.eps']);
end


%% plot_phase_diff: function description
function plot_phase_diff(dists, phase_diffs, seed, output_filename)
    %% phase_diffs: dist x seed x sc-1
    font_size = 18;

    avg_phase_diffs = squeeze(phase_diffs(:,seed,:));
    
    for i = 1:12
        legends{i} = ['sc' int2str(i)];
    end

    fh = figure;
    clf;
    bar(dists, avg_phase_diffs);
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('mean(|phase(t) - phase(t-1)|)', 'FontSize', font_size);
    set(gca, 'XLim', [min(dists)-10 max(dists)+10]);  %% -Inf and Inf for automatic value
    legend(legends, 'Location','EastOutside'); %% North | NorthEast | NorthOutside | Best | BestOutside
    print(fh, '-dpsc', [output_filename '.' int2str(seed) '.eps']);
end


%% plot_phase_diff: function description
function plot_phase_diff_all(dists, phase_diffs, seeds, output_filename)
    %% phase_diffs: dist x seed x sc-1
    font_size = 18;

    avg_phase_diffs = zeros(size(phase_diffs,1), size(phase_diffs,3));
    for seed = seeds
        avg_phase_diffs = avg_phase_diffs + squeeze(phase_diffs(:,seed,:));
    end
    avg_phase_diffs = avg_phase_diffs / length(seeds);
    
    for i = 1:12
        legends{i} = ['sc' int2str(i)];
    end

    fh = figure;
    clf;
    bar(dists, avg_phase_diffs);
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('mean(|phase(t) - phase(t-1)|)', 'FontSize', font_size);
    set(gca, 'XLim', [min(dists)-10 max(dists)+10]);  %% -Inf and Inf for automatic value
    legend(legends, 'Location','EastOutside'); %% North | NorthEast | NorthOutside | Best | BestOutside
    print(fh, '-dpsc', [output_filename '.all.eps']);
end


%% plot_pdp: function description
function plot_pdp(avg_pdps, output_filename)
    font_size = 18;

    size(avg_pdps)

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

