
TR1_TOWARD = 1;
TR1_AWAY   = 1;
TR2_TOWARD = 1;
TR2_AWAY   = 1;
TR3_SPK1   = 1;
TR3_SPK2   = 1;
TR4_SPK1   = 1;
TR4_SPK2   = 1;

avg_errs = zeros(1, 1);
std_errs = zeros(1, 1);
tr_cnt = 0;
f0 = 17999;
f1 = 19999;

%% =================================
%% trace 1 -- toward
%% =================================

if TR1_TOWARD
    tr_cnt = tr_cnt + 1;

    input_dir = '../data/rcv_pkts/exp0930/';
    directions = {'toward'};
    real_dist = 1;
    src = 1;
    exps = [1:5];
    move_dists = zeros(length(directions), length(exps));

    for di = 1:length(directions)
        this_dir = char(directions(di));
        for ei = 1:length(exps)
            this_exp = exps(ei);
            filename = ['freq18k.dist100.s' int2str(src) '.' this_dir '.exp' int2str(this_exp)];
            fprintf('> %s\n', filename);

            [traces, time] = get_doppler_shift(input_dir, filename, f0);
            move_dists(di, ei) = traces(1, end);
        end
    end

    % est_err = mean2(abs(move_dists - real_dist) / mean(real_dist));
    est_err = abs(move_dists - real_dist);
    est_err_avg = mean(est_err(:));
    est_err_std = std(est_err(:));
    fprintf('trace 1 [toward] error: avg=%f, std=%f\n', est_err_avg, est_err_std);

    avg_errs(1, tr_cnt) = est_err_avg;
    std_errs(1, tr_cnt) = est_err_std;
end


%% =================================
%% trace 1 -- away
%% =================================

if TR1_AWAY
    tr_cnt = tr_cnt + 1;

    input_dir = '../data/rcv_pkts/exp0930/';
    directions = {'away'};
    real_dist = -1;
    src = 1;
    exps = [1:5];
    move_dists = zeros(length(directions), length(exps));

    for di = 1:length(directions)
        this_dir = char(directions(di));
        for ei = 1:length(exps)
            this_exp = exps(ei);
            filename = ['freq18k.dist100.s' int2str(src) '.' this_dir '.exp' int2str(this_exp)];
            fprintf('> %s\n', filename);

            [traces, time] = get_doppler_shift(input_dir, filename, f0);
            move_dists(di, ei) = traces(1, end);
        end
    end

    % est_err = mean2(abs(move_dists - real_dist) / mean(real_dist));
    est_err = abs(move_dists - real_dist);
    est_err_avg = mean(est_err(:));
    est_err_std = std(est_err(:));
    fprintf('trace 1 [away] error: avg=%f, std=%f\n', est_err_avg, est_err_std);

    avg_errs(1, tr_cnt) = est_err_avg;
    std_errs(1, tr_cnt) = est_err_std;
end


%% =================================
%% trace 2 -- toward
%% =================================

if TR2_TOWARD
    tr_cnt = tr_cnt + 1;

    input_dir = '../data/rcv_pkts/exp0930/';
    directions = {'toward'};
    real_dist = 1;
    src = 2;
    exps = [1:5];
    move_dists = zeros(length(directions), length(exps));

    for di = 1:length(directions)
        this_dir = char(directions(di));
        for ei = 1:length(exps)
            this_exp = exps(ei);
            filename = ['freq18k.dist100.s' int2str(src) '.' this_dir '.exp' int2str(this_exp)];
            fprintf('> %s\n', filename);

            [traces, time] = get_doppler_shift(input_dir, filename, f0);
            move_dists(di, ei) = traces(1, end);
        end
    end

    % est_err = mean2(abs(move_dists - real_dist) / mean(real_dist));
    est_err = abs(move_dists - real_dist);
    est_err_avg = mean(est_err(:));
    est_err_std = std(est_err(:));
    fprintf('trace 2 [toward] error: avg=%f, std=%f\n', est_err_avg, est_err_std);

    avg_errs(1, tr_cnt) = est_err_avg;
    std_errs(1, tr_cnt) = est_err_std;
end


%% =================================
%% trace 2 -- away
%% =================================

if TR2_AWAY
    tr_cnt = tr_cnt + 1;

    input_dir = '../data/rcv_pkts/exp0930/';
    directions = {'away'};
    real_dist = -1;
    src = 2;
    exps = [1:5];
    move_dists = zeros(length(directions), length(exps));

    for di = 1:length(directions)
        this_dir = char(directions(di));
        for ei = 1:length(exps)
            this_exp = exps(ei);
            filename = ['freq18k.dist100.s' int2str(src) '.' this_dir '.exp' int2str(this_exp)];
            fprintf('> %s\n', filename);

            [traces, time] = get_doppler_shift(input_dir, filename, f0);
            move_dists(di, ei) = traces(1, end);
        end
    end

    % est_err = mean2(abs(move_dists - real_dist) / mean(real_dist));
    est_err = abs(move_dists - real_dist);
    est_err_avg = mean(est_err(:));
    est_err_std = std(est_err(:));
    fprintf('trace 2 [away] error: avg=%f, std=%f\n', est_err_avg, est_err_std);

    avg_errs(1, tr_cnt) = est_err_avg;
    std_errs(1, tr_cnt) = est_err_std;
end


%% =================================
%% trace 3 -- speaker 1
%% =================================

if TR3_SPK1
    tr_cnt = tr_cnt + 1;

    input_dir = '../data/rcv_pkts/exp0930/';
    real_dist = 0.8;
    exps = [1:5];
    move_dists = zeros(1, length(exps));

    for ei = 1:length(exps)
        this_exp = exps(ei);
        filename = ['freq18k20k.dist80_0.exp' int2str(this_exp)];
        fprintf('> %s\n', filename);

        [traces, time] = get_doppler_shift(input_dir, filename, f0);
        move_dists(1, ei) = traces(1, end);
    end

    % est_err = mean2(abs(move_dists - real_dist) / mean(real_dist));
    est_err = abs(move_dists - real_dist);
    est_err_avg = mean(est_err(:));
    est_err_std = std(est_err(:));
    fprintf('trace 3 [spk1] error: avg=%f, std=%f\n', est_err_avg, est_err_std);

    avg_errs(1, tr_cnt) = est_err_avg;
    std_errs(1, tr_cnt) = est_err_std;
end

%% =================================
%% trace 3 -- speaker 2
%% =================================

if TR3_SPK2
    tr_cnt = tr_cnt + 1;

    input_dir = '../data/rcv_pkts/exp0930/';
    real_dist = sqrt(0.5^2+0.5^2) - sqrt(0.5^2+0.3^2);
    exps = [1:5];
    move_dists = zeros(1, length(exps));

    for ei = 1:length(exps)
        this_exp = exps(ei);
        filename = ['freq18k20k.dist80_0.exp' int2str(this_exp)];
        fprintf('> %s\n', filename);

        [traces, time] = get_doppler_shift(input_dir, filename, f1);
        move_dists(1, ei) = traces(1, end);
    end

    % est_err = mean2(abs(move_dists - real_dist) / mean(real_dist));
    est_err = abs(move_dists - real_dist);
    est_err_avg = mean(est_err(:));
    est_err_std = std(est_err(:));
    fprintf('trace 3 [spk2] error: avg=%f, std=%f\n', est_err_avg, est_err_std);

    avg_errs(1, tr_cnt) = est_err_avg;
    std_errs(1, tr_cnt) = est_err_std;
end


%% =================================
%% trace 4 -- speaker 1
%% =================================
if TR4_SPK1
    tr_cnt = tr_cnt + 1;

    input_dir = '../data/rcv_pkts/exp0930/';
    real_dist = 0.50 - sqrt(0.50^2+0.30^2);
    exps = [1:5];
    move_dists = zeros(1, length(exps));

    for ei = 1:length(exps)
        this_exp = exps(ei);
        filename = ['freq18k20k.dist0_30.exp' int2str(this_exp)];
        fprintf('> %s\n', filename);

        [traces, time] = get_doppler_shift(input_dir, filename, f0);
        move_dists(1, ei) = traces(1, end);
    end
    
    % est_err = mean2(abs(move_dists - real_dist) / mean(real_dist));
    est_err = abs(move_dists - real_dist);
    est_err_avg = mean(est_err(:));
    est_err_std = std(est_err(:));
    fprintf('trace 4 [spk2] error: avg=%f, std=%f\n', est_err_avg, est_err_std);

    avg_errs(1, tr_cnt) = est_err_avg;
    std_errs(1, tr_cnt) = est_err_std;
end



%% =================================
%% trace 4 -- speaker 2
%% =================================

if TR4_SPK2
    tr_cnt = tr_cnt + 1;

    input_dir = '../data/rcv_pkts/exp0930/';
    real_dist = 0.3;
    exps = [1:5];
    move_dists = zeros(1, length(exps));

    for ei = 1:length(exps)
        this_exp = exps(ei);
        filename = ['freq18k20k.dist0_30.exp' int2str(this_exp)];
        fprintf('> %s\n', filename);

        [traces, time] = get_doppler_shift(input_dir, filename, f1);
        move_dists(1, ei) = traces(1, end);
    end
    
    % est_err = mean2(abs(move_dists - real_dist) / mean(real_dist));
    est_err = abs(move_dists - real_dist);
    est_err_avg = mean(est_err(:));
    est_err_std = std(est_err(:));
    fprintf('trace 4 [spk2] error: avg=%f, std=%f\n', est_err_avg, est_err_std);

    avg_errs(1, tr_cnt) = est_err_avg;
    std_errs(1, tr_cnt) = est_err_std;
end



for tri = 1:length(avg_errs)
    fprintf('tr%d: avg=%fm, std=%fm\n', tri, avg_errs(1,tri), std_errs(1,tri));
end
