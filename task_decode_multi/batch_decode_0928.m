font_size = 18;

%% =============================================================================
input_dir  = '../data/rcv_pkts/exp0928/';
output_dir = '../processed_data/task_decode_multi/rcv_pkts/exp0928/';

dists = [10:10:100];
exps = [1:2];
angles = [0, 180];

for ai = 1:length(angles)
    this_angle = angles(ai);
    bers = zeros(1, length(dists));

    for di = 1:length(dists)
        this_dist = dists(di);
        avg_ber = 0;
        for this_exp = exps
            filename = ['dist' int2str(this_dist) '.dir' int2str(this_angle) '.exp' int2str(this_exp) '.s3'];
            this_bers = decode_one_file(3, filename, input_dir, output_dir, -1, -1);
            dlmwrite([output_dir filename '.ber.txt'], this_bers, 'delimiter', '\t');
            avg_ber = avg_ber + mean(this_bers);
        end
        avg_ber = avg_ber / length(exps);
        bers(di) = avg_ber;
    end

    fh = figure(1);
    clf;
    lh1 = plot(dists, bers);
    set(lh1, 'Color', 'r');      %% color : r|g|b|c|m|y|k|w|[.49 1 .63]
    set(lh1, 'LineStyle', '-');  %% line  : -|--|:|-.
    set(lh1, 'LineWidth', 4);
    set(lh1, 'marker', 'o');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
    set(lh1, 'MarkerSize', 10);
    set(gca, 'FontSize', font_size);
    xlabel('Distance (cm)', 'FontSize', font_size);
    ylabel('BER', 'FontSize', font_size);
    print(fh, '-dpsc', ['./tmp/exp0928.dir' int2str(this_angle) '.ber.ps']);
end
