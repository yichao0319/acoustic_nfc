function batch_exp0718

input_rcv_dir  = '../data/rcv_pkts/exp0718/';
output_rcv_dir = '../processed_data/task_decode/rcv_pkts/exp0718/';


%% --------------------------------------------
%% experiments
%% --------------------------------------------
seeds = [1:5];
exps = [1:5];
dists = [10, 50, 100, 150, 200, 250, 300];
% seeds = [1:5];
% exps = [1:1];
% dists = [10];


for this_exp = exps
    for dist = dists
        for seed = seeds

            if (this_exp == 4 | this_exp == 5) & (dist ~= 100)
                continue;
            end

            filename = ['rcv_packet.exp' num2str(this_exp) '.dist' num2str(dist) '.s' num2str(seed)];
            fprintf('- file: %s\n', filename);


            start_offset = -1;
            threshold = -1;
            % start_offset = 1;
            % threshold = 0.7;
            BER = evaluate_one(seed, filename, input_rcv_dir, output_rcv_dir, start_offset, threshold);
            fprintf('  BER = %f\n', BER);

        end
    end
end


