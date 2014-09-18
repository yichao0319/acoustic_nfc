function evaluate

addpath('../util/matlab/acoustic_nfc/');

global xxx_fec_decodeTable
global xxx_fec_encodeTable

input_rcv_dir  = '../data/rcv_pkts/exp0523/';
input_sent_dir = '../processed_data/task_decode/sent_pkts/';
output_rcv_dir = '../processed_data/task_decode/rcv_pkts/exp0523/';


% Use 44.1KHz as the sampling frequency for the WAV file, because it
% is almost universally supported on mobile devices.
Fs = 44100;

% Use 22KHz as max frequency as some devices only support 44KHz sampling rate
f_min = 18000;
f_max = 20000;
% f_min = 14000;
% f_max = 16000;

% use 128 samples
Nfft = 256;
% Nfft = 128;
Nfft_useful = 1 + floor(Nfft * (f_max - f_min) / Fs);  % Nfft_useful = 6

% 1/2 symbols for cyclic prefix
Ncp = Nfft/2;

% number of slices in the sandwich.  each slice consists
% of 1 sync symbol plus (slice_width - 1) data symbols
slice_count = 10;
slice_width = 4;

% on average how many channels a bit consumes in an OFDM symbo
channels_per_bit = 2;
data_size = Nfft_useful / channels_per_bit * (slice_width - 1) * slice_count;


fprintf('Fs               = %d\n', Fs);
fprintf('[min:max]        = [%d:%d]\n', f_min, f_max);
fprintf('Nfft             = %d\n', Nfft);
fprintf('Nfft_useful      = %d\n', Nfft_useful);
fprintf('Ncp              = %d\n', Ncp);
fprintf('slice_count      = %d\n', slice_count);
fprintf('slice_width      = %d\n', slice_width);
fprintf('channels_per_bit = %d\n', channels_per_bit);
fprintf('data_size        = %d\n', data_size);

% initialize FEC encoding / decoding 
[xxx_fec_decodeTable,xxx_fec_encodeTable] = fec_init_table();


%% --------------------------------------------
%% experiments
%% --------------------------------------------
seeds = [1:5];
expnames = {'rcv_packet_dist0_', 'rcv_packet_dist0_hand', 'rcv_packet_dist0_obs', ...
            'rcv_packet_dist10_', 'rcv_packet_dist10_hand', 'rcv_packet_dist10_obs', ...
            'rcv_packet_dist20_', 'rcv_packet_dist20_hand', 'rcv_packet_dist20_obs', ...
            'rcv_packet_dist30_', 'rcv_packet_dist30_hand', 'rcv_packet_dist30_obs', ...
            'rcv_packet_dist40_', 'rcv_packet_dist40_hand', 'rcv_packet_dist40_obs', ...
            'rcv_packet_dist50_', ...
            'rcv_packet_dist60_', ...
            'rcv_packet_dist70_', ...
            'rcv_packet_dist80_', ...
            'rcv_packet_dist90_', ...
            'rcv_packet_dist100_', ...
            'rcv_packet_away_', 'rcv_packet_toward_'};
BERS = zeros(length(expnames), length(seeds));
start_offsets{1} = ...
           [40000, 30000, 40000, ...
            40000, 40000, 50000, ...
            40000, 30000, 40000, ...
            50000, 40000, 50000, ...
            50000, 60000, 50000, ...
            60000, ...
            50000, ...
            50000, ...
            50000, ...
            40000, ...
            60000, ...
            60000, 60000];
thresholds{1} = ...
           [0.6, 0.4, 0.4, ...
            0.5, 0.4, 0.5, ...
            0.7, 0.5, 0.6, ...
            0.6, 0.6, 0.5, ...
            0.7, 0.6, 0.6, ...
            0.7, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.2, 0.2];
start_offsets{2} = ...
           [40000, 30000, 70000, ...
            40000, 30000, 40000, ...
            40000, 20000, 60000, ...
            60000, 40000, 40000, ...
            40000, 20000, 60000, ...
            50000, ...
            60000, ...
            40000, ...
            40000, ...
            50000, ...
            60000, ...
            40000, 40000];
thresholds{2} = ...
           [0.6, 0.5, 0.4, ...
            0.5, 0.4, 0.5, ...
            0.7, 0.6, 0.7, ...
            0.7, 0.6, 0.6, ...
            0.7, 0.6, 0.6, ...
            0.7, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.5, ...
            0.2, 0.2];
start_offsets{3} = ...
           [40000, 40000, 40000, ...
            30000, 20000, 40000, ...
            40000, 40000, 40000, ...
            40000, 60000, 50000, ...
            40000, 20000, 50000, ...
            60000, ...
            60000, ...
            60000, ...
            50000, ...
            40000, ...
            60000, ...
            40000, 40000];
thresholds{3} = ...
           [0.6, 0.4, 0.35, ...
            0.5, 0.4, 0.35, ...
            0.7, 0.6, 0.6, ...
            0.6, 0.6, 0.6, ...
            0.6, 0.6, 0.6, ...
            0.7, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.25, 0.25];
start_offsets{4} = ...
           [40000, 40000, 40000, ...
            40000, 30000, 40000, ...
            40000, 40000, 40000, ...
            50000, 30000, 50000, ...
            50000, 50000, 50000, ...
            40000, ...
            60000, ...
            50000, ...
            30000, ...
            40000, ...
            60000, ...
            30000, 30000];
thresholds{4} = ...
           [0.6, 0.4, 0.4, ...
            0.5, 0.4, 0.3, ...
            0.7, 0.6, 0.6, ...
            0.6, 0.5, 0.6, ...
            0.6, 0.6, 0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.5, ...
            0.25, 0.2];
start_offsets{5} = ...
           [40000, 20000, 40000, ...
            40000, 20000, 50000, ...
            40000, 60000, 50000, ...
            60000, 40000, 40000, ...
            50000, 40000, 40000, ...
            60000, ...
            50000, ...
            60000, ...
            50000, ...
            50000, ...
            60000, ...
            30000, 50000];
thresholds{5} = ...
           [0.6, 0.5, 0.3, ...
            0.5, 0.4, 0.4, ...
            0.6, 0.6, 0.6, ...
            0.7, 0.6, 0.6, ...
            0.6, 0.5, 0.6, ...
            0.7, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.6, ...
            0.5, ...
            0.2, 0.25];


%% --------------------------------------------
%% starts
%% --------------------------------------------
for seed = seeds
  %--------------------------------------------------------------------------
  % 2) Prepare modulated preamble (shifted by a half CP)
  %--------------------------------------------------------------------------
  fprintf('seed %d: Prepare modulated preamble\n', seed);

  rng(seed);

  Nofdm = Nfft + Ncp;
  Ncp_half = floor(Ncp/2);

  preamble_length = Nfft_useful * slice_count;
  [preamble, preamble_bytes] = swch_find_preamble(preamble_length);
  modPreamble = step(comm.BPSKModulator,preamble');
  modPreamble = reshape(modPreamble,Nfft_useful,slice_count);

  % each column is a slice
  modPreamble_padded = [modPreamble; zeros(Nfft-Nfft_useful,slice_count)];

  fprintf('  Nofdm = %d\n', Nofdm);
  fprintf('  Ncp_half = %d\n', Ncp_half);
  fprintf('  preamble_length = %d\n', preamble_length);
  fprintf('  preamble (%d x %d): ', size(preamble));
  fprintf('%d, ', preamble(1:20)');
  fprintf('...\n');
  fprintf('  modPreamble (%d x %d)\n', size(modPreamble));
  fprintf('  modPreamble_padded (%d x %d)\n', size(modPreamble_padded));
  
  modPreamble_ifft = zeros(Nfft,slice_count);
  for slice = 1:slice_count

    % fprintf('  slice %d: modPreamble_ifft (%d x %d), ifft modPreamble_padded (%d x %d)\n', slice, size(modPreamble_ifft(:,slice)), size(ifft(modPreamble_padded(:,slice),Nfft)));

    modPreamble_ifft(:,slice) = ifft(modPreamble_padded(:,slice),Nfft);
  end

  % add only 1/2 CP
  modPreamble_ifft = [modPreamble_ifft(Nfft-Ncp_half+1:Nfft,:); modPreamble_ifft(1:Nfft-Ncp_half,:)];

  % perform fft
  modPreamble_fft = zeros(Nfft, slice_count);
  for slice = 1:slice_count

    % fprintf('  slice %d: modPreamble_fft (%d x %d), fft modPreamble_ifft (%d x %d)\n', slice, size(modPreamble_fft(:,slice)), size(fft(modPreamble_ifft(:,slice),Nfft)));

    modPreamble_fft(:,slice) = fft(modPreamble_ifft(:,slice),Nfft);
  end

  % extract only the subcarriers of interest
  modPreamble_fft = modPreamble_fft(1:Nfft_useful,:);
  modPreamble_fft_norm = norm(modPreamble_fft,'fro');

  % NEW: perform digital up conversion to modPreamble_ifft
  for slice=1:slice_count
    base_idx = (slice-1)*Nofdm*slice_width + Ncp-Ncp_half;
    modPreamble_ifft(:,slice) = modPreamble_ifft(:,slice) .* ...
        exp(i*2*pi*f_min*(base_idx:(base_idx+Nfft-1)).'/Fs);
  end

  % NEW: convert modPreamble_ifft into sandwiches

  modPreamble_swch = zeros(1,Nofdm*(slice_count*slice_width+1));
  for slice = 1:slice_count
    base_idx = (slice-1)*slice_width*Nofdm + Ncp-Ncp_half;
    modPreamble_swch((base_idx+1):(base_idx+Nfft)) = modPreamble_ifft(:,slice);
  end

  subCarrier_swch = zeros(Nfft_useful,Nofdm);
  base_idx = Ncp-Ncp_half;
  for sc = 1:Nfft_useful
    % set one subcarrier to 1 and the rest to 0
    subCarrier_indicator = zeros(1,Nfft);
    subCarrier_indicator(sc) = 1;

    % perform ifft
    subCarrier_ifft = ifft(subCarrier_indicator,Nfft);
      
    % add only 1/2 CP
    subCarrier_ifft = [subCarrier_ifft(1,Nfft-Ncp_half+1:Nfft) subCarrier_ifft(1,1:Nfft-Ncp_half)];
      
    % up convert
    subCarrier_ifft = subCarrier_ifft .* exp(i*2*pi*f_min*(base_idx:(base_idx+Nfft-1))/Fs);

    % set the sandwich
    subCarrier_swch(sc,(base_idx+1):(base_idx+Nfft)) = subCarrier_ifft;
  end

  % continue;
  %% --------------------------------------------
  %% read ground truth
  %% --------------------------------------------
  fprintf('  read ground truth\n');
  gt_symbol = load([input_sent_dir 'sent_pkt' int2str(seed) '.symbol.txt']);
  gt_demod  = load([input_sent_dir 'sent_pkt' int2str(seed) '.demod.txt']);
  fprintf('    gt_symbol %d x %d\n', size(gt_symbol));
  fprintf('    gt_demod %d x %d\n', size(gt_demod));



  %% --------------------------------------------
  %% for each file
  %% --------------------------------------------
  for fi = [1:length(expnames)]
    expname = expnames{fi};
    filename = [input_rcv_dir char(expname) int2str(seed) '.wav'];

    if exist([output_rcv_dir char(expname) int2str(seed) '.h.txt'], 'file') == 2
      continue;
    end

    fprintf('- %s\n', filename);
    ber = 0;
    num_bits = 0;


    %--------------------------------------------------------------------------
    % 1) Read the .wav file
    %--------------------------------------------------------------------------
    fprintf('  read file\n');

    [wav_packet_orig, Fs, nbits] = wavread(filename);
    sample_count = length(wav_packet_orig);
    fprintf('    wav_packet_orig %d x %d\n', size(wav_packet_orig));

    figure(4)
    clf
    plot(wav_packet_orig)
    set(gca, 'XLim', [0 200000])


    %--------------------------------------------------------------------------
    % 3) Perform synchronization
    %--------------------------------------------------------------------------
    fprintf('  Perform synchronization\n')

    % size of entire sandwich (note that there is one empty symbol at the end)
    swch_samples = (slice_width * slice_count + 1) * Nofdm;

    sync_ofdm = zeros(1,Nofdm*slice_count);
    sync_ofdm_matrix = zeros(Nofdm,slice_count);
    sync_ofdm_no_cp = zeros(Nfft,slice_count);

    p_lst = [];
    p_max = -1;
    % start_offset = 1;
    start_offset = start_offsets{seed}(fi);
    threshold = thresholds{seed}(fi);
    fprintf('    start_offset = %d\n', start_offset);
    fprintf('    threshold    = %f\n', threshold);

    preamble_offset = start_offset - 1;

    step_size = swch_samples;
    for try_offset = start_offset:step_size:(sample_count - (swch_samples*2 - 1))
      
      max_offset = min(sample_count, try_offset + step_size + (swch_samples*2 - 1));
      try_p_cnt = max_offset - (try_offset + (swch_samples*2 - 1)) + 1;
      
      try_packets = ...
              wav_packet_orig((try_offset:(max_offset-swch_samples))) + ...
          i * wav_packet_orig((try_offset+swch_samples):max_offset);
      try_packets = reshape(try_packets,1,[]);

      modInnerProd = swch_prod(conj(modPreamble_swch), try_packets);
      modInnerProd = modInnerProd(1:try_p_cnt);

      scProd = zeros(1,max_offset-try_offset+1-swch_samples);
      for sc = 1:Nfft_useful
        scProd = scProd + swch_prod(conj(subCarrier_swch(sc,:)), try_packets).^2;
      end
      
      scNormSq = zeros(1,try_p_cnt);
      for slice = 1:slice_count
        base_idx = (slice-1)*Nofdm*slice_width;
        scNormSq = scNormSq + scProd(base_idx+1:base_idx+try_p_cnt);
      end
      
      try_p_lst = modInnerProd ./ (modPreamble_fft_norm * sqrt(scNormSq));

      [p_max,idx] = max(try_p_lst);
      p_lst = [p_lst try_p_lst];

      fprintf('    %d: p_max = %f, idx = %d\n', try_offset, p_max, idx);
  
      % if (p_max >= 0.4) 
      if (p_max >= threshold)
        preamble_offset = idx + try_offset - 1;
        break;
      end
    end

    % fd=fopen('p.txt','w');
    % fprintf(fd,'%f\n',p_lst);
    % fclose(fd);
    % fprintf('p_lst = %d x %d\n', size(p_lst));
    
    % figure(5)
    % clf
    % plot(p_lst)
    % set(gca, 'XLim', [0 200000])




    %% --------------------------------------------
    %% Get channel coefficient
    %% --------------------------------------------
    fprintf('  Get channel coefficient\n');
    for try_offset = preamble_offset
  
      %--------------------------------------------------------------------------
      % 4.1) Get the sync symbols
      %--------------------------------------------------------------------------
      for r = 1:slice_count
        offset1 = try_offset - 1 + (r-1)*Nofdm*slice_width;
        offset2 = offset1 + swch_samples;
        sync_ofdm((Nofdm*(r-1)+1):(Nofdm*r)) = ...
            wav_packet_orig((offset1+1):(offset1+Nofdm)) + ...
            i*wav_packet_orig((offset2+1):(offset2+Nofdm));
      end
      
      %--------------------------------------------------------------------------
      % 4.2) digital down convert (DDC)
      %--------------------------------------------------------------------------
      sync_ofdm_matrix = reshape(sync_ofdm,Nofdm,slice_count);
      for slice=1:slice_count
        base_idx = (slice-1)*Nofdm*slice_width;
        sync_ofdm_matrix(:,slice) = sync_ofdm_matrix(:,slice) .* ...
            exp(-i*2*pi*f_min*(base_idx:(base_idx+Nofdm-1)).'/Fs);
      end
      
      %--------------------------------------------------------------------------
      % 4.3) remove 1/2 CP 
      %--------------------------------------------------------------------------
      
      % Nfft rows x slice_count cols
      sync_ofdm_no_cp = sync_ofdm_matrix((Ncp-Ncp_half+1):(Nofdm-Ncp_half),:);

      %--------------------------------------------------------------------------
      % 4.4) apply FFT to obtain frequency-domain signals
      %--------------------------------------------------------------------------
      sync_fft = zeros(Nfft,slice_count);
      for slice=1:slice_count
        sync_fft(:,slice) = fft(sync_ofdm_no_cp(:,slice),Nfft).';
      end
      
      % extract only those subcarriers of interest
      sync_fft = sync_fft(1:Nfft_useful,:);
      sync_fft_norm = norm(sync_fft,'fro');
      
      %--------------------------------------------------------------------------
      % 4.5: compute channel coefficient
      %--------------------------------------------------------------------------
      preamble_offset = try_offset;
      Hfft = sync_fft ./ modPreamble;
      fprintf('    Hfft %d x %d\n', size(Hfft));

      % figure(6)
      % clf
      % lh1 = plot(abs(Hfft(:,1)));
      % set(lh1, 'Color', 'r');
      % set(lh1, 'LineWidth', 3);
      % set(lh1, 'marker', 'o');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
      % hold on;
      % lh2 = plot(abs(Hfft(:,2)));
      % set(lh2, 'Color', 'b');
      % set(lh2, 'LineWidth', 3);
      % set(lh2, 'marker', '*');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
      % hold on;
      % lh3 = plot(abs(Hfft(:,3)));
      % set(lh3, 'Color', 'g');
      % set(lh3, 'LineWidth', 3);
      % set(lh3, 'marker', 'x');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
      % hold on;
      % lh4 = plot(abs(Hfft(:,4)));
      % set(lh4, 'Color', 'k');
      % set(lh4, 'LineWidth', 3);
      % set(lh4, 'marker', '^');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h

      % figure(7)
      % clf
      % lh1 = plot(angle(Hfft(:,1)));
      % set(lh1, 'Color', 'r');
      % set(lh1, 'LineWidth', 3);
      % set(lh1, 'marker', 'o');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
      % hold on;
      % lh2 = plot(angle(Hfft(:,2)));
      % set(lh2, 'Color', 'b');
      % set(lh2, 'LineWidth', 3);
      % set(lh2, 'marker', '*');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
      % hold on;
      % lh3 = plot(angle(Hfft(:,3)));
      % set(lh3, 'Color', 'g');
      % set(lh3, 'LineWidth', 3);
      % set(lh3, 'marker', 'x');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h
      % hold on;
      % lh4 = plot(angle(Hfft(:,4)));
      % set(lh4, 'Color', 'k');
      % set(lh4, 'LineWidth', 3);
      % set(lh4, 'marker', '^');     %% marker: +|o|*|.|x|s|d|^|>|<|p|h

    end

    decodedData = [];

    if preamble_offset==-1
      fprintf('Unable to find preamble\n');
      return
    end


    %--------------------------------------------------------------------------
    % 4) Perform decoding
    %--------------------------------------------------------------------------
    fprintf('  Perform decoding\n');

    % prepare the complex numbers

    % (slice_width * slice_count + 1) * Nofdm;
    swch_raw = wav_packet_orig(preamble_offset:(preamble_offset+swch_samples*2-1));
    swch_raw = reshape(swch_raw, Nofdm, (slice_width*slice_count+1)*2);

    % do not include the last column, which is supposed to be an all-zero symbol
    swch_data = swch_raw(:,1:(slice_width*slice_count)) + i*swch_raw(:,(slice_width*slice_count+2):(slice_width*slice_count*2+1));

    decodedData = zeros(Nfft_useful/channels_per_bit, (slice_width - 1) * slice_count);

    % compute rand_bits
    rng(seed*2);
    rand_bits = randi([0 1], data_size, 1);

    data_equalized = zeros(Nfft_useful, slice_count*slice_width);
    data_demod     = zeros(Nfft_useful, slice_count*slice_width);

    for slice = 1:slice_count
      hfft = Hfft(:,slice);

      for count = 2:slice_width
        col_idx = (slice-1)*slice_width+count;
        base_tim = (col_idx-1)*Nofdm;

        %--------------------------------------------------------------------------
        % 4.1) down convert swch_data(:,col_idx) to base band
        %--------------------------------------------------------------------------
        swch_data_bb = swch_data(:,col_idx) .* ...
            exp(-i*2*pi*f_min*(base_tim:(base_tim+Nofdm-1)).'/Fs);

        %--------------------------------------------------------------------------
        % 4.2) remove 1/2 CP
        %--------------------------------------------------------------------------
        swch_data_no_cp = swch_data_bb((Ncp-Ncp_half+1):(Nofdm-Ncp_half));
        
        %--------------------------------------------------------------------------
        % 4.3) perform FFT
        %--------------------------------------------------------------------------
        swch_fft = fft(swch_data_no_cp,Nfft);
        swch_fft = swch_fft(1:Nfft_useful);
        
        %--------------------------------------------------------------------------
        % 4.4) channel compensation
        %--------------------------------------------------------------------------
        swch_equalized = swch_fft ./ hfft;
        data_equalized(:, col_idx) = swch_equalized;
        % this_gt_equalized = gt_symbol(:, col_idx);
        
        %--------------------------------------------------------------------------
        % 4.5) demodulation
        %--------------------------------------------------------------------------
        swch_demod = step(comm.BPSKDemodulator, swch_equalized);
        data_demod(:, col_idx) = swch_demod;
        % fprintf('    swch_demod %d x %d\n', size(swch_demod));
        this_gt_demod = gt_demod(:, col_idx);
        this_num_err = nnz(this_gt_demod - swch_demod);
        this_num_bits = length(swch_demod);
        ber = ber + this_num_err;
        num_bits = num_bits + this_num_bits;
        fprintf('    #errs = %d\n', this_num_err);
        fprintf('    #bits = %d\n', this_num_bits);

        %--------------------------------------------------------------------------
        % 4.6) FEC decode
        %--------------------------------------------------------------------------
        swch_dec = [];
        swch_enc = [];
        for k = 1:6:length(swch_demod)
          [sd,se] = swch_fec_decode(swch_demod(k:(k+5)));
          swch_dec(((k+1)/2):((k+5)/2)) = sd;
          swch_enc(k:(k+5)) = se;
        end
        
        use_rand_xor = 0;
        %--------------------------------------------------------------------------
        % 4.7) xor with rand_bits
        %--------------------------------------------------------------------------
        col_idx = (slice-1)*(slice_width-1)+(count-1);
        if (use_rand_xor == 1)
            rand_bits_index = (col_idx-1)*Nfft_useful/channels_per_bit;
            swch_dec_xor = xor(swch_dec', rand_bits((rand_bits_index+1):(rand_bits_index+Nfft_useful/channels_per_bit)));
        else
            swch_dec_xor = swch_dec;
        end
        
        decodedData(:, col_idx) = swch_dec_xor;
        
        %--------------------------------------------------------------------------
        % 4.8) remodulate 
        %--------------------------------------------------------------------------
        swch_mod = step(comm.BPSKModulator, swch_enc.');

        %--------------------------------------------------------------------------
        % 4.9) obtain new estimate of channel coefficients 
        %--------------------------------------------------------------------------    
        swch_hfft = swch_fft ./ swch_mod;
        
        %--------------------------------------------------------------------------
        % 4.10) update hfft
        %--------------------------------------------------------------------------    
        alpha = 0.5;
        hfft = swch_hfft * alpha + hfft * (1-alpha);
        
      end
    end

    decodedData = reshape(decodedData,1,[]);
    dd = size(decodedData);
    [ts, id, len] = swch_decode(decodedData);
    this_ber = ber / num_bits;
    BERS(fi, seed) = this_ber;
    fprintf('    decodedData size = %d x %d\n', dd);
    fprintf('    ts               = %d\n', ts);
    fprintf('    id               = %s\n', id);
    fprintf('    len              = %d\n', len);
    fprintf('    ber              = %d / %d = %1.2g\n', ber, num_bits, this_ber);

    % result = input('');
    % dlmwrite([output_rcv_dir char(expname) int2str(seed) '.h.txt'], Hfft, 'delimiter', '\t');
    Hfft_real = real(Hfft);
    Hfft_imag = imag(Hfft);
    dlmwrite([output_rcv_dir char(expname) int2str(seed) '.h.txt'], [Hfft_real Hfft_imag], 'delimiter', '\t');
    % dlmwrite([output_rcv_dir char(expname) int2str(seed) '.preamble.txt'], sync_fft, 'delimiter', '\t');
    sync_fft_real = real(sync_fft);
    sync_fft_imag = imag(sync_fft);
    dlmwrite([output_rcv_dir char(expname) int2str(seed) '.preamble.txt'], [sync_fft_real sync_fft_imag], 'delimiter', '\t');
    % dlmwrite([output_rcv_dir char(expname) int2str(seed) '.symbol.txt'], data_equalized, 'delimiter', '\t');
    data_equalized_real = real(data_equalized);
    data_equalized_imag = imag(data_equalized);
    dlmwrite([output_rcv_dir char(expname) int2str(seed) '.symbol.txt'], [data_equalized_real data_equalized_imag], 'delimiter', '\t');
    %%
    dlmwrite([output_rcv_dir char(expname) int2str(seed) '.demod.txt'], data_demod, 'delimiter', '\t');

    % return

  end
end

return;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% FEC related functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [decodedData,encodedData] = swch_fec_decode(input)

global xxx_fec_decodeTable
global xxx_fec_encodeTable
  
assert(length(input) == 6);
  
index = 1 + input(1) + input(2)*2 + input(3)*4 + input(4)*8 + input(5)*16 + input(6)*32;
  
decodedData = xxx_fec_decodeTable(index,:);
encodedData = xxx_fec_encodeTable(index,:);


function [decodeTable,encodeTable] = fec_init_table()

decodeTable = zeros(64,3);
encodeTable = zeros(64,6);

for obs = 0:63
  obits = bitget(obs,1:6);
    
  ndif = inf;
  decoded_bits = [];
  encoded_bits = [];
  for input = 0:7
    ibits = bitget(input,1:3);
    x = ibits(1);
    y = ibits(2);
    z = ibits(3);
    p = xor(x,xor(y,z));
    q = xor(x,y);
    r = xor(x,z);
    
    % primary   criteria: minimizing difference on all 6 bits
    % secondary criteria: minimizing difference on observed bits
    dif = sum(xor([x y z p q r], obits)) + ...
          sum(xor([x y z], obits(1:3))) * 0.1;
    if (dif < ndif) 
      ndif = dif;
      decoded_bits = ibits;
      encoded_bits = [x y z p q r];
    end
  end
    
  decodeTable(obs+1,:) = decoded_bits;
  encodeTable(obs+1,:) = encoded_bits;
end

