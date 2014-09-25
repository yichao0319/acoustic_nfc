%% =====================================
%%
%% e.g.
%%   decode_one_file(1, 'rcv_packet.exp1.dist10.s1', '../data/rcv_pkts/exp0718/', '../processed_data/task_decode/rcv_pkts/exp0718/', -1, -1)
%%   decode_one_file(1, 'multiple.v0.exp2.s1', '../data/rcv_pkts/exp0918/', '../processed_data/task_decode_multi/rcv_pkts/exp0918/', -1, -1)
%% =====================================

function [BERs] = decode_one_file(seed, filename, input_rcv_dir, output_rcv_dir, start_offset, threshold)

  addpath('../util/matlab/acoustic_nfc/');

  global xxx_fec_decodeTable
  global xxx_fec_encodeTable

  DEBUG0 = 0;
  DEBUG1 = 1;
  DEBUG2 = 1;  %% progress
  DEBUG3 = 1;  %% basic info
  DEBUG4 = 1;  %% process info
  DEBUG5 = 0;  %% final output


  % input_rcv_dir  = '../data/rcv_pkts/exp0523/';
  input_sent_dir = '../processed_data/task_decode/sent_pkts/';
  % output_rcv_dir = '../processed_data/task_decode/rcv_pkts/exp0523/';


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


  if DEBUG3,
    fprintf('Fs               = %d\n', Fs);
    fprintf('[min:max]        = [%d:%d]\n', f_min, f_max);
    fprintf('Nfft             = %d\n', Nfft);
    fprintf('Nfft_useful      = %d\n', Nfft_useful);
    fprintf('Ncp              = %d\n', Ncp);
    fprintf('slice_count      = %d\n', slice_count);
    fprintf('slice_width      = %d\n', slice_width);
    fprintf('channels_per_bit = %d\n', channels_per_bit);
    fprintf('data_size        = %d\n', data_size);
  end

  % initialize FEC encoding / decoding 
  [xxx_fec_decodeTable,xxx_fec_encodeTable] = fec_init_table();


  %% -----------------------------------
  %% Start
  %% -----------------------------------

  rng(seed);

  Nofdm = Nfft + Ncp;
  Ncp_half = floor(Ncp/2);

  preamble_length = Nfft_useful * slice_count;
  [preamble, preamble_bytes] = swch_find_preamble(preamble_length);
  modPreamble = step(comm.BPSKModulator,preamble');
  modPreamble = reshape(modPreamble,Nfft_useful,slice_count);

  % each column is a slice
  modPreamble_padded = [modPreamble; zeros(Nfft-Nfft_useful,slice_count)];

  if DEBUG3,
    fprintf('  Nofdm = %d\n', Nofdm);
    fprintf('  Ncp_half = %d\n', Ncp_half);
    fprintf('  preamble_length = %d\n', preamble_length);
    fprintf('  preamble (%d x %d): ', size(preamble));
    fprintf('%d, ', preamble(1:20)');
    fprintf('...\n');
    fprintf('  modPreamble (%d x %d)\n', size(modPreamble));
    fprintf('  modPreamble_padded (%d x %d)\n', size(modPreamble_padded));
  end

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
  if DEBUG2, fprintf('Read ground truth\n'); end

  gt_symbol = load([input_sent_dir 'sent_pkt' int2str(seed) '.symbol.txt']);
  gt_demod  = load([input_sent_dir 'sent_pkt' int2str(seed) '.demod.txt']);
  if DEBUG4, fprintf('    gt_symbol %d x %d\n', size(gt_symbol)); end
  if DEBUG4, fprintf('    gt_demod %d x %d\n', size(gt_demod)); end



  %% --------------------------------------------
  %% read received file
  %% --------------------------------------------
  if DEBUG2, fprintf('Read received file\n'); end

  file_path_name = [input_rcv_dir filename '.wav'];

  % if exist([output_rcv_dir filename '.h.txt'], 'file') == 2
  %   return;
  % end

  if DEBUG3, fprintf('- %s\n', file_path_name); end
  ber = 0;
  num_bits = 0;


  %--------------------------------------------------------------------------
  % 1) Read the .wav file
  %--------------------------------------------------------------------------
  if DEBUG2, fprintf('Read file\n'); end

  [wav_packet_orig, Fs, nbits] = wavread(file_path_name);
  sample_count = length(wav_packet_orig);
  if DEBUG4, 
    fprintf('  wav_packet_orig %d x %d (Fs=%d, nbits=%d)\n', size(wav_packet_orig), Fs, nbits); 
  end


  %--------------------------------------------------------------------------
  % 3) Perform synchronization
  %--------------------------------------------------------------------------
  if DEBUG2, fprintf('Perform synchronization\n'); end

  % size of entire sandwich (note that there is one empty symbol at the end)
  swch_samples = (slice_width * slice_count + 1) * Nofdm;

  preamble_offset = sync_pkt(slice_width, slice_count, Nofdm, Nfft, Nfft_useful, subCarrier_swch, modPreamble_fft_norm, start_offset, threshold, modPreamble_swch, wav_packet_orig);
  if DEBUG4, 
    fprintf('  file length=%d\n', sample_count);
    fprintf('  preamble_offset (%d)=', length(preamble_offset)); 
    fprintf('%d,', preamble_offset); fprintf('\n');
    fprintf('  pkt length=%d\n', swch_samples*2);
  end

  if length(preamble_offset) < 1
    fprintf('Unable to find preamble\n\n');
    return
  end


  %% --------------------------------------------
  %% For each packet
  %% --------------------------------------------
  num_pkts = length(preamble_offset);
  BERs = zeros(1, num_pkts);
  CSIs = zeros(slice_count * num_pkts, Nfft_useful);
  SNRs = zeros(slice_count * num_pkts, Nfft_useful);
  p_idx = 0;

  for try_offset = preamble_offset
    p_idx = p_idx + 1;
    %% --------------------------------------------
    %% Get channel coefficient
    %% --------------------------------------------
    if DEBUG2, fprintf('Get channel coefficient\n'); end

    [Hfft, sync_fft] = get_H(wav_packet_orig, try_offset, ...
                 f_min, Fs, Nfft, Nfft_useful, ...
                 Nofdm, slice_width, slice_count, ...
                 Ncp, Ncp_half, ...
                 modPreamble);
    if DEBUG4, fprintf('    Hfft %d x %d\n', size(Hfft)); end
    CSIs( (p_idx-1)*slice_count+1:p_idx*slice_count, :) = Hfft';
    snr = get_snr(modPreamble, sync_fft);
    SNRs( (p_idx-1)*slice_count+1:p_idx*slice_count, :) = snr';

    %--------------------------------------------------------------------------
    % 4) Perform decoding
    %--------------------------------------------------------------------------
    if DEBUG2, fprintf('Perform decoding\n'); end

    decodedData = [];

    % prepare the complex numbers

    % (slice_width * slice_count + 1) * Nofdm;
    if DEBUG4, fprintf('  raw data = %d ~ %d\n', try_offset, (try_offset+swch_samples*2-1)); end
    swch_raw = wav_packet_orig(try_offset:(try_offset+swch_samples*2-1));
    swch_raw = reshape(swch_raw, Nofdm, (slice_width*slice_count+1)*2);

    % do not include the last column, which is supposed to be an all-zero symbol
    swch_data = swch_raw(:,1:(slice_width*slice_count)) + i*swch_raw(:,(slice_width*slice_count+2):(slice_width*slice_count*2+1));

    % swch_data(1:5, 1:5)

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

        % if slice == 1 & count == 2
        %   swch_data_no_cp(1:5)'
        % end
        
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

        % if slice == 1 & count == 2
        %   swch_equalized'
        % end
        
        %--------------------------------------------------------------------------
        % 4.5) demodulation
        %--------------------------------------------------------------------------
        swch_demod = step(comm.BPSKDemodulator, swch_equalized);
        
        % if slice == 1 & count == 2
        %   swch_demod'
        % end

        data_demod(:, col_idx) = swch_demod;
        % fprintf('    swch_demod %d x %d\n', size(swch_demod));
        this_gt_demod = gt_demod(:, col_idx);
        this_num_err = nnz(this_gt_demod - swch_demod);
        this_num_bits = length(swch_demod);
        ber = ber + this_num_err;
        num_bits = num_bits + this_num_bits;
        % fprintf('    #errs = %d, #bits = %d\n', this_num_err, this_num_bits);
        

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
    % BERS(fi, seed) = this_ber;
    BER = this_ber;
    fprintf('  BER=%f\n', BER);
    BERs(1, p_idx) = BER;

    if DEBUG5, 
      fprintf('    decodedData size = %d x %d\n', dd);
      fprintf('    ts               = %d\n', ts);
      fprintf('    id               = %s\n', id);
      fprintf('    len              = %d\n', len);
      fprintf('    ber              = %d / %d = %1.2g\n', ber, num_bits, this_ber);
    end

    sync_fft_real = real(sync_fft);
    sync_fft_imag = imag(sync_fft);
    % dlmwrite([output_rcv_dir filename '.preamble.txt'], [sync_fft_real sync_fft_imag], 'delimiter', '\t');
    data_equalized_real = real(data_equalized);
    data_equalized_imag = imag(data_equalized);
    % dlmwrite([output_rcv_dir filename '.symbol.txt'], [data_equalized_real data_equalized_imag], 'delimiter', '\t');
    %%
    % dlmwrite([output_rcv_dir filename '.demod.txt'], data_demod, 'delimiter', '\t');
  end

  CSIs_real = real(CSIs);
  CSIs_imag = imag(CSIs);
  dlmwrite([output_rcv_dir filename '.h.txt'], [CSIs_real CSIs_imag], 'delimiter', '\t');

  fh = figure(1);
  clf;
  plot(SNRs(:, 1:3));
  print(fh, '-dpng', './tmp/snr.png');

  return;
end


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
end

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
end


%% weighted_freq: weighted center
function [center_freq, avg_rss] = weighted_freq(freq, rss)
  avg_rss = mean(rss);
  center_freq = sum(freq .* (rss/sum(rss)));
end