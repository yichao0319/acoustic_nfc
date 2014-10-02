%% gen_pure_tone
function gen_pure_tone()
    
    output_dir = './tmp/';
    freqs = [440, 17000, 17500, 18000, 18500, 19000, 19500, 20000];
    t = 1*60;
    amp = 1;
    % t = 1*60;
    Fs = 44100;

    signals = zeros(Fs*t, length(freqs));

    for fi = 1:length(freqs)
        f = freqs(fi);
        ts = [linspace(0, t, Fs*t)];
        signals(:, fi) = sin(2*pi*f*ts)';
    end


    %% freq L[17k], R[18k]
    idxl = find(freqs == 17000);
    idxr = find(freqs == 18000);
    out = amp*signals(:, [idxl, idxr]);
    audiowrite([output_dir '17k.18k.300s.wav'], out, Fs);

    %% freq L[19k], R[18k]
    idxl = find(freqs == 19000);
    idxr = find(freqs == 18000);
    out = amp*signals(:, [idxl, idxr]);
    audiowrite([output_dir '19k.18k.300s.wav'], out, Fs);

    %% freq L[17k, 19k], R[18k, 20k]
    idxl1 = find(freqs == 17000);
    idxl2 = find(freqs == 19000);
    idxr1 = find(freqs == 18000);
    idxr2 = find(freqs == 20000);
    out = amp*(signals(:, [idxl1, idxr1]) + signals(:, [idxl2, idxr2])) / 2;
    audiowrite([output_dir '17k19k.18k20k.300s.wav'], out, Fs);

    
    % idxl = find(freqs == 440);
    % idxr = find(freqs == 17000);
    % out = amp*signals(:, [idxl, idxr]);
    % audiowrite([output_dir 'tmp.wav'], out, Fs);


