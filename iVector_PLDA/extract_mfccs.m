function mfcc_files = extract_mfccs(inFolder, soundfile_ext, outFolder, normalizeMFCCs)
%% Extract MFCCs from input sound files 
% Arguments:
%   inFolder:       The folder where the sound files are stored (may contain subdirectories)
%   soundfile_ext:  The extension of the sound files (e.g. .wav, .WAV, .flac, etc.)
%   outFolder:      The folder where the mfcc files are written to
%   normalizeMFCCs: Binary flag to normalize MFCCs before writing to file

    if exist(outFolder, 'dir')
        disp(['Appears MFCCs already extracted to: ' outFolder]);
        disp('Collecting .mfcc files from the directory');
        files = dir([outFolder '\**\*.mfcc']);
        mfcc_files = cell(size(files,1),1);
        for fileIdx = 1:size(files,1)
            mfcc_files(fileIdx) = {strcat(files(fileIdx).folder, '\', files(fileIdx).name)};
        end
        return
    else
        mkdir(outFolder);
    end
    
    % Get files
    files = dir([inFolder '\**\*' soundfile_ext]);
    
    % For each file, calculate MFCCs and normalize if selected
    mfcc_files = cell(size(files,1),1);
    parfor fileIdx = 1:size(files,1)
        filename = files(fileIdx).name;
        folder = files(fileIdx).folder;
        mfccFolder = strrep(folder, inFolder, outFolder);
        if ~exist(mfccFolder, 'dir')
            mkdir(mfccFolder);
        end
        file = strcat(folder, '\', filename);
        [x, fs] = audioread(file);
        [vad,~] = v_vadsohn(x,fs,'a');
        x(size(vad,1)+1:end) = [];
        x(~vad) = [];
        x_feats = get_mfccs_deltas(x,fs);
        if normalizeMFCCs
            x_feats = fea_warping(x_feats);
        end   
        outFile = strcat(mfccFolder, '\', filename);
        outFile = strrep(outFile, soundfile_ext, '.mfcc');
        fileId = fopen(outFile, 'w');
        fwrite(fileId, x_feats);
        fclose(fileId);
        mfcc_files(fileIdx) = {outFile};
    end
end

function mfcc_feats = get_mfccs_deltas(sig, fs)
%% Returns the mfccs, deltas and delta-deltas of a signal using preset parameters
%  via the rastamat toolbox
%   window_size:    window to calculate FFT on, time in seconds
%   window_shift:   window overlap, time in seconds
%   number_coeff:   number of MFCCoefficients
    window_size = 0.025;
    window_shift = 0.010;
    number_coeff = 20;
    [mfcc_x_feats, ~, ~] = melfcc(sig, fs, 'wintime', window_size, 'hoptime', window_shift, 'numcep', number_coeff);
    ds = deltas(mfcc_x_feats, 9);
    dds = deltas(ds, 9);
    mfcc_feats = [mfcc_x_feats; ds; dds];
end

