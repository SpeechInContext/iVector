function mfcc_files = extract_mfccs(inFolder, soundfile_ext, outFolder, normalizeMFCCs)
%% Extract MFCCs from input sound files using get_mfcc_delta (function below)
    if exist(outFolder, 'dir')
        disp(['Appears MFCCs already extracted to: ' outFolder]);
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
    
    % Collect all MFCCs to get normalization parameters
    if normalizeMFCCs
        all_feats = [];
        tic
        parfor fileIdx = 1:size(files,1)
            filename = files(fileIdx).name;
            folder = files(fileIdx).folder;
            file = strcat(folder, '/', filename);
            [x, fs] = audioread(file);
            x_feats = get_mfccs_deltas(x, fs);
            all_feats = [all_feats, x_feats];
        end
        fprintf('Feature extraction from training set complete (%0.0f seconds).\n',toc)
        normMean = mean(all_feats,2,'omitnan');
        normSTD = std(all_feats,[],2,'omitnan');
    end
    
    % Calculate all MFCCs, normalize if needed
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
        x_feats = get_mfccs_deltas(x,fs);
        if normalizeMFCCs
            x_feats = (x_feats-normMean)./normSTD; %normalize
            %x_feats = x_feats - mean(x_feats,'all'); %for channel noise
        end   
        outFile = strcat(mfccFolder, '\', filename);
        outFile = strrep(outFile, '.WAV', '.mfcc');
        outFile = strrep(outFile, '.wav', '.mfcc');
        fileId = fopen(outFile, 'w');
        fwrite(fileId, x_feats);
        fclose(fileId);
        mfcc_files(fileIdx) = {outFile};
    end
end

function mfcc_feats = get_mfccs_deltas(sig, fs)
    % Returns the mfccs, deltas and delta-deltas of a signal with preset
    % parameters
    window_size = 0.025;
    window_shift = 0.010;
    number_coeff = 20;
    [mfcc_x_feats, ~, ~] = melfcc(sig, fs, 'wintime', window_size, 'hoptime', window_shift, 'numcep', number_coeff);
    ds = deltas(mfcc_x_feats, 9);
    dds = deltas(ds, 9);
    mfcc_feats = [mfcc_x_feats; ds; dds];
end

