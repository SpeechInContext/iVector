function extract_and_normalize_all_mfccs(inFolder, numfeats)
    %Set file path and collect background files
    [files_and_speakers, ~] = collect_files(inFolder);
    num_utterances = size(files_and_speakers, 1);
    all_feats = [];
    tic

    parfor idx = 1:num_utterances
            [x, fs] = audioread(files_and_speakers(idx, 1));
            x_feats = get_mfccs_deltas(x, fs, numfeats);
            all_feats = [all_feats, x_feats];
    end
    fprintf('Feature extraction from training set complete (%0.0f seconds).\n',toc)
    normMean = mean(all_feats,2,'omitnan');
    normSTD = std(all_feats,[],2,'omitnan');

    outFolder = 'C:/Users/mFry2/Desktop/SpeeCon/Data/PTDB-TUG/SPEECH DATA/FEMALE/MFCC/';
    parfor idx = 1:num_utterances
        [x, fs] = audioread(files_and_speakers(idx, 1));
        [~,name,~] = fileparts(files_and_speakers(idx, 1));
        x_feats = get_mfccs_deltas(x, fs);
        x_feats = (x_feats-normMean)./normSTD; %normalize
        x_feats = x_feats - mean(x_feats,'all'); %for channel noise
        outFile = strcat(outFolder, name, '.mfcc');
        fileId = fopen(outFile, 'w');
        fwrite(fileId, x_feats);
        fclose(fileId);
    end
end
    
%%% ---------- Helper functions --------------------------------------- %%%
function [files, unique_speakers] = collect_files(folder)
    % Returns the filelist give a folder 
    temp_files = dir(folder);
    files = strings(size(temp_files,1), 2);
    for idx = 1:size(temp_files,1)
        files(idx, 1) = fullfile(temp_files(idx).folder, temp_files(idx).name);
    end
    %How to identify unique speakers
    files(:,2) = extractBetween(files(:,1),'mic_','_');
    unique_speakers = unique(files(:,2));
end

function mfcc_feats = get_mfccs_deltas(sig, fs, numfeats)
    % Returns the mfccs, deltas and delta-deltas of a signal with preset
    % parameters
    [mfcc_x_feats, ~, ~] = melfcc(sig, fs, 'wintime', 0.025, 'hoptime', 0.010, 'numcep', numfeats);
    ds = deltas(mfcc_x_feats, 9);
    dds = deltas(ds, 9);
    mfcc_feats = [mfcc_x_feats; ds; dds];
end