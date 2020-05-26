function plda = train_projm_PLDA(mfccFolder, ubm, tvm, ivector_dim)
    [files_and_speakers, unique_speakers] = collect_files(mfccFolder);
    
    % i-vector extraction
    numSpeakers = size(unique_speakers, 1);
    numFeatures = ubm.numFeatures;
    tic
    disp('Extracting i-vectors')
    train_data = [];
    train_labels = [];
    for speakerIdx = 1:numSpeakers
        idxs = files_and_speakers(:,2)==unique_speakers(speakerIdx);
        speaker_files = files_and_speakers(idxs, 1);
        ivectorPerFile = zeros(tvm.Tdim,size(speaker_files, 1));
        parfor fileIdx = 1:size(speaker_files,1)
            fileId = fopen(speaker_files(fileIdx, 1));
            x_feats = fread(fileId);
            x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
            fclose(fileId);

            % get i-vector
            ivectorPerFile(:,fileIdx) = get_i_vector(x_feats, ubm, tvm);
        end
        train_data = [train_data ivectorPerFile]; %#ok
        train_labels = [train_labels repmat(speakerIdx, 1, size(ivectorPerFile,2))]; %#ok
    end
    fprintf('I-vectors extracted from training set (%0.0f seconds).\n',toc)
    plda = gplda_em(train_data, train_labels, ivector_dim, 5);
end

%%% ----------- Helper functions -------------------------------------- %%%
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