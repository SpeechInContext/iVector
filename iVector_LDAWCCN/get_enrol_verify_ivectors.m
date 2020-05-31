function [speaker_ivectors, verify_ivectors] = get_enrol_verify_ivectors(mfccFolder, numEnrolUtt, ubm, tvm)
    [files_and_speakers, unique_speakers] = collect_files(mfccFolder);
    numSpeakers = size(unique_speakers, 1);
    numFeatures = ubm.numFeatures;
    speaker_ivectors = cell(numSpeakers,1);
    tic
    % Calcuate i-vector for speaker
    for speakerIdx = 1:numSpeakers
        idxs = files_and_speakers(:,2)==unique_speakers(speakerIdx);
        speaker_files = files_and_speakers(idxs, 1);
        numFiles = numEnrolUtt;
        ivectorMat = zeros(tvm.Tdim,numFiles);
        parfor fileIdx = 1:numFiles
            fileId = fopen(speaker_files(fileIdx, 1));
            x_feats = fread(fileId);
            x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
            fclose(fileId);

            %i-vector Extraction
            ivector = get_i_vector(x_feats, ubm, tvm);

            ivectorMat(:,fileIdx) = ivector;
        end
        % i-vector model
        speaker_ivectors{speakerIdx} = mean(ivectorMat,2);
    end
    speaker_ivectors = [cellstr(unique_speakers) speaker_ivectors];
    
    verify_ivectors = [];
    for speakerIdx = 1:numSpeakers
        idxs = files_and_speakers(:,2)==unique_speakers(speakerIdx);
        speaker_files = files_and_speakers(idxs, 1);
        speaker_files(1:numEnrolUtt, :) = [];
        numFiles = size(speaker_files,1);
        ivectorMat = zeros(tvm.Tdim,numFiles);
        parfor fileIdx = 1:numFiles
            fileId = fopen(speaker_files(fileIdx, 1));
            x_feats = fread(fileId);
            x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
            fclose(fileId);

            %i-vector Extraction
            ivector = get_i_vector(x_feats, ubm, tvm)

            ivectorMat(:,fileIdx) = ivector;
        end
        % i-vector model
        verify_ivectors = [verify_ivectors ivectorMat]; %#ok
    end    
    fprintf('Speakers enrolled (%0.0f seconds).\n',toc)
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
    files(:,2) = extractBetween(files(:,1),'MFCC\','_');
    unique_speakers = unique(files(:,2));
end