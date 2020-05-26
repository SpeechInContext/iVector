function speaker_ivectors = extract_speakers_ivectors(mfccFolder, ubm, tvm, projMatrix)
    [files_and_speakers, unique_speakers] = collect_files(mfccFolder);
    numSpeakers = size(unique_speakers, 1);
    numFeatures = ubm.numFeatures;
    speaker_ivectors = cell(numSpeakers,1);
    tic
    % Calcuate i-vector for speaker
    for speakerIdx = 1:numSpeakers
        idxs = files_and_speakers(:,2)==unique_speakers(speakerIdx);
        speaker_files = files_and_speakers(idxs, 1);
        numFiles = size(speaker_files,1);
        ivectorMat = zeros(size(projMatrix,1),numFiles);
        parfor fileIdx = 1:numFiles
            fileId = fopen(speaker_files(fileIdx, 1));
            x_feats = fread(fileId);
            x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
            fclose(fileId);

            %i-vector Extraction
            ivector = get_i_vector(x_feats, ubm, tvm)

            % Intersession Compensation
            ivector = projMatrix*ivector;

            ivectorMat(:,fileIdx) = ivector;
        end
        % i-vector model
        speaker_ivectors{speakerIdx} = ivectorMat;
    end
    speaker_ivectors = [cellstr(unique_speakers) speaker_ivectors];
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