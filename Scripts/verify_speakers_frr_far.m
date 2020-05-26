function [thresholdsToTest, FRR, FAR] = verify_speakers_frr_far(mfccFolder, speakers_ivectors, numEnrolUtt, ubm, tvm, projMatrix)
    [verify_files_and_speakers, ~] = collect_files(mfccFolder);
    numSpeakers = size(speakers_ivectors,1);
    numFeatures = ubm.numFeatures;
    thresholdsToTest = -1:0.001:1;
    cssFRR = cell(numSpeakers,1);
    tic
    for speakerIdx = 1:numSpeakers
        idxs = verify_files_and_speakers(:,2)==speakers_ivectors{speakerIdx, 1};
        speaker_files = verify_files_and_speakers(idxs, 1);
        speaker_files(1:numEnrolUtt, :) = [];
        numFiles = size(speaker_files,1);
        ivectorToTest = speakers_ivectors{speakerIdx, 2};
        css = zeros(numFiles,1);
        parfor fileIdx = 1:numFiles
            fileId = fopen(speaker_files(fileIdx, 1));
            x_feats = fread(fileId);
            x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
            fclose(fileId);
            
            %i-vector Extraction
            ivector = get_i_vector(x_feats, ubm, tvm);
            
            % Intersession Compensation
            ivector = projMatrix*ivector;
            
            % Cosine Similarity Score
            css(fileIdx) = dot(ivectorToTest,ivector)/(norm(ivector)*norm(ivectorToTest));
        end
        cssFRR{speakerIdx} = css;
    end
    cssFRR = cat(1,cssFRR{:});
    FRR = mean(cssFRR<thresholdsToTest);
      
    cssFAR = cell(numSpeakers,1);
    for speakerIdx = 1:numSpeakers
        idxs = verify_files_and_speakers(:,2)~=speakers_ivectors{speakerIdx, 1};
        speaker_files = verify_files_and_speakers(idxs, 1);
        numFiles = size(speaker_files,1);

        ivectorToTest = speakers_ivectors{speakerIdx, 2};
        css = zeros(numFiles,1);
        parfor fileIdx = 1:numFiles
            fileId = fopen(speaker_files(fileIdx, 1));
            x_feats = fread(fileId);
            x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
            fclose(fileId);
            
            %i-vector Extraction
            ivector = get_i_vector(x_feats, ubm, tvm);
            
            % Intersession Compensation
            ivector = projMatrix*ivector;
            
            % Cosine Similarity Score
            css(fileIdx) = dot(ivectorToTest,ivector)/(norm(ivector)*norm(ivectorToTest));
        end
        cssFAR{speakerIdx} = css;
    end
    cssFAR = cat(1,cssFAR{:});    
    FAR = mean(cssFAR>thresholdsToTest);
    fprintf('FRR and FAR calculated (%0.0f seconds).\n',toc)
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