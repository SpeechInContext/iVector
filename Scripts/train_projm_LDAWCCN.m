function projMatrix = train_projm_LDAWCCN(mfccFolder, ubm, tvm, ivector_dim)
    [files_and_speakers, unique_speakers] = collect_files(mfccFolder);
    
    % i-vector extraction
    numSpeakers = size(unique_speakers, 1);
    numFeatures = ubm.numFeatures;
    ivectorPerSpeaker = cell(numSpeakers,1);
    tic
    disp('Extracting i-vectors')
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
        ivectorPerSpeaker{speakerIdx} = ivectorPerFile;
    end
    fprintf('I-vectors extracted from training set (%0.0f seconds).\n',toc)

    % Use LDA and WCCN to generate projection matrix
    w = ivectorPerSpeaker;
    utterancePerSpeaker = cellfun(@(x)size(x,2),w);

    ivectorsTrain = cat(2,w{:});
    projMatrix = eye(size(w{1},1));
    performLDA = true;
    disp('Performing LDA')
    if performLDA
        tic
        Sw = zeros(size(projMatrix,1));
        Sb = zeros(size(projMatrix,1));
        wbar = mean(cat(2,w{:}),2);
        for ii = 1:numel(w)
            ws = w{ii};
            wsbar = mean(ws,2);
            Sb = Sb + (wsbar - wbar)*(wsbar - wbar)';
            Sw = Sw + cov(ws',1);
        end

        [A,~] = eigs(Sb,Sw,ivector_dim); % ivector_dim is number of eigenvectors for i-vector projection
        A = (A./vecnorm(A))';

        ivectorsTrain = A * ivectorsTrain;

        w = mat2cell(ivectorsTrain,size(ivectorsTrain,1),utterancePerSpeaker);

        projMatrix = A * projMatrix;

        fprintf('LDA projection matrix calculated (%0.2f seconds).\n',toc)
    end

    performWCCN = true;
    if performWCCN
        tic
        alpha = 0.9;

        W = zeros(size(projMatrix,1));
        for ii = 1:numel(w)
            W = W + cov(w{ii}',1);
        end
        W = W/numel(w);

        W = (1 - alpha)*W + alpha*eye(size(W,1));

        B = chol(pinv(W),'lower');

        projMatrix = B * projMatrix;

        fprintf('WCCN projection matrix calculated (%0.4f seconds).\n',toc)
    end 
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