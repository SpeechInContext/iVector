function [ubm, tvm] = train_ubm_tvm(mfccFolder, numComponents, numFeatures, tvm_dim)
    [files_and_speakers, ~] = collect_files(mfccFolder);
    
    %Initialize UBM
    alpha = ones(1,numComponents)/numComponents;
    mu = randn(numFeatures,numComponents);
    vari = rand(numFeatures,numComponents) + eps;
    ubm = struct('ComponentProportion',alpha,'mu',mu,'sigma',vari);
    ubm.numFeatures = numFeatures;
    
    % Train UBM with EM algorithm
    disp('Training UBM.')   
    maxIter = 1;
    num_utterances = size(files_and_speakers, 1);
    for iter = 1:maxIter
        tic
        % EXPECTATION
        N = zeros(1,numComponents);
        F = zeros(numFeatures,numComponents);
        S = zeros(numFeatures,numComponents);
        L = 0;
        parfor idx = 1:num_utterances
            fileId = fopen(files_and_speakers(idx, 1));
            x_feats = fread(fileId);
            x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
            fclose(fileId);

            % Compute a posteriori log-liklihood
            logLikelihood = helperGMMLogLikelihood(x_feats,ubm);

            % Compute a posteriori normalized probability
            amax = max(logLikelihood,[],1);
            logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
            gamma = exp(logLikelihood - logLikelihoodSum)';

            % Compute Baum-Welch statistics
            n = sum(gamma,1);
            f = x_feats * gamma;
            s = (x_feats.*x_feats) * gamma;

            % Update the sufficient statistics over utterances
            N = N + n;
            F = F + f;
            S = S + s;

            % Update the log-likelihood
            L = L + sum(logLikelihoodSum);
        end

        % Print current log-likelihood
        fprintf('Training UBM: %d/%d complete (%0.0f seconds), Log-likelihood = %0.0f\n',iter,maxIter,toc,L)

        % MAXIMIZATION
        N = max(N,eps);
        ubm.ComponentProportion = max(N/sum(N),eps);
        ubm.ComponentProportion = ubm.ComponentProportion/sum(ubm.ComponentProportion);
        ubm.mu = F./N;
        ubm.sigma = max(S./N - ubm.mu.^2,eps);
    end
    
    disp('Calculating Baum-Welch statistics to train TVM.')
    % Calculate Baum_Welch statistics for training the 
    % Total Variability Matrix (later)
    tic
    Nc = cell(1,num_utterances);
    Fc = cell(1,num_utterances);
    parfor idx = 1:num_utterances
        fileId = fopen(files_and_speakers(idx, 1));
        x_feats = fread(fileId);
        x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
        fclose(fileId);

        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(x_feats,ubm);

        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';

        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = x_feats * gamma;

        Nc{idx} = reshape(n,1,1,numComponents);
        Fc{idx} = reshape(f,numFeatures,1,numComponents);
    end
    fprintf('Baum-Welch statistics completed (%0.0f seconds).\n',toc)

    % Expand statistics into matrices
    N = Nc;
    F = Fc;
    muc = reshape(ubm.mu,numFeatures,1,[]);
    for s = 1:num_utterances
        N{s} = repelem(reshape(Nc{s},1,[]),numFeatures);
        F{s} = reshape(Fc{s} - Nc{s}.*muc,[],1);
    end

    % Initialize Total Variability Matrix
    Sigma = ubm.sigma(:);
    T = randn(numel(ubm.sigma),tvm_dim);
    T = T/norm(T);
    I = eye(tvm_dim);
    Ey = cell(num_utterances,1);
    Eyy = cell(num_utterances,1);
    Linv = cell(num_utterances,1);

    % Train Total Variability Matrix
    numIterations = 1;
    for iterIdx = 1:numIterations
        tic
        % 1. Calculate the posterior distribution of the hidden variable
        TtimesInverseSSdiag = (T./Sigma)';
        parfor s = 1:num_utterances
            L = (I + TtimesInverseSSdiag.*N{s}*T);
            Linv{s} = pinv(L);
            Ey{s} = Linv{s}*TtimesInverseSSdiag*F{s};
            Eyy{s} = Linv{s} + Ey{s}*Ey{s}';
        end

        % 2. Accumlate statistics across the speakers
        Eymat = cat(2,Ey{:});
        FFmat = cat(2,F{:});
        Kt = FFmat*Eymat';
        K = mat2cell(Kt',tvm_dim,repelem(numFeatures,numComponents));

        newT = cell(numComponents,1);
        for c = 1:numComponents
            AcLocal = zeros(tvm_dim);
            for s = 1:num_utterances
                AcLocal = AcLocal + Nc{s}(:,:,c)*Eyy{s};
            end

        % 3. Update the Total Variability Space
            newT{c} = (pinv(AcLocal)*K{c})';
        end
        T = cat(1,newT{:});

        fprintf('Training Total Variability Space: %d/%d complete (%0.0f seconds).\n',iterIdx,numIterations,toc)
    end
    tvm.T = T;
    tvm.Tdim = size(T, 2);
    tvm.I = eye(tvm_dim);
    tvm.TS = T./ubm.sigma(:);
    tvm.TSi = tvm.TS';
end

%%% ----------------- Helper Functions -------------------------------- %%%
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


%%% ---------- (edited) Helper functions from tutorial ---------------- %%%

function L = helperGMMLogLikelihood(x,gmm)
    xMinusMu = repmat(x,1,1,numel(gmm.ComponentProportion)) - permute(gmm.mu,[1,3,2]);
    permuteSigma = permute(gmm.sigma,[1,3,2]);

    Lunweighted = -0.5*(sum(log(permuteSigma),1) + sum(xMinusMu.*(xMinusMu./permuteSigma),1) + size(gmm.mu,1)*log(2*pi));

    temp = squeeze(permute(Lunweighted,[1,3,2]));
    if size(temp,1)==1
        % If there is only one frame, the trailing singleton dimension was
        % removed in the permute. This accounts for that edge case.
        temp = temp';
    end

    L = temp + log(gmm.ComponentProportion)';
end