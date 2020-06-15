%%%------------------ Define hyper-parameters ------------------------- %%%
num_gaussians = 256;
tv_dim =  200;
plda_dim = 200;
numFeatures = 60;
normalizeMFCCs = true;
train_gender = 'X';             %M, F or X
train_language = 'english';     % english or cantonese
test_gender = 'F';              %M, F or X
test_language = 'cantonese';    % english or cantonese
corpus = 'ls';                  %timit or ls
num_para_workers = 19;
train_file_end = join(['_' num2str(num_gaussians) '_' num2str(tv_dim) '_' ...
    num2str(plda_dim) '_' train_gender '_' train_language '_' corpus '.mat'], '');
test_file_end = join(['_' num2str(num_gaussians) '_' num2str(tv_dim) '_' ...
    num2str(plda_dim) '_' test_gender '_' test_language '_' corpus '.mat'], '');

%%%---------- Get/Process files for background UBM model -------------- %%%
%------------ First, extract MFCCs if needed -----------------------------%
if strcmp(train_language, 'english')
    if strcmp(corpus, 'ls')
        inFolder = 'C:\Users\mf\Documents\Corpora\LibriSpeech\flac';
        outFolder = 'C:\Users\mf\Documents\Corpora\LibriSpeech\MFCC';
        soundfile_ext = '.flac';
    elseif strcmp(corpus, 'timit')
        inFolder = 'C:\Users\mf\Documents\Corpora\TIMIT\TRAIN\WAV';
        outFolder = 'C:\Users\mf\Documents\Corpora\TIMIT\TRAIN\MFCC';
        soundfile_ext = '.WAV';
    end
end

disp('Calculating MFCCs');
tic
mfcc_list = extract_mfccs(inFolder, soundfile_ext, outFolder, normalizeMFCCs);
%----------- Filter to gender if needed ----------------------------------%
if train_gender == 'F'
    gender_idx = contains(mfcc_list, '\F');
    mfcc_list = mfcc_list(gender_idx, :);
elseif train_gender == 'M'
    gender_idx = ~contains(mfcc_list, '\F');
    mfcc_list = mfcc_list(gender_idx, :);
end
fprintf('Feature extraction from background set complete (%0.0f seconds).\n',toc)

%%%------------------ Train UBM-GMM ----------------------------------- %%%
ubm_file = join(['./Files/ubm' train_file_end], '');
if exist(ubm_file, 'file')
    disp('Loading pre-trained UBM')
    load(ubm_file)
else
    %---------- Next, collect all MFCCs of background files --------------%
    all_feats = cell(size(mfcc_list,1),1);
    parfor fileIdx = 1:size(mfcc_list,1)
        fileId = fopen(mfcc_list{fileIdx});
        x_feats = fread(fileId);
        x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
        fclose(fileId);
        all_feats(fileIdx) = {x_feats};
    end
    max_iters = 10;
    scale_fact = 1;
    disp('Training UBM')
    tic
    ubm = gmm_em(all_feats, num_gaussians, max_iters, scale_fact, num_para_workers);
    save(ubm_file, 'ubm');
    fprintf('UBM training complete (%0.0f seconds).\n',toc)
end

%%%------------------ Train Total Variability Matrix ------------------ %%%
tvm_file = join(['./Files/tvm' train_file_end], '');
wrap_bw_stats = @(x) compute_bw_stats(x, ubm);
if exist(tvm_file, 'file')
    disp('Loading pre-trained Total Variability Matrix')
    load(tvm_file)
else
    %------------------ Compute Baum-Welch stats to train TVM ----------- %
    disp('Calculating Baum-Welch stats on background data')
    tic
    temp_arr = distributed(all_feats);
    [N, F] = cellfun(wrap_bw_stats, temp_arr, 'UniformOutput', false);
    N = gather(N);
    F = gather(F);
    all_bw_stats = cell(size(N));
    for cellIdx = 1:size(N,1)
       all_bw_stats(cellIdx,1) = {[N{cellIdx}; F{cellIdx}]};
    end
    fprintf('Baum-Welch stats on background data calculated (%0.0f seconds).\n',toc)

    %---------------- Use EM to train TVM --------------------------------%
    num_iters = 5;
    disp('Calculating Total Variability Matrix (using all background data)')
    tic
    tvm = train_tv_space(all_bw_stats, ubm, tv_dim, num_iters, num_para_workers);
    save(tvm_file, 'tvm')
    fprintf('TVM calculated (%0.0f seconds).\n',toc)
end
clear all_feats N F temp_arr gender_idx

%%%--------------- Extract i-vectors for background data -------------- %%%
wrap_ivector = @(x) extract_ivector(x, ubm, tvm);
background_iv_file = join(['./Files/background_ivectors' train_file_end], '');
if exist(background_iv_file, 'file')
    disp('Loading pre-extracted i-vectors of background data')
    load(background_iv_file)
else
    disp('Extracting i-vectors for all background data')
    tic
    temp_arr = distributed(all_bw_stats);
    background_ivectors = cellfun(wrap_ivector, temp_arr, 'UniformOutput', false);
    background_ivectors = gather(background_ivectors);
    save(background_iv_file, 'background_ivectors')
    fprintf('i-vectors extracted (%0.0f seconds).\n',toc)
end
clear all_bw_stats

%%%------------ Use labels to train LDA projection ------------------- %%%
for fileIdx = 1:size(mfcc_list)
    background_ivectors(fileIdx, 2) = extractBetween(mfcc_list(fileIdx), 'clean-100\', '\');
end
speakerIds = grp2idx(background_ivectors(:,2));
speaker_ivectors = cat(2, background_ivectors{:,1});
lda_file = join(['./Files/lda' train_file_end], '');
if exist(lda_file, 'file')
    disp('Loading pre-trained LDA mapping')
    load(lda_file)
else
    disp('Calculating LDA mapping')
    tic
    lda_out = lda(double(speaker_ivectors), speakerIds');
    save(lda_file, 'lda_out')
    fprintf('LDA mapping calculated (%0.0f seconds).\n',toc)
end

%%%------------ Use labels to train PLDA projection ------------------- %%%
plda_file = join(['./Files/plda' train_file_end], '');
if exist(plda_file, 'file')
    disp('Loading pre-trained PLDA mapping')
    load(plda_file)
else
    disp('Calculating PLDA mapping')
    tic
    speaker_ivectors_lda = lda_out'*speaker_ivectors;
    plda = gplda_em(double(speaker_ivectors_lda), speakerIds, plda_dim, 10);
    save(plda_file, 'plda')
    fprintf('PLDA mapping calculated (%0.0f seconds).\n',toc)
end


%%%---------------------------------------------------------------------%%%
%%% -------------- At this point the model is trained ----------------- %%%
%%%---------------------------------------------------------------------%%%

%%%------------- Extract MFCCs for enrol and verify data -------------- %%%

%------------ First, extract MFCCs if needed -----------------------------%
inFolder = ['C:\Users\mf\Desktop\SpeeCon\SpiCE\WAV\' test_language '_interview_snippets'];
outFolder = ['C:\Users\mf\Desktop\SpeeCon\SpiCE\MFCC\' test_language '_interview_snippets'];
soundfile_ext = '.wav';
normalizeMFCCs = true;
enrol_verify_list = extract_mfccs(inFolder, soundfile_ext, outFolder, normalizeMFCCs);

% Filter to selected gender
gender_idx = contains(enrol_verify_list, ['V' test_gender]);
enrol_verify_list = enrol_verify_list(gender_idx,1);

%---------- Next, collect all MFCCs of enrol/verify files ----------------%
all_enrol_verify_feats = cell(size(enrol_verify_list,1),1);
for fileIdx = 1:size(enrol_verify_list,1)
    fileId = fopen(enrol_verify_list{fileIdx});
    x_feats = fread(fileId);
    x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
    fclose(fileId);
    all_enrol_verify_feats(fileIdx) = {x_feats};
end

%%%---------- Next, get Baum-welch statistics for enrol/verify files ---%%%
enrol_verify_bw_file = join(['./Files/enrol_verify_bw' test_file_end], '');
if exist(enrol_verify_bw_file, 'file')
    disp('Loading pre-calculated Baum-Welch stats on enrol/verify data')
    load(enrol_verify_bw_file);
else
    disp('Calculating Bam-Welch stats on enrol/verify data')
    tic
    temp_arr = distributed(all_enrol_verify_feats);
    [N, F] = cellfun(wrap_bw_stats, temp_arr, 'UniformOutput', false);
    N = gather(N);
    F = gather(F);
    all_enrol_verify_bw_stats = cell(size(N));
    for cellIdx = 1:size(N,1)
       all_enrol_verify_bw_stats(cellIdx,1) = {[N{cellIdx}; F{cellIdx}]};
    end
    %save(enrol_verify_bw_file, 'all_enrol_verify_bw_stats');
    fprintf('Baum-Welch stats on enrol/verify data calculated (%0.0f seconds).\n',toc)
end

%%%------------ Extract i-vectors for enrol and verify data ----------- %%%
enrol_verify_iv_file = join(['./Files/enrol_verify_ivectors' test_file_end], '');
if exist(enrol_verify_iv_file, 'file')
    disp('Loading pre-extracted i-vectors of enrol/verify data')
    load(enrol_verify_iv_file)
else
    disp('Extracting i-vectors for all enrol/verify data')
    tic
    temp_arr = distributed(all_enrol_verify_bw_stats);
    enrol_verify_ivectors = cellfun(wrap_ivector,temp_arr , 'UniformOutput', false);
    enrol_verify_ivectors = gather(enrol_verify_ivectors);
    for fileIdx = 1:size(enrol_verify_list)
        enrol_verify_ivectors(fileIdx, 2) = extractBetween(enrol_verify_list(fileIdx), 'snippets\', '_');
    end
    enrol_verify_ivectors = [enrol_verify_ivectors enrol_verify_list];
    save(enrol_verify_iv_file, 'enrol_verify_ivectors')
    fprintf('i-vectors extracted (%0.0f seconds).\n',toc)
end



%%%------------ Splice enrol set and verify set ----------------------- %%%
% Set proportion of enrol utterances
proportion = 0.4;
[model_iv, speaker_model_size, test_iv, verify_labels] = separate_enrol_verify(enrol_verify_ivectors,proportion);

%%%------------ Perform PLDA on verification set -----------------------%%%
scores = score_gplda_trials(plda, lda_out'*model_iv, lda_out'*test_iv);
prob_scores = zeros(size(scores));
[mm, pred_speaker_id] = max(scores);
for col = 1:size(scores,2)
    prob_scores(:,col) = exp(scores(:,col))/(sum(exp(scores(:,col))));
end
accuracy = pred_speaker_id == grp2idx(verify_labels)';
cp = classperf(grp2idx(verify_labels)', pred_speaker_id);

