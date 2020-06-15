%%%------------------ Train a UBM-GMM/PLDA iVector model ---------------%%%
% This script trains a UBM-GMM/PLDA model to allow for iVector
% representations of an utterance to be extracted

%%%------------------ Define hyper-parameters ------------------------- %%%
num_gaussians = 8;
tv_dim =  10;
plda_dim = 10;
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
%-------------- Set corpus folder/soundfile extension --------------------%
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

%-------------------------- Extract MFCCs --------------------------------%
disp('Calculating MFCCs');
tic
mfcc_list = extract_mfccs(inFolder, soundfile_ext, outFolder, normalizeMFCCs);

%--------- Filter to gender if needed (hardcoded to TIMIT for now)--------%
if train_gender == 'F'
    gender_idx = contains(mfcc_list, '\F');
    mfcc_list = mfcc_list(gender_idx, :);
elseif train_gender == 'M'
    gender_idx = ~contains(mfcc_list, '\F');
    mfcc_list = mfcc_list(gender_idx, :);
end
fprintf('Feature extraction from background set complete (%0.0f seconds).\n',toc)

%%%------------------ Train UBM-GMM ----------------------------------- %%%
%------------------ Collect all MFCCs of background files ----------------%
all_feats = cell(size(mfcc_list,1),1);
parfor fileIdx = 1:size(mfcc_list,1)
    fileId = fopen(mfcc_list{fileIdx});
    x_feats = fread(fileId);
    x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
    fclose(fileId);
    all_feats(fileIdx) = {x_feats};
end
%------------- Train UBM-GMM (if model not already trained) --------------%
ubm_file = join(['./Files/ubm' train_file_end], '');
if exist(ubm_file, 'file')
    disp('Loading pre-trained UBM')
    load(ubm_file)
else
    max_iters = 10;
    scale_fact = 1;
    disp('Training UBM')
    tic
    ubm = gmm_em(all_feats, num_gaussians, max_iters, scale_fact, num_para_workers);
    save(ubm_file, 'ubm');
    fprintf('UBM training complete (%0.0f seconds).\n',toc)
end

%%%------------------ Train Total Variability Matrix ------------------ %%%
%-------------------- Compute Baum-Welch stats to train TVM ------------- %
disp('Calculating Baum-Welch stats on background data')
tic
wrap_bw_stats = @(x) compute_bw_stats(x, ubm);
temp_arr = distributed(all_feats);
[N, F] = cellfun(wrap_bw_stats, temp_arr, 'UniformOutput', false);
N = gather(N);
F = gather(F);
all_bw_stats = cell(size(N));
for cellIdx = 1:size(N,1)
   all_bw_stats(cellIdx,1) = {[N{cellIdx}; F{cellIdx}]};
end
fprintf('Baum-Welch stats on background data calculated (%0.0f seconds).\n',toc)

%-------------------- Train TVM (if not already trained) -----------------%
tvm_file = join(['./Files/tvm' train_file_end], '');
if exist(tvm_file, 'file')
    disp('Loading pre-trained Total Variability Matrix')
    load(tvm_file)
else
    %---------------- Use EM to train TVM --------------------------------%
    num_iters = 5;
    disp('Calculating Total Variability Matrix (using all background data)')
    tic
    tvm = train_tv_space(all_bw_stats, ubm, tv_dim, num_iters, num_para_workers);
    save(tvm_file, 'tvm')
    fprintf('TVM calculated (%0.0f seconds).\n',toc)
end
clear all_feats N F temp_arr gender_idx

%%%-- Extract i-vectors for background data (needed to calculate LDA) --%%%
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

%%%------------------ Train LDA projection ---------------------------- %%%
%--------------- Get talker Id for each file (hardcoded to filepath)------%
for fileIdx = 1:size(mfcc_list)
    background_ivectors(fileIdx, 2) = extractBetween(mfcc_list(fileIdx), 'clean-100\', '\');
end
speakerIds = grp2idx(background_ivectors(:,2));
speaker_ivectors = cat(2, background_ivectors{:,1});
%---------- Get LDA projection (if not calculated yet) -------------------%
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

%%%------------------- Train PLDA projection -------------------------- %%%
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