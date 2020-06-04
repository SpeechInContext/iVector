%%%------------------ Define hyper-parameters ------------------------- %%%
num_gaussians = 32;
tv_dim =  100;
plda_dim = 50;
numFeatures = 60;
gender = 'M';
num_para_workers = 5;
file_end = join(['_' num2str(num_gaussians) '_' num2str(tv_dim) '_' ...
    num2str(plda_dim) '_' gender '.mat'], '');

%%%---------- Get/Process files for background UBM model -------------- %%%

%------------ First, extract MFCCs if needed -----------------------------%
inFolder = 'C:\Users\mFry2\Desktop\SpeeCon\Data\PTDB-TUG\SPEECH DATA\MALE\MIC';
soundfile_ext = '.wav';
outFolder = 'C:\Users\mFry2\Desktop\SpeeCon\Data\PTDB-TUG\SPEECH DATA\MALE\MFCC';
normalizeMFCCs = true;
disp('Calculating MFCCs');
tic
mfcc_list = extract_mfccs(inFolder, soundfile_ext, outFolder, normalizeMFCCs);

%----------- Filter to gender if needed ----------------------------------%
if gender == 'F'
    gender_idx = contains(mfcc_list, '\F');
else
    gender_idx = ~contains(mfcc_list, '\F');
end
mfcc_list = mfcc_list(gender_idx, :);
fprintf('Feature extraction from background set complete (%0.0f seconds).\n',toc)

%---------- Next, collect all MFCCs of background files ------------------%
all_feats = cell(size(mfcc_list,1),1);
parfor fileIdx = 1:size(mfcc_list,1)
    fileId = fopen(mfcc_list{fileIdx});
    x_feats = fread(fileId);
    x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
    fclose(fileId);
    all_feats(fileIdx) = {x_feats};
end

%%%------------------ Train UBM-GMM ----------------------------------- %%%
ubm_file = join(['./Files/ubm' file_end], '');
if exist(ubm_file, 'file')
    disp('Loading pre-trained UBM')
    load(ubm_file)
else
    max_iters = 16;
    scale_fact = 4;
    disp('Training UBM')
    tic
    ubm = gmm_em(all_feats, num_gaussians, max_iters, scale_fact, num_para_workers);
    save(ubm_file, 'ubm');
    fprintf('UBM training complete (%0.0f seconds).\n',toc)
end

%%%------------------ Compute Baum-Welch stats to train TVM ----------- %%%
wrap_bw_stats = @(x) compute_bw_stats(x, ubm);
background_bw_file = join(['./Files/background_bw' file_end], '');
if exist(background_bw_file, 'file')
    disp('Loading pre-calculated Baum-Welch stats on background data')
    load(background_bw_file);
else
    disp('Calculating Baum-Welch stats on background data')
    tic
    [N, F] = cellfun(wrap_bw_stats, all_feats, 'UniformOutput', false);
    all_bw_stats = cell(size(N));
    for cellIdx = 1:size(N,1)
       all_bw_stats(cellIdx,1) = {[N{cellIdx}; F{cellIdx}]};
    end
    save(background_bw_file, 'all_bw_stats');
    fprintf('Baum-Welch stats on background data calculated (%0.0f seconds).\n',toc)
end

%%%------------------ Train Total Variability Matrix ------------------ %%%
tvm_file = join(['./Files/tvm' file_end], '');
if exist(tvm_file, 'file')
    disp('Loading pre-trained Total Variability Matrix')
    load(tvm_file)
else
    num_iters = 20;
    disp('Calculating Total Variability Matrix (using all background data)')
    tic
    tvm = train_tv_space(all_bw_stats, ubm, tv_dim, num_iters, num_para_workers);
    save(tvm_file, 'tvm')
    fprintf('TVM calculated (%0.0f seconds).\n',toc)
end

%%%--------------- Extract i-vectors for background data -------------- %%%
wrap_ivector = @(x) extract_ivector(x, ubm, tvm);
background_iv_file = join(['./Files/background_ivectors' file_end], '');
if exist(background_iv_file, 'file')
    disp('Loading pre-extracted i-vectors of background data')
    load(background_iv_file)
else
    disp('Extracting i-vectors for all background data')
    tic
    background_ivectors = cellfun(wrap_ivector, all_bw_stats, 'UniformOutput', false);
    save(background_iv_file, 'background_ivectors')
    fprintf('i-vectors extracted (%0.0f seconds).\n',toc)
end

%%%------------ Use labels to train PDLA projection ------------------- %%%
for fileIdx = 1:size(mfcc_list)
    background_ivectors(fileIdx, 2) = extractBetween(mfcc_list(fileIdx), 'mic_', '_');
end
speakerIds = grp2idx(background_ivectors(:,2));
speaker_ivectors = cat(2, background_ivectors{:,1});
plda_file = join(['./Files/plda' file_end], '');
if exist(plda_file, 'file')
    disp('Loading pre-trained PLDA mapping')
    load(plda_file)
else
    disp('Calculating PLDA mapping')
    tic
    plda = gplda_em(double(speaker_ivectors), speakerIds, plda_dim, 5);
    save(plda_file, 'plda')
    fprintf('PLDA mapping calculated (%0.0f seconds).\n',toc)
end

%%%---------------------------------------------------------------------%%%
%%% -------------- At this point the model is trained ----------------- %%%
%%%---------------------------------------------------------------------%%%

%%%------------- Extract MFCCs for enrol and verify data -------------- %%%

%------------ First, extract MFCCs if needed -----------------------------%
inFolder = 'C:\Users\mFry2\Desktop\SpeeCon\Data\SpiCE\audio_files\Interview snippets\WAV';
outFolder = 'C:\Users\mFry2\Desktop\SpeeCon\Data\SpiCE\audio_files\Interview snippets\MFCC';
soundfile_ext = '.wav';
normalizeMFCCs = true;
enrol_verify_list = extract_mfccs(inFolder, soundfile_ext, outFolder, normalizeMFCCs);

% Filter to selected gender
gender_idx = contains(enrol_verify_list, ['V' gender]);
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
enrol_verify_bw_file = join(['./Files/enrol_verify_bw' file_end], '');
if exist(enrol_verify_bw_file, 'file')
    disp('Loading pre-calculated Baum-Welch stats on enrol/verify data')
    load(enrol_verify_bw_file);
else
    disp('Calculating Bam-Welch stats on enrol/verify data')
    tic
    [N, F] = cellfun(wrap_bw_stats, all_enrol_verify_feats, 'UniformOutput', false);
    all_enrol_verify_bw_stats = cell(size(N));
    for cellIdx = 1:size(N,1)
       all_enrol_verify_bw_stats(cellIdx,1) = {[N{cellIdx}; F{cellIdx}]};
    end
    save(enrol_verify_bw_file, 'all_enrol_verify_bw_stats');
    fprintf('Baum-Welch stats on enrol/verify data calculated (%0.0f seconds).\n',toc)
end

%%%------------ Extract i-vectors for enrol and verify data ----------- %%%
enrol_verify_ivectors = cellfun(wrap_ivector, all_enrol_verify_bw_stats, 'UniformOutput', false);
for fileIdx = 1:size(enrol_verify_list)
    enrol_verify_ivectors(fileIdx, 2) = extractBetween(enrol_verify_list(fileIdx), 'MFCC\', '_');
end
speakerIds = grp2idx(enrol_verify_ivectors(:,2));

%%%------------ Splice enrol set and verify set ----------------------- %%%
% Set proportion of enrol utterances
proportion = 0.3;
[model_iv, speaker_model_size, test_iv, verify_labels] = separate_enrol_verify(enrol_verify_ivectors,proportion);

%%%------------ Perform PLDA on verification set -----------------------%%%
scores = score_gplda_trials(plda, model_iv, test_iv);
prob_scores = zeros(size(scores));
[mm, pred_speaker_id] = max(scores);
for col = 1:size(scores,2)
    prob_scores(:,col) = exp(scores(:,col))/(sum(exp(scores(:,col))));
end
accuracy = pred_speaker_id == grp2idx(verify_labels)';
cp = classperf(grp2idx(verify_labels)', pred_speaker_id);

