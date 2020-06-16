%%%--------- Apply a trained iVector model to speaker data ------------ %%%
% The goal is to extract iVectors, derived from a trained model, for each 
% utterance in the data. This script will most likely be used to generate 
% enrol/verify data, which is terminology used throughout

% This script assumes you have already trained a model of matching hyper-
% parameters using 'train_ivector_MSR.m'

%%%------------------ Define hyper-parameters ------------------------- %%%
% These parameters must match a model that has already been trained
num_gaussians = 256;
tv_dim =  200;
plda_dim = 200;
numFeatures = 60;
normalizeMFCCs = true;
train_gender = 'X';             %M, F or X
train_language = 'english';     % english or cantonese
test_gender = 'F';              %M, F or X
test_language = 'english';    % english or cantonese
corpus = 'ls';                  %timit or ls
num_para_workers = 19;
train_file_end = join(['_' num2str(num_gaussians) '_' num2str(tv_dim) '_' ...
    num2str(plda_dim) '_' train_gender '_' train_language '_' corpus '.mat'], '');
test_file_end = join(['_' num2str(num_gaussians) '_' num2str(tv_dim) '_' ...
    num2str(plda_dim) '_' test_gender '_' test_language '_' corpus '.mat'], '');

%%%----------------- Load trained model ------------------------------- %%%
ubm_file = join(['./Files/ubm' train_file_end], ''); load(ubm_file);
tvm_file = join(['./Files/tvm' train_file_end], ''); load(tvm_file);

%%%----------------- Extract MFCCs for speaker data ------------------- %%%
%------------ Set corpus folder/soundfile extension ----------------------%
inFolder = ['C:\Users\mf\Desktop\SpeeCon\SpiCE\WAV\' test_language '_interview_snippets'];
outFolder = ['C:\Users\mf\Desktop\SpeeCon\SpiCE\MFCC\' test_language '_interview_snippets'];
soundfile_ext = '.wav';
enrol_verify_list = extract_mfccs(inFolder, soundfile_ext, outFolder, normalizeMFCCs);

%---- Filter to selected gender (hardcoded to SpiCE data now) ------------%
gender_idx = contains(enrol_verify_list, ['V' test_gender]);
enrol_verify_list = enrol_verify_list(gender_idx,1);

%%%-------- Extract iVectors for all (enrol/verify) utterances -----------%
% To do this, we need to collect the MFCCs for an utterance, calculate the
% Baum-Welch stats for the utterance, and decompose the stats into factors
% using the TVM

%----------------------- Collect all MFCCs -------------------------------%
all_enrol_verify_feats = cell(size(enrol_verify_list,1),1);
parfor fileIdx = 1:size(enrol_verify_list,1)
    fileId = fopen(enrol_verify_list{fileIdx});
    x_feats = fread(fileId);
    x_feats = reshape(x_feats, numFeatures, size(x_feats,1)/numFeatures);
    fclose(fileId);
    all_enrol_verify_feats(fileIdx) = {x_feats};
end

%------------------- Calculate Baum-welch statistics ---------------------%
disp('Calculating Baum-Welch stats on enrol/verify data')
tic
wrap_bw_stats = @(x) compute_bw_stats(x, ubm);
temp_arr = distributed(all_enrol_verify_feats); 
[N, F] = cellfun(wrap_bw_stats, temp_arr, 'UniformOutput', false);
N = gather(N);
F = gather(F);
all_enrol_verify_bw_stats = cell(size(N));
for cellIdx = 1:size(N,1)
   all_enrol_verify_bw_stats(cellIdx,1) = {[N{cellIdx}; F{cellIdx}]};
end
fprintf('Baum-Welch stats on enrol/verify data calculated (%0.0f seconds).\n',toc)

%----------------------------- Extract i-vectors -------------------------%
enrol_verify_iv_file = join(['./Files/enrol_verify_ivectors' test_file_end], '');
disp('Extracting i-vectors for all enrol/verify data')
wrap_ivector = @(x) extract_ivector(x, ubm, tvm);
tic
temp_arr = distributed(all_enrol_verify_bw_stats);
enrol_verify_ivectors = cellfun(wrap_ivector,temp_arr , 'UniformOutput', false);
enrol_verify_ivectors = gather(enrol_verify_ivectors);
%--------------- Add file identifier (hardcoded to filepath) and save ----%
for fileIdx = 1:size(enrol_verify_list)
    enrol_verify_ivectors(fileIdx, 2) = extractBetween(enrol_verify_list(fileIdx), 'snippets\', '_');
end
enrol_verify_ivectors = [enrol_verify_ivectors enrol_verify_list];
save(enrol_verify_iv_file, 'enrol_verify_ivectors')
fprintf('i-vectors extracted (%0.0f seconds).\n',toc)


%%%------------------ Enrol and verify utterances --------------------- %%%
% This code is here for demonstration purposes if someone wants to enrol a
% speaker with a certain proportion of their utterances and then verify
% new utterances thereafter

% % Set proportion of enrol utterances
% proportion = 0.4;
% [model_iv, speaker_model_size, test_iv, verify_labels] = separate_enrol_verify(enrol_verify_ivectors,proportion);
% 
% %%%------------ Perform PLDA on verification set -----------------------%%%
% scores = score_gplda_trials(plda, lda_out'*model_iv, lda_out'*test_iv);
% prob_scores = zeros(size(scores));
% [mm, pred_speaker_id] = max(scores);
% for col = 1:size(scores,2)
%     prob_scores(:,col) = exp(scores(:,col))/(sum(exp(scores(:,col))));
% end
% accuracy = pred_speaker_id == grp2idx(verify_labels)';
% cp = classperf(grp2idx(verify_labels)', pred_speaker_id);
