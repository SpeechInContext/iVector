%%%--------------- Computes metrics of speakers -------------------- %%%
% Assumes you have already run and extracted these loaded models using
% 'run_ivector_MSR.m'

%%%------------------ Define hyper-parameters ------------------------- %%%
% These parameters must match a model that has already been trained
num_gaussians = 128;
tv_dim =  200;
plda_dim = 200;
numFeatures = 60;
gender = 'F';
num_para_workers = 5;
file_end = join(['_' num2str(num_gaussians) '_' num2str(tv_dim) '_' ...
    num2str(plda_dim) '_' gender '.mat'], '');

ubm_file = join(['./Files/ubm' file_end], ''); load(ubm_file);
tvm_file = join(['./Files/tvm' file_end], ''); load(tvm_file);
lda_file = join(['./Files/lda' file_end], ''); load(lda_file);
plda_file = join(['./Files/plda' file_end], ''); load(plda_file);

%%%------------------ Load ivectors --------------------------- %%%
enrol_verify_iv_file = join(['./Files/enrol_verify_ivectors' file_end], '');
load(enrol_verify_iv_file);
unique_speakers = unique(enrol_verify_ivectors(:,2));
speakerIds = grp2idx(enrol_verify_ivectors(:,2));
unique_speakerIds = unique(speakerIds);

%%%---------------- Generate analyses --------------------------------- %%%
speaker_analyses = cell(size(unique_speakerIds,1), 14);
grouped_utterances_cols = ["SpeakerId" "SpeakerIdNum" "Files" "iVectors" "SpeakerModel" ...
    "iVectEuDist2Model" "AvgEuDist" "EuStd" "iVectCSDist2Model" "AvgCSDist" "CSStd" ...
    "iVectProb2Model" "AvgProb" "ProbStd"];
for spIdx = 1:size(unique_speakerIds,1)
    utteranceIdx = speakerIds == spIdx;
    speaker_analyses(spIdx,1) = unique_speakers(spIdx);
    speaker_analyses(spIdx,2) = {spIdx};
    speaker_analyses(spIdx,3) = {string(cat(2, enrol_verify_ivectors(utteranceIdx, 3)))'};
    speaker_analyses(spIdx,4) = {cat(2, enrol_verify_ivectors{utteranceIdx, 1})};
    %%%----------Generate Speaker Model (average ivector)
    speaker_model = mean(speaker_analyses{spIdx,4},2);
    speaker_analyses(spIdx,5) = {speaker_model};  
    
    %%%-----------Calculate Euclidean Distance per iVector
    dist_per_ivector = vecnorm(speaker_analyses{spIdx,4}-repmat(speaker_model, 1,size(speaker_analyses{spIdx,4},2)));
    speaker_analyses(spIdx,6) = {dist_per_ivector};
    % get average Euclidean distance
    speaker_analyses(spIdx,7) = {mean(dist_per_ivector)};
    % get std of Euclidean distances
    speaker_analyses(spIdx,8) = {std(dist_per_ivector)};
    
    %%%-----------Calculate Cosine Distance per iVector
    csd_per_ivector = cosine_dist_mat(speaker_model, speaker_analyses{spIdx,4});
    speaker_analyses(spIdx,9) = {csd_per_ivector'};
    speaker_analyses(spIdx,10) = {mean(csd_per_ivector)};
    speaker_analyses(spIdx,11) = {std(csd_per_ivector)};    
end

%%%------------Now generate PLDA probabilities
model_iv = cat(2, speaker_analyses{:,5});
test_iv =  cat(2, speaker_analyses{:,4});
scores = score_gplda_trials(plda, model_iv, test_iv);
prob_scores = zeros(size(scores));
[mm, pred_speaker_id] = max(scores);
for col = 1:size(scores,2)
    prob_scores(:,col) = exp(scores(:,col))/(sum(exp(scores(:,col))));
end
st_loc = 1;
for spIdx = 1:size(unique_speakerIds,1)
    en_loc = st_loc + size(speaker_analyses{spIdx,4},2) - 1;
    %%%-------------Calculate Probability using PLDA per iVector
    prob_per_ivector = prob_scores(spIdx, st_loc:en_loc);
    speaker_analyses(spIdx,12) = {prob_per_ivector};
    speaker_analyses(spIdx,13) = {mean(prob_per_ivector)}; 
    speaker_analyses(spIdx,14) = {std(prob_per_ivector)}; 
    st_loc = en_loc + 1;
end

%%%---------------------- HelperFunctions ------------------------------%%%
function csd_m= cosine_dist_mat(speaker_model, ivectors)
    csd_m = zeros(size(ivectors,2), 1);
    for idx = 1:size(csd_m,1)
        csd_m(idx) = cosine_dist(ivectors(:,idx), speaker_model);
    end
end
function csd = cosine_dist(v1, v2)
    csd = 1 - dot(v1,v2)/norm(v1)/norm(v2);
end
