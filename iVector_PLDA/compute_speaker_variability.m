%%%--------------- Computes metrics of speakers' variability ----------- %%%
% This script calculates variability metrics for speakers' utterances

% This script assumes you have already trained a model matching the hyper-
% parameters below using the `train_ivector_MSR.m` script.

% It also assumes you have already extracted iVector representations for
% your enrol/verify data set using the `apply_ivector_MSR.m` script.

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
test_language = 'cantonese';    % english or cantonese
corpus = 'ls';                  %timit or ls
num_para_workers = 19;
train_file_end = join(['_' num2str(num_gaussians) '_' num2str(tv_dim) '_' ...
    num2str(plda_dim) '_' train_gender '_' train_language '_' corpus '.mat'], '');
test_file_end = join(['_' num2str(num_gaussians) '_' num2str(tv_dim) '_' ...
    num2str(plda_dim) '_' test_gender '_' test_language '_' corpus '.mat'], '');

%%%---- Load pretrained LDA/PLDA analyses and enrol/verify iVectors-----%%%
lda_file = join(['./Files/lda' train_file_end], ''); load(lda_file);
plda_file = join(['./Files/plda' train_file_end], ''); load(plda_file);
enrol_verify_iv_file = join(['./Files/enrol_verify_ivectors' test_file_end], '');
load(enrol_verify_iv_file);


%%%----------------- Analyze speaker variability ---------------------- %%%
%------------------ Get unique speakers ------------------------------- %%%
unique_speakers = unique(enrol_verify_ivectors(:,2));
speakerIds = grp2idx(enrol_verify_ivectors(:,2));
unique_speakerIds = unique(speakerIds);

%------------------- Initialize speaker analyses matrix ------------------%
speaker_analyses = cell(size(unique_speakerIds,1), 14);
speaker_analyses_cols = ["SpeakerId" "SpeakerIdNum" "Files" "iVectors" "SpeakerModel" ...
    "iVectEucDist2Model" "AvgEucDist" "EucDistStd" "iVectCosDist2Model" "AvgCosDist" "CosDistStd" ...
    "iVectPLDAScore2Model" "AvgPLDAScore" "PLDAScoreStd" "PLDAAcc" "PLDAFRR" "PLDAFAR"];

%----------------- Go through all utterances for each talker -------------%
for spIdx = 1:size(unique_speakerIds,1)
    utteranceIdx = speakerIds == spIdx;
    speaker_analyses(spIdx,1) = unique_speakers(spIdx);     % SpeakerId
    speaker_analyses(spIdx,2) = {spIdx};                    % Speaker Numeric Id
    speaker_analyses(spIdx,3) = {string(cat(2, enrol_verify_ivectors(utteranceIdx, 3)))'}; %Speaker Filenames
    speaker_analyses(spIdx,4) = {lda_out'*cat(2, enrol_verify_ivectors{utteranceIdx, 1})}; %Filenames' iVector representation
    %%%----------Generate Speaker Model (average ivector)
    speaker_model = mean(speaker_analyses{spIdx,4},2);      %Note this is the mean of all utterances
    speaker_analyses(spIdx,5) = {speaker_model};  
    
    %%%-----------Calculate Euclidean Distance per iVector
    dist_per_ivector = vecnorm2(speaker_analyses{spIdx,4}-repmat(speaker_model, 1,size(speaker_analyses{spIdx,4},2)));
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
model_iv = cat(2, speaker_analyses{:,5}); %Speaker iVector models
test_iv =  cat(2, speaker_analyses{:,4}); %iVectors to compare to speaker models
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
    speaker_analyses(spIdx,13) = {nanmean(prob_per_ivector)}; 
    speaker_analyses(spIdx,14) = {nanstd(prob_per_ivector)};
    speaker_accuracy_per_utt = speakerIds(speakerIds==spIdx)' == pred_speaker_id(speakerIds==spIdx);
    false_accept_per_utt = pred_speaker_id(speakerIds~=spIdx) == spIdx;
    speak_accuracy = sum(speaker_accuracy_per_utt)/size(speaker_accuracy_per_utt,2);
    false_accept = sum(false_accept_per_utt)/size(false_accept_per_utt,2);
    speaker_analyses(spIdx,15) = {speak_accuracy};
    speaker_analyses(spIdx,16) = {1-speak_accuracy};
    speaker_analyses(spIdx,17) = {false_accept};
    st_loc = en_loc + 1;
end
speaker_analyses = [cellstr(speaker_analyses_cols); speaker_analyses];
%sort command: 
speaker_analyses(2:end,:) = sortrows(speaker_analyses(2:end,:), 13, 'descend');
cp = classperf(speakerIds', pred_speaker_id);

%%%---------------------- Write to CSV per talker ----------------------%%%
utt_outs = [];
for spIdx = 2:size(speaker_analyses,1)
    %Out data
    talker = speaker_analyses(spIdx,1);
    lang = {test_language};
    files = cellstr(speaker_analyses{spIdx,3});
    filenames = cell(size(files));
    for fileIdx = 1:size(files,2)
        [filepath,name,ext] = fileparts(files{fileIdx});
        filenames(fileIdx) = {[name '.wav']};
    end
    talker = repmat(talker, 1, size(files,2));
    lang = repmat(lang, 1, size(files,2));
    plda_score = num2cell(speaker_analyses{spIdx,12});
    euc_dist = num2cell(speaker_analyses{spIdx,6});
    cs_dist = num2cell(speaker_analyses{spIdx,9});
    outmat = [talker;lang;filenames;euc_dist;cs_dist;plda_score]';
    utt_outs = [utt_outs; outmat]; 
end
header = {'SpeakerId' 'Language' 'Utterance' 'EucDist' 'CosDist' 'PLDAScore'};
outFile = ['./Files/utterances' test_file_end];
outFile = strrep(outFile, '.mat','.csv');

%Write data
T = cell2table(utt_outs,'VariableNames',header);
writetable(T,outFile);
    
speakers_out = speaker_analyses(:, [1 7 8 10 11 13 14 15 16 17]);
lang = [{'Language'}; repmat({test_language}, size(speakers_out,1)-1, 1)];
speakers_out = [speakers_out(:, 1) lang speakers_out(:, 2:end)];
T = cell2table(speakers_out(2:end,:),'VariableNames',speakers_out(1, :));
outFile = ['./Files/speakers' test_file_end];
outFile = strrep(outFile, '.mat','.csv');
writetable(T, outFile);

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

function vn = vecnorm2(m)
    vn = zeros(1,size(m,2));
    for idx = 1:size(m,2)
        vn(idx) = norm(m(:,idx));
    end
end