%%% ----------------- Set path to background audio files -------------- %%%
%inFolder = 'C:/Users/mFry2/Desktop/SpeeCon/Data/PTDB-TUG/SPEECH DATA/MALE/MIC/**/*.wav';
inFolder = 'C:/Users/mFry2/Desktop/SpeeCon/Data/PTDB-TUG/SPEECH DATA/FEMALE/MIC/**/*.wav';
numMFCCs = 20; %number of MFCCs
numFeats = numMFCCs * 3;
%extract_and_normalize_all_mfccs(inFolder, numMFCCs)

%%% ----------- Train UBM-GMM and total variability matrix (tvm) ------ %%%
%mfccFolder = 'C:/Users/mFry2/Desktop/SpeeCon/Data/PTDB-TUG/SPEECH DATA/MALE/MFCC/*.mfcc';
mfccFolder = 'C:/Users/mFry2/Desktop/SpeeCon/Data/PTDB-TUG/SPEECH DATA/FEMALE/MFCC/*.mfcc';
num_gaussians = 1024;
ivector_dim = 400;
proj_dim = 200;
gender = 'F';
file_end = join(['_' num2str(num_gaussians) '_' num2str(ivector_dim) '_' num2str(proj_dim) ...
                '_' gender '.mat'], '');        
ubm_file = join(['./Models/ubm' file_end], '');
if exist(ubm_file, 'file')
    disp('ivector system already trained.')
    load(ubm_file)
else    
    [ubm, tvm] = train_ubm_tvm(mfccFolder, num_gaussians, numFeats, ivector_dim);
    save(ubm_file, 'ubm', 'tvm')
end

%%% ------------ Calculate PLDA parameters over test set -------------- %%%
pdla_file = join(['./Models/pdla' file_end], '');
if exist(pdla_file, 'file') 
    load(pdla_file)
else
    plda = train_projm_PLDA(mfccFolder, ubm, tvm, proj_dim);
    save(pdla_file, 'plda')
end

%%% ------- Get enrolled speaker i-vector and set of verify ivectors -- %%%
enrolFolder = 'C:/Users/mFry2/Desktop/SpeeCon/Data/SpiCE/MFCC/*.mfcc';
num_enrolment_utterances = 75;
speaker_iv_plda_file = join(['./Models/speaker_iv_PLDA' file_end], '');
if exist(speaker_iv_plda_file, 'file')
    load(speaker_iv_plda_file);
else
    [speaker_ivectors, verify_ivectors] = get_enrol_verify_ivectors(enrolFolder, num_enrolment_utterances, ubm, tvm);
    save(speaker_iv_plda_file, 'speaker_ivectors', 'verify_ivectors');
end

speaker_iv = cat(2, speaker_ivectors{:,2});
scores = score_gplda_trials(plda, speaker_iv, verify_ivectors);
prob_scores = zeros(size(scores));
[mm, pred_speaker_id] = max(scores);
for col = 1:size(scores,2)
    prob_scores(:,col) = exp(scores(:,col))/(sum(exp(scores(:,col))));
end