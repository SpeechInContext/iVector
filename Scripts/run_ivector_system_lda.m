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

%%% ------ Project i-vectors to new space using LDA/WCCN -------------- %%%
projm_file = join(['./Models/projm' file_end], '');
if exist(projm_file, 'file') 
    load(projm_file)
else
    projMatrix = train_projm_LDAWCCN(mfccFolder, ubm, tvm, proj_dim);
    save(projm_file, 'projMatrix')
end

%%% -------------- Enrol speakers ------------------------------------- %%%
enrolFolder = 'C:/Users/mFry2/Desktop/SpeeCon/Data/SpiCE/MFCC/*.mfcc';
num_enrolment_utterances = 100;
speaker_iv_files = join(['./Models/lda_speaker_ivectors' file_end], '');
if exist(speaker_iv_files, 'file')
    load(speaker_iv_files);
else
    speaker_ivectors = enrol_speakers(enrolFolder, num_enrolment_utterances, ubm, tvm, projMatrix);
    save(speaker_iv_files, 'speaker_ivectors')
end

%%% ----------------- Verify Speakers with FRR/FAR -------------------- %%%
frr_far_file = join(['./Models/lda_frr_far' file_end], '');
if exist(frr_far_file, 'file')
    load(frr_far_file)
else
    [thresholds, FRR, FAR] = verify_speakers_frr_far(enrolFolder, speaker_ivectors, ...
                                num_enrolment_utterances, ubm, tvm, projMatrix);
    save(frr_far_file, 'thresholds', 'FRR', 'FAR');
end

%%% ----------------- Plot Equal Error Rate --------------------------- %%%
% Equal Error rate
[~,EERThresholdIdx] = min(abs(FAR - FRR));
EERThreshold = thresholds(EERThresholdIdx);
EER = mean([FAR(EERThresholdIdx),FRR(EERThresholdIdx)]);
figure
plot(thresholds,FAR,'k', ...
     thresholds,FRR,'b', ...
     EERThreshold,EER,'ro','MarkerFaceColor','r')
title(sprintf('Equal Error Rate = %0.4f, Threshold = %0.4f',EER,EERThreshold))
xlabel('Threshold')
ylabel('Error Rate')
legend('False Acceptance Rate (FAR)','False Rejection Rate (FRR)','Equal Error Rate (EER)','Location','southwest')
grid on
axis([thresholds(find(FAR~=1,1)) thresholds(find(FRR==1,1)) 0 1])