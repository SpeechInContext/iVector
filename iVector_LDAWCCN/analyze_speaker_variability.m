%%% ---------------- Load trained models ------------------------------ %%%
num_gaussians = 1024;
ivector_dim = 400;
proj_dim = 200;
gender = 'F';
file_end = join(['_' num2str(num_gaussians) '_' num2str(ivector_dim) '_' num2str(proj_dim) ...
                '_' gender '.mat'], '');
ubm_file = join(['../Models/ubm' file_end], '');
projm_file = join(['../Models/projm' file_end], '');
speaker_variability_file = join(['../Models/speaker_variability' file_end], '');

load(ubm_file);
load(projm_file);
%%% ------ Collect ivectors for each utterance for each speaker ------- %%% 
if exist(speaker_variability_file, 'file')
    load(speaker_variability_file);
else
    mfccFolder = 'C:/Users/mFry2/Desktop/SpeeCon/Data/SpiCE/MFCC/*.mfcc';
    speaker_ivectors = extract_speakers_ivectors(mfccFolder, ubm, tvm, projMatrix);
    save(speaker_variability_file, 'speaker_ivectors');
end
   
    %[ivectors, speaker] = expand_ivectors(speaker_ivectors);

%%% ---- Calculate average cosine distance ------------------------ %%%
for speakerIdx = 1:size(speaker_ivectors,1)
    [curr_csd, curr_csd_m] = average_cosine_dist(speaker_ivectors{speakerIdx,2}');
    speaker_ivectors(speakerIdx,3) = {curr_csd};
    speaker_ivectors(speakerIdx,4) = {curr_csd_m};
end

for plotIdx = 1:4
    subplot(4,1,plotIdx)
    histogram(speaker_ivectors{plotIdx,4});
    title(speaker_ivectors{plotIdx,1});
    xlim([0.2,1.4]);
    xlabel('Cosine distance (i-vector pairwise comparison)')
    ylabel('Frequency')
end

%%% -------- Analyze ivectors per speaker for dispersion -------------- %%%
%eva1 = evalclusters(ivectors,speaker,'CalinskiHarabasz');
%eva2 = evalclusters(ivectors,speaker,'DaviesBouldin');


function [ivectors, speaker] = expand_ivectors(speakers_ivectors)
    ivectors = cat(2, speakers_ivectors{:,2})';
    speaker = zeros(size(ivectors,2), 1);
    curr_idx = 1;
    for speakerIdx = 1:size(speakers_ivectors,1)
        numUtt = size(speakers_ivectors{speakerIdx,2},2);
        speaker(curr_idx:curr_idx+numUtt-1) = speakerIdx;
        curr_idx = curr_idx+numUtt;
    end
end

function [avg_csd, avg_csd_m] = average_cosine_dist(m1)
    avg_csd_m = zeros(size(m1,1)^2, 1);
    idx = 1;
    for idx1 = 1:size(m1,1)
        for idx2 = 1:size(m1,1)
            avg_csd_m(idx) = cosine_dist(m1(idx1,:), m1(idx2,:));
            idx = idx + 1;
        end
    end
    avg_csd = mean(avg_csd_m);
end
function csd = cosine_dist(v1, v2)
    csd = 1 - dot(v1,v2)/norm(v1)/norm(v2);
end