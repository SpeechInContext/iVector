function [speaker_ivector_models, speaker_model_size, verify_ivectors, verify_labels] = separate_enrol_verify(enrol_verify_ivectors, proportion)
    unique_speakers = unique(enrol_verify_ivectors(:,2));
    speaker_ivector_models = [];
    speaker_model_size = [];
    verify_ivectors = [];
    verify_labels = [];
    for speakerIdx = 1:size(unique_speakers,1)
        speaker_idxs = string(enrol_verify_ivectors(:,2)) == unique_speakers(speakerIdx);
        speaker_ivectors = enrol_verify_ivectors(speaker_idxs, 1);
        num_for_model = floor(proportion*size(speaker_ivectors,1));
        model_ivectors = cat(2,speaker_ivectors{1:num_for_model, 1});
        speaker_ivector_models = [speaker_ivector_models mean(model_ivectors,2)]; %#ok
        speaker_model_size = [speaker_model_size num_for_model]; %#ok
        other_ivectors = cat(2,speaker_ivectors{num_for_model+1:end, 1});
        verify_ivectors = [verify_ivectors other_ivectors]; %#ok
        verify_labels = [verify_labels repmat(unique_speakers(speakerIdx), 1, size(other_ivectors,2))]; %#ok
    end
end

