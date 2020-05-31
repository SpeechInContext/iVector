function i_vector = get_i_vector(sig_feats, ubm, tvm)
    logLikelihood = helperGMMLogLikelihood(sig_feats,ubm);
    % Compute a posteriori normalized probability
    amax = max(logLikelihood,[],1);
    logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
    gamma = exp(logLikelihood - logLikelihoodSum)';
    % Compute Baum-Welch statistics
    n = sum(gamma,1);
    f = sig_feats * gamma - n.*(ubm.mu);
    i_vector = pinv(tvm.I + (tvm.TS.*repelem(n(:),ubm.numFeatures))' * tvm.T) * tvm.TSi * f(:);
end

function L = helperGMMLogLikelihood(x,gmm)
    xMinusMu = repmat(x,1,1,numel(gmm.ComponentProportion)) - permute(gmm.mu,[1,3,2]);
    permuteSigma = permute(gmm.sigma,[1,3,2]);

    Lunweighted = -0.5*(sum(log(permuteSigma),1) + sum(xMinusMu.*(xMinusMu./permuteSigma),1) + size(gmm.mu,1)*log(2*pi));

    temp = squeeze(permute(Lunweighted,[1,3,2]));
    if size(temp,1)==1
        % If there is only one frame, the trailing singleton dimension was
        % removed in the permute. This accounts for that edge case.
        temp = temp';
    end

    L = temp + log(gmm.ComponentProportion)';
end