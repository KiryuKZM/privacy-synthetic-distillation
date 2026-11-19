"""Whole-set residual model with the independently verified moment-mixture prior."""
from privacy_synth.spatial_reference_diffusion import ReferenceSetDenoiser
from privacy_synth.spatial_mixture_reference import GaussianMixtureReference


class MixtureReferenceSetDenoiser(ReferenceSetDenoiser):
    def __init__(self, gaussian_parameters, mixture_parameters, prior):
        # Identical trainable initialization/RNG consumption to the Gaussian arm.
        super().__init__(gaussian_parameters, prior)
        replacement = GaussianMixtureReference(mixture_parameters, prior, query_chunk=64)
        if replacement.classes != self.classes:
            raise ValueError('Same registered category vocabulary required')
        self.reference = replacement
