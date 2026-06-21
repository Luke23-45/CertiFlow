"""CertiQ index architecture - model, interaction, and geometry."""

from certiq_net.dispatcher.certiq.geometry import CertifiedGeometry
from certiq_net.dispatcher.certiq.index_model import CertiQIndexModel, MarginalIndexHead
from certiq_net.dispatcher.certiq.interaction import (
    DispatchInteractionEncoder,
    index_token_features,
)

__all__ = [
    "CertiQIndexModel",
    "DispatchInteractionEncoder",
    "MarginalIndexHead",
    "CertifiedGeometry",
    "index_token_features",
]
