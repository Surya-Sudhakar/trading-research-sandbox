"""Provider-neutral, immutable research market data."""
from .models import CanonicalMarketBar, ProviderMetadata, DatasetKind
from .sanitation import sanitize
from .gaps import SessionProfile, ContinuityPolicy

