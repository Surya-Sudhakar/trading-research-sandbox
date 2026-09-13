class ResearchError(RuntimeError): pass
class ImmutableRecordError(ResearchError): pass
class InvalidTransitionError(ResearchError): pass
class PreRegistrationError(ResearchError): pass
class DuplicateExperimentError(ResearchError):
    def __init__(self, existing_experiment_id: str):
        self.existing_experiment_id = existing_experiment_id
        super().__init__(f"DUPLICATE_EXPERIMENT: existing experiment {existing_experiment_id}")
class DatasetVerificationError(ResearchError): pass
class JournalIntegrityError(ResearchError): pass
