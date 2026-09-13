from uuid import uuid4

import pytest

from sandbox.research.errors import JournalIntegrityError
from test_stage3_registry import setup_registry


def test_hash_chain_verifies_and_append_preserves_integrity():
    reg,_,_=setup_registry(); assert reg.verify_journal_integrity()
    reg.append_journal("TEST_EVENT","TEST_PROGRAM","append test",metadata={"b":2,"a":1})
    assert reg.verify_journal_integrity()


def test_historical_tampering_is_detected():
    reg,_,_=setup_registry()
    with reg.catalog.connection() as con: con.execute("UPDATE research_journal SET message='tampered' WHERE sequence=(SELECT MIN(sequence) FROM research_journal)")
    with pytest.raises(JournalIntegrityError): reg.verify_journal_integrity()

