from __future__ import annotations

import json
from pathlib import Path

from sandbox.audit.models import AuditClassification,ResearchAudit


def write_audit_report(audit:ResearchAudit,root:Path)->tuple[Path,Path]:
    folder=root/"audits"/audit.experiment_id;folder.mkdir(parents=True,exist_ok=True)
    json_path=folder/f"{audit.audit_id}.json";md_path=folder/f"{audit.audit_id}.md"
    serialized=json.dumps(audit.model_dump(mode="json"),indent=2,sort_keys=True)
    json_path.write_text(serialized,encoding="utf-8")
    execution="BLOCKED" if audit.overall_classification==AuditClassification.BLOCKED else "OVERRIDE REQUIRED" if audit.overall_classification==AuditClassification.HIGH_RISK else "MAY PROCEED"
    lines=["# RESEARCH AUDIT",f"## {audit.experiment_id}","",f"- Audit: `{audit.audit_id}`",f"- Phase: {audit.phase.value}",f"- Classification: **{audit.overall_classification.value}**",f"- Risk score: {audit.risk_score} / 100",f"- Auditor: {audit.auditor_version}",f"- Config fingerprint: `{audit.config_fingerprint}`",f"- Execution: **{execution}**","","## Findings",""]
    for finding in audit.findings:
        lines.extend([f"### [{finding.severity.value}] {finding.code}","",finding.explanation,"",f"Evidence: `{json.dumps(finding.evidence,sort_keys=True)}`","",f"Remediation: {finding.remediation}",""])
    lines.extend(["## Hard blocks","",*(f"- {x}" for x in audit.hard_block_reasons)] if audit.hard_block_reasons else ["## Hard blocks","","None"])
    md_path.write_text("\n".join(lines)+"\n",encoding="utf-8");return json_path,md_path
