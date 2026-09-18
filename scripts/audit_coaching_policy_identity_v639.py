from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_coaching_policy_identity_v635 as v635_audit
from runtime_v639_composer import (
    build_week1_runtime_inputs_v639,
    write_runtime_fingerprint_v639,
)


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    # Keep audit mathematics identical to v6.3.5 and swap only the runtime builder.
    v635_audit.build_week1_runtime_inputs_v635 = build_week1_runtime_inputs_v639
    v635_audit.write_runtime_fingerprint_v635 = write_runtime_fingerprint_v639
    v635_audit.main()

    out = _arg_path("--out")
    renames = (
        (
            "v635_coaching_policy_identity_matrix.csv",
            "v639_coaching_policy_identity_matrix.csv",
        ),
        (
            "v635_coaching_policy_identity_summary.csv",
            "v639_coaching_policy_identity_summary.csv",
        ),
    )
    for old_name, new_name in renames:
        old_path = out / old_name
        if old_path.exists():
            old_path.replace(out / new_name)

    old_report = out / "v635_coaching_policy_identity.json"
    new_report = out / "v639_coaching_policy_identity.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v6.3.9-coaching-policy-identity"
    payload["audit_logic"] = "exact v6.3.5 coaching audit, v6.3.9 runtime injected"
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
