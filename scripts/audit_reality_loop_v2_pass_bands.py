from __future__ import annotations

import audit_reality_loop_v2_play_gain as audit

from monster.sim import resolution_ecology
from monster.sim.pass_resolution_bands_v2 import sample_yac_v2


def main() -> None:
    """Run the full Reality Loop audit with shallow-completion gain bands in shadow."""
    resolution_ecology.sample_yac = sample_yac_v2
    audit.main()


if __name__ == "__main__":
    main()
