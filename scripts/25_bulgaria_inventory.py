"""Count the Bulgarian ground records and list which UAV mosaics can match them.

No reflectance is extracted here. A campaign is marked matched only when the
corresponding Zenodo mosaic zip is present and larger than a placeholder page.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "external" / "bulgaria"
UAV = {
    "March_2017": ["S16-17_U1-3_2017-03-21.zip", "S16-17_U2_2017-03-21.zip", "S16-17_U4-5-6_2017-03-22.zip"],
    "April_2017": ["S16-17_U1-3_2017-04-25.zip", "S16-17_U2_2017-04-25.zip", "S16-17_U4-5-6_2017-04-26.zip"],
    "May_2017": ["S16-17_U1-3_2017-05-18.zip", "S16-17_U2_2017-05-18.zip", "S16-17_U4-5-6_2017-05-15.zip"],
    "November_2016": ["S16-17_U1-3_2016-11-07.zip", "S16-17_U2_2016-11-11.zip", "S16-17_U4-5-6_2016-11-10.zip"],
    "November_2017": ["S17-18_U4-5-6_2017-11-10.zip"],
    "April_2018": ["S17-18_U1_2018-04-03.zip", "S17-18_U2_2018-04-04.zip", "S17-18_U3_2018-04-04.zip", "S17-18_U4-5-6_2018-04-03.zip"],
}


def _real(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100_000:
        return False
    with path.open("rb") as handle:
        return not handle.read(10).lstrip().startswith(b"<")


def main() -> None:
    bio_path = TABLE / "ESU_Biophysical_and_biochemical_crop_data.txt"
    bio = pd.read_csv(bio_path, sep="\t")
    rows = []
    for campaign, part in bio.groupby("Campaign"):
        cc = pd.to_numeric(part["CC_mean"], errors="coerce")
        lai = pd.to_numeric(part["LAI_mean"], errors="coerce")
        files = UAV.get(str(campaign), [])
        present = [name for name in files if _real(TABLE / "uav" / name)]
        rows.append({
            "campaign": str(campaign),
            "n_esu": int(len(part)),
            "n_chlorophyll": int(cc.notna().sum()),
            "n_lai": int(lai.notna().sum()),
            "chlorophyll_mean": None if cc.notna().sum() == 0 else float(cc.mean()),
            "lai_mean": None if lai.notna().sum() == 0 else float(lai.mean()),
            "uav_zips_expected": files,
            "uav_zips_present": present,
        })
    payload = {
        "source": "10.5281/zenodo.17475742",
        "n_rows": int(len(bio)),
        "campaigns": rows,
        "field_spectra_present": (TABLE / "ESU_Field_spectral_data.txt").exists() and (TABLE / "ESU_Field_spectral_data.txt").stat().st_size > 100_000,
        "meteo_present": (TABLE / "Meteo_data.txt").exists() and (TABLE / "Meteo_data.txt").stat().st_size > 100_000,
        "note": "Chlorophyll in this table is CCM-300 canopy or leaf chlorophyll as released, not the Gaocheng SPAD conversion. UAV reflectance is not used until the mosaic zip is on disk.",
    }
    dest = ROOT / "outputs" / "metrics" / "bulgaria_inventory.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(dest)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
