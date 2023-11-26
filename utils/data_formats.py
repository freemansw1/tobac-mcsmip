import datetime
import pathlib
from dataclasses import dataclass


@dataclass
class MCS_dataset:
    """Class for keeping track of an item in inventory."""

    name: str
    time_dim: str
    precip_var: str
    bt_var: str
    convert_olr: bool
    file_prefix: str
    has_summer: bool
    has_winter: bool

    def glob_date(
        self, path: pathlib.Path, season: str, date: datetime
    ) -> list[pathlib.Path]:
        if season == "summer":
            if not self.has_summer:
                raise ValueError(f"Summer not available for model {self.name}")
            glob_str = (
                f"Summer/{self.name}/{self.file_prefix}*{date.strftime('%Y%m%d')}*.nc"
            )
        elif season == "winter":
            if not self.has_winter:
                raise ValueError(f"Winter not available for model {self.name}")
            glob_str = (
                f"Winter/{self.name}/{self.file_prefix}*{date.strftime('%Y%m%d')}*.nc"
            )
        else:
            raise ValueError("Season must be one of ['summer', 'winter']")

        
        return sorted(list(path.glob(glob_str)))


ARPEGE_summer = MCS_dataset(
    name="ARPEGE",
    time_dim="time",
    precip_var="param8.1.0",
    bt_var="ttr",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_arpnh_summer_",
    has_summer=True,
    has_winter=False,
)

ARPEGE_winter = MCS_dataset(
    name="ARPEGE",
    time_dim="time",
    precip_var="pr",
    bt_var="rlt",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_arpnh_winter_",
    has_summer=False,
    has_winter=True,
)

FV3 = MCS_dataset(
    name="FV3",
    time_dim="time",
    precip_var="pr",
    bt_var="flut",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_fv3_",
    has_summer=True,
    has_winter=False,
)

GEOS = MCS_dataset(
    name="GEOS",
    time_dim="time",
    precip_var="pr",
    bt_var="rlut",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_geos_winter_",
    has_summer=False,
    has_winter=True,
)

GRIST = MCS_dataset(
    name="GRIST",
    time_dim="time",
    precip_var="pr",
    bt_var="rlt",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_grist_",
    has_summer=False,
    has_winter=True,
)

ICON = MCS_dataset(
    name="ICON",
    time_dim="time",
    precip_var="pr",
    bt_var="rlut",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_icon_",
    has_summer=False,
    has_winter=True,
)

IFS_summer = MCS_dataset(
    name="IFS",
    time_dim="time",
    precip_var="tp",
    bt_var="ttr",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_ifs_summer_",
    has_summer=True,
    has_winter=False,
)

IFS_winter = MCS_dataset(
    name="IFS",
    time_dim="time",
    precip_var="pracc",
    bt_var="rltacc",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_ecmwf_",
    has_summer=False,
    has_winter=True,
)

MPAS_summer = MCS_dataset(
    name="MPAS",
    time_dim="xtime",
    precip_var="pr",
    bt_var="olrtoa",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_mpas_",
    has_summer=True,
    has_winter=False,
)

MPAS_winter = MCS_dataset(
    name="MPAS",
    time_dim="xtime",
    precip_var="pr",
    bt_var="rltacc",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_mpas_winter_",
    has_summer=False,
    has_winter=True,
)

NICAM = MCS_dataset(
    name="NICAM",
    time_dim="time",
    precip_var="sa_tppn",
    bt_var="sa_lwu_toa",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_nicam_summer_",
    has_summer=True,
    has_winter=False,
)

OBS = MCS_dataset(
    name="OBS",
    time_dim="time",
    precip_var="precipitationCal",
    bt_var="Tb",
    convert_olr=False,
    file_prefix="olr_pcp/merg_",
    has_summer=True,
    has_winter=True,
)

SAM_summer = MCS_dataset(
    name="SAM",
    time_dim="time",
    precip_var="Precac",
    bt_var="LWNTA",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_sam_summer_",
    has_summer=True,
    has_winter=False,
)

SAM_winter = MCS_dataset(
    name="SAM",
    time_dim="time",
    precip_var="pracc",
    bt_var="rltacc",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_sam_winter_",
    has_summer=False,
    has_winter=True,
)

SCREAM = MCS_dataset(
    name="SCREAM",
    time_dim="time",
    precip_var="pr",
    bt_var="rlt",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_scream_",
    has_summer=False,
    has_winter=True,
)

UM_summer = MCS_dataset(
    name="UM",
    time_dim="time",
    precip_var="precipitation_flux",
    bt_var="toa_outgoing_longwave_flux",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_um_summer_",
    has_summer=True,
    has_winter=False,
)

UM_winter = MCS_dataset(
    name="UM",
    time_dim="time",
    precip_var="pr",
    bt_var="rlut",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_um_winter_",
    has_summer=False,
    has_winter=True,
)

XSHIELD = MCS_dataset(
    name="XSHiELD",
    time_dim="time",
    precip_var="pr",
    bt_var="rlut",
    convert_olr=True,
    file_prefix="olr_pcp_instantaneous/pr_rlut_SHiELD-3km_",
    has_summer=False,
    has_winter=True,
)

all_datasets = [
    ARPEGE_summer,
    ARPEGE_winter,
    FV3,
    GEOS,
    GRIST,
    ICON,
    IFS_summer,
    IFS_winter,
    MPAS_summer,
    MPAS_winter,
    NICAM,
    OBS,
    SAM_summer,
    SAM_winter,
    SCREAM,
    UM_summer,
    UM_winter,
    XSHIELD,
]

summer_datasets = [dataset for dataset in all_datasets if dataset.has_summer]

winter_datasets = [dataset for dataset in all_datasets if dataset.has_winter]
