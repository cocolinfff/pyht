import numpy as np


def configure_tables():
    from pyht.fluid import CoolPropWrapper
    CoolPropWrapper.set_table_range(p_lim=(5e4, 3e7), T_lim=(300.0, 1073.15),
                                    n_p=50, n_T=150, fluid="Water")
    CoolPropWrapper.set_table_range(p_lim=(5e5, 1e7), T_lim=(300.0, 1173.15),
                                    n_p=40, n_T=120, fluid="Helium")
