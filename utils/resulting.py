import h5py
from openmc.deplete import ResultsList

from config import UnitConfig


class ResultReader:
    def __init__(self, file_path: str):
        self.file = h5py.File(file_path, 'r')


class DepletionResultReader(ResultReader):
    def __init__(self, file_path: str, fuel_mat: str = 'Fuel', isotopes_list=None):
        super().__init__(file_path)
        self.data = ResultsList.from_hdf5(file_path)
        self.units = UnitConfig()
        self.fuel_mat = fuel_mat
        if isotopes_list is None:
            self.isotopes = [
                "U235",
                "U238"
            ]

    @property
    def timestamps(self):
        return [timestep.time[0] for timestep in self.data]

    def get_k(self):
        _, k = self.data.get_eigenvalue()
        return k

    def get_atoms(self,
                  nuc_units: str = 'atoms',
                  ):
        res = {}
        for iso in self.isotopes:
            _, atoms = self.data.get_atoms(mat=self.fuel_mat, nuc=iso, nuc_units=nuc_units)
            res[iso] = atoms
        return res

    def get_power(self, units: str = 'W'):
        _, heat = self.data.get_decay_heat(
            mat=self.fuel_mat,
            units=units
        )
        return heat

    def prepare_data(self, units: UnitConfig = None):
        if units is None:
            units = self.units
        data = {
            'timestamps': self.timestamps,
            'k_inf': self.get_k(),
            "heat": self.get_power(units=units.heat),
        } | self.get_atoms(nuc_units=units.atoms)

        return data
