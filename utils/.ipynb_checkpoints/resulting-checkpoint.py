import h5py
from openmc.deplete import Results

from config import UnitConfig


class ResultReader:
    def __init__(self, file_path: str):
        self.file = h5py.File(file_path, 'r')


class DepletionResultReader(ResultReader):
    def __init__(self, file_path: str, fuel_mat: str = 'Fuel', isotopes_list=None):
        super().__init__(file_path)
        data = Results(file_path)
        self.data = data
        self.units = UnitConfig()
        self.fuel_mat = fuel_mat
        if isotopes_list is None:
            smp = data[-1]
            isotopes = [ x.name for x in smp.get_material('2').nuclides]
            isotopes_list = isotopes
        self.isotopes = isotopes_list


    @property
    def timestamps(self):
        return [timestep.time[0] for timestep in self.data]

    def get_k(self):
        _, k = self.data.get_keff()
        return k[:, 0]

    def get_atoms(self,
                  nuc_units: str = 'atoms',
                  filter_ = True
                  ):
        res = {}
        for iso in self.isotopes:
            _, atoms = self.data.get_atoms(mat=self.fuel_mat, nuc=iso, nuc_units=nuc_units)
            if (filter_ and float(atoms.max() - atoms.min()) != 0 and atoms.max()>1e10) or not filter_:
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
