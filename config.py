from singleton_decorator import singleton


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class GeometryConfig(Config):
    tvel_r = 0.91
    tvel_dist = 2.2
    cladding_thick = 0.1
    lat_type = 'hex'
    high = 380

    @property
    def pitch(self):
        if self.lat_type == 'hex':
            return self.tvel_dist * 3 ** 0.5 / 2
        elif self.lat_type == 'sqr':
            return self.tvel_dist
        else:
            raise ValueError(f'Lattice type {self.lat_type} not supported')


class MaterialConfig(Config):
    fuel_enr = 5
    fuel_density = 8.3
    coolant_density = 1
    cladding_density = 2


class UnitConfig(Config):
    time = 's'
    heat = 'W'
    atoms = 'atoms'
