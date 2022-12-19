from geometry import Simplex, Pentagon, Cylinder, Cone, Tetrahedron

from easydict import EasyDict as edict
from mindelec.data import Dataset
from mindelec.geometry import Rectangle, Disk, create_config_from_edict


shape_factory = {
    "rectangle": Rectangle,
    "disk": Disk,
    "triangle": Simplex,
    "pentagon": Pentagon,
    "cylinder": Cylinder,
    "cone": Cone,
    "tetrahedron": Tetrahedron,
}


def create_dataset(n_samps_domain, n_samps_bc, batch_size, shape_name, shape_kwargs):
    # Craete dataset
    config = edict({
        'domain': edict({
            'random_sampling': True,
            'size': n_samps_domain,
            'sampler': 'uniform',
        }),
        'BC': edict({
            'random_sampling': True,
            'size': n_samps_bc,
            'sampler': 'uniform',
        })
    })
    sampling_config = create_config_from_edict(config)
    region = shape_factory[shape_name](shape_name, **shape_kwargs, sampling_config=sampling_config)
    return Dataset({region: ['domain', 'BC']}).create_dataset(
        batch_size=batch_size, shuffle=True, prebatched_data=True, drop_remainder=True
    )

