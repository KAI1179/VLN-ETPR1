from __future__ import annotations
import habitat_sim.registry
from habitat_sim.sensors.noise_models.gaussian_noise_model import GaussianNoiseModel
from habitat_sim.sensors.noise_models.no_noise_model import NoSensorNoiseModel
from habitat_sim.sensors.noise_models.poisson_noise_model import PoissonNoiseModel
from habitat_sim.sensors.noise_models.redwood_depth_noise_model import RedwoodDepthNoiseModel
from habitat_sim.sensors.noise_models.salt_and_pepper_noise_model import SaltAndPepperNoiseModel
from habitat_sim.sensors.noise_models.sensor_noise_model import SensorNoiseModel
from habitat_sim.sensors.noise_models.speckle_noise_model import SpeckleNoiseModel
from . import gaussian_noise_model
from . import no_noise_model
from . import poisson_noise_model
from . import redwood_depth_noise_model
from . import salt_and_pepper_noise_model
from . import sensor_noise_model
from . import speckle_noise_model
__all__: list = ['make_sensor_noise_model', 'SensorNoiseModel', 'RedwoodDepthNoiseModel', 'NoSensorNoiseModel', 'GaussianNoiseModel', 'SaltAndPepperNoiseModel', 'PoissonNoiseModel', 'SpeckleNoiseModel']
def make_sensor_noise_model(name: str, kwargs: typing.Dict[str, typing.Any]) -> sensor_noise_model.SensorNoiseModel:
    """
    Constructs a noise model using the given name and keyword arguments
    
        :param name: The name of the noise model in the `habitat_sim.registry`
        :param kwargs: The keyword arguments to be passed to the constructor of the noise model
        
    """
registry: habitat_sim.registry._Registry  # value = <habitat_sim.registry._Registry object>
