"""Independent XHand synthetic rollout environment."""

from .scene_environment import FingerEyeSyntheLabEnv


class FingerEyeXHandSyntheLabEnv(FingerEyeSyntheLabEnv):
    """Named subclass so XHand registration never changes the LEAP task."""

    pass
