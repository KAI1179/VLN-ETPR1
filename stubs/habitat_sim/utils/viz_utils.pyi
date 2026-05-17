from PIL import Image
from __future__ import annotations
import base64 as base64
from functools import partial
import imageio as imageio
import io as io
import numpy as np
import numpy
import os as os
import subprocess as subprocess
import sys as sys
from tqdm.asyncio import tqdm_asyncio as tqdm
__all__: list[str] = ['Image', 'base64', 'border_frames_from_overlay', 'd3_40_colors_rgb', 'depth_to_rgb', 'display_video', 'get_fast_video_writer', 'imageio', 'io', 'is_notebook', 'make_video', 'make_video_frame', 'np', 'observation_to_image', 'os', 'partial', 'save_video', 'semantic_to_rgb', 'subprocess', 'sys', 'tqdm']
def border_frames_from_overlay(overlay_settings, observation_to_image = observation_to_image):
    ...
def depth_to_rgb(depth_image: numpy.ndarray, clip_max: float = 10.0) -> numpy.ndarray:
    """
    Normalize depth image into [0, 1] and convert to grayscale rgb
    
        :param depth_image: Raw depth observation image from sensor output.
        :param clip_max: Max depth distance for clipping and normalization.
    
        :return: Clipped grayscale depth image data.
        
    """
def display_video(video_file: str, height: int = 400):
    """
    Displays a video both locally and in a notebook. Will display the video
        as an HTML5 video if in a notebook, otherwise it opens the video file using
        the default system viewer.
    
        :param video_file: the filename of the video to display
        :param height: the height to display the video in a notebook.
        
    """
def get_fast_video_writer(video_file: str, fps: int = 60):
    ...
def is_notebook() -> bool:
    """
    This utility function detects if the code is running in a notebook
    """
def make_video(observations: typing.List[numpy.ndarray], primary_obs: str, primary_obs_type: str, video_file: str, fps: int = 60, open_vid: bool = True, video_dims: typing.Union[typing.Tuple[int], NoneType] = None, overlay_settings: typing.Union[typing.List[typing.Dict[str, typing.Any]], NoneType] = None, depth_clip: typing.Union[float, NoneType] = 10.0, observation_to_image = observation_to_image):
    """
    Build a video from a passed observations array, with some images optionally overlayed.
        :param observations: List of observations from which the video should be constructed.
        :param primary_obs: Sensor name in observations to be used for primary video images.
        :param primary_obs_type: Primary image observation type ("color", "depth", "semantic" supported).
        :param video_file: File to save resultant .mp4 video.
        :param fps: Desired video frames per second.
        :param open_vid: Whether or not to open video upon creation.
        :param video_dims: Height by Width of video if different than observation dimensions. Applied after overlays.
        :param overlay_settings: List of settings Dicts, optional.
        :param depth_clip: Defines default depth clip normalization for all depth images.
        :param observation_to_image: Allows overriding the observation_to_image function
        With **overlay_settings** dicts specifying per-entry: 
    
            "type": observation type ("color", "depth", "semantic" supported)
    
            "dims": overlay dimensions (Tuple : (width, height))
    
            "pos": overlay position (top left) (Tuple : (width, height))
    
            "border": overlay image border thickness (int)
    
            "border_color": overlay image border color [0-255] (3d: array, list, or tuple). Defaults to gray [150]
    
            "obs": observation key (string)
    
        
    """
def make_video_frame(ob, primary_obs: str, primary_obs_type: str, video_dims, overlay_settings = None, observation_to_image = observation_to_image):
    ...
def observation_to_image(observation_image: numpy.ndarray, observation_type: str, depth_clip: typing.Union[float, NoneType] = 10.0):
    """
    Generate an rgb image from a sensor observation. Supported types are: "color", "depth", "semantic"
    
        :param observation_image: Raw observation image from sensor output.
        :param observation_type: Observation type ("color", "depth", "semantic" supported)
        :param depth_clip: Defines default depth clip normalization for all depth images.
    
        :return: PIL Image object or None if fails.
        
    """
def save_video(video_file: str, frames, fps: int = 60):
    """
    Saves the video using imageio. Will try to use GPU hardware encoding on
        Google Colab for faster video encoding. Will also display a progressbar.
    
        :param video_file: the file name of where to save the video
        :param frames: the actual frame objects to save
        :param fps: the fps of the video (default 60)
        
    """
def semantic_to_rgb(semantic_image: numpy.ndarray) -> numpy.ndarray:
    """
    Map semantic ids to colors and genereate an rgb image
    
        :param semantic_image: Raw semantic observation image from sensor output.
    
        :return: rgb semantic image data.
        
    """
d3_40_colors_rgb: numpy.ndarray  # value = array([[ 31, 119, 180],...
