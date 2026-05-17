"""
Data format exchange
"""
from __future__ import annotations
import _magnum
import corrade.containers
import corrade.pluginmanager
import typing
__all__: list[str] = ['AbstractImporter', 'ImageData1D', 'ImageData2D', 'ImageData3D', 'ImporterManager', 'MeshData']
class AbstractImporter:
    """
    Interface for importer plugins
    """
    def close(self) -> None:
        """
        Close currently opened file
        """
    def image1d(self, id: int, level: int = 0) -> ImageData1D:
        """
        One-dimensional image
        """
    def image1d_for_name(self, arg0: str) -> int:
        """
        One-dimensional image ID for given name
        """
    def image1d_level_count(self, id: int) -> int:
        """
        One-dimensional image level count
        """
    def image1d_name(self, id: int) -> str:
        """
        One-dimensional image name
        """
    def image2d(self, id: int, level: int = 0) -> ImageData2D:
        """
        Two-dimensional image
        """
    def image2d_for_name(self, arg0: str) -> int:
        """
        Two-dimensional image ID for given name
        """
    def image2d_level_count(self, id: int) -> int:
        """
        Two-dimensional image level count
        """
    def image2d_name(self, id: int) -> str:
        """
        Two-dimensional image name
        """
    def image3d(self, id: int, level: int = 0) -> ImageData3D:
        """
        Three-dimensional image
        """
    def image3d_for_name(self, arg0: str) -> int:
        """
        Three-dimensional image ID for given name
        """
    def image3d_level_count(self, id: int) -> int:
        """
        Three-dimensional image level count
        """
    def image3d_name(self, id: int) -> str:
        """
        Three-dimensional image name
        """
    def mesh(self, id: int, level: int = 0) -> MeshData:
        """
        Mesh
        """
    def mesh_for_name(self, arg0: str) -> int:
        """
        Mesh ID for given name
        """
    def mesh_level_count(self, id: int) -> int:
        """
        Mesh level count
        """
    def mesh_name(self, id: int) -> str:
        """
        Mesh name
        """
    def open_data(self, data: corrade.containers.ArrayView) -> None:
        """
        Open raw data
        """
    def open_file(self, filename: str) -> None:
        """
        Open a file
        """
    @property
    def image1d_count(self) -> int:
        """
        One-dimensional image count
        """
    @property
    def image2d_count(self) -> int:
        """
        Two-dimensional image count
        """
    @property
    def image3d_count(self) -> int:
        """
        Three-dimensional image count
        """
    @property
    def is_opened(self) -> bool:
        """
        Whether any file is opened
        """
    @property
    def manager(self) -> typing.Any:
        """
        Manager owning this plugin instance
        """
    @property
    def mesh_count(self) -> int:
        """
        Mesh count
        """
class ImageData1D:
    """
    One-dimensional image data
    """
    @property
    def data(self) -> corrade.containers.ArrayView:
        """
        Image data
        """
    @property
    def format(self) -> _magnum.PixelFormat:
        """
        Format of pixel data
        """
    @property
    def is_compressed(self) -> bool:
        """
        Whether the image is compressed
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.StridedArrayView2D:
        """
        View on pixel data
        """
    @property
    def size(self) -> int:
        """
        Image size
        """
    @property
    def storage(self) -> _magnum.PixelStorage:
        """
        Storage of pixel data
        """
class ImageData2D:
    """
    Two-dimensional image data
    """
    @property
    def data(self) -> corrade.containers.ArrayView:
        """
        Image data
        """
    @property
    def format(self) -> _magnum.PixelFormat:
        """
        Format of pixel data
        """
    @property
    def is_compressed(self) -> bool:
        """
        Whether the image is compressed
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.StridedArrayView3D:
        """
        View on pixel data
        """
    @property
    def size(self) -> _magnum.Vector2i:
        """
        Image size
        """
    @property
    def storage(self) -> _magnum.PixelStorage:
        """
        Storage of pixel data
        """
class ImageData3D:
    """
    Three-dimensional image data
    """
    @property
    def data(self) -> corrade.containers.ArrayView:
        """
        Image data
        """
    @property
    def format(self) -> _magnum.PixelFormat:
        """
        Format of pixel data
        """
    @property
    def is_compressed(self) -> bool:
        """
        Whether the image is compressed
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.StridedArrayView4D:
        """
        View on pixel data
        """
    @property
    def size(self) -> _magnum.Vector3i:
        """
        Image size
        """
    @property
    def storage(self) -> _magnum.PixelStorage:
        """
        Storage of pixel data
        """
class ImporterManager(corrade.pluginmanager.AbstractManager):
    """
    Plugin manager for importer plugins
    """
    def __init__(self, plugin_directory: str = '') -> None:
        """
        Constructor
        """
    def instantiate(self, arg0: str) -> AbstractImporter:
        ...
    def load_and_instantiate(self, arg0: str) -> AbstractImporter:
        ...
class MeshData:
    """
    Mesh data
    """
    @property
    def attribute_count(self) -> int:
        ...
    @property
    def index_count(self) -> int:
        ...
    @property
    def is_indexed(self) -> bool:
        """
        Whether the mesh is indexed
        """
    @property
    def primitive(self) -> _magnum.MeshPrimitive:
        """
        Primitive
        """
    @property
    def vertex_count(self) -> int:
        ...
