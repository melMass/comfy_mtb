import os

import numpy as np
import open3d as o3d

from ..utils import (
    create_box,
    get_transformation_matrix,
    log,
    spread_geo,
    tensor2b64,
)

# create_grid,
# create_sphere,
# create_torus,
# mesh_to_json,
# json_to_mesh
# rotate_mesh,
# euler_to_rotation_matrix,


# class GeoPrimitive:
#     """Primitive 3D geometry"""

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "kind": (["Box", "Sphere", "Cylinder", "Torus"], {"default": "Box"})
#             }
#         }

#     RETURN_TYPES = ("UV_MAP",)
#     RETURN_NAMES = ("uv_map",)
#     FUNCTION = "distort_uvs"
#     CATEGORY = "mtb/uv"


def default_material(color=None):
    return {
        "color": color or "#00ff00",
        "roughness": 1.0,
        "metalness": 0.0,
        "emissive": "#000000",
        "displacementScale": 1.0,
        "displacementMap": None,
    }


class MTB_Camera:
    """Make a Camera."""

    @classmethod
    def INPUT_TYPES(cls):
        base = default_material()
        return {
            "required": {
                "color": ("COLOR", {"default": base["color"]}),
                "roughness": (
                    "FLOAT",
                    {
                        "default": base["roughness"],
                        "min": 0.005,
                        "max": 4.0,
                        "step": 0.01,
                    },
                ),
                "flatShading": ("BOOLEAN",),
                "metalness": (
                    "FLOAT",
                    {
                        "default": base["metalness"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "emissive": ("COLOR", {"default": base["emissive"]}),
                "displacementScale": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0},
                ),
            },
            "optional": {"displacementMap": ("IMAGE",)},
        }

    RETURN_TYPES = ("CAMERA",)
    RETURN_NAMES = ("camera",)
    FUNCTION = "make_camera"
    CATEGORY = "mtb/3D"

    def make_camera(self, **kwargs):
        return (kwargs,)


class MTB_GeometryDraw:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "geometry": ("GEOMETRY",),
            },
            "optional": {
                "camera": ("CAMERA",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_image",)
    FUNCTION = "render"
    CATEGORY = "mtb/3D"

    def render(self, geometry, camera):
        mesh, material = spread_geo(geometry)
        o3d.visualization.draw_geometries([mesh], **camera)


# class MTB_RGBD_Image:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "depth": ("IMAGE",),
#             }
#         }

#     RETURN_TYPES = ("RGBD_IMAGE",)
#     RETURN_NAMES = ("rgbd",)
#     FUNCTION = "make_rgbd"
#     CATEGORY = "mtb/3D"

#     def make_rgbd(self, image, depth):
#         color_raw = o3d.io.read_image("../../test_data/RGBD/color/00000.jpg")
#         depth_raw = o3d.io.read_image("../../test_data/RGBD/depth/00000.png")
#         rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             color_raw, depth_raw
#         )
#         print(rgbd_image)


class MTB_GeometryMaterial:
    """Make a std material."""

    @classmethod
    def INPUT_TYPES(cls):
        base = default_material()
        return {
            "required": {
                "color": ("COLOR", {"default": base["color"]}),
                "roughness": (
                    "FLOAT",
                    {
                        "default": base["roughness"],
                        "min": 0.005,
                        "max": 4.0,
                        "step": 0.01,
                    },
                ),
                "flatShading": ("BOOLEAN",),
                "metalness": (
                    "FLOAT",
                    {
                        "default": base["metalness"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "emissive": ("COLOR", {"default": base["emissive"]}),
                "displacementScale": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0},
                ),
            },
            "optional": {"displacementMap": ("IMAGE",)},
        }

    RETURN_TYPES = ("GEO_MATERIAL",)
    RETURN_NAMES = ("material",)
    FUNCTION = "make_material"
    CATEGORY = "mtb/3D"

    def make_material(
        self, **kwargs
    ):  # color, roughness, metalness, emissive, displacementScalen displacementMap=None):
        # TODO: convert image to b64 and remove the key/add the B64 one
        # TODO: we can just use the "wireframe" property instead of my current solution
        if kwargs.get("displacementMap") is not None:
            tens = kwargs.pop("displacementMap")
            # TODO: alert about batch size > 1 ?
            b64images = tensor2b64(tens)[0]
            kwargs["displacementB64"] = b64images

        return (kwargs,)


class MTB_GeometryApplyMaterial:
    """Apply a Material to a geometry."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "geometry": ("GEOMETRY",),
                "color": ("COLOR", {"default": "#000000"}),
            },
            "optional": {"material": ("GEO_MATERIAL",)},
        }

    RETURN_TYPES = ("GEOMETRY",)
    RETURN_NAMES = ("geometry",)
    FUNCTION = "apply"
    CATEGORY = "mtb/3D"

    def apply(
        self,
        geometry,
        color,
        material=None,
    ):
        if material is None:
            material = default_material(color)
        #
        geometry["material"] = material

        return (geometry,)


class MTB_GeometryTransform:
    """Transforms the input geometry."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("GEOMETRY",),
                "position_x": (
                    "FLOAT",
                    {"default": 0.0, "step": 0.1, "min": -10000, "max": 10000},
                ),
                "position_y": (
                    "FLOAT",
                    {"default": 0.0, "step": 0.1, "min": -10000, "max": 10000},
                ),
                "position_z": (
                    "FLOAT",
                    {"default": 0.0, "step": 0.1, "min": -10000, "max": 10000},
                ),
                "rotation_x": (
                    "FLOAT",
                    {"default": 0.0, "step": 1, "min": -10000, "max": 10000},
                ),
                "rotation_y": (
                    "FLOAT",
                    {"default": 0.0, "step": 1, "min": -10000, "max": 10000},
                ),
                "rotation_z": (
                    "FLOAT",
                    {"default": 0.0, "step": 1, "min": -10000, "max": 10000},
                ),
                "scale_x": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "scale_y": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "scale_z": ("FLOAT", {"default": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("GEOMETRY",)
    RETURN_NAMES = ("geometry",)
    FUNCTION = "transform_geometry"
    CATEGORY = "mtb/3D"

    def transform_geometry(
        self,
        mesh: o3d.geometry.TriangleMesh,
        position_x=0.0,
        position_y=0.0,
        position_z=0.0,
        rotation_x=0,
        rotation_y=0,
        rotation_z=0,
        scale_x=1,
        scale_y=1,
        scale_z=1,
    ):
        # mesh = o3d.geometry.TriangleMesh.create_box(
        #     width,
        #     height,
        #     depth,

        # )
        # mesh.compute_vertex_normals()

        position = np.array([position_x, position_y, position_z])
        rotation = (rotation_x, rotation_y, rotation_z)
        scale = np.array([scale_x, scale_y, scale_z])

        transformation_matrix = get_transformation_matrix(
            position, rotation, scale
        )
        mesh, material = spread_geo(mesh, cp=True)

        return (
            {
                "mesh": mesh.transform(transformation_matrix),
                "material": material,
            },
        )


class MTB_GeometrySphere:
    """Makes a Sphere 3D geometry.."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "create_uv_map": ("BOOLEAN", {"default": True}),
                "radius": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "resolution": ("INT", {"default": 20, "min": 1}),
            }
        }

    RETURN_TYPES = ("GEOMETRY",)
    RETURN_NAMES = ("geometry",)
    FUNCTION = "make_sphere"
    CATEGORY = "mtb/3D"

    def make_sphere(self, create_uv_map, radius, resolution):
        mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius,
            resolution,
            create_uv_map,
        )
        mesh.compute_vertex_normals()

        return ({"mesh": mesh},)


class MTB_GeometryTest:
    """Fetches an Open3D data geometry.."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": (
                    [
                        "ArmadilloMesh",
                        "AvocadoModel",
                        "BunnyMesh",
                        "CrateModel",
                        "DamagedHelmetModel",
                        "FlightHelmetModel",
                        "KnotMesh",
                        "MonkeyModel",
                        "SwordModel",
                    ],
                    {
                        "default": "KnotMesh",
                    },
                )
            }
        }

    RETURN_TYPES = ("GEOMETRY",)
    RETURN_NAMES = ("geometry",)
    FUNCTION = "fetch_data"
    CATEGORY = "mtb/3D"

    def fetch_data(self, name):
        model = getattr(o3d.data, name)()
        mesh = o3d.io.read_triangle_mesh(model.path)
        mesh.compute_vertex_normals()
        return ({"mesh": mesh},)


class MTB_GeometryBox:
    """Makes a Box 3D geometry."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "create_uv_map": ("BOOLEAN", {"default": True}),
                "uniform_scale": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "width": ("FLOAT", {"default": 1.0, "step": 0.05}),
                "height": ("FLOAT", {"default": 1.0, "step": 0.05}),
                "depth": ("FLOAT", {"default": 1.0, "step": 0.05}),
                "divisions_x": ("INT", {"default": 1}),
                "divisions_y": ("INT", {"default": 1}),
                "divisions_z": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("GEOMETRY",)
    RETURN_NAMES = ("geometry",)
    FUNCTION = "make_box"
    CATEGORY = "mtb/3D"

    def make_box(
        self,
        uniform_scale,
        width,
        height,
        depth,
        divisions_x,
        divisions_y,
        divisions_z,
    ):
        width, height, depth = (width, height, depth) * uniform_scale

        # mesh = o3d.geometry.TriangleMesh.create_box(
        #     width,
        #     height,
        #     depth,

        # )
        # mesh.compute_vertex_normals()

        mesh = create_box(
            (width, height, depth), (divisions_x, divisions_y, divisions_z)
        )

        return ({"mesh": mesh},)


class MTB_GeometryLoad:
    """Load a 3D geometry."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("GEOMETRY",)
    RETURN_NAMES = ("geometry",)
    FUNCTION = "load_geo"
    CATEGORY = "mtb/3D"

    def load_geo(self, path):
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        mesh = o3d.io.read_triangle_mesh(path)

        if len(mesh.vertices) == 0:
            mesh = o3d.io.read_triangle_model(path)
            mesh_count = len(mesh.meshes)
            if mesh_count == 0:
                raise ValueError("Couldn't parse input file")

            if mesh_count > 1:
                log.warn(
                    f"Found {mesh_count} meshes, only the first will be used..."
                )

            mesh = mesh.meshes[0].mesh

        mesh.compute_vertex_normals()

        return {
            "result": ({"mesh": mesh},),
        }


class MTB_GeometryInfo:
    """Retrieve information about a 3D geometry."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"geometry": ("GEOMETRY", {})}}

    RETURN_TYPES = ("INT", "INT", "MATERIAL")
    RETURN_NAMES = ("num_vertices", "num_triangles", "material")
    FUNCTION = "get_info"
    CATEGORY = "mtb/3D"

    def get_info(self, geometry):
        mesh, material = spread_geo(geometry)
        log.debug(mesh)
        return (len(mesh.vertices), len(mesh.triangles), material)


class MTB_GeometryDecimater:
    """Optimized the geometry to match the target number of triangles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("GEOMETRY", {}),
                "target": ("INT", {"default": 1500, "min": 3, "max": 500000}),
            }
        }

    RETURN_TYPES = ("GEOMETRY",)
    RETURN_NAMES = ("geometry",)
    FUNCTION = "decimate"
    CATEGORY = "mtb/3D"

    def decimate(self, mesh, target):
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target
        )
        mesh.compute_vertex_normals()

        return ({"mesh": mesh},)


class MTB_GeometrySceneSetup:
    """Scene setup for the renderer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "geometry": ("GEOMETRY",),
            }
        }

    RETURN_TYPES = ("SCENE",)
    RETURN_NAMES = ("scene",)
    FUNCTION = "setup"
    CATEGORY = "mtb/3D"

    def setup(self, mesh, target):
        return ({"geometry": {"mesh": mesh}, "camera": cam},)


class MTB_GeometryRender:
    """Renders a Geometry to an image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "geometry": ("SCENE", {}),
                "width": ("INT", {"default": 512, "min": 1}),
                "height": ("INT", {"default": 512, "min": 1}),
                "background": ("COLOR", {"default": [0.0, 0.0, 0.0]}),
                "camera": ("CAMERA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "mtb/3D"

    def render(self, geometry, width, height, background, camera):
        # create a renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.set_camera(camera)
        renderer.clear(background)
        renderer.add_geometry(geometry)
        renderer.render()
        image = renderer.get_image()
        return (image,)


__nodes__ = [
    MTB_Camera,
    MTB_GeometryApplyMaterial,
    MTB_GeometryBox,
    MTB_GeometryDecimater,
    MTB_GeometryDraw,
    MTB_GeometryInfo,
    MTB_GeometryLoad,
    MTB_GeometryMaterial,
    MTB_GeometryRender,
    MTB_GeometrySceneSetup,
    MTB_GeometrySphere,
    MTB_GeometryTest,
    MTB_GeometryTransform,
]
