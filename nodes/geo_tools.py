import copy
import itertools
import json
import os

import numpy as np
import open3d as o3d

from ..utils import log


def spread_geo(geo, *, cp=False):
    """Spreads a GEOMETRY type into (mesh,material)."""
    mesh = geo["mesh"] if not cp else copy.copy(geo["mesh"])
    material = geo.get("material", {})
    return (mesh, material)


def euler_to_rotation_matrix(x_deg, y_deg, z_deg):
    # Convert degrees to radians
    x = np.radians(x_deg)
    y = np.radians(y_deg)
    z = np.radians(z_deg)

    # Rotation matrix around x-axis
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
    )

    # Rotation matrix around y-axis
    Ry = np.array(
        [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
    )

    # Rotation matrix around z-axis
    Rz = np.array(
        [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]
    )

    return Rz @ Ry @ Rx


def rotate_mesh(mesh, x_deg, y_deg, z_deg, center=None):
    R = euler_to_rotation_matrix(x_deg, y_deg, z_deg)
    return mesh.rotate(R, center) if center is not None else mesh.rotate(R)


def get_transformation_matrix(position, rotation, scale):
    # Construct the translation matrix
    T = np.eye(4)
    T[:3, 3] = position

    # Get the rotation matrix from Euler angles
    R = euler_to_rotation_matrix(*rotation)
    R_homo = np.eye(4)
    R_homo[:3, :3] = R

    # Construct the scaling matrix
    S = np.eye(4)
    S[0, 0] = scale[0]
    S[1, 1] = scale[1]
    S[2, 2] = scale[2]

    # Combined transforms
    return T @ R_homo @ S


def json_to_mesh(json_data: str):
    """Convert JSON to an Open3D mesh."""
    data = json.loads(json_data)
    mesh = o3d.geometry.TriangleMesh()

    if "vertices" in data:
        mesh.vertices = o3d.utility.Vector3dVector(
            np.array(data["vertices"]).reshape(-1, 3)
        )

    if "triangles" in data:
        mesh.triangles = o3d.utility.Vector3iVector(
            np.array(data["triangles"]).reshape(-1, 3)
        )

    if "vertex_normals" in data:
        mesh.vertex_normals = o3d.utility.Vector3dVector(
            np.array(data["vertex_normals"]).reshape(-1, 3)
        )

    if "vertex_colors" in data:
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array(data["vertex_colors"]).reshape(-1, 3)
        )

    if "triangle_uvs" in data:
        mesh.triangle_uvs = o3d.utility.Vector2dVector(
            np.array(data["triangle_uvs"]).reshape(-1, 2)
        )

    return mesh


def mesh_to_json(mesh: o3d.geometry.MeshBase):
    """Convert an Open3D mesh to JSON."""
    mesh_dict = {
        "vertices": np.asarray(mesh.vertices).tolist(),
        "triangles": np.asarray(mesh.triangles).tolist(),
    }

    if mesh.has_vertex_normals():
        mesh_dict["vertex_normals"] = np.asarray(mesh.vertex_normals).tolist()

    if mesh.has_vertex_colors():
        mesh_dict["vertex_colors"] = np.asarray(mesh.vertex_colors).tolist()

    if mesh.has_triangle_uvs():
        mesh_dict["triangle_uvs"] = np.asarray(mesh.triangle_uvs).tolist()

    return json.dumps(mesh_dict)


def create_grid(scale=(1, 1, 1), rows=10, columns=10):
    dx, dy, dz = scale

    # Create vertices
    vertices = []
    for i in np.linspace(-dy / 2, dy / 2, rows + 1):
        vertices.extend(
            [j, 0, i] for j in np.linspace(-dx / 2, dx / 2, columns + 1)
        )
    # Generate triangles
    triangles = []
    for i, j in itertools.product(range(rows), range(columns)):
        p1 = i * (columns + 1) + j
        p2 = i * (columns + 1) + j + 1
        p3 = (i + 1) * (columns + 1) + j
        p4 = (i + 1) * (columns + 1) + j + 1

        triangles.extend(([p1, p2, p3], [p2, p4, p3]))
    vertices = o3d.utility.Vector3dVector(np.array(vertices))
    triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)

    mesh.compute_vertex_normals()

    return mesh


def create_box(scale=(1, 1, 1), divisions=(1, 1, 1)):
    dx, dy, dz = scale
    div_x, div_y, div_z = divisions

    vertices = []
    for i in np.linspace(-dx / 2, dx / 2, div_x + 1):
        for j in np.linspace(-dy / 2, dy / 2, div_y + 1):
            vertices.extend(
                [i, j, k] for k in np.linspace(-dz / 2, dz / 2, div_z + 1)
            )
    # Generate triangles for the box faces
    triangles = []
    for x, y in itertools.product(range(div_x), range(div_y)):
        for z in range(div_z):
            # Define base index for this cube
            base = z * (div_x + 1) * (div_y + 1) + y * (div_x + 1) + x

            # Indices for the 8 vertices of the cube
            v0 = base
            v1 = base + 1
            v2 = base + (div_x + 1) + 1
            v3 = base + (div_x + 1)
            v4 = base + (div_x + 1) * (div_y + 1)
            v5 = v4 + 1
            v6 = v4 + (div_x + 1) + 1
            v7 = v4 + (div_x + 1)

            triangles.extend(
                (
                    [v0, v1, v2],
                    [v2, v3, v0],
                    [v4, v5, v6],
                    [v6, v7, v4],
                    [v0, v3, v7],
                    [v7, v4, v0],
                    [v1, v5, v6],
                    [v6, v2, v1],
                    [v0, v4, v5],
                    [v5, v1, v0],
                    [v3, v2, v6],
                    [v6, v7, v3],
                )
            )
    vertices = o3d.utility.Vector3dVector(np.array(vertices))
    triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)

    mesh.compute_vertex_normals()

    return mesh


def create_sphere(radius=1, columns=10, rows=10):
    # Create vertex positions
    vertices = []
    for i in range(rows + 1):
        lat = i * np.pi / rows
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)

        for j in range(columns + 1):
            lon = j * 2 * np.pi / columns
            sin_lon = np.sin(lon)
            cos_lon = np.cos(lon)

            x = radius * cos_lon * sin_lat
            y = radius * cos_lat
            z = radius * sin_lon * sin_lat
            vertices.append([x, y, z])

    # Create triangles
    triangles = []
    for i in range(rows):
        for j in range(columns):
            p1 = i * (columns + 1) + j
            p2 = i * (columns + 1) + j + 1
            p3 = (i + 1) * (columns + 1) + j
            p4 = (i + 1) * (columns + 1) + j + 1

            triangles.extend(([p1, p2, p3], [p2, p4, p3]))
    vertices = o3d.utility.Vector3dVector(np.array(vertices))
    triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)

    # Assigning random colors to vertices
    N = len(vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.random.uniform(0, 1, size=(N, 3))
    )
    mesh.compute_vertex_normals()

    return mesh


def create_torus(torus_radius=1, ring_radius=0.5, rows=10, columns=10):
    vertices = []
    for i in range(rows + 1):
        theta = i * 2 * np.pi / rows
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        circle_center = torus_radius + ring_radius * cos_theta

        for j in range(columns + 1):
            phi = j * 2 * np.pi / columns
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            x = circle_center * cos_phi
            y = ring_radius * sin_theta
            z = circle_center * sin_phi
            vertices.append([x, y, z])

    triangles = []
    for i in range(rows):
        for j in range(columns):
            p1 = i * (columns + 1) + j
            p2 = i * (columns + 1) + j + 1
            p3 = (i + 1) * (columns + 1) + j
            p4 = (i + 1) * (columns + 1) + j + 1

            triangles.extend(([p1, p2, p3], [p2, p4, p3]))
    vertices = o3d.utility.Vector3dVector(np.array(vertices))
    triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)

    mesh.compute_vertex_normals()

    return mesh


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


class MTBCamera:
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


class MTBDrawGeometry:
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


# class MTBRGBD_Image:
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


class MTBMaterial:
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
        return (kwargs,)


class MTBApplyMaterial:
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


class TransformGeometry:
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


class GeometrySphere:
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


class GeometryTest:
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


class GeometryBox:
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


class LoadGeometry:
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


class GeometryInfo:
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


class GeometryDecimater:
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


class GeometrySceneSetup:
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


class GeometryRender:
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
    LoadGeometry,
    GeometryInfo,
    GeometryTest,
    GeometryDecimater,
    GeometrySphere,
    GeometryRender,
    TransformGeometry,
    GeometryBox,
    MTBApplyMaterial,
    MTBMaterial,
]
