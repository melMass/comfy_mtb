/**
 * File: geometry_nodes.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

import { app } from '../../scripts/app.js'

import * as THREE from './extern/three.module.js'

export const make_wireframe = (mesh) => {
  const wireframeGeometry = new THREE.WireframeGeometry(mesh.geometry)
  const wireframeMaterial = new THREE.LineBasicMaterial({
    color: 0x000000,
    linewidth: 1,
  })
  const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial)

  //   return {
  //     wireframe,
  //     wireframeGeometry,
  //     wireframeMaterial,
  //   }
  return wireframe
}

export const three_to_o3d = (mesh) => {
  const meshData = {};

  // vertices
  if (mesh.geometry.getAttribute('position')) {
    meshData.vertices = Array.from(mesh.geometry.getAttribute('position').array);
  }

  // triangles (indices)
  if (mesh.geometry.index) {
    meshData.triangles = Array.from(mesh.geometry.index.array);
  }

  // vertex normals
  if (mesh.geometry.getAttribute('normal')) {
    meshData.vertex_normals = Array.from(mesh.geometry.getAttribute('normal').array);
  }

  // vertex colors
  if (mesh.geometry.getAttribute('color')) {
    meshData.vertex_colors = Array.from(mesh.geometry.getAttribute('color').array);
  }

  // UVs
  if (mesh.geometry.getAttribute('uv')) {
    meshData.triangle_uvs = Array.from(mesh.geometry.getAttribute('uv').array);
  }

  // Convert to JSON and send to Python backend
  return JSON.stringify(meshData);
};

export const o3d_to_three = (data, material_opts) => {

  material_opts = material_opts || { color: "0x00ff00" }
  // Parse the JSON data
  const meshData = JSON.parse(data)

  // Create a geometry
  const geometry = new THREE.BufferGeometry()

  // Set vertices and triangles
  const vertices = new Float32Array(meshData.vertices.flat())
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))

  const indices = new Uint32Array(meshData.triangles.flat())
  geometry.setIndex(new THREE.BufferAttribute(indices, 1))

  // If vertex_normals are available
  if (meshData.vertex_normals) {
    const normals = new Float32Array(meshData.vertex_normals.flat())
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
  }

  // If vertex_colors are available
  if (meshData.vertex_colors) {
    const colors = new Float32Array(meshData.vertex_colors.flat())
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  }

  // If triangle_uvs are available
  if (meshData.triangle_uvs) {
    const uvs = new Float32Array(meshData.triangle_uvs.flat())
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  }

  if (material_opts.displacementB64 != undefined) {
    material_opts.displacementMap = THREE.ImageUtils.loadTexture(material_opts.displacementB64)
    delete (material_opts.displacementB64)
  }

  // For visualization, you might choose to use the MeshPhongMaterial to get the benefit of lighting with normals
  const material = meshData.vertex_colors
    ? new THREE.MeshStandardMaterial({ ...material_opts, vertexColors: true })
    : new THREE.MeshStandardMaterial({ ...material_opts })

  const threeMesh = new THREE.Mesh(geometry, material)

  return threeMesh
}

app.registerExtension({
  name: 'mtb.geometry_nodes',

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    switch (nodeData.name) {
      case 'Load Geometry (mtb)':
        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
          onExecuted?.apply(this, arguments)
          console.log('Executed Load Geometry', arguments)
        }
        break
    }
  },
})
