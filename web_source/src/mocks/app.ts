// src/mocks/app.ts

import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'

export const app: ComfyApp = {
  canvas: document.createElement('canvas'),
  graph: {
    // Mock graph methods if you use them
    add: (node: any) => console.log('Mock graph.add', node),
    // ...
  } as any, // Cast to any if you don't want to fully mock LGraph
  ui: {
    // Mock UI methods
    dialog: {
      show: (content: any) => console.log('Mock dialog.show', content),
      // ...
    },
    // ...
  } as any, // Cast to any if you don't want to fully mock UI
  // Mock common methods
  onReady: async () => {
    console.log('Mock app.onReady called')
    // Simulate a delay if your code expects async behavior
    await new Promise((resolve) => setTimeout(resolve, 100))
  },
  registerCustomNodeMapping: (mapping: any) =>
    console.log('Mock registerCustomNodeMapping', mapping),
  // Add any other properties/methods your code directly accesses
  // that would cause errors if undefined.
  // You can use `jest.fn()` or similar if you're using a testing framework.
}
