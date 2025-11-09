/// <reference types="comfyui-frontend-types" />

// Declare the ComfyUI 'app' module using the alias path
declare module '@/scripts/app' {
  // Assuming ComfyApp is defined in src/types/comfyui.d.ts or an installed @types package
  import type { ComfyApp } from '@comfyorg/comfyui-frontend-types' // Adjust path if ComfyApp is in a different file
  export const app: ComfyApp
  // You might also export other things if they are part of this module
  // export const api: ComfyApi;
}

// Declare the ComfyUI 'app' module using the relative path
// This is less ideal as it's a runtime path, but covers your existing imports
declare module '../../scripts/app.js' {
  import type { ComfyApp } from '@comfyorg/comfyui-frontend-types' // Adjust path
  export const app: ComfyApp
}

// If you import other ComfyUI modules, declare them here too:
declare module '@/scripts/api' {
  import type { ComfyApp } from '@comfyorg/comfyui-frontend-types' // Adjust path
  export const api: ComfyApi
}

// declare module '@/scripts/utils' {
//   // Define types for utils if available
//   export const utils: any // Replace with actual type
// }

// Add declarations for any other ComfyUI internal modules you import
