/**
 * MTB API Authoring Layer
 * Main entry point for the API extension
 *
 * NOTE: This bundle does NOT auto-register to avoid side-effects.
 * ComfyUI loads all .js files in web/, so dist/ files must be side-effect free.
 * Call registerMtbApiExtension() explicitly from the main entry point.
 */

// Import CSS for injection (this is bundled, not a side-effect at runtime)
import './api_nodes.css'

// Re-export types
export * from './types'

// Re-export constants
export { API_COLOR, OUTPUT_COLOR } from './constants'

// Re-export core modules
export { graphToPrompt } from './graph-to-prompt'
export { APISettingsWidgetManager, apiSettingsWidgetManager } from './widget-manager'
export { APIPanel, getAPIPanel } from './panel.svelte'
export { registerMtbApiExtension } from './extension'
