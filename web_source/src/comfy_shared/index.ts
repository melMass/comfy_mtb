/**
 * MTB Shared Utilities
 * Main entry point - re-exports all modules
 *
 * NOTE: This bundle does NOT auto-execute anything.
 * Import the functions you need directly.
 */

// Types
export * from './types'

// Utilities
export {
  getConvexHull,
  makeUUID,
  debounce,
  deepMerge,
  safe_json,
  getNodes,
  type Debounced,
} from './utils'

// Storage
export { LocalStorageManager } from './storage'

// Logging
export {
  infoLogger,
  warnLogger,
  errorLogger,
  successLogger,
  log,
} from './logger'

// Widgets
export {
  CONVERTED_TYPE,
  hideWidget,
  showWidget,
  convertToWidget,
  convertToInput,
  hideWidgetForGood,
  fixWidgets,
  inner_value_change,
  getNamedWidget,
  nodesFromLink,
  hasWidgets,
  cleanupNode,
  offsetDOMWidget,
  getWidgetType,
} from './widgets'

// Dynamic connections
export {
  setupDynamicConnections,
  dynamic_connection,
} from './dynamic-connections'

// Colors
export { isColorBright } from './colors'

// DOM utilities
export {
  calculateTotalChildrenHeight,
  loadScript,
} from './dom'

// Documentation
export {
  ensureMarkdownParser,
  addDocumentation,
} from './documentation'

// Node extensions
export {
  chainCallback,
  addMenuHandler,
  addDeprecation,
} from './node-extensions'

// Server API
export {
  runAction,
  getServerInfo,
  setServerInfo,
} from './api'
