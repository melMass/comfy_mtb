// Some manual types I use to facilitate developing on top of
// Comfy's Litegraph implementation.

import type {
  ContextMenuItem,
  LGraphNode,
  IWidget,
  LGraph,
} from '../../../web/types/litegraph'

export type {
  ComfyExtension,
  ComfyObjectInfo,
  ComfyObjectInfoConfig,
} from '../../../web/types/comfy'

export type {
  ContextMenuItem,
  IWidget,
  LLink,
  INodeInputSlot,
  INodeOutputSlot,
} from '../../../web/types/litegraph'

export type VectorWidget = IWidget<number[], { default: number[] }>
export interface NodeData {
  category: str
  description: str
  display_name: str
  input: NodeInput
  name: str
  output: [str]
  output_is_list: [boolean]
  output_name: [str]
  output_node: boolean
}

export interface ComfyDialog {
  element: Element
  close: () => void
  show: (html: str) => void
}

export interface ComfySettingsDialog {
  app: ComfyApp
  element: Element
  settingsValues: Record<string, unknown>
  settingsLookup: Record<string, unknown>
  load: () => Promise<void>
  setSettingValueAsync: (id: string, value: unknown) => Promise<void>
}

export interface ComfyUI {
  app: ComfyApp
  dialog: ComfyDialog
  settings: ComfySettingsDialog
  autoQueueMode: 'instant' | 'change'
  batchCount: number
  lastQueueSize: number
  graphHasChanged: boolean
  queue: ComfyList
  history: ComfyList
}

/**Very incomplete Comfy App definition*/
interface ComfyApp {
  graph: LGraph
  queueItems: { number: number; batchCount: number }[]
  processingQueue: boolean
  ui: ComfyUI
  extensions: ComfyExtension[]
  nodeOutputs: Record<string, unknown>
  nodePreviewImages: Record<string, Image>
  shiftDown: boolean
  isImageNode: (node: LGraphNodeExtended) => boolean
  queuePrompt: (number: number, batchCount: number) => Promise<void>
  /** Loads workflow data from the specified file*/
  handleFile: (file: File) => Promise<void>
}

export type { ComfyApp as App }

export interface LGraphNodeExtension {
  addDOMWidget: (
    name: string,
    type: string,
    element: Element,
    options: Record<string, unknown>,
  ) => IWidget
  onNodeCreated: () => void
  getExtraMenuOptions: () => ContextMenuItem[]
  prototype: LGraphNodeExtended
}

export type LGraphNodeExtended = LGraphNode & LGraphNodeExtension

export interface NodeType /*extends LGraphNode*/ {
  category: str
  comfyClass: str
  length: 0
  name: str
  nodeData: NodeData
  prototype: LGraphNodeExtended
  title: str
  type: str
}

export interface NodeInput {
  required: object
}

// NOTE: for prototype overriding
export type OnDrawWidgetParams = Parameters<IWidget['draw']>
export type OnDrawForegroundParams = Parameters<LGraphNode['onDrawForeground']>
export type OnMouseDownParams = Parameters<LGraphNode['onMouseDown']>
export type OnConnectionsChangeParams = Parameters<
  LGraphNode['onConnectionsChange']
>
export type OnNodeCreatedParams = Parameters<
  LGraphNodeExtension['onNodeCreated']
>

export interface DocumentationOptions {
  icon_size?: number
  icon_margin?: number
}
