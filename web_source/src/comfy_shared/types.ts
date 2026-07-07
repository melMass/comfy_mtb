/**
 * Type definitions for comfy_shared
 */

import type { LGraphNode, IWidget, LLink, INodeInputSlot, INodeOutputSlot } from '@comfyorg/litegraph'

// Extend window for MTB globals
declare global {
  interface Window {
    MTB?: {
      DEBUG?: boolean
      mdParser?: MarkdownParser
    }
    MTBMarkdown?: { getParser: () => Promise<MarkdownParser> }
    MTBMarkdownPlus?: { getParser: () => Promise<MarkdownParser> }
  }
  const LiteGraph: {
    NODE_SLOT_HEIGHT: number
    NODE_TITLE_HEIGHT: number
    NODE_COLLAPSED_RADIUS: number
  }
}

export interface MarkdownParser {
  parse: (content: string) => Promise<string>
}

// Widget types
export interface MTBWidget extends IWidget {
  origType?: string
  origComputeSize?: () => [number, number]
  origSerializeValue?: () => unknown
  hidden?: boolean
  linkedWidgets?: MTBWidget[]
  last_y?: number
  canvas?: HTMLCanvasElement
  inputEl?: HTMLElement
  onRemoved?: () => void
  parent?: { inputHeight?: number }
}

export interface MTBNode extends LGraphNode {
  widgets?: MTBWidget[]
  inputs: (INodeInputSlot & { widget?: { name: string; config?: unknown }; _isDynamic?: boolean })[]
  outputs: INodeOutputSlot[]
  properties: Record<string, unknown>
  color?: string
  flags: { collapsed?: boolean }
  show_doc?: boolean
  docCtrl?: AbortController
  docPos?: { width: string; height: string }
  connectionTransit?: boolean
  setProperty: (name: string, value: unknown) => void
  addInput: (name: string, type: string, extra?: Record<string, unknown>) => INodeInputSlot
  removeInput: (index: number) => void
  setSize: (size: [number, number]) => void
  graph: {
    getLink: (id: number) => LLink
    getNodeById: (id: number) => MTBNode
    beforeChange: () => void
    afterChange: () => void
  }
  slots: (INodeInputSlot | INodeOutputSlot)[]
}

export interface NodeType {
  title?: string
  prototype: {
    onNodeCreated?: (this: MTBNode) => void
    onConfigure?: (this: MTBNode, data: unknown) => void
    onConnectionsChange?: (this: MTBNode, ...args: OnConnectionsChangeParams) => void
    onDrawForeground?: (this: MTBNode, ctx: CanvasRenderingContext2D, canvas: unknown) => void
    onMouseDown?: (this: MTBNode, ...args: OnMouseDownParams) => boolean | void
    onRemoved?: (this: MTBNode) => void
    getExtraMenuOptions?: (this: MTBNode, app: unknown, options: ContextMenuItem[]) => ContextMenuItem[]
    getSlotMenuOptions?: (slot: SlotMenuContext) => ContextMenuItem[]
  }
}

export interface NodeData {
  name: string
  description?: string
}

export type OnConnectionsChangeParams = [
  type: number,
  slotIndex: number,
  isConnected: boolean,
  link: LLink | null,
  ioSlot: INodeInputSlot | INodeOutputSlot,
]

export type OnMouseDownParams = [
  event: MouseEvent,
  localPos: [number, number],
  graphCanvas: unknown,
]

export interface ContextMenuItem {
  content: string
  callback?: (...args: unknown[]) => void
}

export interface SlotMenuContext {
  input?: INodeInputSlot & { label?: string; name: string }
  node: MTBNode
}

export interface DocumentationOptions {
  icon_size?: number
  icon_margin?: number
}

export interface DynamicConnectionOptions {
  separator?: string
  rename_menu?: 'label' | 'name'
  start_index?: number
  link?: LLink | null
  ioSlot?: INodeInputSlot | INodeOutputSlot
  nameArray?: string[]
  DEBUG?: unknown
}

export interface LinkInfo {
  to: MTBNode
  from: MTBNode
  type: 'error' | 'incoming' | 'outgoing'
}
