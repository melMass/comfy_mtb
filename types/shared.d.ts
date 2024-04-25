// Some manual types I use to facilitate developing on top of
// Comfy's Litegraph implementation.

import type { ContextMenuItem, LGraphNode } from '../web/types/litegraph'

export type { ContextMenuItem } from '../web/types/litegraph'
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

export interface ExtendedLGraphNode {
  onNodeCreated: () => void
  getExtraMenuOptions: () => ContextMenuItem[]
}

export interface NodeType /*extends LGraphNode*/ {
  category: str
  comfyClass: str
  length: 0
  name: str
  nodeData: NodeData
  prototype: LGraphNode & ExtendedLGraphNode
  title: str
  type: str
}

export interface NodeInput {
  required: object
}

// NOTE: for prototype overriding
export type OnDrawForegroundParams = Parameters<LGraphNode['onDrawForeground']>
export type OnMouseDownParams = Parameters<LGraphNode['onMouseDown']>
export type OnConnectionsChangeParams = Parameters<
  LGraphNode['onConnectionsChange']
>
export type OnNodeCreatedParams = Parameters<
  ExtendedLGraphNode['onNodeCreated']
>

export interface DocumentationOptions {
  icon_size?: number
  icon_margin?: number
}
