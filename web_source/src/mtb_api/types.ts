import type { LGraphNode, IWidget, LGraphCanvas } from '@comfyorg/litegraph'

/** Supported API input types */
export const API_INPUT_TYPES = [
  'STRING',
  'IMAGE',
  'COMBO',
  'MODEL',
  'NUMBER',
  'FLOATS',
  'BOOLEAN',
] as const

export type APIInputType = (typeof API_INPUT_TYPES)[number]

/** Settings for a single API input on a node */
export interface APIInputSettings {
  enabled: boolean
  type: APIInputType
  name: string
}

/** API configuration stored in node.properties.mtb_api */
export interface APINodeSettings {
  isAPIOutput?: boolean
  showDisabled?: boolean
  inputs?: Record<string, APIInputSettings>
}

/** Extended node properties with MTB API fields */
export interface MTBNodeProperties {
  useAPI?: boolean
  mtb_api?: APINodeSettings
  [key: string]: unknown
}

/** LGraphNode with MTB API properties */
export interface MTBNode extends LGraphNode {
  properties: MTBNodeProperties
}

/** Collected API input with widget references */
export interface APIInput {
  id: number
  name: string
  original_name?: string
  type: APIInputType
  node_id: number
  widgets: IWidget[]
  options?: string[]
  widget?: IWidget
  enabled?: boolean
}

/** Result of graphToPrompt conversion */
export interface GraphPromptResult {
  workflow: object
  output: Record<
    string,
    {
      inputs: Record<string, unknown>
      class_type: string
      _meta?: {
        title: string
        apiSettings?: Partial<APINodeSettings>
      }
    }
  >
}

/** Props for the API Panel component */
export interface APIPanelProps {
  visible?: boolean
  inputs?: Record<string, APIInput>
}

/** Interface for the API Settings Widget Manager */
export interface IAPISettingsWidgetManager {
  createAPISettingsWidget(node: MTBNode, force?: boolean): void
  removeAPISettingsWidget(node: MTBNode, widget?: IWidget): void
  applySettings(
    node: MTBNode,
    settings: Partial<APINodeSettings>,
  ): APINodeSettings
  ensureWidgets(node: MTBNode, force?: boolean): void
  drawForeground(
    node: MTBNode,
    ctx: CanvasRenderingContext2D,
    canvas: LGraphCanvas,
  ): void
}
