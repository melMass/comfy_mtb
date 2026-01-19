/**
 * Graph to Prompt conversion for MTB API
 * Uses ComfyUI's graphToPrompt and post-processes to add MTB API metadata
 */

import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import type { GraphPromptResult, APINodeSettings, MTBNode } from './types'

/**
 * Converts the current graph workflow for sending to the API
 * Includes MTB API metadata for SDK consumers
 */
export async function graphToPrompt(app: ComfyApp): Promise<GraphPromptResult> {
  // Use ComfyUI's built-in graphToPrompt which handles all complexity
  // (subgraphs, DTOs, virtual nodes, etc.)
  const result = await app.graphToPrompt()

  // Post-process to add MTB API metadata to nodes that have it
  for (const nodeId of Object.keys(result.output)) {
    const nodeData = result.output[nodeId]

    // Find the original node in the graph to get MTB API settings
    const node = app.graph.getNodeById(Number(nodeId)) as MTBNode | null

    if (node?.properties?.mtb_api) {
      const apiSettings = { ...node.properties.mtb_api }
      const enabledInputs: Record<string, Partial<APINodeSettings['inputs']>[string]> = {}

      // Clean up internal-only settings
      delete apiSettings.showDisabled

      // Filter to only enabled inputs
      if (apiSettings.inputs) {
        for (const k of Object.keys(apiSettings.inputs)) {
          const current = apiSettings.inputs[k]
          if (current.enabled) {
            enabledInputs[k] = { ...current }
            delete (enabledInputs[k] as { enabled?: boolean }).enabled
          }
        }
        apiSettings.inputs = enabledInputs as typeof apiSettings.inputs
      }

      // Add MTB API metadata
      nodeData._meta = {
        ...nodeData._meta,
        apiSettings,
      }
    }
  }

  return result as GraphPromptResult
}
