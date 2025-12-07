/**
 * Graph to Prompt conversion for MTB API
 * Adapted from ComfyUI core method with MTB API metadata support
 */

import type { LGraphNode, IWidget, LLink } from '@comfyorg/litegraph'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import type { GraphPromptResult, APINodeSettings, MTBNode } from './types'

interface NodeData {
  inputs: Record<string, unknown>
  class_type: string
  _meta?: {
    title: string
    apiSettings?: Partial<APINodeSettings>
  }
}

interface ExtendedNode extends LGraphNode {
  isVirtualNode?: boolean
  comfyClass?: string
  getInnerNodes?: () => LGraphNode[]
  applyToGraph?: () => void
  updateLink?: (link: LLink) => LLink | null
}

/**
 * Converts the current graph workflow for sending to the API
 * This includes MTB API metadata when DevMode is enabled
 */
export async function graphToPrompt(app: ComfyApp): Promise<GraphPromptResult> {
  const graph = app.graph

  // First pass: call beforeQueued on widgets and apply virtual nodes
  for (const outerNode of graph.computeExecutionOrder(false) as ExtendedNode[]) {
    if (outerNode.widgets) {
      for (const widget of outerNode.widgets) {
        // Allow widgets to run callbacks before a prompt has been queued
        // e.g. random seed before every gen
        ;(widget as IWidget & { beforeQueued?: () => void }).beforeQueued?.()
      }
    }

    const innerNodes = outerNode.getInnerNodes
      ? outerNode.getInnerNodes()
      : [outerNode]

    for (const node of innerNodes as ExtendedNode[]) {
      if (node.isVirtualNode) {
        // Don't serialize frontend only nodes but let them make changes
        node.applyToGraph?.()
      }
    }
  }

  const workflow = graph.serialize()
  const output: Record<string, NodeData> = {}

  // Process nodes in order of execution
  for (const outerNode of graph.computeExecutionOrder(false) as ExtendedNode[]) {
    const skipNode = outerNode.mode === 2 || outerNode.mode === 4
    const innerNodes =
      !skipNode && outerNode.getInnerNodes
        ? outerNode.getInnerNodes()
        : [outerNode]

    for (const node of innerNodes as ExtendedNode[]) {
      if (node.isVirtualNode) {
        continue
      }

      if (node.mode === 2 || node.mode === 4) {
        // Don't serialize muted nodes
        continue
      }

      const inputs: Record<string, unknown> = {}
      const widgets = node.widgets

      // Store all widget values
      if (widgets) {
        for (const i in widgets) {
          const widget = widgets[i] as IWidget & {
            serializeValue?: (node: LGraphNode, index: string) => Promise<unknown>
          }
          if (!widget.options || widget.options.serialize !== false) {
            inputs[widget.name] = widget.serializeValue
              ? await widget.serializeValue(node, i)
              : widget.value
          }
        }
      }

      // Store all node links
      if (node.inputs) {
        for (const i in node.inputs) {
          let parent = node.getInputNode(Number(i)) as ExtendedNode | null
          if (parent) {
            let link = node.getInputLink(Number(i))
            while (parent && (parent.mode === 4 || parent.isVirtualNode)) {
              let found = false
              if (parent.isVirtualNode) {
                link = parent.getInputLink(link?.origin_slot ?? 0)
                if (link) {
                  parent = parent.getInputNode(link.target_slot) as ExtendedNode | null
                  if (parent) {
                    found = true
                  }
                }
              } else if (link && parent.mode === 4) {
                let allInputs: (number | string)[] = [link.origin_slot]
                if (parent.inputs) {
                  allInputs = allInputs.concat(Object.keys(parent.inputs))
                  for (let parentInput of allInputs) {
                    parentInput = allInputs[parentInput as number]
                    if (
                      parent.inputs[parentInput as number]?.type ===
                      node.inputs[Number(i)].type
                    ) {
                      link = parent.getInputLink(parentInput as number)
                      if (link) {
                        parent = parent.getInputNode(
                          parentInput as number,
                        ) as ExtendedNode | null
                      }
                      found = true
                      break
                    }
                  }
                }
              }

              if (!found) {
                break
              }
            }

            if (link) {
              if (parent?.updateLink) {
                link = parent.updateLink(link)
              }
              if (link) {
                inputs[node.inputs[Number(i)].name] = [
                  String(link.origin_id),
                  Number.parseInt(String(link.origin_slot)),
                ]
              }
            }
          }
        }
      }

      const nodeData: NodeData = {
        inputs,
        class_type: node.comfyClass ?? node.type ?? '',
      }

      // Add MTB API metadata in DevMode
      if (app.ui?.settings?.getSettingValue?.('Comfy.DevMode')) {
        const mtbNode = node as unknown as MTBNode
        nodeData._meta = {
          title: node.title ?? '',
        }

        if (mtbNode.properties?.mtb_api) {
          const apiSettings = { ...mtbNode.properties.mtb_api }
          const enabledInputs: Record<string, Partial<APINodeSettings['inputs']>[string]> = {}

          delete apiSettings.showDisabled

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

          nodeData._meta = {
            ...nodeData._meta,
            apiSettings,
          }
        }
      }

      output[String(node.id)] = nodeData
    }
  }

  // Remove inputs connected to removed nodes
  for (const o in output) {
    for (const i in output[o].inputs) {
      const inputValue = output[o].inputs[i]
      if (
        Array.isArray(inputValue) &&
        inputValue.length === 2 &&
        !output[inputValue[0] as string]
      ) {
        delete output[o].inputs[i]
      }
    }
  }

  return { workflow, output }
}
