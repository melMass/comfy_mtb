/**
 * Widget utilities for LiteGraph nodes
 */

import type { LLink } from '@comfyorg/litegraph'
import type { MTBNode, MTBWidget, LinkInfo } from './types'
import { log } from './logger'

export const CONVERTED_TYPE = 'converted-widget'

export function hideWidget(node: MTBNode, widget: MTBWidget, suffix = ''): void {
  widget.origType = widget.type as string
  widget.hidden = true
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = CONVERTED_TYPE + suffix
  widget.serializeValue = () => {
    // Prevent serializing the widget if we have no input linked
    const input = node.inputs?.find((i) => i.widget?.name === widget.name)
    if (input?.link == null) {
      return undefined
    }
    return widget.origSerializeValue ? widget.origSerializeValue() : widget.value
  }

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidget(node, w, `:${widget.name}`)
    }
  }
}

export function showWidget(widget: MTBWidget): void {
  widget.type = widget.origType!
  widget.computeSize = widget.origComputeSize!
  widget.serializeValue = widget.origSerializeValue

  delete widget.origType
  delete widget.origComputeSize
  delete widget.origSerializeValue

  // Show any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      showWidget(w)
    }
  }
}

export function convertToWidget(node: MTBNode, widget: MTBWidget): void {
  showWidget(widget)
  const sz = node.size
  const inputIndex = node.inputs.findIndex((i) => i.widget?.name === widget.name)
  if (inputIndex >= 0) {
    node.removeInput(inputIndex)
  }

  if (node.widgets) {
    for (const w of node.widgets) {
      if (w.last_y !== undefined) {
        w.last_y -= LiteGraph.NODE_SLOT_HEIGHT
      }
    }
  }

  // Restore original size but grow if needed
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

/**
 * Extracts the type and link type from a widget config object.
 */
export function getWidgetType(config: unknown[]): { type: string; linkType: string } {
  let type = config?.[0]
  let linkType = type as string
  if (Array.isArray(type)) {
    linkType = type.join(',')
    type = 'COMBO'
  }
  return { type: type as string, linkType }
}

export function convertToInput(
  node: MTBNode,
  widget: MTBWidget,
  config: unknown[],
): void {
  hideWidget(node, widget)

  const { linkType } = getWidgetType(config)

  // Add input and store widget config for creating on primitive node
  const sz = node.size
  node.addInput(widget.name, linkType, {
    widget: { name: widget.name, config },
  })

  if (node.widgets) {
    for (const w of node.widgets) {
      if (w.last_y !== undefined) {
        w.last_y += LiteGraph.NODE_SLOT_HEIGHT
      }
    }
  }

  // Restore original size but grow if needed
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function hideWidgetForGood(node: MTBNode, widget: MTBWidget, suffix = ''): void {
  widget.origType = widget.type as string
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4]
  widget.hidden = true
  widget.type = CONVERTED_TYPE + suffix

  // Hide any linked widgets
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidgetForGood(node, w, `:${widget.name}`)
    }
  }
}

export function fixWidgets(node: MTBNode): void {
  if (!node.inputs) return

  for (const input of node.inputs) {
    log(input)
    if (input.widget || node.widgets) {
      const matching_widget = node.widgets?.find((w) => w.name === input.name)
      if (matching_widget) {
        const w = node.widgets?.find((w) => w.name === matching_widget.name)
        if (w && w.type !== CONVERTED_TYPE) {
          log(w)
          log(`hiding ${w.name}(${w.type}) from ${node.type}`)
          log(node)
          hideWidget(node, w)
        } else {
          log(`converting to widget ${w}`)
          if (w) convertToWidget(node, w)
        }
      }
    }
  }
}

export function inner_value_change(
  node: MTBNode,
  widget: MTBWidget,
  val: unknown,
  pos?: [number, number],
  event?: Event,
): void {
  let value = val
  if (widget.type === 'number' || widget.type === 'BBOX') {
    value = Number(value)
  } else if (widget.type === 'BOOL') {
    value = Boolean(value)
  }
  widget.value = value

  const property = (widget.options as { property?: string } | undefined)?.property
  if (property && node.properties[property] !== undefined) {
    node.setProperty(property, value)
  }

  if (widget.callback) {
    // @ts-expect-error - app is global in ComfyUI
    widget.callback(widget.value, app.canvas, node, pos, event)
  }
}

export const getNamedWidget = <T extends string>(
  node: MTBNode,
  ...names: T[]
): Record<T, MTBWidget | undefined> => {
  const out = {} as Record<T, MTBWidget | undefined>

  for (const name of names) {
    out[name] = node.widgets?.find((w) => w.name === name)
  }

  return out
}

export const nodesFromLink = (node: MTBNode, link: LLink | number): LinkInfo => {
  let resolvedLink: LLink
  if (typeof link === 'number') {
    resolvedLink = node.graph.getLink(link)
  } else {
    resolvedLink = link
  }

  const fromNode = node.graph.getNodeById(resolvedLink.origin_id)
  const toNode = node.graph.getNodeById(resolvedLink.target_id)

  let tp: 'error' | 'incoming' | 'outgoing' = 'error'

  if (fromNode.id === node.id) {
    tp = 'outgoing'
  } else if (toNode.id === node.id) {
    tp = 'incoming'
  }

  return { to: toNode, from: fromNode, type: tp }
}

export const hasWidgets = (node: MTBNode): boolean => {
  if (!node.widgets || !node.widgets?.[Symbol.iterator]) {
    return false
  }
  return true
}

export const cleanupNode = (node: MTBNode): void => {
  if (!hasWidgets(node)) {
    return
  }
  for (const w of node.widgets!) {
    if (w.canvas) {
      w.canvas.remove()
    }
    if (w.inputEl) {
      w.inputEl.remove()
    }
    w.onRemoved?.()
  }
}

export function offsetDOMWidget(
  widget: MTBWidget,
  ctx: CanvasRenderingContext2D,
  node: MTBNode,
  widgetWidth: number,
  widgetY: number,
  height?: number,
): void {
  if (!widget.inputEl) return

  const margin = 10
  const elRect = ctx.canvas.getBoundingClientRect()
  const transform = new DOMMatrix()
    .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
    .multiplySelf(ctx.getTransform())
    .translateSelf(margin, margin + widgetY)

  const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
  Object.assign(widget.inputEl.style, {
    transformOrigin: '0 0',
    transform: scale.toString(),
    left: `${transform.a + transform.e}px`,
    top: `${transform.d + transform.f}px`,
    width: `${widgetWidth - margin * 2}px`,
    height: `${(height || widget.parent?.inputHeight || 32) - margin * 2}px`,
    position: 'absolute',
    background: !node.color ? '' : node.color,
    color: !node.color ? '' : 'white',
    zIndex: '5',
  })
}
