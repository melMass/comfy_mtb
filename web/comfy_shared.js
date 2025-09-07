/**
 * @module Shared utilities
 * File: comfy_shared.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 * Copyright (c) 2023-2024 Mel Massadian
 */

// Reference the shared typedefs file
/// <reference path="../types/typedefs.js" />

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

// #region base utils

/**
 * Computes the convex hull of a set of points using the Monotone Chain algorithm.
 *
 * @param {Array<Array<number>>} points An array of points, where each point is an array of two numbers [x, y].
 * @returns {Array<Array<number>>} The points forming the convex hull, in counter-clockwise order.
 */
export const getConvexHull = (points) => {
  if (points.length <= 3) {
    return points
  }

  points.sort((a, b) => a[0] - b[0] || a[1] - b[1])

  const lower = []
  for (const p of points) {
    while (
      lower.length >= 2 &&
      cross_product(lower[lower.length - 2], lower[lower.length - 1], p) <= 0
    ) {
      lower.pop()
    }
    lower.push(p)
  }

  const upper = []
  for (let i = points.length - 1; i >= 0; i--) {
    const p = points[i]
    while (
      upper.length >= 2 &&
      cross_product(upper[upper.length - 2], upper[upper.length - 1], p) <= 0
    ) {
      upper.pop()
    }
    upper.push(p)
  }

  function cross_product(o, a, b) {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  }

  return lower
    .slice(0, lower.length - 1)
    .concat(upper.slice(0, upper.length - 1))
}

// - crude uuid
export function makeUUID() {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

// - basic debounce decorator
export function debounce(func, delay) {
  let timeout
  let debounced = function (...args) {
    clearTimeout(timeout)
    timeout = setTimeout(() => func.apply(this, args), delay)
  }
  debounced.cancel = () => {
    clearTimeout(timeout)
  }
  return debounced
}

//- local storage manager
export class LocalStorageManager {
  constructor(namespace) {
    this.namespace = namespace
  }

  _namespacedKey(key) {
    return `${this.namespace}:${key}`
  }

  set(key, value) {
    const serializedValue = JSON.stringify(value)
    localStorage.setItem(this._namespacedKey(key), serializedValue)
  }

  get(key, default_val = null) {
    const value = localStorage.getItem(this._namespacedKey(key))
    return value ? JSON.parse(value) : default_val
  }

  remove(key) {
    localStorage.removeItem(this._namespacedKey(key))
  }

  clear() {
    for (const key of Object.keys(localStorage).filter((k) =>
      k.startsWith(`${this.namespace}:`),
    )) {
      localStorage.removeItem(key)
    }
  }
}

// - log utilities

const consoleMethodToSeverity = (method) => {
  switch (method) {
    case 'log': {
      return 'info'
    }
    case 'error' | 'warn': {
      return method
    }
    default: {
      return 'secondary'
    }
  }
}

function createLogger(emoji, color, consoleMethod = 'log') {
  return (message, ...args) => {
    if (window.MTB?.DEBUG) {
      // biome-ignore lint/suspicious/noConsole: logger wrapper
      console[consoleMethod](
        `%c${emoji} ${message}`,
        `color: ${color};`,
        ...args,
      )
    }
    return {
      notify: (timeout = 3000) => {
        app.extensionManager.toast.add({
          severity: consoleMethodToSeverity(consoleMethod),
          summary: 'MTB',
          detail: `${emoji} ${message}`,
          life: timeout,
        })
      },
    }
  }
}

export const infoLogger = createLogger('ℹ️', 'yellow')
export const warnLogger = createLogger('⚠️', 'orange', 'warn')
export const errorLogger = createLogger('🔥', 'red', 'error')
export const successLogger = createLogger('✅', 'green')

export const log = (...args) => {
  if (window.MTB?.DEBUG) {
    // biome-ignore lint/suspicious/noConsole: logger wrapper
    console.debug(...args)
  }
}

export const safe_json = (object, replacer) => {
  var objects = new WeakMap()
  return (function derez(value, path) {
    let old_path
    let nu
    if (replacer !== undefined) {
      value = replacer(value)
    }
    if (
      typeof value === 'object' &&
      value !== null &&
      !(value instanceof Boolean) &&
      !(value instanceof Date) &&
      !(value instanceof Number) &&
      !(value instanceof RegExp) &&
      !(value instanceof String)
    ) {
      old_path = objects.get(value)
      if (old_path !== undefined) {
        return { $ref: old_path }
      }
      objects.set(value, path)

      if (Array.isArray(value)) {
        nu = []
        value.forEach((element, i) => {
          nu[i] = derez(element, `${path}[${i}]`)
        })
      } else {
        nu = {}
        Object.keys(value).forEach((name) => {
          nu[name] = derez(value[name], path + '[' + JSON.stringify(name) + ']')
        })
      }
      return nu
    }
    return value
  })(object, '$')
}

/**
 * Deep merge two objects.
 * @param {Object} target - The target object to merge into.
 * @param {...Object} sources - The source objects to merge from.
 * @returns {Object} - The merged object.
 */
export function deepMerge(target, ...sources) {
  if (!sources.length) return target
  const source = sources.shift()

  for (const key in source) {
    if (source[key] instanceof Object) {
      if (!target[key]) Object.assign(target, { [key]: {} })
      deepMerge(target[key], source[key])
    } else {
      Object.assign(target, { [key]: source[key] })
    }
  }

  return deepMerge(target, ...sources)
}

// #endregion

// #region widget utils
export const CONVERTED_TYPE = 'converted-widget'

export function hideWidget(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.hidden = true
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = CONVERTED_TYPE + suffix
  widget.serializeValue = () => {
    // Prevent serializing the widget if we have no input linked
    const { link } = node.inputs.find((i) => i.widget?.name === widget.name)
    if (link == null) {
      return undefined
    }
    return widget.origSerializeValue
      ? widget.origSerializeValue()
      : widget.value
  }

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidget(node, w, `:${widget.name}`)
    }
  }
}

/**
 * Show widget
 *
 * @param {import("../../../web/types/litegraph.d.ts").IWidget} widget - target widget
 */
export function showWidget(widget) {
  widget.type = widget.origType
  widget.computeSize = widget.origComputeSize
  widget.serializeValue = widget.origSerializeValue

  delete widget.origType
  delete widget.origComputeSize
  delete widget.origSerializeValue

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      showWidget(w)
    }
  }
}

export function convertToWidget(node, widget) {
  showWidget(widget)
  const sz = node.size
  node.removeInput(node.inputs.findIndex((i) => i.widget?.name === widget.name))

  for (const widget of node.widgets) {
    widget.last_y -= LiteGraph.NODE_SLOT_HEIGHT
  }

  // Restore original size but grow if needed
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function convertToInput(node, widget, config) {
  hideWidget(node, widget)

  const { linkType } = getWidgetType(config)

  // Add input and store widget config for creating on primitive node
  const sz = node.size
  node.addInput(widget.name, linkType, {
    widget: { name: widget.name, config },
  })

  for (const widget of node.widgets) {
    widget.last_y += LiteGraph.NODE_SLOT_HEIGHT
  }

  // Restore original size but grow if needed
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function hideWidgetForGood(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.hidden = true
  widget.type = CONVERTED_TYPE + suffix
  // widget.serializeValue = () => {
  //     // Prevent serializing the widget if we have no input linked
  //     const w = node.inputs?.find((i) => i.widget?.name === widget.name);
  //     if (w?.link == null) {
  //         return undefined;
  //     }
  //     return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
  // };

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidgetForGood(node, w, `:${widget.name}`)
    }
  }
}

export function fixWidgets(node) {
  if (node.inputs) {
    for (const input of node.inputs) {
      log(input)
      if (input.widget || node.widgets) {
        // if (newTypes.includes(input.type)) {
        const matching_widget = node.widgets.find((w) => w.name === input.name)
        if (matching_widget) {
          // if (matching_widget.hidden) {
          //     log(`Already hidden skipping ${matching_widget.name}`)
          //     continue
          // }
          const w = node.widgets.find((w) => w.name === matching_widget.name)
          if (w && w.type !== CONVERTED_TYPE) {
            log(w)
            log(`hidding ${w.name}(${w.type}) from ${node.type}`)
            log(node)
            hideWidget(node, w)
          } else {
            log(`converting to widget ${w}`)

            convertToWidget(node, input)
          }
        }
      }
    }
  }
}
export function inner_value_change(widget, val, event = undefined) {
  let value = val
  if (widget.type === 'number' || widget.type === 'BBOX') {
    value = Number(value)
  } else if (widget.type === 'BOOL') {
    value = Boolean(value)
  }
  widget.value = corrected_value
  if (
    widget.options?.property &&
    node.properties[widget.options.property] !== undefined
  ) {
    node.setProperty(widget.options.property, value)
  }
  if (widget.callback) {
    widget.callback(widget.value, app.canvas, node, pos, event)
  }
}

export const getNamedWidget = (node, ...names) => {
  const out = {}

  for (const name of names) {
    out[name] = node.widgets.find((w) => w.name === name)
  }

  return out
}

/**
 * @param {LGraphNode} node
 * @param {LLink} link
 * @returns {{to:LGraphNode, from:LGraphNode, type:'error' | 'incoming' | 'outgoing'}}
 */
export const nodesFromLink = (node, link) => {
  if (typeof link === 'number') {
    link = app.graph.getLink(link)
  }

  const fromNode = app.graph.getNodeById(link.origin_id)
  const toNode = app.graph.getNodeById(link.target_id)

  let tp = 'error'

  if (fromNode.id === node.id) {
    tp = 'outgoing'
  } else if (toNode.id === node.id) {
    tp = 'incoming'
  }

  return { to: toNode, from: fromNode, type: tp }
}

export const hasWidgets = (node) => {
  if (!node.widgets || !node.widgets?.[Symbol.iterator]) {
    return false
  }
  return true
}

export const cleanupNode = (node) => {
  if (!hasWidgets(node)) {
    return
  }
  for (const w of node.widgets) {
    if (w.canvas) {
      w.canvas.remove()
    }
    if (w.inputEl) {
      w.inputEl.remove()
    }
    // calls the widget remove callback
    w.onRemoved?.()
  }
}

export function offsetDOMWidget(
  widget,
  ctx,
  node,
  widgetWidth,
  widgetY,
  height,
) {
  const margin = 10
  const elRect = ctx.canvas.getBoundingClientRect()
  const transform = new DOMMatrix()
    .scaleSelf(
      elRect.width / ctx.canvas.width,
      elRect.height / ctx.canvas.height,
    )
    .multiplySelf(ctx.getTransform())
    .translateSelf(margin, margin + widgetY)

  const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
  Object.assign(widget.inputEl.style, {
    transformOrigin: '0 0',
    transform: scale,
    left: `${transform.a + transform.e}px`,
    top: `${transform.d + transform.f}px`,
    width: `${widgetWidth - margin * 2}px`,
    // height: `${(widget.parent?.inputHeight || 32) - (margin * 2)}px`,
    height: `${(height || widget.parent?.inputHeight || 32) - margin * 2}px`,

    position: 'absolute',
    background: !node.color ? '' : node.color,
    color: !node.color ? '' : 'white',
    zIndex: 5, //app.graph._nodes.indexOf(node),
  })
}

/**
 * Extracts the type and link type from a widget config object.
 * @param {*} config
 * @returns
 */
export function getWidgetType(config) {
  // Special handling for COMBO so we restrict links based on the entries
  let type = config?.[0]
  let linkType = type
  if (Array.isArray(type)) {
    type = 'COMBO'
    linkType = linkType.join(',')
  }
  return { type, linkType }
}

// #endregion

// function to test if input is a dynamic one
const isDynamicInput = (input) => {
  // infoLogger('Checking if input dynamic', { input })
  // return input.name.startsWith(connectionPrefix)
  return input._isDynamic === true
}

// Add a dynamic input, update node properties and slot colors!
const addDynamicInput = (node, name, kind) => {
  const input = node.addInput(name, kind)
  input._isDynamic = true

  update_dynamic_properties(node)
  set_slot_colors(node, ['cyan', undefined], isDynamicInput)

  return input
}

const set_slot_colors = (node, colors, condition) => {
  if (!condition) {
    condition = (_s) => true
  }

  for (const slot of node.slots) {
    // infoLogger('Candidate', { slot, accepted: condition(slot) })
    if (condition(slot)) {
      slot.color_off = colors[0]
      slot.color_on = colors[1]
    }
  }
}

const update_dynamic_properties = (node) => {
  const dyn = []
  for (const input of node.inputs) {
    if (isDynamicInput(input)) {
      dyn.push(input.name)
    }
  }
  node.setProperty('dynamic_connections', dyn)
}

// #region dynamic connections
/**
 * @param {NodeType} nodeType The nodetype to attach the documentation to
 * @param {str} prefix A prefix added to each dynamic inputs
 * @param {str | [str]} inputType The datatype(s) of those dynamic inputs
 * @param {{separator?:string,rename_menu?:'label'|'name', start_index?:number, link?:LLink, ioSlot?:INodeInputSlot | INodeOutputSlot}?} [opts] Extra options
 * @returns
 */
export const setupDynamicConnections = (
  nodeType,
  prefix,
  inputType,
  opts = undefined,
) => {
  infoLogger(
    'Setting up dynamic connections for',
    Object.getOwnPropertyDescriptors(nodeType).title.value,
  )

  /** @type {{separator:string,rename_menu?:"label"|"name" start_index:number, link?:LLink, ioSlot?:INodeInputSlot | INodeOutputSlot}?} */
  const options = Object.assign(
    {
      separator: '_',
      start_index: 1,
      rename_menu: 'label',
    },
    opts || {},
  )
  const is_valid_name = (node, val) => {
    return true
  }
  nodeType.prototype.getSlotMenuOptions = (slot) => {
    if (!slot.input) {
      return
    }
    infoLogger('Slot Menu', { slot })
    return [
      {
        content: `Rename Input (${options.rename_menu})`,
        callback: () => {
          const dialog = app.canvas.createDialog(
            "<span class='name'>Name</span><input autofocus type='text'/><button>OK</button>",
            {},
          )
          const dialogInput = dialog.querySelector('input')
          if (dialogInput) {
            if (options.rename_menu === 'label') {
              dialogInput.value = slot.input.label || slot.input.name || ''
            } else if (options.rename_menu === 'name') {
              dialogInput.value = slot.input.name || ''
            }
          }
          const inner = () => {
            // TODO: check if name exists or other guards
            const val = dialogInput.value
            if (!is_valid_name(slot.node, val)) {
              dialog.close()
              return
            }

            app.graph.beforeChange()
            if (options.rename_menu === 'label') {
              slot.input.label = val
            } else if (options.rename_menu === 'name') {
              slot.input.name = val
              slot.input.label = val
            }

            app.graph.afterChange()

            dialog.close()
          }
          dialog.querySelector('button').addEventListener('click', inner)
          dialogInput.addEventListener('keydown', (e) => {
            dialog.is_modified = true
            if (e.keyCode === 27) {
              dialog.close()
            } else if (e.keyCode === 13) {
              inner()
            } else if (e.keyCode !== 13 && e.target?.localName !== 'textarea') {
              return
            }
            e.preventDefault()
            e.stopPropagation()
          })
          dialogInput.focus()
        },
      },
    ]
  }

  const onConfigure = nodeType.prototype.onConfigure

  nodeType.prototype.onConfigure = function (data) {
    const r = onConfigure ? onConfigure.apply(this, data) : undefined

    // Set or restore serialized properties, lt seems to auto serialize/deserialize to/from string
    if (!('dynamic_connections' in this.properties)) {
      // this.addProperty('dynamic_connections', [], 'string')
      this.setProperty('dynamic_connections', [])
    } else {
      const dynamic_connections = this.properties.dynamic_connections
      if (typeof dynamic_connections !== 'object') {
        return r
      }
      for (const name of dynamic_connections) {
        infoLogger(`Would dynamize: ${name}`)
        const input = this.inputs.find((i) => i.name === name)
        if (input) {
          infoLogger('Input found', { input })
          input._isDynamic = true
        }
      }
    }
    // set color
    set_slot_colors(this, ['cyan', undefined], isDynamicInput)

    return r
  }

  const onNodeCreated = nodeType.prototype.onNodeCreated
  const inputList = typeof inputType === 'object'

  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated ? onNodeCreated.apply(this, []) : undefined

    const input = addDynamicInput(
      this,
      `${prefix}${options.separator}${options.start_index}`,
      inputList ? '*' : inputType,
    )
    return r
  }

  const onConnectionsChange = nodeType.prototype.onConnectionsChange
  /**
   * @param {OnConnectionsChangeParams} args
   */
  nodeType.prototype.onConnectionsChange = function (...args) {
    const [type, slotIndex, isConnected, link, ioSlot] = args

    options.link = link
    options.ioSlot = ioSlot
    const r = onConnectionsChange
      ? onConnectionsChange.apply(this, [
          type,
          slotIndex,
          isConnected,
          link,
          ioSlot,
        ])
      : undefined
    options.DEBUG = {
      node: this,
      type,
      slotIndex,
      isConnected,
      link,
      ioSlot,
    }

    dynamic_connection(
      this,
      slotIndex,
      isConnected,
      `${prefix}${options.separator}`,
      inputType,
      options,
    )
    return r
  }
}

/**
 * Main logic around dynamic inputs
 *
 * @param {LGraphNode} node - The target node
 * @param {number} index - The slot index of the currently changed connection
 * @param {bool} connected - Was this event connecting or disconnecting
 * @param {string} [connectionPrefix] - The common prefix of the dynamic inputs
 * @param {string|[string]} [connectionType] - The type of the dynamic connection
 * @param {{start_index?:number, link?:LLink, ioSlot?:INodeInputSlot | INodeOutputSlot}} [opts] - extra options
 */
export const dynamic_connection = (
  node,
  index,
  connected,
  connectionPrefix = 'input_',
  connectionType = '*',
  opts = undefined,
) => {
  /* {{start_index:number, link?:LLink, ioSlot?:INodeInputSlot | INodeOutputSlot}} [opts] - extra options*/
  const options = Object.assign(
    {
      start_index: 1,
    },
    opts || {},
  )

  if (node.inputs.length > 0 && !isDynamicInput(node.inputs[index])) {
    return
  }

  const listConnection = typeof connectionType === 'object'

  const conType = listConnection ? '*' : connectionType
  const nameArray = options.nameArray || []

  const clean_inputs = () => {
    if (node.id < 0) return // being duplicated
    if (node.inputs.length === 0) return

    let w_count = node.widgets?.length || 0
    let i_count = node.inputs?.length || 0
    infoLogger(`Cleaning inputs: [BEFORE] (w: ${w_count} | inputs: ${i_count})`)

    const to_remove = []
    for (let n = 1; n < node.inputs.length; n++) {
      const element = node.inputs[n]
      if (!element.link && isDynamicInput(element)) {
        if (node.widgets) {
          const w = node.widgets.find((w) => w.name === element.name)
          if (w) {
            w.onRemoved?.()
            node.widgets.length = node.widgets.length - 1
          }
        }
        infoLogger(`Removing input ${n}`)
        to_remove.push(n)
      }
    }
    for (let i = 0; i < to_remove.length; i++) {
      const id = to_remove[i]
      try {
        node.removeInput(id)
        i_count -= 1
      } catch (err) {
        errorLogger('Cannot remove input', err)
      }
    }
    node.inputs.length = i_count

    w_count = node.widgets?.length || 0
    i_count = node.inputs?.length || 0
    infoLogger(`Cleaning inputs: [AFTER] (w: ${w_count} | inputs: ${i_count})`)

    infoLogger('Cleaning inputs: making it sequential again')
    // make inputs sequential again
    let prefixed_idx = options.start_index
    for (let i = 0; i < node.inputs.length; i++) {
      let name = ''
      // rename only prefixed inputs
      if (node.inputs[i].name.startsWith(connectionPrefix)) {
        // prefixed => rename and increase index
        name = `${connectionPrefix}${prefixed_idx}`
        prefixed_idx += 1
      } else {
        // not prefixed => keep same name
        name = node.inputs[i].name
      }

      if (nameArray.length > 0) {
        name = i < nameArray.length ? nameArray[i] : name
      }

      // preserve label if it exists
      node.inputs[i].label = node.inputs[i].label || name
      node.inputs[i].name = name
    }
  }
  if (!connected) {
    if (!options.link) {
      infoLogger('Disconnecting', { options })

      clean_inputs()
    } else {
      if (!options.ioSlot.link) {
        node.connectionTransit = true
      } else {
        node.connectionTransit = false
        clean_inputs()
      }
      infoLogger('Reconnecting', { options })
    }
  }

  if (connected) {
    if (options.link) {
      const { from, to, type } = nodesFromLink(node, options.link)
      if (type === 'outgoing') return
      infoLogger('Connecting', { options, from, to, type })
    } else {
      infoLogger('Connecting', { options })
    }

    if (node.connectionTransit) {
      infoLogger('In Transit')
      node.connectionTransit = false
    }

    // Remove inputs and their widget if not linked.
    clean_inputs()

    if (node.inputs.length === 0) return
    // add an extra input
    if (node.inputs[node.inputs.length - 1].link !== null) {
      const nextIndex = node.inputs.reduce(
        (acc, cur) => (isDynamicInput(cur) ? ++acc : acc),
        0,
      )

      const name =
        nextIndex < nameArray.length
          ? nameArray[nextIndex]
          : `${connectionPrefix}${nextIndex + options.start_index}`

      infoLogger(`Adding input ${nextIndex + 1} (${name})`)
      addDynamicInput(node, name, conType)
    }
  }
}
// #endregion

// #region color utils
export function isColorBright(rgb, threshold = 240) {
  const brightess = getBrightness(rgb)
  return brightess > threshold
}

function getBrightness(rgbObj) {
  return Math.round(
    (Number.parseInt(rgbObj[0]) * 299 +
      Number.parseInt(rgbObj[1]) * 587 +
      Number.parseInt(rgbObj[2]) * 114) /
      1000,
  )
}
// #endregion

// #region html/css utils

/**
 * Calculate total height of DOM element child
 *
 * @param {HTMLElement} parentElement - The target dom element
 * @returns {number} the total height
 */
export function calculateTotalChildrenHeight(parentElement) {
  let totalHeight = 0

  if (!parentElement || !parentElement.children) {
    return 0
  }

  for (const child of parentElement.children) {
    const style = window.getComputedStyle(child)

    const height = Number.parseFloat(style.height)
    const marginTop = Number.parseFloat(style.marginTop)
    const marginBottom = Number.parseFloat(style.marginBottom)

    totalHeight += height + marginTop + marginBottom
  }

  return Math.ceil(totalHeight)
}

export const loadScript = (
  FILE_URL,
  async = true,
  type = 'text/javascript',
) => {
  return new Promise((resolve, reject) => {
    try {
      // Check if the script already exists
      let scriptEle = document.querySelector(`script[src="${FILE_URL}"]`)
      if (scriptEle) {
        scriptEle.addEventListener('load', (_ev) => {
          resolve({ status: true })
        })
        return
      }

      scriptEle = document.createElement('script')
      scriptEle.type = type
      scriptEle.async = async
      scriptEle.src = FILE_URL

      scriptEle.addEventListener('load', (_ev) => {
        resolve({ status: true })
      })

      scriptEle.addEventListener('error', (_ev) => {
        reject({
          status: false,
          message: `Failed to load the script ${FILE_URL}`,
        })
      })

      document.body.appendChild(scriptEle)
    } catch (error) {
      reject(error)
    } finally {
      infoLogger(`Finally loaded script: ${FILE_URL}`)
    }
  })
}

// #endregion

// #region documentation widget

const create_documentation_stylesheet = () => {
  const tag = 'mtb-documentation-stylesheet'

  let styleTag = document.head.querySelector(tag)

  if (!styleTag) {
    styleTag = document.createElement('style')
    styleTag.type = 'text/css'
    styleTag.id = tag

    styleTag.innerHTML = `
.documentation-popup {
    background: var(--comfy-menu-bg);
    position: absolute;
    color: var(--fg-color);
    font: 12px monospace;
    line-height: 1.5em;
    padding: 10px;
    border-radius: 6px;
    pointer-events: "inherit";
    z-index: 5;
    overflow: hidden;
}
.documentation-wrapper {
    padding: 0 2em;
    overflow: auto;
    max-height: 100%;
    /* Scrollbar styling for Chrome */
    &::-webkit-scrollbar {
       width: 6px;
    }
    &::-webkit-scrollbar-track {
       background: var(--bg-color);
    }
    &::-webkit-scrollbar-thumb {
       background-color: var(--fg-color);
       border-radius: 6px;
       border: 3px solid var(--bg-color);
    }

    /* Scrollbar styling for Firefox */
    scrollbar-width: thin;
    scrollbar-color: var(--fg-color) var(--bg-color);
    a {
      color: yellow;
    }
    a:visited {
      color: orange;
    }
    a:hover {
      color: red;
    }
}

.documentation-popup img {
  max-width: 100%;
}
.documentation-popup table {
  border-collapse: collapse;
  border: 1px var(--border-color) solid;
}
.documentation-popup th,
.documentation-popup td {
  border: 1px var(--border-color) solid;
}
.documentation-popup th {
  background-color: var(--comfy-input-bg);
}`
    document.head.appendChild(styleTag)
  }
}
let parserPromise
const callbackQueue = []

function runQueuedCallbacks() {
  while (callbackQueue.length) {
    const cb = callbackQueue.shift()
    cb(window.MTB.mdParser)
  }
}

function loadParser(shiki) {
  if (!parserPromise) {
    parserPromise = import(
      shiki
        ? '/mtb_async/mtb_markdown_plus.umd.js'
        : '/mtb_async/mtb_markdown.umd.js'
    )
      .then((_module) =>
        shiki ? MTBMarkdownPlus.getParser() : MTBMarkdown.getParser(),
      )
      .then((instance) => {
        window.MTB.mdParser = instance
        runQueuedCallbacks()
        return instance
      })
      .catch((error) => {
        console.error('Error loading the parser:', error)
      })
  }
  return parserPromise
}

export const ensureMarkdownParser = async (callback) => {
  infoLogger('Ensuring md parser')
  const use_shiki = app.extensionManager.setting.get(
    'mtb.noteplus.use-shiki',
    false,
  )

  if (window.MTB?.mdParser) {
    infoLogger('Markdown parser found')
    callback?.(window.MTB.mdParser)
    return window.MTB.mdParser
  }

  if (!parserPromise) {
    infoLogger('Running promise to fetch parser')

    try {
      loadParser(use_shiki) //.then(() => {
      // callback?.(window.MTB.mdParser)
      // })
    } catch (error) {
      console.error('Error loading the parser:', error)
    }
  } else {
    infoLogger('A similar promise is already running, waiting for it to finish')
  }
  if (callback) {
    callbackQueue.push(callback)
  }

  await await parserPromise

  return window.MTB.mdParser
}

/**
 * Add documentation widget to the given node.
 *
 * This method will add a `docCtrl` property to the node
 * that contains the AbortController that manages all the events
 * defined inside it (global and instance ones) without explicit
 * cleanup method for each.
 *
 * @param {NodeData} nodeData
 * @param {NodeType}  nodeType
 * @param {DocumentationOptions} opts
 */
export const addDocumentation = (
  nodeData,
  nodeType,
  opts = { icon_size: 14, icon_margin: 4 },
) => {
  if (!nodeData.description) {
    infoLogger(
      `Skipping ${nodeData.name} doesn't have a description, skipping...`,
    )
    return
  }

  const options = opts || {}
  const iconSize = options.icon_size || 14
  const iconMargin = options.icon_margin || 4

  let docElement = null
  let wrapper = null

  const onRem = nodeType.prototype.onRemoved

  nodeType.prototype.onRemoved = function () {
    const r = onRem ? onRem.apply(this, []) : undefined

    if (docElement) {
      docElement.remove()
      docElement = null
    }

    if (wrapper) {
      wrapper.remove()
      wrapper = null
    }
    return r
  }

  const drawFg = nodeType.prototype.onDrawForeground

  /**
   * @param {OnDrawForegroundParams} args
   */
  nodeType.prototype.onDrawForeground = function (...args) {
    const [ctx, _canvas] = args
    const r = drawFg ? drawFg.apply(this, args) : undefined

    if (this.flags.collapsed) return r

    // icon position
    const x = this.size[0] - iconSize - iconMargin

    let resizeHandle
    // create it
    if (this.show_doc && docElement === null) {
      create_documentation_stylesheet()

      docElement = document.createElement('div')
      docElement.classList.add('documentation-popup')
      document.body.appendChild(docElement)

      wrapper = document.createElement('div')
      wrapper.classList.add('documentation-wrapper')
      docElement.appendChild(wrapper)

      // wrapper.innerHTML = documentationConverter.makeHtml(nodeData.description)

      ensureMarkdownParser().then(() => {
        MTB.mdParser.parse(nodeData.description).then((e) => {
          wrapper.innerHTML = e
          // resize handle
          resizeHandle = document.createElement('div')
          resizeHandle.classList.add('doc-resize-handle')
          resizeHandle.style.width = '0'
          resizeHandle.style.height = '0'
          resizeHandle.style.position = 'absolute'
          resizeHandle.style.bottom = '0'
          resizeHandle.style.right = '0'

          resizeHandle.style.cursor = 'se-resize'
          resizeHandle.style.userSelect = 'none'

          resizeHandle.style.borderWidth = '15px'
          resizeHandle.style.borderStyle = 'solid'

          resizeHandle.style.borderColor =
            'transparent var(--border-color) var(--border-color) transparent'

          wrapper.appendChild(resizeHandle)
          let isResizing = false

          let startX
          let startY
          let startWidth
          let startHeight

          resizeHandle.addEventListener(
            'mousedown',
            (e) => {
              e.stopPropagation()
              isResizing = true
              startX = e.clientX
              startY = e.clientY
              startWidth = Number.parseInt(
                document.defaultView.getComputedStyle(docElement).width,
                10,
              )
              startHeight = Number.parseInt(
                document.defaultView.getComputedStyle(docElement).height,
                10,
              )
            },

            { signal: this.docCtrl.signal },
          )

          document.addEventListener(
            'mousemove',
            (e) => {
              if (!isResizing) return
              const scale = app.canvas.ds.scale
              const newWidth = startWidth + (e.clientX - startX) / scale
              const newHeight = startHeight + (e.clientY - startY) / scale

              docElement.style.width = `${newWidth}px`
              docElement.style.height = `${newHeight}px`

              this.docPos = {
                width: `${newWidth}px`,
                height: `${newHeight}px`,
              }
            },
            { signal: this.docCtrl.signal },
          )

          document.addEventListener(
            'mouseup',
            () => {
              isResizing = false
            },
            { signal: this.docCtrl.signal },
          )
        })
      })
    } else if (!this.show_doc && docElement !== null) {
      docElement.remove()
      docElement = null
    }

    // reposition
    if (this.show_doc && docElement !== null) {
      const rect = ctx.canvas.getBoundingClientRect()

      const dpi = Math.max(1.0, window.devicePixelRatio)
      const scaleX = rect.width / ctx.canvas.width
      const scaleY = rect.height / ctx.canvas.height
      const transform = new DOMMatrix()
        .scaleSelf(scaleX, scaleY)
        .multiplySelf(ctx.getTransform())
        .translateSelf(this.size[0] * scaleX * dpi, 0)
        .translateSelf(10, -32)

      const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)

      Object.assign(docElement.style, {
        transformOrigin: '0 0',
        transform: scale,
        left: `${transform.a + rect.x + transform.e}px`,
        top: `${transform.d + rect.y + transform.f}px`,
        width: this.docPos ? this.docPos.width : `${this.size[0] * 1.5}px`,
        height: this.docPos?.height,
      })

      if (this.docPos === undefined) {
        this.docPos = {
          width: docElement.style.width,
          height: docElement.style.height,
        }
      }
    }

    ctx.save()
    ctx.translate(x, iconSize - 34)
    ctx.scale(iconSize / 32, iconSize / 32)
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'

    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    ctx.lineWidth = 2.4
    ctx.font = 'bold 36px monospace'
    ctx.fillText('?', 0, 24)

    // ctx.font = `bold ${this.show_doc ? 36 : 24}px monospace`
    // ctx.fillText(`${this.show_doc ? '▼' : '▶'}`, 24, 24)
    ctx.restore()

    return r
  }
  const mouseDown = nodeType.prototype.onMouseDown

  /**
   * @param {OnMouseDownParams} args
   */
  nodeType.prototype.onMouseDown = function (...args) {
    const [_event, localPos, _graphCanvas] = args
    const r = mouseDown ? mouseDown.apply(this, args) : undefined
    const iconX = this.size[0] - iconSize - iconMargin
    const iconY = iconSize - 34
    if (
      localPos[0] > iconX &&
      localPos[0] < iconX + iconSize &&
      localPos[1] > iconY &&
      localPos[1] < iconY + iconSize
    ) {
      // Pencil icon was clicked, open the editor
      // this.openEditorDialog();
      if (this.show_doc === undefined) {
        this.show_doc = true
      } else {
        this.show_doc = !this.show_doc
      }
      if (this.show_doc) {
        this.docCtrl = new AbortController()
      } else {
        this.docCtrl.abort()
      }
      return true // Return true to indicate the event was handled
    }

    return r // Return false to let the event propagate

    // return r;
  }
}

// #endregion

// #region node extensions

/**
 * Extend an object, either replacing the original property or extending it.
 * @param {Object} object - The object to which the property belongs.
 * @param {string} property - The name of the property to chain the callback to.
 * @param {Function} callback - The callback function to be chained.
 */
export function chainCallback(object, property, callback) {
  if (object === undefined) {
    errorLogger('Could not extend undefined object', { object, property })
    return
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function (...args) {
      const r = callback_orig?.apply(this, args)
      const n = callback.apply(this, args)
      return r || n
    }
  } else {
    object[property] = callback
  }
}

/**
 * Appends a callback to the extra menu options of a given node type.
 * @param {NodeType} nodeType
 * @param {(app,options) => ContextMenuItem[]} cb
 */
export function addMenuHandler(nodeType, cb) {
  const getOpts = nodeType.prototype.getExtraMenuOptions
  /**
   * @returns {ContextMenuItem[]} items
   */
  nodeType.prototype.getExtraMenuOptions = function (app, options) {
    const r = getOpts.apply(this, [app, options]) || []
    const newItems = cb.apply(this, [app, options]) || []
    return [...r, ...newItems]
  }
}

/** Prefixes the node title with '[DEPRECATED]' and log the deprecation reason to the console.*/
export const addDeprecation = (nodeType, reason) => {
  const title = nodeType.title
  nodeType.title = `[DEPRECATED] ${title}`
  // console.log(nodeType)

  const styles = {
    title: 'font-size:1.3em;font-weight:900;color:yellow; background: black',
    reason: 'font-size:1.2em',
  }
  console.log(
    `%c!  ${title} is deprecated:%c ${reason}`,
    styles.title,
    styles.reason,
  )
}

// #endregion

// #region Actions API
export const runAction = async (name, ...args) => {
  const req = await api.fetchApi('/mtb/actions', {
    method: 'POST',
    body: JSON.stringify({
      name,
      args,
    }),
  })

  const res = await req.json()
  return res.result
}
export const getServerInfo = async () => {
  const res = await api.fetchApi('/mtb/server-info')
  return await res.json()
}
export const setServerInfo = async (opts) => {
  await api.fetchApi('/mtb/server-info', {
    method: 'POST',
    body: JSON.stringify(opts),
  })
}

// #endregion
