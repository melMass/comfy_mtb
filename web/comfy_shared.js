/**
 * File: comfy_shared.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

import { app } from '../../scripts/app.js'

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

function createLogger(emoji, color, consoleMethod = 'log') {
  return (message, ...args) => {
    if (window.MTB?.DEBUG) {
      console[consoleMethod](
        `%c${emoji} ${message}`,
        `color: ${color};`,
        ...args,
      )
    }
  }
}

export const infoLogger = createLogger('i', 'yellow')
export const warnLogger = createLogger('!', 'orange', 'warn')
export const errorLogger = createLogger('🔥', 'red', 'error')
export const successLogger = createLogger('✅', 'green')

export const log = (...args) => {
  if (window.MTB?.DEBUG) {
    console.debug(...args)
  }
}

//- WIDGET UTILS
export const CONVERTED_TYPE = 'converted-widget'

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
export const setupDynamicConnections = (nodeType, prefix, inputType, opts) => {
  infoLogger('Setting up dynamic connections for', nodeType)
  const options = opts || {}
  const onNodeCreated = nodeType.prototype.onNodeCreated
  const inputList = typeof inputType === 'object'

  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated ? onNodeCreated.apply(this) : undefined
    this.addInput(`${prefix}_1`, inputList ? '*' : inputType)
    return r
  }

  const onConnectionsChange = nodeType.prototype.onConnectionsChange
  nodeType.prototype.onConnectionsChange = function (
    type,
    slotIndex,
    isConnected,
    link,
    ioSlot,
  ) {
    infoLogger(`Connection changed for ${this.type}`, {
      node: this,
      type,
      slotIndex,
      isConnected,
      link,
      ioSlot,
    })
    options.link = link
    options.ioSlot = ioSlot

    const r = onConnectionsChange
      ? onConnectionsChange.apply(
          this,
          type,
          slotIndex,
          isConnected,
          link,
          ioSlot,
        )
      : undefined
    dynamic_connection(
      this,
      slotIndex,
      isConnected,
      `${prefix}_`,
      inputType,
      options,
    )
    return r
  }
}
/**
 * cleanup dynamic inputs
 *
 * @param {import("../../../web/types/litegraph.d.ts").LGraphNode} node - The target node
 * @param {bool} connected - Was this event connecting or disconnecting
 * @param {string} connectionPrefix - The common prefix of the dynamic inputs
 * @param {string|[string]} connectionType - The type of the dynamic connection
 * @param {{nameInput?:[string]}} [opts] - extra options
 */

const clean_dynamic_state = (
  node,
  connected,
  connectionPrefix,
  connectionType,
  opts,
) => {
  infoLogger('CLEANING', { node, connectionPrefix, connectionType, opts })
  const options = opts || {}
  const nameArray = options.nameArray || []

  const listConnection = typeof connectionType === 'object'
  const conType = listConnection ? '*' : connectionType
  infoLogger('connected', connected)

  if (connected) {
    // Remove inputs and their widget if not linked.
    for (let n = 0; n < node.inputs.length; n++) {
      const element = node.inputs[n]
      if (!element.link) {
        if (node.widgets) {
          const w = node.widgets.find((w) => w.name === element.name)
          if (w) {
            w.onRemoved?.()
            node.widgets.length = node.widgets.length - 1
          }
        }
        node.removeInput(n)
      }
    }
  }
  // make inputs sequential again
  for (let i = 0; i < node.inputs.length; i++) {
    let name = `${connectionPrefix}${i + 1}`

    if (nameArray.length > 0) {
      name = i < nameArray.length ? nameArray[i] : name
    }

    node.inputs[i].label = name
    node.inputs[i].name = name
  }
  // add an extra input
  if (node.inputs[node.inputs.length - 1].link !== undefined) {
    const nextIndex = node.inputs.length
    let name = `${connectionPrefix}${nextIndex + 1}`
    if (nameArray.length > 0) {
      name = nextIndex < nameArray.length ? nameArray[nextIndex] : name
    }
    log(`Adding input ${nextIndex + 1} (${name})`)
    node.addInput(name, conType)
  }
}

/**
 * Main logic around dynamic inputs
 *
 * @param {import("../../../web/types/litegraph.d.ts").LGraphNode} node - The target node
 * @param {number} index - The slot index of the currently changed connection
 * @param {bool} connected - Was this event connecting or disconnecting
 * @param {string} [connectionPrefix] - The common prefix of the dynamic inputs
 * @param {string|[string]} [connectionType] - The type of the dynamic connection
 * @param {{nameInput?:[string]}} [opts] - extra options
 */
export const dynamic_connection = (
  node,
  index,
  connected,
  connectionPrefix = 'input_',
  connectionType = '*',
  opts = undefined,
) => {
  infoLogger('MTB Dynamic Connection', {
    node,
    node_inputs: node.inputs,
    index,
    connected,
    connectionPrefix,
    connectionType,
    opts,
  })
  const options = opts || {}
  if (!node.inputs[index].name.startsWith(connectionPrefix)) {
    return
  }

  const listConnection = typeof connectionType === 'object'

  const conType = listConnection ? '*' : connectionType
  const nameArray = options.nameArray || []

  // clean_dynamic_state(
  //   node,
  //   connected,
  //   connectionPrefix,
  //   connectionType,
  //   options,
  // )
  //

  if (connected) {
    // Remove inputs and their widget if not linked.
    for (let n = 0; n < node.inputs.length; n++) {
      const element = node.inputs[n]
      if (!element.link) {
        if (node.widgets) {
          const w = node.widgets.find((w) => w.name === element.name)
          if (w) {
            w.onRemoved?.()
            node.widgets.length = node.widgets.length - 1
          }
        }
        node.removeInput(n)
      }
    }
  }
  // make inputs sequential again
  for (let i = 0; i < node.inputs.length; i++) {
    let name = `${connectionPrefix}${i + 1}`

    if (nameArray.length > 0) {
      name = i < nameArray.length ? nameArray[i] : name
    }

    node.inputs[i].label = name
    node.inputs[i].name = name
  }

  // add an extra input
  if (node.inputs.length === 0) {
    let name = `${connectionPrefix}1`
    if (nameArray.length > 0) {
      name = nameArray.length[0]
    }
    log(`Adding input 1 (${name})`)
    node.addInput(name, conType)
  } else {
    if (node.inputs[node.inputs.length - 1].link !== undefined) {
      const nextIndex = node.inputs.length
      let name = `${connectionPrefix}${nextIndex + 1}`
      if (nameArray.length > 0) {
        name = nextIndex < nameArray.length ? nameArray[nextIndex] : name
      }
      log(`Adding input ${nextIndex + 1} (${name})`)
      node.addInput(name, conType)
    }
  }
}

/**
 * Calculate total height of DOM element child
 *
 * @param {HTMLElement} parentElement - The target dom element
 * @returns {number} the total height
 */
export function calculateTotalChildrenHeight(parentElement) {
  let totalHeight = 0

  for (const child of parentElement.children) {
    const style = window.getComputedStyle(child)

    // Get height as an integer (without 'px')
    const height = Number.parseInt(style.height, 10)

    // Get vertical margin as integers
    const marginTop = Number.parseInt(style.marginTop, 10)
    const marginBottom = Number.parseInt(style.marginBottom, 10)

    // Sum up height and vertical margins
    totalHeight += height + marginTop + marginBottom
  }

  return totalHeight
}
/**
 * Appends a callback to the extra menu options of a given node type.
 * @param {*} nodeType
 * @param {*} cb
 */
export function addMenuHandler(nodeType, cb) {
  const getOpts = nodeType.prototype.getExtraMenuOptions
  nodeType.prototype.getExtraMenuOptions = function (node, options) {
    const r = getOpts.apply(this, [node, options])
    cb.apply(this, [node, options])
    return r
  }
}

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
export function inner_value_change(widget, value, event = undefined) {
  let corrected_value = value
  if (widget.type === 'number' || widget.type === 'BBOX') {
    corrected_value = Number(value)
  } else if (widget.type === 'BOOL') {
    corrected_value = Boolean(value)
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

//- COLOR UTILS
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
//- HTML / CSS UTILS
export const loadScript = (
  FILE_URL,
  async = true,
  type = 'text/javascript',
) => {
  return new Promise((resolve, reject) => {
    try {
      // Check if the script already exists
      const existingScript = document.querySelector(`script[src="${FILE_URL}"]`)
      if (existingScript) {
        resolve({ status: true, message: 'Script already loaded' })
        return
      }

      const scriptEle = document.createElement('script')
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
    }
  })
}

export function defineClass(className, classStyles) {
  const styleSheets = document.styleSheets

  // Helper function to check if the class exists in a style sheet
  function classExistsInStyleSheet(styleSheet) {
    const rules = styleSheet.rules || styleSheet.cssRules
    for (const rule of rules) {
      if (rule.selectorText === `.${className}`) {
        return true
      }
    }
    return false
  }

  // Check if the class is already defined in any of the style sheets
  let classExists = false
  for (const styleSheet of styleSheets) {
    if (classExistsInStyleSheet(styleSheet)) {
      classExists = true
      break
    }
  }

  // If the class doesn't exist, add the new class definition to the first style sheet
  if (!classExists) {
    if (styleSheets[0].insertRule) {
      styleSheets[0].insertRule(`.${className} { ${classStyles} }`, 0)
    } else if (styleSheets[0].addRule) {
      styleSheets[0].addRule(`.${className}`, classStyles, 0)
    }
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
