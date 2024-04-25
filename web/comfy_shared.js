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
export const setupDynamicConnections = (nodeType, prefix, inputType) => {
  const onNodeCreated = nodeType.prototype.onNodeCreated
  // check if it's a list
  const inputList = typeof inputType === 'object'
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
    this.addInput(`${prefix}_1`, inputList ? '*' : inputType)
    return r
  }

  const onConnectionsChange = nodeType.prototype.onConnectionsChange
  nodeType.prototype.onConnectionsChange = function (
    type,
    index,
    connected,
    link_info,
  ) {
    const r = onConnectionsChange
      ? onConnectionsChange.apply(this, arguments)
      : undefined
    dynamic_connection(this, index, connected, `${prefix}_`, inputList)
  }
}
export const dynamic_connection = (
  node,
  index,
  connected,
  connectionPrefix = 'input_',
  connectionType = 'PSDLAYER',
  nameArray = [],
) => {
  if (!node.inputs[index].name.startsWith(connectionPrefix)) {
    return
  }
  const listConnection = typeof connectionType === 'object'

  // remove all non connected inputs
  if (!connected && node.inputs.length > 1) {
    log(`Removing input ${index} (${node.inputs[index].name})`)
    if (node.widgets) {
      const w = node.widgets.find((w) => w.name === node.inputs[index].name)
      if (w) {
        w.onRemoved?.()
        node.widgets.length = node.widgets.length - 1
      }
    }
    node.removeInput(index)

    // make inputs sequential again
    for (let i = 0; i < node.inputs.length; i++) {
      const name =
        i < nameArray.length ? nameArray[i] : `${connectionPrefix}${i + 1}`
      node.inputs[i].label = name
      node.inputs[i].name = name
    }
  }

  // add an extra input
  if (node.inputs[node.inputs.length - 1].link != undefined) {
    const nextIndex = node.inputs.length
    const name =
      nextIndex < nameArray.length
        ? nameArray[nextIndex]
        : `${connectionPrefix}${nextIndex + 1}`

    log(`Adding input ${nextIndex + 1} (${name})`)

    node.addInput(name, listConnection ? '*' : connectionType)
  }
}

export function calculateTotalChildrenHeight(parentElement) {
  let totalHeight = 0

  for (const child of parentElement.children) {
    const style = window.getComputedStyle(child)

    // Get height as an integer (without 'px')
    const height = parseInt(style.height, 10)

    // Get vertical margin as integers
    const marginTop = parseInt(style.marginTop, 10)
    const marginBottom = parseInt(style.marginBottom, 10)

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
  nodeType.prototype.getExtraMenuOptions = function () {
    const r = getOpts.apply(this, arguments)
    cb.apply(this, arguments)
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
      hideWidget(node, w, ':' + widget.name)
    }
  }
}

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
      hideWidgetForGood(node, w, ':' + widget.name)
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
          if (w && w.type != CONVERTED_TYPE) {
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
  if (widget.type == 'number' || widget.type == 'BBOX') {
    value = Number(value)
  } else if (widget.type == 'BOOL') {
    value = Boolean(value)
  }
  widget.value = value
  if (
    widget.options &&
    widget.options.property &&
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
    (parseInt(rgbObj[0]) * 299 +
      parseInt(rgbObj[1]) * 587 +
      parseInt(rgbObj[2]) * 114) /
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

      scriptEle.addEventListener('load', (ev) => {
        resolve({ status: true })
      })

      scriptEle.addEventListener('error', (ev) => {
        reject({
          status: false,
          message: `Failed to load the script ＄{FILE_URL}`,
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
  nodeType.title = '[DEPRECATED] ' + title
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

const create_documentation_stylesheet = () => {
  const tag = 'mtb-documentation-stylesheet'

  let styleTag = document.head.querySelector(tag)

  if (!styleTag) {
    styleTag = document.createElement('style')
    styleTag.type = 'text/css'
    styleTag.id = tag

    styleTag.innerHTML = `
    .documentation-popup {
       background: var(--bg-color);
	     position: absolute;
			 color: var(--fg-color);
			 font: 12px monospace;
			 line-height: 1.5em;
		   padding: 3px;
			 border-radius: 4px;
			 pointer-events: "inherit";
			 z-index: 5;
    	 overflow:scroll;
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
		}
		`
    document.head.appendChild(styleTag)
  }
}
let documentationConverter

/** Add documentation widget to the selected node */
export const addDocumentation = (
  nodeData,
  nodeType,
  opts = { icon_size: 14, icon_margin: 4 },
) => {
  if (!documentationConverter) {
    documentationConverter = new showdown.Converter({
      tables: true,
      strikethrough: true,
      emoji: true,
      ghCodeBlocks: true,
      tasklists: true,
      ghMentions: true,
      smoothLivePreview: true,
      simplifiedAutoLink: true,
      parseImgDimensions: true,
      openLinksInNewWindow: true,
    })
  }

  opts = opts || {}
  const iconSize = opts.icon_size ? opts.icon_size : 14
  const iconMargin = opts.icon_margin ? opts.icon_margin : 4
  let docElement = null
  let offsetX = 0
  let offsetY = 0

  if (!nodeData.description) {
    return
  }
  const drawFg = nodeType.prototype.onDrawForeground
  nodeType.prototype.onDrawForeground = function (ctx, canvas) {
    const r = drawFg ? drawFg.apply(this, arguments) : undefined
    if (this.flags.collapsed) return r

    // icon position
    const x = this.size[0] - iconSize - iconMargin
    // const y = iconMargin * 1.5

    // const questionMark = new Path2D(
    //   'm15.901 25.36h3.84v-3.84h-3.84v3.84zm1.92-15.36c-2.88 0-5.76 2.88-5.76 5.76h3.84c0-.96.96-1.92 1.92-1.92s1.92.96 1.92 1.92c0 1.92-3.84 1.92-3.84 3.84h3.84c1.92-.66 3.84-1.92 3.84-4.8s-2.88-4.8-5.76-4.8zm0-7.68c-8.49 0-15.36 6.87-15.36 15.36s6.87 15.36 15.36 15.36 15.36-6.87 15.36-15.36-6.87-15.36-15.36-15.36zm0 26.88c-6.36 0-11.52-5.16-11.52-11.52s5.16-11.52 11.52-11.52 11.52 5.16 11.52 11.52-5.16 11.52-11.52 11.52z',
    // )
    //
    // ctx.save()

    if (this.show_doc && docElement === null) {
      create_documentation_stylesheet()
      docElement = document.createElement('div')
      docElement.classList.add('documentation-popup')
      docElement.innerHTML = documentationConverter.makeHtml(
        nodeData.description,
      )
      // resize handle
      const resizeHandle = document.createElement('div')
      resizeHandle.style.width = '10px'
      resizeHandle.style.height = '10px'
      resizeHandle.style.background = 'gray'
      resizeHandle.style.position = 'absolute'
      resizeHandle.style.bottom = '0'
      resizeHandle.style.right = '0'
      resizeHandle.style.cursor = 'se-resize'

      // TODO: fix resize logic
      docElement.appendChild(resizeHandle)
      let isResizing = false
      let startX, startY, startWidth, startHeight

      resizeHandle.addEventListener('mousedown', function (e) {
        e.stopPropagation()
        isResizing = true
        startX = e.clientX
        startY = e.clientY
        startWidth = parseInt(
          document.defaultView.getComputedStyle(docElement).width,
          10,
        )
        startHeight = parseInt(
          document.defaultView.getComputedStyle(docElement).height,
          10,
        )
      })

      document.addEventListener('mousemove', function (e) {
        if (!isResizing) return
        const newWidth = startWidth + e.clientX - startX
        const newHeight = startHeight + e.clientY - startY
        offsetX += newWidth - startWidth
        offsetY += newHeight - startHeight

        startWidth = newWidth
        startHeight = newHeight
      })

      document.addEventListener('mouseup', function () {
        isResizing = false
      })
      document.body.appendChild(docElement)
    } else if (!this.show_doc && docElement !== null) {
      docElement.parentNode.removeChild(docElement)
      docElement = null
    }

    if (this.show_doc && docElement !== null) {
      const rect = ctx.canvas.getBoundingClientRect()

      const scaleX = rect.width / ctx.canvas.width
      const scaleY = rect.height / ctx.canvas.height
      const transform = new DOMMatrix()
        .scaleSelf(scaleX, scaleY)
        .multiplySelf(ctx.getTransform())

        .translateSelf(this.size[0] * scaleX, 0)
        .translateSelf(10, -32)
      const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
      Object.assign(docElement.style, {
        transformOrigin: '0 0',
        transform: scale,
        left: `${transform.a + transform.e}px`,
        top: `${transform.d + transform.f}px`,
        width: `${this.size[0] * 2}px`,
        // height: `${(widget.parent?.inputHeight || 32) - (margin * 2)}px`,
        height: `${this.size[1] || this.parent?.inputHeight || 32}px`,

        // background: !node.color ? "" : node.color,
        // color: "blue", //!node.color ? "" : "white",
      })
      // docElement.style.left = 140 - rect.right + "px";
      // docElement.style.top = rect.top + "px";
    }
    ctx.translate(x, iconSize - 34) // Position the icon on the canvas
    ctx.scale(iconSize / 32, iconSize / 32) // Scale the icon to the desired size
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'

    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    ctx.lineWidth = 2.4
    // ctx.stroke(questionMark);
    ctx.font = '36px monospace'
    ctx.fillText('?', 0, 24)
    ctx.restore()

    return r
  }
  const mouseDown = nodeType.prototype.onMouseDown

  nodeType.prototype.onMouseDown = function (e, localPos, canvas) {
    const r = mouseDown ? mouseDown.apply(this, arguments) : undefined
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
      return true // Return true to indicate the event was handled
    }

    return r // Return false to let the event propagate

    // return r;
  }
}
