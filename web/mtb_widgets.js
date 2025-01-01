/**
 * File: mtb_widgets.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

/// <reference path="../types/typedefs.js" />

// TODO: Use the builtin addDOMWidget everywhere appropriate

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

import * as mtb_ui from './mtb_ui.js'
import parseCss from './extern/parse-css.js'
import * as shared from './comfy_shared.js'
import { infoLogger } from './comfy_shared.js'
import { NumberInputWidget } from './numberInput.js'

// NOTE: new widget types registered by MTB Widgets
const newTypes = [/*'BOOL'*/ 'COLOR', 'BBOX']

const deprecated_nodes = {
  //  'Animation Builder':
  //    'Kept to avoid breaking older script but replaced by TimeEngine',
}

const withFont = (ctx, font, cb) => {
  const oldFont = ctx.font
  ctx.font = font
  cb()
  ctx.font = oldFont
}

const calculateTextDimensions = (ctx, value, width, fontSize = 16) => {
  const words = value.split(' ')
  const lines = []
  let currentLine = ''
  for (const word of words) {
    const testLine = currentLine.length === 0 ? word : `${currentLine} ${word}`
    const testWidth = ctx.measureText(testLine).width
    if (testWidth > width) {
      lines.push(currentLine)
      currentLine = word
    } else {
      currentLine = testLine
    }
  }
  if (lines.length === 0) lines.push(value)
  const textHeight = (lines.length + 1) * fontSize
  const maxLineWidth = lines.reduce(
    (maxWidth, line) => Math.max(maxWidth, ctx.measureText(line).width),
    0,
  )
  return { textHeight, maxLineWidth }
}

export function addMultilineWidget(node, name, opts, callback) {
  const inputEl = document.createElement('textarea')
  inputEl.className = 'comfy-multiline-input'
  inputEl.value = opts.defaultVal
  inputEl.placeholder = opts.placeholder || name

  const widget = node.addDOMWidget(name, 'textmultiline', inputEl, {
    getValue() {
      return inputEl.value
    },
    setValue(v) {
      inputEl.value = v
    },
  })
  widget.inputEl = inputEl

  inputEl.addEventListener('input', () => {
    callback?.(widget.value)
    widget.callback?.(widget.value)
  })
  widget.onRemove = () => {
    inputEl.remove()
  }

  return { minWidth: 400, minHeight: 200, widget }
}

export const VECTOR_AXIS = {
  0: 'x',
  1: 'y',
  2: 'z',
  3: 'w',
}

export function addVectorWidgetW(
  node,
  name,
  value,
  vector_size,
  _callback,
  app,
) {
  // const inputEl = document.createElement('div')
  // const vecEl = document.createElement('div')
  //
  // inputEl.style.background = 'red'
  //
  // inputEl.className = 'comfy-vector-container'
  // vecEl.className = 'comfy-vector-input'
  //
  // vecEl.style.display = 'flex'
  // inputEl.appendChild(vecEl)
  const inputs = []

  for (let i = 0; i < vector_size; i++) {
    // const input = document.createElement('input')
    // input.type = 'number'
    // input.value = value[VECTOR_AXIS[i]]
    const input = node.addWidget(
      'number',
      `${name}_${VECTOR_AXIS[i]}`,
      value[VECTOR_AXIS[i]],
      (val) => {},
    )

    inputs.push(input)
    // vecEl.appendChild(input)
  }
  //
  // const widget = node.addDOMWidget(name, 'vector', inputEl, {
  //   getValue() {
  //     return JSON.stringify(widget._value)
  //   },
  //   setValue(v) {
  //     widget._value = v
  //   },
  //   afterResize(node, widget) {
  //     console.log('After resize', { that: this, node, widget })
  //   },
  // })
  //
  // console.log('prev callback', widget.callback)
  // widget.callback = callback
  // widget._value = value
  //
  // for (let i = 0; i < vector_size; i++) {
  //   const input = inputs[i]
  //   input.addEventListener('change', (event) => {
  //     widget._value[VECTOR_AXIS[i]] = Number.parseFloat(event.target.value)
  //     widget.callback?.(widget._value)
  //     node.graph._version++
  //     node.setDirtyCanvas(true, true)
  //   })
  // }
  // // document.body.append(inputEl)
  //
  // widget.inputEl = inputEl
  // widget.vecEl = vecEl
  //
  // inputEl.addEventListener('input', () => {
  //   widget.callback?.(widget.value)
  // })
  //
  return { minWidth: 400, minHeight: 200, widget }
}
export function addVectorWidget(node, name, value, vector_size, callback, app) {
  const inputEl = document.createElement('div')
  const vecEl = document.createElement('div')

  inputEl.className = 'comfy-vector-container'
  vecEl.className = 'comfy-vector-input'
  vecEl.id = 'vecEl'

  vecEl.style.display = 'flex'
  vecEl.style.flexDirection = 'column'
  inputEl.appendChild(vecEl)
  const inputs = []

  //
  // for (let i = 0; i < vector_size; i++) {
  //   const input = document.createElement('input')
  //   input.type = 'number'
  //   input.value = value[VECTOR_AXIS[i]]
  //   inputs.push(input)
  //   vecEl.appendChild(input)
  // }

  const widget = node.addDOMWidget(name, 'vector', inputEl, {
    getValue() {
      return JSON.stringify(widget._value)
    },
    setValue(v) {
      widget._value = v
    },
  })
  const vec = new NumberInputWidget('vecEl', vector_size, true)
  vec.setValue(...Object.values(value))
  vec.onChange = (value) => {
    for (let i = 0; i < value.length; i++) {
      const val = value[i]
      widget._value[VECTOR_AXIS[i]] = Number.parseFloat(val)
    }

    widget.callback?.(widget._value)
    // widget._value[VECTOR_AXIS[index]] = Number.parseFloat(value)
  }

  console.log('prev callback', widget.callback)
  widget.callback = callback
  widget._value = value

  // for (let i = 0; i < vector_size; i++) {
  //   const input = inputs[i]
  //   input.addEventListener('change', (event) => {
  //     widget._value[VECTOR_AXIS[i]] = Number.parseFloat(event.target.value)
  //     widget.callback?.(widget._value)
  //     node.graph._version++
  //     node.setDirtyCanvas(true, true)
  //   })
  // }

  widget.inputEl = inputEl
  widget.vecEl = vecEl
  widget.vec = vec

  return { minWidth: 400, minHeight: 200 * vector_size, widget }
}
export const MtbWidgets = {
  //TODO: complete this properly

  /**
   * Creates a vector widget.
   * @param {string} key - The key for the widget.
   * @param {number[]} [val] - The initial value for the widget.
   * @param {number} size - The size of the vector.
   * @returns {VectorWidget} The vector widget.
   */
  VECTOR: (key, val, size) => {
    shared.infoLogger('Adding VECTOR widget', { key, val, size })
    /** @type {VectorWidget} */
    const widget = {
      name: key,
      type: `vector${size}`,
      y: 0,
      options: { default: Array.from({ length: size }, () => 0.0) },
      _value: val || Array.from({ length: size }, () => 0.0),
      draw: (ctx, node, width, widgetY, height) => {
        ctx.textAlign = 'left'
        ctx.strokeStyle = outline_color
        ctx.fillStyle = background_color
        ctx.beginPath()
        if (show_text)
          ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5])
        else ctx.rect(margin, y, widget_width - margin * 2, H)
        ctx.fill()
        if (show_text) {
          if (!w.disabled) ctx.stroke()
          ctx.fillStyle = text_color
          if (!w.disabled) {
            ctx.beginPath()
            ctx.moveTo(margin + 16, y + 5)
            ctx.lineTo(margin + 6, y + H * 0.5)
            ctx.lineTo(margin + 16, y + H - 5)
            ctx.fill()
            ctx.beginPath()
            ctx.moveTo(widget_width - margin - 16, y + 5)
            ctx.lineTo(widget_width - margin - 6, y + H * 0.5)
            ctx.lineTo(widget_width - margin - 16, y + H - 5)
            ctx.fill()
          }
          ctx.fillStyle = secondary_text_color
          ctx.fillText(w.label || w.name, margin * 2 + 5, y + H * 0.7)
          ctx.fillStyle = text_color
          ctx.textAlign = 'right'
          if (w.type === 'number') {
            ctx.fillText(
              Number(w.value).toFixed(
                w.options.precision !== undefined ? w.options.precision : 3,
              ),
              widget_width - margin * 2 - 20,
              y + H * 0.7,
            )
          } else {
            let v = w.value
            if (w.options.values) {
              let values = w.options.values
              if (values.constructor === Function) values = values()
              if (values && values.constructor !== Array) v = values[w.value]
            }
            ctx.fillText(v, widget_width - margin * 2 - 20, y + H * 0.7)
          }
        }
      },
      get value() {
        return this._value
      },
      set value(val) {
        this._value = val
        this.callback?.(this._value)
      },
    }

    return widget
  },
  BBOX: (key, val) => {
    /** @type {import("./types/litegraph").IWidget} */
    const widget = {
      name: key,
      type: 'BBOX',
      // options: val,
      y: 0,
      value: val?.default || [0, 0, 0, 0],
      options: {},

      draw: function (ctx, _node, widget_width, widgetY, _height) {
        const hide = this.type !== 'BBOX' && app.canvas.ds.scale > 0.5

        const show_text = true
        const outline_color = LiteGraph.WIDGET_OUTLINE_COLOR
        const background_color = LiteGraph.WIDGET_BGCOLOR
        const text_color = LiteGraph.WIDGET_TEXT_COLOR
        const secondary_text_color = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
        const H = LiteGraph.NODE_WIDGET_HEIGHT

        const margin = 15
        const numWidgets = 4 // Number of stacked widgets

        if (hide) return

        for (let i = 0; i < numWidgets; i++) {
          const currentY = widgetY + i * (H + margin) // Adjust Y position for each widget

          ctx.textAlign = 'left'
          ctx.strokeStyle = outline_color
          ctx.fillStyle = background_color
          ctx.beginPath()
          if (show_text)
            ctx.roundRect(margin, currentY, widget_width - margin * 2, H, [
              H * 0.5,
            ])
          else ctx.rect(margin, currentY, widget_width - margin * 2, H)
          ctx.fill()
          if (show_text) {
            if (!this.disabled) ctx.stroke()
            ctx.fillStyle = text_color
            if (!this.disabled) {
              ctx.beginPath()
              ctx.moveTo(margin + 16, currentY + 5)
              ctx.lineTo(margin + 6, currentY + H * 0.5)
              ctx.lineTo(margin + 16, currentY + H - 5)
              ctx.fill()
              ctx.beginPath()
              ctx.moveTo(widget_width - margin - 16, currentY + 5)
              ctx.lineTo(widget_width - margin - 6, currentY + H * 0.5)
              ctx.lineTo(widget_width - margin - 16, currentY + H - 5)
              ctx.fill()
            }
            ctx.fillStyle = secondary_text_color
            ctx.fillText(
              this.label || this.name,
              margin * 2 + 5,
              currentY + H * 0.7,
            )
            ctx.fillStyle = text_color
            ctx.textAlign = 'right'

            ctx.fillText(
              Number(this.value).toFixed(
                this.options?.precision !== undefined
                  ? this.options.precision
                  : 3,
              ),
              widget_width - margin * 2 - 20,
              currentY + H * 0.7,
            )
          }
        }
      },
      mouse: function (event, pos, node) {
        let old_value = this.value
        let x = pos[0] - node.pos[0]
        let y = pos[1] - node.pos[1]
        let width = node.size[0]
        let H = LiteGraph.NODE_WIDGET_HEIGHT
        let margin = 5
        let numWidgets = 4 // Number of stacked widgets

        for (let i = 0; i < numWidgets; i++) {
          let currentY = y + i * (H + margin) // Adjust Y position for each widget

          if (
            event.type == LiteGraph.pointerevents_method + 'move' &&
            this.type == 'BBOX'
          ) {
            if (event.deltaX)
              this.value += event.deltaX * 0.1 * (this.options?.step || 1)
            if (this.options.min != null && this.value < this.options.min) {
              this.value = this.options.min
            }
            if (this.options.max != null && this.value > this.options.max) {
              this.value = this.options.max
            }
          } else if (event.type == LiteGraph.pointerevents_method + 'down') {
            let values = this.options?.values
            if (values && values.constructor === Function) {
              values = this.options.values(w, node)
            }
            let values_list = null

            let delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0
            if (this.type == 'BBOX') {
              this.value += delta * 0.1 * (this.options.step || 1)
              if (this.options.min != null && this.value < this.options.min) {
                this.value = this.options.min
              }
              if (this.options.max != null && this.value > this.options.max) {
                this.value = this.options.max
              }
            } else if (delta) {
              //clicked in arrow, used for combos
              let index = -1
              this.last_mouseclick = 0 //avoids dobl click event
              if (values.constructor === Object)
                index = values_list.indexOf(String(this.value)) + delta
              else index = values_list.indexOf(this.value) + delta
              if (index >= values_list.length) {
                index = values_list.length - 1
              }
              if (index < 0) {
                index = 0
              }
              if (values.constructor === Array) this.value = values[index]
              else this.value = index
            }
          } //end mousedown
          else if (
            event.type == LiteGraph.pointerevents_method + 'up' &&
            this.type == 'BBOX'
          ) {
            let delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0
            if (event.click_time < 200 && delta == 0) {
              this.prompt(
                'Value',
                this.value,
                function (v) {
                  // check if v is a valid equation or a number
                  if (/^[0-9+\-*/()\s]+|\d+\.\d+$/.test(v)) {
                    try {
                      //solve the equation if possible
                      v = eval(v)
                    } catch (e) {}
                  }
                  this.value = Number(v)
                  shared.inner_value_change(this, this.value, event)
                }.bind(w),
                event,
              )
            }
          }

          if (old_value != this.value)
            setTimeout(
              function () {
                shared.inner_value_change(this, this.value, event)
              }.bind(this),
              20,
            )

          app.canvas.setDirty(true)
        }
      },
      computeSize: function (width) {
        return [width, LiteGraph.NODE_WIDGET_HEIGHT * 4]
      },
      // onDrawBackground: function (ctx) {
      //     if (!this.flags.collapsed) return;
      //     this.inputEl.style.display = "block";
      //     this.inputEl.style.top = this.graphcanvas.offsetTop + this.pos[1] + "px";
      //     this.inputEl.style.left = this.graphcanvas.offsetLeft + this.pos[0] + "px";
      // },
      // onInputChange: function (e) {
      //     const property = e.target.dataset.property;
      //     const bbox = this.getInputData(0);
      //     if (!bbox) return;
      //     bbox[property] = parseFloat(e.target.value);
      //     this.setOutputData(0, bbox);
      // }
    }

    widget.desc = 'Represents a Bounding Box with x, y, width, and height.'
    return widget
  },

  COLOR: (key, val, compute = false) => {
    /** @type {import("/types/litegraph").IWidget} */
    const widget = {}
    widget.y = 0
    widget.name = key
    widget.type = 'COLOR'
    widget.options = { default: '#ff0000' }
    widget.value = val || '#ff0000'
    widget.draw = function (ctx, node, widgetWidth, widgetY, height) {
      const hide = this.type !== 'COLOR' && app.canvas.ds.scale > 0.5
      if (hide) {
        return
      }
      const border = 3
      ctx.fillStyle = '#000'
      ctx.fillRect(0, widgetY, widgetWidth, height)
      ctx.fillStyle = this.value
      ctx.fillRect(
        border,
        widgetY + border,
        widgetWidth - border * 2,
        height - border * 2,
      )
      const color = parseCss(this.value.default || this.value)
      if (!color) {
        return
      }
      ctx.fillStyle = shared.isColorBright(color.values, 125) ? '#000' : '#fff'

      ctx.font = '14px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(this.name, widgetWidth * 0.5, widgetY + 14)
    }
    widget.mouse = function (e, pos, node) {
      if (e.type === 'pointerdown') {
        const widgets = node.widgets.filter((w) => w.type === 'COLOR')

        for (const w of widgets) {
          // color picker
          const rect = [w.last_y, w.last_y + 32]
          if (pos[1] > rect[0] && pos[1] < rect[1]) {
            const picker = document.createElement('input')
            picker.type = 'color'
            picker.value = this.value

            Object.assign(picker.style, {
              position: 'fixed',
              left: `${e.clientX}px`,
              top: `${e.clientY}px`,
              height: '0px',
              width: '0px',
              padding: '0px',
              opacity: 0,
            })

            picker.addEventListener('blur', () => {
              this.callback?.(this.value)
              node.graph._version++
              picker.remove()
            })
            picker.addEventListener('input', () => {
              if (!picker.value) return

              this.value = picker.value
              app.canvas.setDirty(true)
            })

            document.body.appendChild(picker)

            requestAnimationFrame(() => {
              picker.showPicker()
              picker.focus()
            })
          }
        }
      }
    }
    widget.computeSize = function (width) {
      return [width, 32]
    }

    return widget
  },

  DEBUG_IMG: (name, val) => {
    const w = {
      name,
      type: 'image',
      value: val,
      draw: function (ctx, node, widgetWidth, widgetY, height) {
        const [cw, ch] = this.computeSize(widgetWidth)
        shared.offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, ch)
      },
      computeSize: function (width) {
        const ratio = this.inputRatio || 1
        if (width) {
          return [width, width / ratio + 4]
        }
        return [128, 128]
      },
      onRemoved: function () {
        if (this.inputEl) {
          this.inputEl.remove()
        }
      },
    }

    w.inputEl = document.createElement('img')
    w.inputEl.src = w.value
    w.inputEl.onload = function () {
      w.inputRatio = w.inputEl.naturalWidth / w.inputEl.naturalHeight
    }
    document.body.appendChild(w.inputEl)
    return w
  },
  DEBUG_STRING: (name, val) => {
    const fontSize = 16
    const w = {
      name,
      type: 'debug_text',

      draw: function (ctx, node, widgetWidth, widgetY, height) {
        // const [cw, ch] = this.computeSize(widgetWidth)
        shared.offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, height)
      },
      computeSize(width) {
        if (!this.value) {
          return [32, 32]
        }
        if (!width) {
          console.debug(`No width ${this.parent.size}`)
        }
        let dimensions
        withFont(app.ctx, `${fontSize}px monospace`, () => {
          dimensions = calculateTextDimensions(app.ctx, this.value, width)
        })
        const widgetWidth = Math.max(
          width || this.width || 32,
          dimensions.maxLineWidth,
        )
        const widgetHeight = dimensions.textHeight * 1.5
        return [widgetWidth, widgetHeight]
      },
      onRemoved: function () {
        if (this.inputEl) {
          this.inputEl.remove()
        }
      },
      get value() {
        return this.inputEl.innerHTML
      },
      set value(val) {
        this.inputEl.innerHTML = val
        this.parent?.setSize?.(this.parent?.computeSize())
      },
    }

    w.inputEl = document.createElement('p')
    w.inputEl.style = `
      text-align: center;
      font-size: ${fontSize}px;
      color: var(--input-text);
      line-height: 1em;
      font-family: monospace;
    `
    w.value = val
    document.body.appendChild(w.inputEl)

    return w
  },
}

/**
 * @returns {import("./types/comfy").ComfyExtension} extension
 */
const mtb_widgets = {
  name: 'mtb.widgets',

  init: async () => {
    infoLogger('Registering mtb.widgets')
    try {
      const msg = await shared.getServerInfo()
      if (!window.MTB) {
        window.MTB = {}
      }
      window.MTB.DEBUG = msg.debug
    } catch (e) {
      console.error('Error:', e)
    }
  },

  setup: () => {
    app.ui.settings.addSetting({
      id: 'mtb.postshot.path',
      category: ['mtb', 'PostShot', 'path'],
      name: 'Path to Postshot CLI',
      type: 'string',
      defaultValue: 'C:/Program Files/Jawset Postshot/bin/postshot-cli.exe',
      tooltip: 'The path to the postshot CLI',
    })

    app.ui.settings.addSetting({
      id: 'mtb.Main.debug-enabled',
      category: ['mtb', 'Main', 'debug-enabled'],
      name: 'Enable Debug (py and js)',
      type: 'boolean',
      defaultValue: false,

      tooltip:
        'This will enable debug messages in the console and in the python console respectively, no need to restart the server, but do reload the webui',
      attrs: {
        style: {
          // fontFamily: 'monospace',
        },
      },
      async onChange(value) {
        if (!window.MTB) {
          window.MTB = {}
        }
        window.MTB.DEBUG = value
        if (value) {
          infoLogger('Enabled DEBUG mode')
        }

        try {
          shared.setServerInfo({ debug: value })
        } catch (err) {
          console.error('Error:', err)
        }
      },
    })
  },

  getCustomWidgets: () => {
    return {
      // BOOL: (node, inputName, inputData, _app) => {
      //   console.debug('Registering bool')
      //
      //   return {
      //     widget: node.addCustomWidget(
      //       MtbWidgets.BOOL(inputName, inputData[1]?.default || false),
      //     ),
      //     minWidth: 150,
      //     minHeight: 30,
      //   }
      // },

      COLOR: (node, inputName, inputData, _app) => {
        console.debug('Registering color')
        return {
          widget: node.addCustomWidget(
            MtbWidgets.COLOR(inputName, inputData[1]?.default || '#ff0000'),
          ),
          minWidth: 150,
          minHeight: 30,
        }
      },
      // BBOX: (node, inputName, inputData, app) => {
      //     console.debug("Registering bbox")
      //     return {
      //         widget: node.addCustomWidget(MtbWidgets.BBOX(inputName, inputData[1]?.default || [0, 0, 0, 0])),
      //         minWidth: 150,
      //         minHeight: 30,
      //     }

      // }
    }
  },
  /**
   * @param {NodeType} nodeType
   * @param {NodeData} nodeData
   * @param {import("./types/comfy").App} app
   */
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // const rinputs = nodeData.input?.required

    let has_custom = false
    if (nodeData.input?.required) {
      for (const i of Object.keys(nodeData.input.required)) {
        const input_type = nodeData.input.required[i][0]

        if (newTypes.includes(input_type)) {
          has_custom = true
          break
        }
      }
    }
    if (has_custom) {
      //- Add widgets on node creation
      const onNodeCreated = nodeType.prototype.onNodeCreated
      nodeType.prototype.onNodeCreated = function (...args) {
        const r = onNodeCreated ? onNodeCreated.apply(this, args) : undefined
        this.serialize_widgets = true
        this.setSize?.(this.computeSize())

        this.onRemoved = function () {
          // When removing this node we need to remove the input from the DOM
          shared.cleanupNode(this)
        }
        return r
      }

      //- Extra menus
      const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions
      nodeType.prototype.getExtraMenuOptions = function (_, options) {
        const r = origGetExtraMenuOptions
          ? origGetExtraMenuOptions.apply(this, arguments)
          : undefined
        if (this.widgets) {
          const toInput = []
          const toWidget = []
          for (const w of this.widgets) {
            if (w.type === shared.CONVERTED_TYPE) {
              //- This is already handled by widgetinputs.js
              // toWidget.push({
              //     content: `Convert ${w.name} to widget`,
              //     callback: () => shared.convertToWidget(this, w),
              // });
            } else if (newTypes.includes(w.type)) {
              const config = nodeData?.input?.required[w.name] ||
                nodeData?.input?.optional?.[w.name] || [w.type, w.options || {}]

              toInput.push({
                content: `Convert ${w.name} to input`,
                callback: () => shared.convertToInput(this, w, config),
              })
            }
          }
          if (toInput.length) {
            options.push(...toInput, null)
          }

          if (toWidget.length) {
            options.push(...toWidget, null)
          }
        }

        return r
      }
    }

    if (!nodeData.name.endsWith('(mtb)')) {
      return
    }
    // console.log('MTB Node', { description: nodeData.description, nodeType })

    shared.addDocumentation(nodeData, nodeType)

    const deprecation = deprecated_nodes[nodeData.name.replace(' (mtb)', '')]

    if (deprecation) {
      shared.addDeprecation(nodeType, deprecation)
    }
    //- Extending Python Nodes
    switch (nodeData.name) {
      //TODO: remove this non sense
      case 'Get Batch From History (mtb)':
      case 'Get Batch From History V2 (mtb)': {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
          const r = onNodeCreated ? onNodeCreated.apply(this, []) : undefined
          const internal_count = this.widgets.find(
            (w) => w.name === 'internal_count',
          )
          shared.hideWidgetForGood(this, internal_count)
          internal_count.afterQueued = function () {
            this.value++
          }

          return r
        }

        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
          const r = onExecuted ? onExecuted.apply(this, message) : undefined
          return r
        }

        break
      }
      case 'Postshot Train (mtb)':
      case 'Postshot Export (mtb)': {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function (...args) {
          const r = onNodeCreated ? onNodeCreated.apply(this, args) : undefined
          const { postshot_cli } = shared.getNamedWidget(this, 'postshot_cli')

          shared.hideWidgetForGood(this, postshot_cli)

          api.getSetting('mtb.postshot.path').then((p) => {
            postshot_cli._value = p
          })
        }

        break
      }
      case 'Save Gif (mtb)':
      case 'Save Animated Image (mtb)': {
        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
          const prefix = 'anything_'
          const r = onExecuted ? onExecuted.apply(this, message) : undefined

          if (this.widgets) {
            const pos = this.widgets.findIndex((w) => w.name === `${prefix}_0`)
            if (pos !== -1) {
              for (let i = pos; i < this.widgets.length; i++) {
                this.widgets[i].onRemoved?.()
              }
              this.widgets.length = pos
            }

            let imgURLs = []
            if (message) {
              if (message.gif) {
                imgURLs = imgURLs.concat(
                  message.gif.map((params) => {
                    return api.apiURL(
                      `/view?${new URLSearchParams(params).toString()}`,
                    )
                  }),
                )
              }
              if (message.apng) {
                imgURLs = imgURLs.concat(
                  message.apng.map((params) => {
                    return api.apiURL(
                      `/view?${new URLSearchParams(params).toString()}`,
                    )
                  }),
                )
              }
              let i = 0
              for (const img of imgURLs) {
                const w = this.addCustomWidget(
                  MtbWidgets.DEBUG_IMG(`${prefix}_${i}`, img),
                )
                w.parent = this
                i++
              }
            }
            const onRemoved = this.onRemoved
            this.onRemoved = () => {
              shared.cleanupNode(this)
              return onRemoved?.()
            }
          }
          this.setSize?.(this.computeSize())
          return r
        }

        break
      }
      case 'Animation Builder (mtb)': {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function (...args) {
          const r = onNodeCreated ? onNodeCreated.apply(this, args) : undefined

          this.changeMode(LiteGraph.ALWAYS)
          const { raw_iteration, raw_loop, total_frames, loop_count } =
            shared.getNamedWidget(
              this,
              'raw_iteration',
              'raw_loop',
              'total_frames',
              'loop_count',
            )

          shared.hideWidgetForGood(this, raw_iteration)
          shared.hideWidgetForGood(this, raw_loop)

          raw_iteration._value = 0

          // const value_preview = this.addCustomWidget(
          // MtbWidgets.DEBUG_STRING('value_preview', 'Idle'),
          // )

          const dom_value_preview = mtb_ui.makeElement('p', {
            fontWeigth: '700',
            textAlign: 'center',
            fontSize: '1.5em',
            margin: 0,
          })
          const value_preview = this.addDOMWidget(
            'value_preview',
            'DISPLAY',
            dom_value_preview,
            {
              hideOnZoom: false,
              setValue: (val) => {
                if (val) {
                  value_preview.element.innerHTML = val
                }
              },
            },
          )
          value_preview.value = 'Idle'

          const dom_loop_preview = mtb_ui.makeElement('p', {
            textAlign: 'center',
            margin: 0,
          })

          const loop_preview = this.addDOMWidget(
            'loop_preview',
            'DISPLAY',
            dom_loop_preview,
            {
              hideOnZoom: false,
              setValue: (val) => {
                if (val) {
                  dom_loop_preview.innerHTML = val
                }
              },
              getValue: () => {
                dom_loop_preview.innerHTML
              },
            },
          )
          loop_preview.value = 'Iteration: Idle'

          const onReset = () => {
            raw_iteration.value = 0
            raw_loop.value = 0

            value_preview.value = 'Idle'
            loop_preview.value = 'Iteration: Idle'

            app.canvas.setDirty(true)
          }

          // reset button
          this.addWidget('button', 'Reset', 'reset', onReset)

          // run button
          this.addWidget('button', 'Queue', 'queue', () => {
            onReset() // this could maybe be a setting or checkbox
            app.queuePrompt(0, total_frames.value * loop_count.value)
            window.MTB?.notify?.(
              `Started a queue of ${total_frames.value} frames (for ${
                loop_count.value
              } loop, so ${total_frames.value * loop_count.value})`,
              5000,
            )
          })

          this.onRemoved = () => {
            shared.cleanupNode(this)
            app.canvas.setDirty(true)
          }

          raw_iteration.afterQueued = function () {
            this.value++
            raw_loop.value = Math.floor(this.value / total_frames.value)

            value_preview.value = `frame: ${
              raw_iteration.value % total_frames.value
            } / ${total_frames.value - 1}`

            if (raw_loop.value + 1 > loop_count.value) {
              loop_preview.value = 'Done 😎!'
            } else {
              loop_preview.value = `current loop: ${raw_loop.value + 1}/${
                loop_count.value
              }`
            }
          }

          return r
        }

        break
      }
      case 'Interpolate Clip Sequential (mtb)': {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function (...args) {
          const r = onNodeCreated
            ? onNodeCreated.apply(this, ...args)
            : undefined
          const addReplacement = () => {
            const input = this.addInput(
              `replacement_${this.widgets.length}`,
              'STRING',
              '',
            )
            console.log(input)
            this.addWidget('STRING', `replacement_${this.widgets.length}`, '')
          }
          //- add
          this.addWidget('button', '+', 'add', (value, widget, node) => {
            console.log('Button clicked', value, widget, node)
            addReplacement()
          })
          //- remove
          this.addWidget('button', '-', 'remove', (value, widget, node) => {
            console.log(`Button clicked: ${value}`, widget, node)
          })

          return r
        }
        break
      }
      case 'Styles Loader (mtb)': {
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
          const r = origGetExtraMenuOptions
            ? origGetExtraMenuOptions.apply(this, arguments)
            : undefined

          const getStyle = async (node) => {
            try {
              const getStyles = await runAction(
                'getStyles',
                node.widgets?.[0].value ? node.widgets[0].value : '',
              )

              const output = await getStyles.json()
              return output?.result
            } catch (e) {
              console.error(e)
            }
          }
          const extracters = [
            {
              content: 'Extract Positive to Text node',
              callback: async () => {
                const style = await getStyle(this)
                if (style && style.length >= 1) {
                  if (style[0]) {
                    window.MTB?.notify?.(
                      `Extracted positive from ${this.widgets[0].value}`,
                    )
                    // const tn = LiteGraph.createNode('Text box')
                    const tn = LiteGraph.createNode('CLIPTextEncode')
                    app.graph.add(tn)
                    tn.title = `${this.widgets[0].value} (Positive)`
                    tn.widgets[0].value = style[0]
                  } else {
                    window.MTB?.notify?.(
                      `No positive to extract for ${this.widgets[0].value}`,
                    )
                  }
                }
              },
            },
            {
              content: 'Extract Negative to Text node',
              callback: async () => {
                const style = await getStyle(this)
                if (style && style.length >= 2) {
                  if (style[1]) {
                    window.MTB?.notify?.(
                      `Extracted negative from ${this.widgets[0].value}`,
                    )
                    const tn = LiteGraph.createNode('CLIPTextEncode')
                    app.graph.add(tn)
                    tn.title = `${this.widgets[0].value} (Negative)`
                    tn.widgets[0].value = style[1]
                  } else {
                    window.MTB.notify(
                      `No negative to extract for ${this.widgets[0].value}`,
                    )
                  }
                }
              },
            },
          ]
          options.push(...extracters)
        }

        break
      }

      //NOTE: dynamic nodes
      case 'Apply Text Template (mtb)': {
        shared.setupDynamicConnections(nodeType, 'var', '*')
        break
      }
      case 'Save Data Bundle (mtb)': {
        shared.setupDynamicConnections(nodeType, 'data', '*') // [MASK,IMAGE]
        break
      }
      case 'Add To Playlist (mtb)': {
        shared.setupDynamicConnections(nodeType, 'video', 'VIDEO')
        break
      }
      case 'Interpolate Condition (mtb)': {
        shared.setupDynamicConnections(nodeType, 'condition', 'CONDITIONING')
        break
      }
      case 'Psd Save (mtb)': {
        shared.setupDynamicConnections(nodeType, 'input_', 'PSDLAYER')
        break
      }
      // case 'Text Encode Frames (mtb)' : {
      //   shared.setupDynamicConnections(nodeType, 'input_', 'IMAGE')
      //   break
      // }
      case 'Stack Images (mtb)':
      case 'Concat Images (mtb)': {
        shared.setupDynamicConnections(nodeType, 'image', 'IMAGE')
        break
      }
      case 'Audio Sequence (mtb)':
      case 'Audio Stack (mtb)': {
        shared.setupDynamicConnections(nodeType, 'audio', 'AUDIO')
        break
      }
      case 'Batch Float Assemble (mtb)':
      case 'Batch Float Math (mtb)':
      case 'Plot Batch Float (mtb)': {
        shared.setupDynamicConnections(nodeType, 'floats', 'FLOATS')
        break
      }
      case 'Batch Sequence (mtb)':
      case 'Batch Sequence Plus (mtb)':
      case 'Batch Merge (mtb)': {
        shared.setupDynamicConnections(nodeType, 'batches', 'IMAGE')

        break
      }
      // TODO: remove this, recommend pythongoss's version that is much better
      case 'Math Expression (mtb)': {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
          const r = onNodeCreated
            ? onNodeCreated.apply(this, arguments)
            : undefined
          this.addInput('x', '*')
          return r
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (
          _type,
          index,
          connected,
          link_info,
        ) {
          const r = onConnectionsChange
            ? onConnectionsChange.apply(this, arguments)
            : undefined
          shared.dynamic_connection(this, index, connected, 'var_', '*', {
            nameArray: ['x', 'y', 'z'],
          })

          //- infer type
          if (link_info) {
            const fromNode = this.graph._nodes.find(
              (otherNode) => otherNode.id !== link_info.origin_id,
            )
            const type = fromNode.outputs[link_info.origin_slot].type
            this.inputs[index].type = type
            // this.inputs[index].label = type.toLowerCase()
          }
          //- restore dynamic input
          if (!connected) {
            this.inputs[index].type = '*'
            this.inputs[index].label = `number_${index + 1}`
          }
        }

        break
      }

      case 'Batch Shape (mtb)':
      case 'Mask To Image (mtb)':
      case 'Text To Image (mtb)': {
        shared.addMenuHandler(nodeType, function (_app, options) {
          /** @type {ContextMenuItem} */
          const item = {
            content: 'swap colors',
            title: 'Swap BG/FG Color ⚡',
            callback: (_menuItem) => {
              const color_w = this.widgets.find((w) => w.name === 'color')
              const bg_w = this.widgets.find(
                (w) => w.name === 'background' || w.name === 'bg_color',
              )

              const color = color_w.value
              const bg = bg_w.value

              color_w.value = bg
              bg_w.value = color
            },
          }

          options.push(item)
          return [item]
        })
        break
      }
      case 'Scene Detect (mtb)': {
        break
      }
      case 'Loop Start (mtb)': {
        const onDrawBackground = nodeType.prototype.onDrawBackground
        nodeType.prototype.onDrawBackground = function (...args) {
          const r = onDrawBackground
            ? onDrawBackground.apply(this, args)
            : undefined
          const [ctx, /*canvas,*/ ..._rest] = args
          if (this.flags.collapsed) return r
          if (!this.computed_flow) {
            const related = new Set([this.id])
            const visited = new Set()
            if (this.outputs[0].links) {
              const initLink = this.outputs[0].links[0]
              const { to: loopEnd } = shared.nodesFromLink(this, initLink)
              const canReachEnd = (node, visited = new Set()) => {
                if (node === loopEnd) return true
                if (visited.has(node.id)) return false
                visited.add(node.id)
                for (const output of node.outputs || []) {
                  if (!output.links) continue
                  for (const linkId of output.links) {
                    const { to: nextNode } = shared.nodesFromLink(node, linkId)
                    if (!nextNode) continue
                    if (canReachEnd(nextNode, visited)) {
                      return true
                    }
                  }
                }
                return false
              }
              const traverseNodes = (node) => {
                if (visited.has(node.id)) return
                visited.add(node.id)

                // can reach the end
                if (node !== this && node !== loopEnd && !canReachEnd(node)) {
                  return
                }

                related.add(node.id)
                for (const output of node.outputs || []) {
                  if (!output.links) continue

                  for (const linkId of output.links) {
                    const { to: nextNode } = shared.nodesFromLink(node, linkId)
                    if (!nextNode) continue

                    traverseNodes(nextNode)
                  }
                }
              }

              traverseNodes(this)
            }
            this.related_to_flow = Array.from(related)
            this.computed_flow = true
          }
          if (this.related_to_flow) {
            ctx.save()
            const points = []
            const padding = 20

            const graph = this.graph
            const offset = this._pos

            for (const nodeId of this.related_to_flow) {
              const node = graph.getNodeById(nodeId)
              if (!node) continue

              const scale = 1.0
              const x = node._pos[0] * scale - offset[0]
              const y = node._pos[1] * scale - offset[1]
              const width = node.size[0] * scale
              const height = node.size[1] * scale
              const scaledPadding = padding * scale
              // console.log({ main: this, x, y, width, height })

              points.push(
                [x - scaledPadding, y - scaledPadding],
                [x + width + scaledPadding, y - scaledPadding],
                [x + width + scaledPadding, y + height + scaledPadding],
                [x - scaledPadding, y + height + scaledPadding],
              )
            }
            // console.log({ points })
            const hull = shared.getConvexHull(points)

            ctx.beginPath()

            ctx.moveTo(hull[0][0], hull[0][1])
            for (let i = 1; i < hull.length; i++) {
              ctx.lineTo(hull[i][0], hull[i][1])
            }

            ctx.closePath()

            ctx.fillStyle = 'rgba(255, 0, 0, 0.1)'
            ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)'
            ctx.lineWidth = 2
            ctx.fill()
            ctx.stroke()

            ctx.restore()
          } else {
            ctx.save()
            ctx.fillStyle = 'red'
            ctx.fillRect(-50, -50, this.size[0] + 100, this.size[1] + 100)
            ctx.fillStyle = 'white'
            ctx.font = 'bold 12px Arial'
            ctx.fillText(
              `pos: ${this.x}x${this.y}`,
              this.size[0] / 2,
              this.size[1],
            )
            ctx.fillText(
              `size:${this._posSize}`,
              this.size[0] / 2,
              this.size[1] - 30,
            )
            ctx.fillText(
              `dpi: ${window.devicePixelRatio}`,
              this.size[0] / 2,
              this.size[1] - 60,
            )
            ctx.fillText(
              `next: ${graph.getNodeById(this.related_to_flow[1])._posSize}`,
              this.size[0] / 2,
              this.size[1] - 90,
            )

            ctx.restore()
          }
          return r
        }
        break
      }
      default: {
        break
      }
    }
  },
}

app.registerExtension(mtb_widgets)
