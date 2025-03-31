/// <reference path="../types/typedefs.js" />

import { app } from '../../scripts/app.js'

import * as shared from './comfy_shared.js'
import { infoLogger, successLogger, errorLogger } from './comfy_shared.js'
import {
  DEFAULT_CSS,
  DEFAULT_HTML,
  DEFAULT_MD,
  DEFAULT_MODE,
  DEFAULT_THEME,
  THEMES,
  CSS_RESET,
  DEMO_CONTENT,
} from './note_plus.constants.js'
import { LocalStorageManager } from './comfy_shared.js'

const storage = new LocalStorageManager('mtb')

/**
 * Uses `@mtb/markdown-parser` (a fork of marked)
 * It is statically stored to avoid having
 * more than 1 instance ever.
 * The size difference between both libraries...
 * ╭───┬────────────────────────────────┬──────────╮
 * │ # │              name              │   size   │
 * ├───┼────────────────────────────────┼──────────┤
 * │ 0 │ web-dist/mtb_markdown_plus.mjs │   1.2 MB │ <- with shiki
 * │ 1 │ web-dist/mtb_markdown.mjs      │  44.7 KB │
 * ╰───┴────────────────────────────────┴──────────╯
 */
let useShiki = storage.get('np-use-shiki', false)

const makeResizable = (dialog) => {
  dialog.style.resize = 'both'
  dialog.style.transformOrigin = 'top left'
  dialog.style.overflow = 'auto'
}

const makeDraggable = (dialog, handle) => {
  let offsetX = 0
  let offsetY = 0
  let isDragging = false

  const onMouseMove = (e) => {
    if (isDragging) {
      dialog.style.left = `${e.clientX - offsetX}px`
      dialog.style.top = `${e.clientY - offsetY}px`
    }
  }

  const onMouseUp = () => {
    isDragging = false
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
  }

  handle.addEventListener('mousedown', (e) => {
    isDragging = true
    offsetX = e.clientX - dialog.offsetLeft
    offsetY = e.clientY - dialog.offsetTop
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  })
}

/** @extends {LGraphNode} */
class NotePlus extends LiteGraph.LGraphNode {
  // same values as the comfy note
  color = LGraphCanvas.node_colors.yellow.color
  bgcolor = LGraphCanvas.node_colors.yellow.bgcolor
  groupcolor = LGraphCanvas.node_colors.yellow.groupcolor

  /* NOTE: this is not serialized and only there to make multiple
   *  note+ nodes in the same graph unique.
   */
  uuid

  /** Stores the dialog observer*/
  resizeObserver

  /** Live update the preview*/
  live = true
  /** DOM height by adding child size together*/
  calculated_height = 0

  /** ????*/
  _raw_html

  /** might not be needed anymore */
  inner

  /** the dialog DOM widget*/
  dialog

  /** widgets*/

  /** used to store the raw value and display the parsed html at the same time*/
  html_widget

  /** hidden widgets for serialization*/
  css_widget
  edit_mode_widget
  theme_widget

  editorsContainer
  /** ACE editors instances*/
  html_editor
  css_editor

  constructor() {
    super()
    this.uuid = shared.makeUUID()

    infoLogger('Constructing Note+ instance')
    shared.ensureMarkdownParser((_p) => {
      this.updateHTML()
    })
    // - litegraph settings
    this.collapsable = true
    this.isVirtualNode = true
    this.shape = LiteGraph.BOX_SHAPE
    this.serialize_widgets = true

    // - default values, serialization is done through widgets
    this._raw_html = DEFAULT_MODE === 'html' ? DEFAULT_HTML : DEFAULT_MD

    // - state
    this.live = true
    this.calculated_height = 0

    // - add widgets
    const cinner = document.createElement('div')
    this.inner = document.createElement('div')

    cinner.append(this.inner)
    this.inner.classList.add('note-plus-preview')
    cinner.style.margin = '0'
    cinner.style.padding = '0'
    this.html_widget = this.addDOMWidget('HTML', 'html', cinner, {
      setValue: (val) => {
        this._raw_html = val
      },
      getValue: () => this._raw_html,
      getMinHeight: () => this.calculated_height, // (the edit button),
      onDraw: () => {
        // HACK: dirty hack for now until it's addressed upstream...
        this.html_widget.element.style.pointerEvents = 'none'
        // NOTE: not sure about this, it avoid the visual "bugs" but scrolling over the wrong area will affect zoom...
        // this.html_widget.element.style.overflow = 'scroll'
      },
      hideOnZoom: false,
    })

    this.setupSerializationWidgets()
    this.setupDialog()
    this.loadAceEditor()
  }

  /**
   * @param {CanvasRenderingContext2D} ctx canvas context
   * @param {any} _graphcanvas
   */
  onDrawForeground(ctx, _graphcanvas) {
    if (this.flags.collapsed) return
    this.drawEditIcon(ctx)
    this.drawSideHandle(ctx)

    // DEBUG BACKGROUND
    // ctx.fillStyle = 'rgba(0, 255, 0, 0.3)'
    // const rect = this.rect
    // ctx.fillRect(rect.x, rect.y, rect.width, rect.height)
  }
  drawSideHandle(ctx) {
    const handleRect = this.sideHandleRect
    const chamfer = 20
    ctx.beginPath()

    // top left
    ctx.moveTo(handleRect.x, handleRect.y + chamfer)
    // top right
    ctx.lineTo(handleRect.x + handleRect.width, handleRect.y)

    // bottom right
    ctx.lineTo(
      handleRect.x + handleRect.width,
      handleRect.y + handleRect.height,
    )
    // bottom left
    ctx.lineTo(handleRect.x, handleRect.y + handleRect.height - chamfer)
    ctx.closePath()

    ctx.fillStyle = 'rgba(255, 255, 255, 0.05)'
    ctx.fill()
  }

  drawEditIcon(ctx) {
    const rect = this.iconRect
    // DEBUG ICON POSITION
    // ctx.fillStyle = 'rgba(0, 255, 0, 0.3)'
    // ctx.fillRect(rect.x, rect.y, rect.width, rect.height)

    const pencilPath = new Path2D(
      'M21.28 6.4l-9.54 9.54c-.95.95-3.77 1.39-4.4.76-.63-.63-.2-3.45.75-4.4l9.55-9.55a2.58 2.58 0 1 1 3.64 3.65z',
    )
    const folderPath = new Path2D(
      'M11 4H6a4 4 0 0 0-4 4v10a4 4 0 0 0 4 4h11c2.21 0 3-1.8 3-4v-5',
    )

    ctx.save()
    ctx.translate(rect.x, rect.y)
    ctx.scale(rect.width / 32, rect.height / 32)
    ctx.strokeStyle = 'rgba(255,255,255,0.4)'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.lineWidth = 2.4
    ctx.stroke(pencilPath)
    ctx.stroke(folderPath)
    ctx.restore()
  }
  /**
   * @param {number} x
   * @param {number} y
   * @param {{x:number,y:number,width:number,height:number}} rect
   * @returns {}
   */
  inRect(x, y, rect) {
    rect = rect || this.iconRect
    return (
      x >= rect.x &&
      x <= rect.x + rect.width &&
      y >= rect.y &&
      y <= rect.y + rect.height
    )
  }
  get rect() {
    return {
      x: 0,
      y: 0,
      width: this.size[0],
      height: this.size[1],
    }
  }
  get sideHandleRect() {
    const w = this.size[0]
    const h = this.size[1]

    const bw = 32
    const bho = 64

    return {
      x: w - bw,
      y: bho,
      width: bw,
      height: h - bho * 1.5,
    }
  }
  get iconRect() {
    const iconSize = 32
    const iconMargin = 16
    return {
      x: this.size[0] - iconSize - iconMargin,
      y: iconMargin * 1.5,
      width: iconSize,
      height: iconSize,
    }
  }
  onMouseDown(_e, localPos, _graphcanvas) {
    if (this.inRect(localPos[0], localPos[1])) {
      this.openEditorDialog()
      return true
    }
    return false
  }

  /* Hidden widgets to store note+ settings in the workflow (stripped in API)*/
  setupSerializationWidgets() {
    infoLogger('Setup Serializing widgets')

    this.edit_mode_widget = this.addWidget(
      'combo',
      'Mode',
      DEFAULT_MODE,
      (me) => successLogger('Updating edit_mode', me),
      {
        values: ['html', 'markdown', 'raw'],
      },
    )
    this.css_widget = this.addWidget('text', 'CSS', DEFAULT_CSS, (val) => {
      successLogger(`Updating css ${val}`)
    })
    this.theme_widget = this.addWidget(
      'text',
      'Theme',
      DEFAULT_THEME,
      (val) => {
        successLogger(`Setting theme ${val}`)
      },
    )

    shared.hideWidgetForGood(this, this.edit_mode_widget)
    shared.hideWidgetForGood(this, this.css_widget)
    shared.hideWidgetForGood(this, this.theme_widget)
  }

  setupDialog() {
    infoLogger('Setup dialog')

    this.dialog = new app.ui.dialog.constructor()
    this.dialog.element.classList.add('comfy-settings')

    Object.assign(this.dialog.element.style, {
      position: 'absolute',
      boxShadow: 'none',
    })

    const subcontainer = this.dialog.textElement.parentElement

    if (subcontainer) {
      Object.assign(subcontainer.style, {
        width: '100%',
      })
    }
    const closeButton = this.dialog.element.querySelector('button')
    closeButton.textContent = 'CANCEL'
    closeButton.id = 'cancel-editor-dialog'
    closeButton.title =
      "Cancel the changes since last opened (doesn't support live mode)"
    closeButton.disabled = this.live

    closeButton.style.background = this.live
      ? 'repeating-linear-gradient(45deg,#606dbc,#606dbc 10px,#465298 10px,#465298 20px)'
      : ''

    const saveButton = document.createElement('button')
    saveButton.textContent = 'SAVE'
    saveButton.onclick = () => {
      this.closeEditorDialog(true)
    }
    closeButton.onclick = () => {
      this.closeEditorDialog(false)
    }
    closeButton.before(saveButton)
  }

  teardownEditors() {
    this.css_editor.destroy()
    this.css_editor.container.remove()

    this.html_editor.destroy()
    this.html_editor.container.remove()
  }

  closeEditorDialog(accept) {
    infoLogger('Closing editor dialog', accept)
    if (accept && !this.live) {
      this.updateHTML(this.html_editor.getValue())
      this.updateCSS(this.css_editor.getValue())
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }
    this.teardownEditors()
    this.dialog.close()
  }

  /**
   * @param {HTMLElement} elem
   */
  hookResize(elem) {
    if (!this.resizeObserver) {
      const observer = () => {
        this.html_editor.resize()
        this.css_editor.resize()
        Object.assign(this.editorsContainer.style, {
          minHeight: `${(this.dialog.element.clientHeight / 100) * 50}px`, //'200px',
        })
      }
      this.resizeObserver = new ResizeObserver(observer).observe(elem)
    }
  }
  openEditorDialog() {
    infoLogger(`Current edit mode ${this.edit_mode_widget.value}`)
    this.hookResize(this.dialog.element)
    const container = document.createElement('div')
    Object.assign(container.style, {
      display: 'flex',
      gap: '10px',
      flexDirection: 'column',
    })

    this.editorsContainer = document.createElement('div')

    Object.assign(this.editorsContainer.style, {
      display: 'flex',
      gap: '10px',
      flexDirection: 'row',
      minHeight: this.dialog.element.offsetHeight, //'200px',
      width: '100%',
    })

    container.append(this.editorsContainer)

    this.dialog.show('')
    this.dialog.textElement.append(container)

    const aceHTML = document.createElement('div')
    aceHTML.id = 'noteplus-html-editor'
    Object.assign(aceHTML.style, {
      width: '100%',
      height: '100%',

      minWidth: '300px',
      minHeight: 'inherit',
    })

    this.editorsContainer.append(aceHTML)

    const aceCSS = document.createElement('div')
    aceCSS.id = 'noteplus-css-editor'
    Object.assign(aceCSS.style, {
      width: '100%',
      height: '100%',
      minHeight: 'inherit',
    })

    this.editorsContainer.append(aceCSS)

    const live_edit = document.createElement('input')
    live_edit.type = 'checkbox'
    live_edit.checked = this.live
    live_edit.onchange = () => {
      this.live = live_edit.checked
      const cancel_button = this.dialog.element.querySelector(
        '#cancel-editor-dialog',
      )
      if (cancel_button) {
        cancel_button.disabled = this.live
        cancel_button.style.background = this.live
          ? 'repeating-linear-gradient(45deg,#606dbc,#606dbc 10px,#465298 10px,#465298 20px)'
          : ''
      }
    }

    //- "Dynamic" elements
    const firstButton = this.dialog.element.querySelector('button')
    const syncUI = () => {
      let convert_to_html =
        this.dialog.element.querySelector('#convert-to-html')
      if (this.edit_mode_widget.value === 'markdown') {
        if (convert_to_html == null) {
          convert_to_html = document.createElement('button')
          convert_to_html.textContent = 'Convert to HTML (NO UNDO!)'
          convert_to_html.id = 'convert-to-html'
          convert_to_html.onclick = () => {
            const select_mode = this.dialog.element.querySelector('#edit_mode')

            const md = this.html_editor.getValue()
            this.edit_mode_widget.value = 'html'
            select_mode.value = 'html'
            MTB.mdParser.parse(md).then((content) => {
              this.html_widget.value = content
              this.html_editor.setValue(content)
              this.html_editor.session.setMode('ace/mode/html')
              this.updateHTML(this.html_widget.value)
              convert_to_html.remove()
            })
          }
          firstButton.before(convert_to_html)
        }
      } else {
        if (convert_to_html != null) {
          convert_to_html.remove()
          convert_to_html = null
        }
      }
      select_mode.value = this.edit_mode_widget.value

      // the header for dragging the dialog
      const header = document.createElement('div')
      header.style.padding = '8px'
      header.style.cursor = 'move'
      header.style.backgroundColor = 'rgba(0,0,0,0.5)'
      header.style.userSelect = 'none'

      header.style.borderBottom = '1px solid #ddd'
      header.textContent = 'MTB Note+ Editor'
      container.prepend(header)
      makeDraggable(this.dialog.element, header)
      makeResizable(this.dialog.element)
    }
    //- combobox
    let theme_select = this.dialog.element.querySelector('#theme_select')
    if (!theme_select) {
      infoLogger('Creating combobox for select')
      theme_select = document.createElement('select')
      theme_select.name = 'theme'
      theme_select.id = 'theme_select'

      const addOption = (label) => {
        const option = document.createElement('option')
        option.value = label
        option.textContent = label
        theme_select.append(option)
      }
      for (const t of THEMES) {
        addOption(t)
      }

      theme_select.addEventListener('change', (event) => {
        const val = event.target.value
        this.setTheme(val)
      })

      container.prepend(theme_select)
    }

    theme_select.value = this.theme_widget.value

    let select_mode = this.dialog.element.querySelector('#edit_mode')

    if (!select_mode) {
      infoLogger('Creating combobox for select')
      select_mode = document.createElement('select')
      select_mode.name = 'mode'
      select_mode.id = 'edit_mode'

      const addOption = (label) => {
        const option = document.createElement('option')
        option.value = label
        option.textContent = label
        select_mode.append(option)
      }
      addOption('markdown')
      addOption('html')

      select_mode.addEventListener('change', (event) => {
        const val = event.target.value
        this.edit_mode_widget.value = val
        if (this.html_editor) {
          this.html_editor.session.setMode(`ace/mode/${val}`)
          this.updateHTML(this.html_editor.getValue())

          syncUI()
        }
      })

      container.append(select_mode)
    }
    select_mode.value = this.edit_mode_widget.value

    syncUI()

    const live_edit_label = document.createElement('label')
    live_edit_label.textContent = 'Live Edit'

    // add a tooltip
    live_edit_label.title =
      'When this is on, the editor will update the note+ whenever you change the text.'

    live_edit_label.append(live_edit)

    // select_mode.before(live_edit_label)
    container.append(live_edit_label)

    this.setupEditors()
  }
  loadAceEditor() {
    shared.loadScript('/mtb_async/ace/ace.js').catch((e) => {
      errorLogger(e)
    })
  }
  onCreate() {
    errorLogger('NotePlus onCreate')
  }
  restoreNodeState(info) {
    this.html_widget.element.id = `note-plus-${this.uuid}`
    this.setMode(this.edit_mode_widget.value)
    this.setTheme(this.theme_widget.value)
    this.updateHTML(this.html_widget.value)
    this.updateCSS(this.css_widget.value)
    if (info?.size) {
      this.setSize(info.size)
    }
  }
  configure(info) {
    super.configure(info)
    infoLogger('Restoring serialized values', info)
    this.restoreNodeState(info)
    // - update view from serialzed data
  }
  onNodeCreated() {
    infoLogger('Node created', this.uuid)
    this.restoreNodeState({})
    // this.html_widget.element.id = `note-plus-${this.uuid}`
    // this.setMode(this.edit_mode_widget.value)
    // this.setTheme(this.theme_widget.value)
    // this.updateHTML(this.html_widget.value) // widget is populated here since we called super
    // this.updateCSS(this.css_widget.value)
  }
  // onRemoved() {
  //   infoLogger('Node removed', this?.uuid)
  // }
  getExtraMenuOptions() {
    const currentMode = this.edit_mode_widget.value
    const newMode = currentMode === 'html' ? 'markdown' : 'html'

    const debugItems = window.MTB?.DEBUG
      ? [
          {
            content: 'Replace with demo content (debug)',
            callback: () => {
              this.html_widget.value = DEMO_CONTENT
            },
          },
        ]
      : []

    return [
      ...debugItems,
      {
        content: `Set to ${newMode}`,
        callback: () => {
          this.edit_mode_widget.value = newMode
          this.updateHTML(this.html_widget.value)
        },
      },
    ]
  }

  _setupEditor(editor) {
    this.setTheme(this.theme_widget.value)

    editor.setShowPrintMargin(false)
    editor.session.setUseWrapMode(true)
    editor.renderer.setShowGutter(false)
    editor.session.setTabSize(4)
    editor.session.setUseSoftTabs(true)
    editor.setFontSize(14)
    editor.setReadOnly(false)
    editor.setHighlightActiveLine(false)
    editor.setShowFoldWidgets(true)
    return editor
  }

  setTheme(theme) {
    this.theme_widget.value = theme
    if (this.html_editor) {
      this.html_editor.setTheme(`ace/theme/${theme}`)
    }
    if (this.css_editor) {
      this.css_editor.setTheme(`ace/theme/${theme}`)
    }
  }

  setMode(mode) {
    this.edit_mode_widget.value = mode
    if (this.html_editor) {
      this.html_editor.session.setMode(`ace/mode/${mode}`)
    }

    this.updateHTML(this.html_widget.value)
  }
  setupEditors() {
    infoLogger('NotePlus setupEditor')

    this.html_editor = ace.edit('noteplus-html-editor')

    this.css_editor = ace.edit('noteplus-css-editor')
    this.css_editor.session.setMode('ace/mode/css')

    this.setMode(DEFAULT_MODE)

    this._setupEditor(this.html_editor)
    this._setupEditor(this.css_editor)

    this.css_editor.session.on('change', (_delta) => {
      // delta.start, delta.end, delta.lines, delta.action
      if (this.live) {
        this.updateCSS(this.css_editor.getValue())
      }
    })

    this.html_editor.session.on('change', (_delta) => {
      // delta.start, delta.end, delta.lines, delta.action
      if (this.live) {
        this.updateHTML(this.html_editor.getValue())
      }
    })

    this.html_editor.setValue(this.html_widget.value)
    this.css_editor.setValue(this.css_widget.value)
  }

  scopeCss(css, scopeId) {
    return css
      .split('}')
      .map((rule) => {
        if (rule.trim() === '') {
          return ''
        }
        const scopedRule = rule
          .split('{')
          .map((segment, index) => {
            if (index === 0) {
              return `#${scopeId} ${segment.trim()}`
            }
            return `{${segment.trim()}`
          })
          .join(' ')
        return `${scopedRule}}`
      })
      .join('\n')
  }
  getCssDom() {
    const styleTagId = `note-plus-stylesheet-${this.uuid}`

    let styleTag = document.head.querySelector(`#${styleTagId}`)

    if (!styleTag) {
      styleTag = document.createElement('style')
      styleTag.type = 'text/css'
      styleTag.id = styleTagId
      document.head.appendChild(styleTag)
      infoLogger(`Creating note-plus-stylesheet-${this.uuid}`, styleTag)
    }

    return styleTag
  }
  calculateHeight() {
    this.calculated_height = shared.calculateTotalChildrenHeight(
      this.html_widget.element,
    )
    this.setDirtyCanvas(true, true)
  }
  updateCSS(css) {
    infoLogger('NotePlus updateCSS')
    // this.html_widget.element.style = css
    const scopedCss = this.scopeCss(
      `${CSS_RESET}\n${css}`,
      `note-plus-${this.uuid}`,
    )

    const cssDom = this.getCssDom()
    cssDom.innerHTML = scopedCss

    this.css_widget.value = css
    this.calculateHeight()
    infoLogger('NotePlus updateCSS', this.calculated_height)
    // this.setSize(this.computeSize())
  }

  parserInitiated() {
    if (window.MTB?.mdParser) return true
    return false
  }

  /** to easilty swap purification methods*/
  purify(content) {
    return DOMPurify.sanitize(content, {
      ADD_TAGS: ['iframe', 'detail', 'summary'],
    })
  }

  updateHTML(val) {
    if (!this.parserInitiated()) {
      return
    }
    val = val || this.html_widget.value
    const isHTML = this.edit_mode_widget.value === 'html'

    const cleanHTML = this.purify(val)

    const value = isHTML
      ? cleanHTML
      : cleanHTML.replaceAll('&gt;', '>').replaceAll('&lt;', '<')
    // .replaceAll('&amp;', '&')
    // .replaceAll('&quot;', '"')
    // .replaceAll('&#039;', "'")

    this.html_widget.value = value

    if (isHTML) {
      this.inner.innerHTML = value
    } else {
      MTB.mdParser.parse(value).then((e) => {
        this.inner.innerHTML = e
      })
    }
    // this.html_widget.element.innerHTML = `<div id="note-plus-spacer"></div>${value}`
    this.calculateHeight()
    // this.setSize(this.computeSize())
  }
}

app.registerExtension({
  name: 'mtb.noteplus',
  setup: () => {
    app.ui.settings.addSetting({
      id: 'mtb.noteplus.use-shiki',
      category: ['mtb', 'Note+', 'use-shiki'],
      name: 'Use shiki to highlight code',
      tooltip:
        'This will load a larger version of @mtb/markdown-parser that bundles shiki, it supports all shiki transformers (supported langs: html,css,python,markdown)',

      type: 'boolean',
      defaultValue: false,
      attrs: {
        style: {
          // fontFamily: 'monospace',
        },
      },
      async onChange(value) {
        storage.set('np-use-shiki', value)
        useShiki = value
      },
    })
  },

  registerCustomNodes() {
    LiteGraph.registerNodeType('Note Plus (mtb)', NotePlus)

    NotePlus.category = 'mtb/utils'
    NotePlus.title = 'Note+ (mtb)'

    NotePlus.title_mode = LiteGraph.NO_TITLE
  },
})
