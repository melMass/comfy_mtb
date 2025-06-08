/// <reference path="../types/typedefs.js" />

import { app } from '../../scripts/app.js'

import * as shared from './comfy_shared.js'
import * as mtb_ui from './mtb_ui.js'
import { warnLogger, infoLogger, errorLogger } from './comfy_shared.js'
import {
  DEFAULT_CSS,
  // DEFAULT_HTML,
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

let _css_reset = app.extensionManager.setting.get(
  'mtb.noteplus.css-reset',
  CSS_RESET,
)

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
  preview_widget

  editorsContainer
  /** ACE editors instances*/
  html_editor
  css_editor

  /** quick edit mode*/
  isEditing = false
  quickEditor = null
  quickEditorContainer = null

  constructor() {
    super()
    this.uuid = shared.makeUUID()

    infoLogger('Constructing Note+ instance')
    shared.ensureMarkdownParser((_p) => {
      this.updateHTML()
    })
    // - litegraph settings
    this.properties = {
      css: DEFAULT_CSS,
      theme: DEFAULT_THEME,
    }
    this.collapsable = true
    this.isVirtualNode = true
    this.shape = LiteGraph.BOX_SHAPE
    this.serialize_widgets = true

    // - default values, serialization is done through widgets
    this._raw_html = DEFAULT_MD

    // - state
    this.calculated_height = 0

    // -
    this.setupDebounce(100)

    // - add widgets
    const cinner = document.createElement('div')
    this.inner = document.createElement('div')

    this.title = 'Note+'

    cinner.append(this.inner)
    this.inner.classList.add('note-plus-preview')
    cinner.style.margin = '0'
    cinner.style.padding = '0'
    this.preview_widget = this.addDOMWidget('HTML', 'html', cinner, {
      setValue: (val) => {
        this._raw_html = val
      },
      getValue: () => this._raw_html,
      getMinHeight: () => {
        this.calculateHeight()
        return this.calculated_height
      },

      onDraw: () => {
        // HACK: dirty hack for now until it's addressed upstream...
        // TODO: check if still needed
        this.preview_widget.element.style.pointerEvents = 'none'
        // NOTE: not sure about this, it avoid the visual "bugs" but scrolling over the wrong area will affect zoom...
        // this.html_widget.element.style.overflow = 'scroll'
      },
      hideOnZoom: false,
    })

    this.setupDialog()
    this.loadAceEditor()
    this.setupDoubleClickEdit()
  }

  /**
   * @param {CanvasRenderingContext2D} ctx canvas context
   * @param {any} _graphcanvas
   */
  onDrawForeground(ctx, _graphcanvas) {
    if (this.flags.collapsed) return
    this.drawEditIcon(ctx)

    // DEBUG BACKGROUND
    // ctx.fillStyle = 'rgba(0, 255, 0, 0.3)'
    // const rect = this.rect
    // ctx.fillRect(rect.x, rect.y, rect.width, rect.height)
  }

  setupDebounce(ms) {
    if (this.calculatedHeight) {
      this.calculateHeight.cancel()
    }
    this.calculateHeight = shared.debounce(() => {
      this.calculated_height = shared.calculateTotalChildrenHeight(
        this.inner,
      )
    }, ms)
  }

  // drawSideHandle(ctx) {

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
  get iconRect() {
    let icon = {
      size: 24,
      margin: 5,
      yoffset: -25,
    }

    if (window.inspector) {
      if (window.inspector.get('note_icon') !== null) {
        icon = window.inspector.get('note_icon')
      } else {
        window.inspector.set('note_icon', icon)
        window.inspector.subscribe('note_icon', (k, v) => {
          console.log(v)
        })

        window.inspector.set('note_icon_debug', true)

        window.inspector.subscribe('note_icon_debug', (k, v) => {
          this.debug_icon = v
        })
      }
    }

    return {
      x: this.size[0] - icon.size - icon.margin,
      y: icon.yoffset,
      width: icon.size,
      height: icon.size,
    }
  }
  onMouseDown(_e, localPos, _graphcanvas) {
    if (this.inRect(localPos[0], localPos[1])) {
      this.openEditorDialog()
      return true
    }
    return false
  }

  onDblClick(e, localPos, graphcanvas) {
    this.openEditorDialog()
    return true
  }

  setupDialog() {
    infoLogger('Setup dialog')

    this.dialog = new app.ui.dialog.constructor()
    this.dialog.element.style.width = '680px'
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
    const saveButton = this.dialog.element.querySelector('button')
    saveButton.textContent = 'SAVE'
    saveButton.onclick = () => {
      this.closeEditorDialog(true)
    }
  }

  teardownEditors() {
    this.css_editor.destroy()
    this.css_editor.container.remove()

    this.html_editor.destroy()
    this.html_editor.container.remove()

    this.html_editor = null
    this.css_editor = null
  }

  closeEditorDialog(accept) {
    infoLogger('Closing editor dialog', accept)
    if (accept) {
      this.updateHTML(this.html_editor.getValue())
      this.updateCSS(this.css_editor.getValue())
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }
    this.teardownEditors()
    this.dialog.close()
    this.setupDebounce(100)
  }

  /**
   * @param {HTMLElement} elem
   */
  hookResize(elem) {
    if (!this.resizeObserver) {
      const observer = () => {
        Object.assign(this.editorsContainer.style, {
          minHeight: `${(this.dialog.element.clientHeight / 100) * 50}px`, //'200px',
        })
        // avoid the few ticks that can happen between destroying the editors
        // and the watched dialog
        if (this.html_editor) {
          this.html_editor.resize()
          this.css_editor.resize()
        }
      }
      this.resizeObserver = new ResizeObserver(observer).observe(elem)
    }
  }
  openEditorDialog() {
    this.setupDebounce(500)
    this.hookResize(this.dialog.element)
    const container = mtb_ui.makeElement('div', {
      display: 'flex',
      gap: '10px',
      flexDirection: 'column',
    })

    this.editorsContainer = mtb_ui.makeElement(
      'div',
      {
        display: 'flex',
        gap: '10px',
        flexDirection: 'row',
        minHeight: this.dialog.element.offsetHeight, //'200px',
        width: '100%',
      },
      container,
    )

    this.dialog.show('')
    this.dialog.textElement.append(container)

    mtb_ui.makeElement(
      'div.#noteplus-html-editor',
      {
        width: '100%',
        height: '100%',

        minWidth: '300px',
        minHeight: 'inherit',
      },
      this.editorsContainer,
    )

    mtb_ui.makeElement(
      'div.#noteplus-css-editor',
      {
        width: '100%',
        height: '100%',
        minHeight: 'inherit',
      },
      this.editorsContainer,
    )


    //- "Dynamic" elements
    const syncUI = () => {
      // let convert_to_html =

      // the header for dragging the dialog
      const header = mtb_ui.makeElement('div', {
        padding: '8px',
        cursor: 'move',
        backgroundColor: 'rgba(0,0,0,0.5)',
        userSelect: 'none',
        borderBottom: '1px solid #ddd',
      })

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

    theme_select.value = this.properties.theme

    // let select_mode = this.dialog.element.querySelector('#edit_mode')

    syncUI()


    this.setupEditors()
  }
  loadAceEditor() {
    if (window.MTB?.ace_loaded) {
      return
    }
    let NEED_PATCH = false
    if (window.ace) {
      infoLogger(
        'A global ace was found in scope not loaded by mtb, this might lead to issues.',
      )
      NEED_PATCH = true
      // window._backupAce = window.ace
      // window.ace = null
    }

    shared
      .loadScript('/mtb_async/ace/ace.js')
      .then((m) => {
        infoLogger('ACE was loaded', m)
        // window.MTB_ACE = window.ace
        if (!window.MTB) {
          errorLogger(
            "window.MTB not found, this shouldn't happen at this stage",
          )
          window.MTB = {}
        }
        window.MTB.ace_loaded = true
        this.setMode('markdown')
        this.setTheme()
        this.updateView()
      })
      .catch((e) => {
        errorLogger(e)
      })
      .finally(() => {
        if (NEED_PATCH) {
          // console.log('Patching back window object')
          // window.ace = window._backupAce
        } else {
          // delete window.ace
        }
      })
  }
  configure(info) {
    super.configure(info)
    infoLogger('Restoring serialized values', info)
    this.preview_widget.element.id = `note-plus-${this.uuid}`
    this.updateView()
  }
  getExtraMenuOptions() {
    const debugItems = window.MTB?.DEBUG
      ? [
          {
            content: 'Replace with demo content (debug)',
            callback: () => {
              this.preview_widget.value = DEMO_CONTENT
            },
          },
        ]
      : []

    return [...debugItems]
  }

  get fontSize() {
    return app.extensionManager.setting.get('Comfy.TextareaWidget.FontSize', 16)
  }

  _setupEditor(editor) {
    this.setTheme()

    editor.setShowPrintMargin(false)
    editor.session.setUseWrapMode(true)
    editor.renderer.setShowGutter(false)
    editor.session.setTabSize(4)
    editor.session.setUseSoftTabs(true)
    editor.setFontSize(this.fontSize)
    editor.setReadOnly(false)
    editor.setHighlightActiveLine(false)
    editor.setShowFoldWidgets(true)
    return editor
  }

  setTheme(theme) {
    theme = theme || this.properties.theme
    this.properties.theme = theme
    if (this.html_editor) {
      this.html_editor.setTheme(`ace/theme/${theme}`)
    }
    if (this.css_editor) {
      this.css_editor.setTheme(`ace/theme/${theme}`)
    }
    if (this.quickEditor) {
      this.quickEditor.setTheme(`ace/theme/${theme}`)
    }
  }

  updateView() {
    this.updateHTML()
    this.updateCSS()
    this.calculateHeight()
  }

  setMode(mode) {
    if (this.html_editor) {
      this.html_editor.session.setMode(`ace/mode/${mode}`)
    }
    if (this.quickEditor) {
      this.quickEditor.session.setMode(`ace/mode/${mode}`)
    }

    this.updateView()
  }

  setupEditors() {
    infoLogger('NotePlus setupEditor')

    if (!window.MTB?.ace_loaded) {
      errorLogger('ACE editor not loaded. Cannot set up editors.')
      return
    }

    if (!this.html_editor) {
      this.html_editor = ace.edit('noteplus-html-editor')
      this._setupEditor(this.html_editor)
      this.html_editor.session.on('change', (_delta) => {
        this.updateHTML(this.html_editor.getValue())
      })
    } else {
      infoLogger('Reusing html editor')
    }

    if (!this.css_editor) {
      this.css_editor = ace.edit('noteplus-css-editor')
      this.css_editor.session.setMode('ace/mode/css')
      this._setupEditor(this.css_editor)
      this.css_editor.session.on('change', (_delta) => {
        this.updateCSS(this.css_editor.getValue())
      })
    }

    this.setMode(DEFAULT_MODE)

    this.html_editor.setValue(this.preview_widget.value, -1)
    this.css_editor.setValue(this.properties.css, -1)
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
  updateCSS(css) {
    infoLogger('NotePlus updateCSS')
    css = css || this.properties.css
    // this.html_widget.element.style = css
    const scopedCss = this.scopeCss(
      `${_css_reset}\n${css}`,
      `note-plus-${this.uuid}`,
    )

    const cssDom = this.getCssDom()
    cssDom.innerHTML = scopedCss

    this.properties.css = css
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
    if (val) {
      this._raw_html = val
    }

    if (
      !this.parserInitiated() ||
      !window.MTB?.ace_loaded ||
      !this.preview_widget
    ) {
      return
    }
    val = val || this._raw_html

    const cleanHTML = this.purify(val)

    const value = cleanHTML.replaceAll('&gt;', '>').replaceAll('&lt;', '<')
    // .replaceAll('&amp;', '&')
    // .replaceAll('&quot;', '"')
    // .replaceAll('&#039;', "'")

    this.preview_widget.value = value

    MTB.mdParser
      .parse(value)
      .then((e) => {
        this.inner.innerHTML = e
      })
      .catch((e) => {
        if (e.name === 'ShikiError') {
          warnLogger(e.message)
          return
        }
        throw e
      })
  }

  /**
   * Attaches the double-click listener to the preview area.
   */
  setupDoubleClickEdit() {
    this.preview_widget.element.addEventListener('dblclick', (e) => {
      e.stopPropagation()
      e.preventDefault()

      if (this.isEditing) {
        return
      }

      if (!window.MTB?.ace_loaded) {
        errorLogger('Ace editor not loaded. Cannot open quick edit.')
        return
      }

      this.enterEditMode()
    })
  }

  /**
   * Switches the preview div to an editable textarea.
   */
  enterEditMode() {
    this.isEditing = true

    this.setupDebounce(10000)

    const id = `noteplus-quick-editor-${this.uuid}`
    this.quickEditorContainer = mtb_ui.makeElement(`div.#${id}`, {
      position: 'absolute',
      top: '0',
      pointerEvents: 'auto',
      left: '0',
      right: '0',
      bottom: '0',
      zIndex: '10',
    })

    // hide the preview div and append the editor container
    this.inner.style.display = 'none'
    this.preview_widget.element.appendChild(this.quickEditorContainer)

    // init ace
    this.quickEditor = ace.edit(this.quickEditorContainer, {
      mode: 'ace/mode/markdown',
      theme: this.properties.theme,
    })

    this.quickEditor.setOptions({
      autoScrollEditorIntoView: true,
      copyWithEmptySelection: true,
      hasCssTransforms: true,
    })
    this._setupEditor(this.quickEditor)

    this.quickEditor.setValue(this._raw_html, -1)
    this.quickEditor.focus()

    this.quickEditor.session.on('change', () => {
      this.updateHTML(this.quickEditor.getValue())
    })

    this.quickEditor.textInput
      .getElement()
      .addEventListener('blur', () => this.exitEditMode(true))

    this.quickEditor.textInput.getElement().addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        this.exitEditMode(true)
      } else if (e.key === 'Escape') {
        e.preventDefault()
        this.exitEditMode(false)
      }
    })

    this.quickEditor.resize()
    this.setDirtyCanvas(true, true)
  }

  /**
   * Switches the editable textarea back to the preview div.
   * @param {boolean} saveChanges - Whether to save the textarea content.
   */
  exitEditMode(saveChanges) {
    if (!this.isEditing) {
      return
    }

    this.isEditing = false
    this.setupDebounce(100)

    if (this.quickEditor) {
      if (saveChanges) {
        const newValue = this.quickEditor.getValue()
        if (newValue !== this._raw_html) {
          this._raw_html = newValue
          this.updateView()
        }
      }
      this.quickEditor.destroy()
      this.quickEditor = null
    }

    if (this.quickEditorContainer?.parentNode) {
      this.quickEditorContainer.parentNode.removeChild(
        this.quickEditorContainer,
      )
      this.quickEditorContainer = null
    }

    this.inner.style.display = ''
    this.setDirtyCanvas(true, true)
    this.calculateHeight()
  }
}

app.registerExtension({
  name: 'mtb.noteplus',
  settings: [
    {
      id: 'mtb.noteplus.use-shiki',
      category: ['mtb', 'Note+', 'use-shiki'],
      name: 'Use shiki to highlight code',
      tooltip:
        'This will load a larger version of mtb/markdown-parser that bundles shiki, it supports all shiki transformers (supported langs: html,css,python,markdown)',

      type: 'boolean',
      defaultValue: false,
      attrs: {
        style: {
          // fontFamily: 'monospace',
        },
      },
    },
    {
      id: 'mtb.noteplus.css-reset',
      category: ['mtb', 'Note+', 'css-reset'],
      name: 'CSS reset',
      tooltip: 'This is prepended to all notes',
      type: 'string',
      defaultValue: CSS_RESET,
      attrs: {
        style: {
          // fontFamily: 'monospace',
        },
      },
      async onChange(value) {
        _css_reset = value
      },
    },
  ],

  registerCustomNodes() {
    LiteGraph.registerNodeType('Note Plus (mtb)', NotePlus)

    NotePlus.category = 'mtb/utils'
    NotePlus.title = 'Note+ (mtb)'
  },
})
