import { app } from '../../scripts/app.js'
import * as shared from './comfy_shared.js'
import {
  infoLogger,
  warnLogger,
  successLogger,
  errorLogger,
} from './comfy_shared.js'

const DEFAULT_CSS = ``
const DEFAULT_HTML = `<p style='color:red;font-family:monospace'>
    Note+
</p>`
const DEFAULT_MD = '## Note+'
const DEFAULT_MODE = 'markdown'

const CSS_RESET = `
* {
  font-family: monospace;
  line-height: 1.25em;
}

h1, h2, h3, h4, h5, h6 {
  margin: 0;
  padding: 0;
  font-weight: normal;
}

p, ul, ol, dl, blockquote {
  margin: 0.3em;
  padding: 0;
}


ul, ol {
  
  padding-left: 1em;
  
}

a {
  color: inherit;
  text-decoration: none;
  pointer-events: all;
  color: cyan;
}

img {
  padding: 1em 0;
  max-width: 100%;
}

iframe {
  width: 100%;
  height: auto;
  border:none;
  pointer-events:all;
}

blockquote {
  border-left: 4px solid #ccc;
  padding-left: 1em;
  margin-left: 0;
  font-style: italic;
}

pre, code {
  font-family: monospace;
}

table {
  border-collapse: collapse;
  width: 100%;
  border-bottom: 1px solid #000;
  margin: 1em 0;
}

th, td {
  border-left: 1px solid #000;
  border-right: 1px solid #000;
  padding: 8px;
  text-align: left;
}

th {
  border: 1px solid #000;

  background-color: rgba(0,0,0,0.5);
}

input[type="checkbox"] {
  margin-right: 10px;
}

`
class NotePlus extends LiteGraph.LGraphNode {
  title = 'Note+ (mtb)'
  category = 'mtb/utils'

  // same values as the comfy note
  color = LGraphCanvas.node_colors.yellow.color
  bgcolor = LGraphCanvas.node_colors.yellow.bgcolor
  groupcolor = LGraphCanvas.node_colors.yellow.groupcolor

  constructor() {
    super()
    this.uuid = shared.makeUUID()

    infoLogger(`Constructing Note+ instance`)
    // - litegraph settings
    this.collapsable = true
    this.isVirtualNode = true
    this.shape = LiteGraph.BOX_SHAPE
    this.serialize_widgets = true

    // - default values, serialization is done through widgets
    this._raw_html = DEFAULT_MODE == 'html' ? DEFAULT_HTML : DEFAULT_MD

    // - mardown converter
    this.markdownConverter = new showdown.Converter({
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

    // - state
    this.live = true
    this.calculated_height = 0

    // - add widgets
    const inner = document.createElement('div')
    inner.style.margin = '0'
    inner.style.padding = '0'
    inner.style.pointerEvents = 'none'
    this.html_widget = this.addDOMWidget('HTML', 'html', inner, {
      setValue: (val) => {
        this._raw_html = val
      },
      getValue: () => this._raw_html,
      getMinHeight: () => this.calculated_height, // (the edit button),
      hideOnZoom: false,
    })

    this.setupSerializationWidgets()
    this.setupDialog()
    this.loadAceEditor()
  }

  /**
   *
   * @param {CanvasRenderingContext2D} ctx
   * @param {LGraphCanvas} graphcanvas
   * @returns
   */

  onDrawForeground(ctx, graphcanvas) {
    if (this.flags.collapsed) return

    // Define the size and position of the icon
    const iconSize = 14 // Size of the icon
    const iconMargin = 8 // Margin from the edges
    const x = this.size[0] - iconSize - iconMargin
    const y = iconMargin * 1.5

    // Create a new Path2D object from SVG path data
    const pencilPath = new Path2D(
      'M21.28 6.4l-9.54 9.54c-.95.95-3.77 1.39-4.4.76-.63-.63-.2-3.45.75-4.4l9.55-9.55a2.58 2.58 0 1 1 3.64 3.65z'
    )
    const folderPath = new Path2D(
      'M11 4H6a4 4 0 0 0-4 4v10a4 4 0 0 0 4 4h11c2.21 0 3-1.8 3-4v-5'
    )

    // Draw the paths
    ctx.save()
    ctx.translate(x, y) // Position the icon on the canvas
    ctx.scale(iconSize / 32, iconSize / 32) // Scale the icon to the desired size
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'

    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    ctx.lineWidth = 2.4
    ctx.stroke(pencilPath)
    ctx.stroke(folderPath)
    ctx.restore()
  }
  onMouseDown(e, localPos, graphcanvas) {
    // Check if the click is within the pencil icon bounds
    const iconSize = 14
    const iconMargin = 8
    const iconX = this.size[0] - iconSize - iconMargin
    const iconY = iconMargin * 1.5

    if (
      localPos[0] > iconX &&
      localPos[0] < iconX + iconSize &&
      localPos[1] > iconY &&
      localPos[1] < iconY + iconSize
    ) {
      // Pencil icon was clicked, open the editor
      this.openEditorDialog()
      return true // Return true to indicate the event was handled
    }

    return false // Return false to let the event propagate
  }

  setupSerializationWidgets() {
    infoLogger(`Setup Serializing widgets`)

    this.edit_mode_widget = this.addWidget(
      'combo',
      'Mode',
      DEFAULT_MODE,
      (me) => successLogger(`Updating edit_mode`, me),
      {
        values: ['html', 'markdown', 'raw'],
      }
    )

    this.css_widget = this.addWidget('text', 'CSS', DEFAULT_CSS, (val) => {
      successLogger(`Updating css ${val}`)
    })

    shared.hideWidgetForGood(this, this.edit_mode_widget)
    shared.hideWidgetForGood(this, this.css_widget)
  }
  setupDialog() {
    infoLogger(`Setup dialog`)
    // this.addWidget('button', 'Edit', 'Edit', this.openEditorDialog.bind(this))

    this.dialog = new app.ui.dialog.constructor()
    this.dialog.element.classList.add('comfy-settings')

    const closeButton = this.dialog.element.querySelector('button')
    closeButton.textContent = 'CANCEL'
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
    infoLogger(`Closing editor dialog`, accept)
    if (accept) {
      this.updateHTML(this.html_editor.getValue())
      this.updateCSS(this.css_editor.getValue())
    }
    this.teardownEditors()
    this.dialog.close()
  }

  openEditorDialog() {
    infoLogger(`Current edit mode ${this.edit_mode_widget.value}`)
    const container = document.createElement('div')

    Object.assign(container.style, {
      display: 'flex',
      gap: '10px',
      flexDirection: 'column',
    })

    const editorsContainer = document.createElement('div')
    Object.assign(editorsContainer.style, {
      display: 'flex',
      gap: '10px',
      flexDirection: 'row',
    })

    container.append(editorsContainer)

    this.dialog.show('')
    this.dialog.textElement.append(container)

    const aceHTML = document.createElement('div')
    aceHTML.id = 'noteplus-html-editor'
    Object.assign(aceHTML.style, {
      width: '300px',
      height: '300px',
      backgroundColor: 'rgb(30,30,30)',
      color: 'whitesmoke',
    })

    editorsContainer.append(aceHTML)

    const aceCSS = document.createElement('div')
    aceCSS.id = 'noteplus-css-editor'
    Object.assign(aceCSS.style, {
      width: '300px',
      height: '300px',
      backgroundColor: 'rgb(30,30,30)',
      color: 'whitesmoke',
    })

    editorsContainer.append(aceCSS)

    const live_edit = document.createElement('input')
    live_edit.type = 'checkbox'
    live_edit.checked = this.live
    live_edit.onchange = () => {
      this.live = live_edit.checked
    }

    //- "Dynamic" elements
    const firstButton = this.dialog.element.querySelector('button')
    const syncUI = () => {
      let convert_to_html =
        this.dialog.element.querySelector('#convert-to-html')
      if (this.edit_mode_widget.value == 'markdown') {
        if (convert_to_html == null) {
          convert_to_html = document.createElement('button')
          convert_to_html.textContent = 'Convert to HTML (NO UNDO!)'
          convert_to_html.id = 'convert-to-html'
          convert_to_html.onclick = () => {
            let select_mode = this.dialog.element.querySelector('#edit_mode')

            let md = this.html_editor.getValue()
            this.edit_mode_widget.value = 'html'
            select_mode.value = 'html'
            const html = this.markdownConverter.makeHtml(md)
            this.html_widget.value = html
            this.html_editor.setValue(html)
            this.html_editor.session.setMode('ace/mode/html')
            this.updateHTML(this.html_widget.value)

            convert_to_html.remove()
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
    }
    //- combobox
    let select_mode = this.dialog.element.querySelector('#edit_mode')

    if (!select_mode) {
      infoLogger(`Creating combobox for select`)
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

      // firstButton.before(select_mode)
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
    shared
      .loadScript(
        'https://cdn.jsdelivr.net/npm/ace-builds@1.16.0/src-min-noconflict/ace.min.js'
      )
      .catch((e) => {
        errorLogger(e)
      })
  }
  onCreate() {
    errorLogger('NotePlus onCreate')
  }
  configure(info) {
    super.configure(info)
    infoLogger('Restoring serialized values', info)
    // - update view from serialzed data
    this.html_widget.element.id = `note-plus-${this.uuid}`
    this.setMode(this.edit_mode_widget.value)
    this.updateHTML(this.html_widget.value)
    this.updateCSS(this.css_widget.value)
    this.setSize(info.size)
  }
  onNodeCreated() {
    infoLogger('Node created', this.uuid)
    this.html_widget.element.id = `note-plus-${this.uuid}`
    this.setMode(this.edit_mode_widget.value)
    this.updateHTML(this.html_widget.value) // widget is populated here since we called super
    this.updateCSS(this.css_widget.value)
  }
  onRemoved() {
    infoLogger('Node removed', this.uuid)
  }
  getExtraMenuOptions() {
    var options = []
    // {
    //       content: string;
    //       callback?: ContextMenuEventListener;
    //       /** Used as innerHTML for extra child element */
    //       title?: string;
    //       disabled?: boolean;
    //       has_submenu?: boolean;
    //       submenu?: {
    //           options: ContextMenuItem[];
    //       } & IContextMenuOptions;
    //       className?: string;
    //   }
    options.push({
      content: `Set to ${
        this.edit_mode_widget.value === 'html' ? 'markdown' : 'html'
      }`,
      callback: () => {
        this.edit_mode_widget.value =
          this.edit_mode_widget.value === 'html' ? 'markdown' : 'html'
        this.updateHTML(this.html_widget.value)
      },
    })

    return options
  }

  _setupEditor(editor) {
    editor.setTheme('ace/theme/dracula')

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

    this.css_editor.session.on('change', (delta) => {
      // delta.start, delta.end, delta.lines, delta.action
      if (this.live) {
        this.updateCSS(this.css_editor.getValue())
      }
    })

    this.html_editor.session.on('change', (delta) => {
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
      this.html_widget.element
    )
    this.setDirtyCanvas(true, true)
  }
  updateCSS(css) {
    infoLogger('NotePlus updateCSS')
    // this.html_widget.element.style = css
    const scopedCss = this.scopeCss(
      `${CSS_RESET}\n${css}`,
      `note-plus-${this.uuid}`
    )

    const cssDom = this.getCssDom()
    cssDom.innerHTML = scopedCss

    this.css_widget.value = css
    this.calculateHeight()
    infoLogger('NotePlus updateCSS', this.calculated_height)
    // this.setSize(this.computeSize())
  }

  updateHTML(val) {
    const cleanHTML = DOMPurify.sanitize(val, { ADD_TAGS: ['iframe'] })
    this.html_widget.value = cleanHTML

    // update our widget preview
    if (this.edit_mode_widget.value === 'html') {
      this.html_widget.element.innerHTML = cleanHTML
    } else if (this.edit_mode_widget.value === 'markdown') {
      this.html_widget.element.innerHTML =
        this.markdownConverter.makeHtml(cleanHTML)
    }
    this.calculateHeight()
    // this.setSize(this.computeSize())
  }
}

app.registerExtension({
  name: 'mtb.noteplus',

  registerCustomNodes() {
    LiteGraph.registerNodeType('Note Plus (mtb)', NotePlus)

    NotePlus.title_mode = LiteGraph.NO_TITLE
  },
})
