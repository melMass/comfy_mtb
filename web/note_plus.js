import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'
import * as shared from './comfy_shared.js'

class NotePlus extends LiteGraph.LGraphNode {
  title = 'Note+ (mtb)'
  category = 'mtb/utils'

  constructor() {
    super()

    this.isVirtualNode = true
    this.serialize_widgets = true

    this.editing = false
    this.live = true
    this.rawVal = "<p style='color:red;font-family:monospace'\n>    Note+\n</p>"

    this.calculated_height = 36

    const inner = document.createElement('div')
    inner.style.margin = '0'
    inner.style.padding = '0'
    this.html_widget = this.addDOMWidget('HTML', 'html', inner, {
      setValue: (v) => {
        // update our widget preview
        this.html_widget.element.innerHTML = v
        // calculate height
        this.calculated_height = this.html_widget.element.scrollHeight + 36
      },
      getValue: () => this.rawVal,
      getMinHeight: () => this.calculated_height, // (the edit button),
    })

    // console.log(`Value of HTML: ${this.html_widget.value}`)
    this.html_widget.element.innerHTML = this.html_widget.value

    //- ace based editor
    this.addWidget('button', 'Edit', 'Edit', () => {
      const container = document.createElement('div')
      Object.assign(container.style, {
        display: 'flex',
        gap: '10px',
      })

      dialog.show('')
      dialog.textElement.append(container)

      const value = document.createElement('div')
      value.id = 'noteplus-editor'
      Object.assign(value.style, {
        width: '300px',
        height: '200px',
        backgroundColor: 'rgb(30,30,30)',
        color: 'whitesmoke',
      })

      container.append(value)

      const live_edit = document.createElement('input')
      live_edit.type = 'checkbox'
      live_edit.checked = this.live
      live_edit.onchange = () => {
        this.live = live_edit.checked
      }

      const live_edit_label = document.createElement('label')
      live_edit_label.textContent = 'Live Edit'
      live_edit_label.append(live_edit)

      value.after(live_edit_label)

      this.setupEditor()
      this.editor.setValue(this.html_widget.element.innerHTML)
    })

    const dialog = new app.ui.dialog.constructor()
    dialog.element.classList.add('comfy-settings')

    const closeButton = dialog.element.querySelector('button')
    closeButton.textContent = 'CANCEL'
    const saveButton = document.createElement('button')
    saveButton.textContent = 'SAVE'
    saveButton.onclick = () => {
      this.updateHTML(this.editor.getValue())

      this.editor.destroy()
      this.editor.container.remove()

      dialog.close()
    }

    closeButton.before(saveButton)

    shared
      .loadScript(
        'https://cdn.jsdelivr.net/npm/ace-builds@1.16.0/src-min-noconflict/ace.min.js'
      )
      .catch((e) => {
        console.error(e)
      })
  }

  setupEditor() {
    this.editor = ace.edit('noteplus-editor')
    this.editor.setTheme('ace/theme/dracula')
    this.editor.session.setMode('ace/mode/html')

    this.editor.setShowPrintMargin(false)
    this.editor.session.setUseWrapMode(true)
    this.editor.renderer.setShowGutter(false)
    this.editor.session.setTabSize(4)
    this.editor.session.setUseSoftTabs(true)
    this.editor.setFontSize(14)
    this.editor.setReadOnly(false)
    this.editor.setHighlightActiveLine(false)
    this.editor.setShowFoldWidgets(true)

    this.editor.session.on('change', (delta) => {
      // delta.start, delta.end, delta.lines, delta.action
      if (this.live) {
        this.updateHTML(this.editor.getValue())
      }
    })
  }

  updateHTML(val) {
    // if (CONTAINER_HTML.includes('${html}')) {
    //   console.log('found template')
    //   val = CONTAINER_HTML.replace('${html}', val)
    // }

    this.html_widget.value = val
    this.rawVal = val

    this.calculated_height = this.html_widget.element.scrollHeight

    this.setSize(this.computeSize())
  }

  // //   onRemoved() {
  // //     console.log('Removing', this)
  // //     for (const w of this.widgets) {
  // //       console.log('Removing', w)
  // //       w.onRemove?.()
  // //       w.onRemoved?.()
  // //     }
  // //   }
}

app.registerExtension({
  name: 'mtb.noteplus',

  setup() {
    // app.ui.settings.addSetting({
    //     id: "mtb.noteplus.Container",
    //     name: "📦 HTML container",
    //     type: "text",
    //     defaultValue: "<div>${html}</div>",
    //     tooltip:
    //         "This defines the wrapper for the noteplus html content, use '${html}' to define the location of the placeholder",
    //     attrs: {
    //         style: {
    //             fontFamily: "monospace",
    //         },
    //     },
    //     onChange(value) {
    //         if (!value) {
    //             CONTAINER_HTML = null;
    //             return;
    //         }
    //         console.log(`NOTEPLUS| value changed: ${value}`)
    //         CONTAINER_HTML = value
    //     },
    // });
  },

  registerCustomNodes() {
    LiteGraph.registerNodeType('Note Plus (mtb)', NotePlus)
  },
})
